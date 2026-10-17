#!/usr/bin/env python3
"""
Real CSI from a Raspberry Pi running nexmon_csi.

Every other source in this package produces frames from a model or a file. This
one takes them off the wire, from the patched Broadcom firmware that makes the
Pi's own WiFi radio report the channel instead of just decoding it:

    .venv/bin/python -m wifi_radar.nexmon                 # is anything arriving?
    .venv/bin/python -m wifi_radar.watch --source udp:5500
    .venv/bin/python -m wifi_radar.live  --source udp:5500

The extractor sends one UDP packet per received WiFi frame, by default to
10.10.10.10:5500. Each packet is a short header followed by one complex value
per subcarrier, as an int16 real part and an int16 imaginary part, little
endian — 64 tones at 20 MHz, 128 at 40, 256 at 80.

Two details decide whether a packet is read correctly or plausibly wrongly:

- **The header length has changed.** The 18-byte header described by `csiread`
  and the 20-byte one in the current README put the CSI at different offsets,
  so a parser that assumes either one misreads the other into a frame that still
  looks like a frame. Payload length fingerprints it exactly: the CSI is the
  only part that varies, 4 bytes per tone, so 276 bytes can only be a 20-byte
  header and 64 tones, and 274 can only be 18 and 64. Anything else is rejected
  with the accepted sizes rather than half-parsed. The header fields are then
  read backwards from the start of the CSI, which is the one origin both
  revisions agree on.
- **The numbers are not the simulation's numbers.** They are the receiver's
  uncalibrated int16 output, thousands rather than about one, and the automatic
  gain control re-scales them frame to frame. That is exactly what `detect.py`
  was built to survive, and it needs no change: the metric divides each frame by
  its own mean magnitude, so a frame-wide gain change is not movement.

A 2x2 radio sends one packet per core per frame, and interleaving the two
streams looks like violent motion because it is not one channel being sampled
twice, it is two. Only the first core seen is used; the rest are counted, and
`describe()` says so, so a misconfigured extractor is visible rather than
quietly ruining the data.

There is deliberately no dependence on `csiread` or `numpy`: this is 60 lines
of `struct` against a documented layout, and the packet builder in the tests is
what keeps it honest without a Pi on the desk.
"""

import argparse
import socket
import struct
import sys
import time
from collections import deque

from wifi_radar import csi

#: Where `nexmon_csi` broadcasts unless its destination is changed.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5500

#: Everything the extractor emits starts with this. The 20-byte header repeats
#: it to six bytes; the RSSI fork keeps only the first two and puts its fields
#: where the stock header has the rest.
MAGIC = b"\x11\x11"

#: How long to wait for a packet before saying so. A raw socket that blocks
#: forever is indistinguishable from a working radar in a quiet room, and the
#: difference matters on the first run.
DEFAULT_TIMEOUT = 10.0

#: Packets to tolerate before concluding the stream is not CSI at all. Traffic
#: on 5500 is not necessarily ours, but 25 wrong lengths in a row is a
#: misconfigured extractor, not a coincidence.
MAX_REJECTED = 25

#: Payload length -> subcarriers. Derivable (4 bytes per tone on top of a
#: 18- or 20-byte header) but written out so the rejection message can list it.
LAYOUTS = {
    274: 64, 276: 64,        # 20 MHz
    530: 128, 532: 128,      # 40 MHz
    1042: 256, 1044: 256,    # 80 MHz
}

#: Subcarriers -> channel width in MHz.
BANDWIDTH_MHZ = {64: 20, 128: 40, 256: 80}


class Packet:
    """One decoded nexmon frame: its CSI, plus the header fields worth seeing."""

    __slots__ = ("values", "sequence", "core", "chanspec", "chip", "rssi",
                 "header", "bandwidth_mhz")

    def __init__(self, values, sequence, core, chanspec, chip, rssi, header):
        self.values = values
        self.sequence = sequence
        self.core = core
        self.chanspec = chanspec
        self.chip = chip
        self.rssi = rssi
        self.header = header
        self.bandwidth_mhz = BANDWIDTH_MHZ[len(values)]

    def __repr__(self):
        return (f"Packet({len(self.values)} tones, {self.bandwidth_mhz} MHz, "
                f"seq={self.sequence}, core={self.core}, header={self.header}B)")


def parse_payload(payload):
    """
    A UDP payload -> a `Packet`, or `ValueError` explaining why it is not one.

    Strict on purpose: every failure here is a length or magic mismatch, and the
    alternative to raising is reading the wrong 256 bytes as CSI and reporting
    confident nonsense for the rest of the session.
    """
    tones = LAYOUTS.get(len(payload))
    if tones is None:
        raise ValueError(
            f"{len(payload)}-byte payload is not a nexmon CSI frame; expected "
            f"{', '.join(str(n) for n in sorted(LAYOUTS))} bytes "
            f"(64/128/256 tones behind an 18- or 20-byte header)")
    if payload[:2] != MAGIC:
        raise ValueError(
            f"payload does not start with the nexmon magic {MAGIC.hex()}; "
            f"first bytes were {payload[:8].hex()}")

    # The header fields sit at fixed offsets *before* the CSI, and that origin
    # is the same for both header revisions — which is what makes one parser
    # enough for both. Header length falls out of the payload length.
    start = len(payload) - 4 * tones
    sequence, core, chanspec, chip = struct.unpack_from("<HHHH", payload, start - 8)

    # The stock 18- and 20-byte headers both begin with four 0x11 bytes; the
    # RSSI fork begins with two and spends the next byte on a signed RSSI.
    stock = payload[:4] == b"\x11\x11\x11\x11"
    rssi = None if stock else struct.unpack_from("<b", payload, 2)[0]

    raw = struct.unpack_from(f"<{2 * tones}h", payload, start)
    values = [complex(raw[2 * i], raw[2 * i + 1]) for i in range(tones)]
    return Packet(values, sequence, core, chanspec, chip, rssi, start)


class NexmonSource(csi.Source):
    """
    Frames off a UDP socket, timestamped on arrival.

    nexmon puts no host time in the packet and no frame rate in the header — the
    rate is whatever the transmitter's traffic produces — so `t` is measured
    here, from the first packet, and `rate` is the nominal rate the detector
    scales its Doppler axis by. `measured_rate` reports what is actually
    arriving, which is the number to pass as `--rate` if the two disagree.
    """

    def __init__(self, port=DEFAULT_PORT, host=DEFAULT_HOST, rate=20.0,
                 timeout=DEFAULT_TIMEOUT, sock=None, clock=time.monotonic):
        self.host = host
        self.port = int(port)
        self.rate = float(rate)
        self.length = None
        self.timeout = timeout
        self.frames = 0
        self.rejected = 0
        self.other_cores = 0
        self.rejection = None
        self.packet = None              # the last decoded Packet
        self._clock = clock
        self._started = None
        self._arrivals = deque(maxlen=64)
        self._core = None
        self._closed = False
        self.sock = sock if sock is not None else socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM)
        if sock is None:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((host, self.port))
        self.port = self.sock.getsockname()[1]
        self.sock.settimeout(timeout)

    # ------------------------------------------------------------- state
    @property
    def measured_rate(self):
        """Frames per second as they actually arrived, or None below two frames."""
        if len(self._arrivals) < 2:
            return None
        span = self._arrivals[-1] - self._arrivals[0]
        return (len(self._arrivals) - 1) / span if span > 0 else None

    def describe(self):
        where = f"nexmon CSI on {self.host}:{self.port}"
        if self.packet is None:
            return f"{where}, waiting for packets"
        shape = f"{len(self.packet.values)} tones, {self.packet.bandwidth_mhz} MHz"
        rate = self.measured_rate
        arrival = "rate not measured yet" if rate is None else f"{rate:.0f} frames/s"
        return (f"{where}, {shape}, {self.packet.header}-byte header, {arrival}"
                + (f", {self.other_cores} packets on other cores ignored"
                   if self.other_cores else ""))

    def close(self):
        if not self._closed:
            self._closed = True
            self.sock.close()

    # ------------------------------------------------------------ stream
    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                payload, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                raise TimeoutError(
                    f"no CSI packets on {self.host}:{self.port} for "
                    f"{self.timeout:g}s. nexmon_csi extracts one frame per "
                    f"received WiFi frame, so check that it is running on the Pi, "
                    f"that its destination is this host and port, and that "
                    f"something on the channel is transmitting.") from None

            try:
                packet = parse_payload(payload)
            except ValueError as error:
                self.rejected += 1
                if self.rejection is None:
                    self.rejection = str(error)
                if self.frames == 0 and self.rejected >= MAX_REJECTED:
                    raise ValueError(
                        f"{self.rejected} packets arrived on {self.host}:{self.port} "
                        f"and none was a nexmon CSI frame; last reason: {error}"
                    ) from None
                continue

            # One stream, not two: two cores interleaved would look like motion.
            if self._core is None:
                self._core = packet.core
            elif packet.core != self._core:
                self.other_cores += 1
                continue

            now = self._clock()
            if self._started is None:
                self._started = now
            self._arrivals.append(now)
            self.packet = packet
            self.frames += 1
            return csi.Frame(now - self._started, packet.values)


def build_payload(values, sequence=0, core=0, chanspec=0, chip=0, header=18,
                  rssi=None):
    """
    The inverse of `parse_payload`: a payload a real extractor could have sent.

    This exists so the parser is tested against the documented layout rather
    than against itself being plausible, and so the first bring-up can push a
    known frame at a listener and watch it arrive. It is not used at runtime.
    """
    tones = len(values)
    if tones not in BANDWIDTH_MHZ:
        raise ValueError(f"cannot build {tones} tones; expected 64, 128 or 256")
    if header not in (18, 20):
        raise ValueError(f"header must be 18 or 20 bytes, not {header}")
    mac = b"\x02\x00\x00\x00\x00\x01"
    if rssi is None:
        # Stock: four magic bytes, or six in the 20-byte revision, then the MAC.
        prefix = MAGIC * (3 if header == 20 else 2) + mac
    else:
        if header != 18:
            raise ValueError("the RSSI header is the 18-byte one")
        prefix = MAGIC + struct.pack("<b", rssi) + b"\x08" + mac
    # Everything before the CSI is (header) bytes, of which the four fields
    # take the last eight; assert the layout rather than padding it blindly.
    assert len(prefix) == header - 8, (len(prefix), header)
    fields = struct.pack("<HHHH", sequence, core, chanspec, chip)
    body = b"".join(struct.pack("<hh", int(round(v.real)), int(round(v.imag)))
                    for v in values)
    return prefix + fields + body


def main(argv=None):
    """Print what is arriving, which is the first question on a new Pi."""
    parser = argparse.ArgumentParser(
        description="Listen for nexmon CSI packets and describe them.",
        epilog="The Pi must be sending to this host:port. nexmon_csi broadcasts "
               "to 10.10.10.10:5500 unless told otherwise.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="address to bind (default every interface)")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="how long to listen; 0 waits forever")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="give up if nothing arrives for this long")
    parser.add_argument("--show", type=int, default=3,
                        help="packets to print in full")
    args = parser.parse_args(argv)

    try:
        source = NexmonSource(port=args.port, host=args.host, timeout=args.timeout)
    except OSError as error:
        print(f"error: cannot listen on {args.host}:{args.port}: {error}",
              file=sys.stderr)
        return 2

    print(f"listening on {args.host}:{source.port} for nexmon CSI packets",
          flush=True)
    started = time.monotonic()
    shown = 0
    try:
        for frame in source:
            packet = source.packet
            if shown < args.show:
                shown += 1
                rssi = "n/a" if packet.rssi is None else f"{packet.rssi} dBm"
                print(f"  {len(packet.values):3d} tones  {packet.bandwidth_mhz:2d} MHz  "
                      f"{packet.header}B header  seq={packet.sequence:<6d} "
                      f"core={packet.core}  chanspec=0x{packet.chanspec:04x}  "
                      f"chip=0x{packet.chip:04x}  rssi={rssi}  "
                      f"first={frame.values[0]:.0f}", flush=True)
            if args.seconds and time.monotonic() - started >= args.seconds:
                break
    except (TimeoutError, KeyboardInterrupt) as error:
        print(f"\nstopped: {error}" if str(error) else "\nstopped", flush=True)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        source.close()

    rate = source.measured_rate
    print(f"\nframes          : {source.frames}")
    print(f"measured rate   : {'n/a' if rate is None else f'{rate:.1f} frames/s'}")
    print(f"rejected        : {source.rejected}"
          + (f" ({source.rejection})" if source.rejection else ""))
    if source.other_cores:
        print(f"other cores     : {source.other_cores} packets ignored")
    if source.packet is None:
        print("no CSI was received. nexmon sends nothing on an idle channel, and a "
              "stock Pi sends nothing at all.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
