#!/usr/bin/env python3
"""
Frames arriving on a USB serial port, from whatever is on the other end.

Two kinds of device can be plugged into this, and they are not equivalent:

**A CSI receiver** (an ESP32 running esp-csi, or anything else that can quote
per-subcarrier channel state) sends the same numbers a WiFi card would, so the
whole pipeline applies unchanged, Doppler included.

**A sensor node** — an Arduino Mega 2560, say — sends something else entirely.
An ATmega2560 has no radio at all: no WiFi, no CSI, no way to see the channel
state of anything. What it has is a 10-bit ADC and sixteen analog pins, so what
it can report is a *profile of sensor channels* — light falling on an LDR, sound
on a piezo, distance from an ultrasonic sensor, the capacitance of a wire with a
hand near it. That profile has the same shape as a CSI frame (one amplitude per
channel, sampled twenty times a second), so the detector's metric, calibration,
hysteresis and presence logic all apply: the metric asks "how much did the
profile change", and a profile that changes because someone walked in front of a
sensor is a real answer to a real question.

What does *not* carry over is the Doppler axis. For CSI, the metric's spectrum
peaks at `speed / wavelength`, so 1.88 Hz means 23 cm/s. For a channel profile
the same spectrum measures how fast the *profile* is changing, which depends on
where the sensors point and how much of the room each one sees — not on how fast
anybody is walking. `SerialSource.doppler` is therefore False for a node, and the
views say so instead of printing a number that would look like a speed.

The line protocol, one sample per line, ASCII, `\\n` terminated:

    S,<millis>,<v0>,<v1>,...,<vN>      a sensor node: N+1 integer channel values
    C,<millis>,<re0>,<im0>,...         a CSI receiver: int16 real/imag pairs
    # free text                        ignored, so firmware can print a banner

`millis` is the device's own clock, which is the only clock that knows when a
sample was actually taken. It arrives as an unsigned 32-bit count and wraps every
49.7 days; the wrap is undone here rather than being allowed to jump the session
backwards, because a timestamp that goes backwards breaks calibration.

Nothing in this module imports pyserial until a real port is opened, so the
tests run on a string, and the rest of the package keeps its promise of having no
dependencies.
"""

import io
import time
from collections import deque

from wifi_radar import csi

#: What the included sketch and most CSI loggers print at. The Mega's USART0 is
#: happy well past this; 115200 is the rate every Arduino serial monitor defaults
#: to, which is the one that has to work without being explained.
DEFAULT_BAUD = 115200

#: Give up after this long with nothing readable on the port.
DEFAULT_TIMEOUT = 10.0

#: A port that sends nothing but noise is misconfigured, not quiet. Same rule as
#: the nexmon source: stop after this many rejected lines rather than idling.
MAX_REJECTED = 50

#: `millis()` is a uint32; this is how far it can go before it wraps.
MILLIS_WRAP = 1 << 32

#: USB vendor IDs that turn up on the end of a cable with an Arduino on it.
#: 0x1A86 is WCH, whose CH340 is on every Elegoo clone and on most cheap
#: Mega/Nano boards, so a Mega bought as "Elegoo" does not identify as Arduino
#: at all — which is exactly why looking for the word "Arduino" is not enough.
BOARD_VIDS = {0x1A86, 0x2341, 0x2A03, 0x1B4F}


def parse_line(line):
    """
    One line from the port as (kind, millis, values), or None for a comment.

    `kind` is "channels" for an `S` line and "csi" for a `C` line; `values` is a
    list of complex numbers either way, so the detector never learns which kind
    it got. A malformed line raises ValueError with the shape it expected, which
    is what makes a wrong baud rate or a stray `Serial.println("hello")` in the
    sketch show up as a counted rejection instead of a mystery.
    """
    text = line.strip()
    if not text or text.startswith("#") or text.startswith("//"):
        return None

    fields = text.split(",")
    kind = fields[0].strip()
    if kind not in ("S", "C"):
        raise ValueError(
            f"a line starting {text[:12]!r} is neither a sensor sample (S,) "
            f"nor CSI (C,)")
    if len(fields) < 3:
        raise ValueError(f"a {kind} line needs a timestamp and at least one "
                         f"value, got {text[:40]!r}")

    try:
        millis = int(fields[1])
    except ValueError:
        raise ValueError(f"timestamp {fields[1]!r} is not an integer "
                         f"(check the line ends with a newline)") from None
    if millis < 0:
        raise ValueError(f"timestamp {millis} is negative")
    millis %= MILLIS_WRAP

    try:
        numbers = [int(f) for f in fields[2:]]
    except ValueError:
        bad = next(f for f in fields[2:] if not _is_int(f))
        raise ValueError(f"{bad!r} is not an integer value") from None

    if kind == "S":
        return "channels", millis, [complex(v, 0) for v in numbers]

    if len(numbers) % 2:
        raise ValueError(
            f"a C line carries real/imaginary pairs, and {len(numbers)} values "
            f"is an odd number of them")
    return "csi", millis, [complex(numbers[i], numbers[i + 1])
                           for i in range(0, len(numbers), 2)]


def _is_int(text):
    try:
        int(text)
        return True
    except ValueError:
        return False


class SerialSource(csi.Source):
    """
    A stream of frames from a text stream of protocol lines.

    The stream is anything with `readline()` returning bytes or str: a pyserial
    `Serial`, or a `StringIO` in the tests. Nothing here needs a COM port to be
    exercised, which is the only reason a bring-up failure can be told apart from
    a signal-processing failure.

    A file-like stream ends at EOF — that is a saved capture being replayed —
    while a port that goes quiet raises TimeoutError. Both return an empty read,
    and the difference matters: a replay that stopped should stop, and a board
    that stopped talking should say so.
    """

    def __init__(self, stream, port="?", rate=20.0, timeout=DEFAULT_TIMEOUT,
                 baud=DEFAULT_BAUD, clock=time.monotonic):
        self.stream = stream
        self.port = port
        self.baud = baud
        self.rate = float(rate)
        self.length = None
        self.timeout = timeout
        self.frames = 0
        self.rejected = 0
        self.rejection = None
        self.channels = None
        self.doppler = False           # becomes True on the first CSI line
        self._clock = clock
        self._last_line = clock()
        self._t0 = None
        self._last_ms = None
        self._wrap = 0
        self._arrivals = deque(maxlen=64)

    # ------------------------------------------------------------- state
    @property
    def measured_rate(self):
        """Samples per second as they actually arrived, or None below two."""
        if len(self._arrivals) < 2:
            return None
        span = self._arrivals[-1] - self._arrivals[0]
        return (len(self._arrivals) - 1) / span if span > 0 else None

    def describe(self):
        if self.channels is None:
            kind = "waiting for lines"
        elif self.doppler:
            kind = f"CSI, {self.channels // 2} tones"
        else:
            kind = f"{self.channels} sensor channels, no Doppler"
        rate = self.measured_rate
        arrived = "rate not measured yet" if rate is None else f"{rate:.0f}/s"
        return (f"serial {self.port} at {self.baud} baud, {kind}, {arrived}"
                + (f", {self.rejected} unreadable lines" if self.rejected else ""))

    def close(self):
        """Release the port. A stream that cannot be closed is not an error."""
        try:
            self.stream.close()
        except (AttributeError, OSError):
            pass

    # ------------------------------------------------------------ stream
    def __iter__(self):
        return self

    def __next__(self):
        while True:
            raw = self.stream.readline()
            if not raw:
                if isinstance(self.stream, io.IOBase):
                    raise StopIteration
                self._check_idle()
                continue

            if isinstance(raw, bytes):
                try:
                    text = raw.decode("utf-8", "replace")
                except Exception:                       # pragma: no cover
                    text = ""
            else:
                text = raw

            try:
                parsed = parse_line(text)
            except ValueError as error:
                self.rejected += 1
                if self.rejection is None:
                    self.rejection = str(error)
                if self.frames == 0 and self.rejected >= MAX_REJECTED:
                    raise ValueError(
                        f"{self.rejected} lines arrived on {self.port} and none "
                        f"was a sample; last reason: {error}. Check the baud "
                        f"rate ({DEFAULT_BAUD} in the sketch and here) and that "
                        f"the firmware prints S, or C, lines. See README.md."
                    ) from None
                continue

            if parsed is None:
                self._last_line = self._clock()
                continue

            kind, millis, values = parsed
            if kind == "csi":
                self.doppler = True
            self.channels = len(values)

            # Undo the 32-bit wrap: a jump of more than half the range backwards
            # is the counter rolling over, not a device that went back in time.
            if self._last_ms is not None and millis < self._last_ms - MILLIS_WRAP // 2:
                self._wrap += MILLIS_WRAP
            self._last_ms = millis
            absolute = millis + self._wrap

            if self._t0 is None:
                self._t0 = absolute
            now = self._clock()
            self._last_line = now
            self._arrivals.append(now)
            self.frames += 1
            return csi.Frame((absolute - self._t0) / 1000.0, values)

    def _check_idle(self):
        """
        An empty read means either a timeout on the port or the end of a file.

        Both are silence, and silence from a board that is meant to be talking is
        the failure this message exists for: it names the three things that are
        actually wrong when a Mega is plugged in and nothing appears.
        """
        waited = self._clock() - self._last_line
        if waited < self.timeout:
            return
        raise TimeoutError(
            f"nothing readable on {self.port} for {waited:.0f}s. Check that the "
            f"sketch is running (the Mega's reset button, then watch its TX LED), "
            f"that the port is the board's (Windows: Device Manager > Ports > "
            f"USB-SERIAL CH340; Linux: /dev/ttyACM0 or /dev/ttyUSB0), and that no "
            f"serial monitor is holding the port open.")


def parse_spec(rest):
    """
    Split the tail of a `serial:` spec into (port, baud).

    `serial:COM3` and `serial:/dev/ttyACM0` use the default baud; an explicit
    `serial:COM3:9600` overrides it. Only a trailing numeric field is read as a
    baud rate, because a Windows port name has a colon in it on no platform and
    a pathspec may.
    """
    if not rest:
        raise ValueError(
            "serial: needs a port, e.g. serial:COM3 on Windows or "
            "serial:/dev/ttyACM0 on Linux")
    if rest.startswith(":"):
        raise ValueError(f"serial:{rest} has no port before the baud rate")
    port, baud = rest, DEFAULT_BAUD
    head, _, tail = rest.rpartition(":")
    if head and tail.isdigit():
        port, baud = head, int(tail)
    if not port:
        raise ValueError(f"serial:{rest} has no port before the baud rate")
    return port, baud


def identify_port(port):
    """
    (confidence, reason) for one port: how likely a board is on the other end.

    A heuristic, not a promise — a CH340 is a USB-to-serial chip that many
    things use, and the built-in port on a desktop is never a board. The reason
    is returned with the number so the answer can be shown rather than trusted.
    """
    vid = getattr(port, "vid", None)
    description = getattr(port, "description", "") or ""
    hwid = getattr(port, "hwid", "") or ""
    if vid == 0x1A86:
        return 3, "CH340 USB-serial (Elegoo and most clone boards)"
    if vid in BOARD_VIDS:
        return 3, "Arduino USB-serial"
    if "arduino" in description.lower() or "arduino" in hwid.lower():
        return 2, "described as an Arduino"
    if vid is not None:
        return 1, "USB serial, but not a known board chip"
    return 0, "not a USB serial port"


def choose_port(ports):
    """
    The port most likely to be talking, or None when there are none.

    Ties break on the port name so the choice is stable between runs: a script
    that opens the board twice should not pick different ports when both are
    equally plausible.
    """
    ranked = sorted(ports, key=lambda p: (-identify_port(p)[0], p.device))
    return ranked[0] if ranked else None


def available_ports():
    """
    Every serial port the operating system reports, sorted by name.

    pyserial is needed to enumerate them for the same reason it is needed to
    open one; the message says so rather than returning an empty list, which
    would be indistinguishable from a machine with nothing plugged in.
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        raise OSError(
            "listing ports needs pyserial, which is not installed. Run "
            "`python -m pip install pyserial` and try again; the simulator, "
            "recordings and nexmon need nothing."
        ) from None
    return sorted(list_ports.comports(), key=lambda p: p.device)


def open_serial(port, baud=DEFAULT_BAUD, timeout=0.2):
    """
    Open a real port, or say precisely why not.

    pyserial is imported here rather than at module import: it is the only
    third-party dependency this package would have, and everything except the
    last few centimetres of cable works without it.
    """
    try:
        import serial
    except ImportError:
        raise OSError(
            f"opening {port} needs pyserial, which is not installed. Run "
            f"`python -m pip install pyserial` and try again; the simulator, "
            f"recordings and nexmon need nothing."
        ) from None

    try:
        return serial.Serial(port, baud, timeout=timeout)
    except (serial.SerialException, OSError, ValueError) as error:
        raise OSError(
            f"cannot open {port} at {baud} baud: {error}. On Windows the board "
            f"appears as a COM port only after the CH340 driver is installed; "
            f"on Linux, check that your user is in the dialout group and that no "
            f"serial monitor already holds the port."
        ) from None


def describe_ports(ports, chosen=None):
    """A table for a person staring at Device Manager, one line per port."""
    if not ports:
        return ("no serial ports found. If the board is plugged in, Windows needs "
                "the CH340 driver before a COM port exists at all; on Linux, "
                "check `dmesg | tail` after plugging it in.")
    lines = [f"{len(ports)} serial port{'s' if len(ports) != 1 else ''}:"]
    for port in ports:
        confidence, reason = identify_port(port)
        mark = "*" if chosen is not None and port.device == chosen.device else " "
        line = f" {mark} {port.device:<14} {reason}"
        extra = " ".join(part for part in (getattr(port, "description", ""),
                                           getattr(port, "hwid", "")) if part)
        if extra and extra not in reason:
            line += f"  [{extra}]"
        lines.append(line)
    if chosen is not None:
        lines.append("* = what --port auto would open")
    return "\n".join(lines)


def main(argv=None):
    """What is arriving on the port? The first question on a new board."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Read sample lines from a serial port and describe them.",
        epilog="The firmware prints one line per sample: S,<millis>,<values> "
               "for channels, C,<millis>,<re>,<im>,... for CSI. Stop it before "
               "opening the port in the Arduino IDE's serial monitor.")
    parser.add_argument("--port", default="auto",
                        help="COM3 on Windows, /dev/ttyACM0 on Linux; "
                             "auto (the default) picks the likely board")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="how long to listen; 0 waits forever")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="give up after this long with nothing readable")
    parser.add_argument("--list", action="store_true",
                        help="list the serial ports and exit")
    args = parser.parse_args(argv)

    if args.list:
        try:
            ports = available_ports()
        except OSError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(describe_ports(ports, choose_port(ports)))
        return 0

    port = args.port
    chosen = None
    if port.strip().lower() == "auto":
        try:
            chosen = choose_port(available_ports())
        except OSError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        if chosen is None:
            print("error: no serial port to open. Plug the board in and check "
                  "that a COM port appears (Windows needs the CH340 driver for "
                  "Elegoo boards); `--list` shows what was found.",
                  file=sys.stderr)
            return 2
        port = chosen.device
        confidence, reason = identify_port(chosen)
        print(f"opening {port}: {reason}", flush=True)

    try:
        stream = open_serial(port, args.baud)
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    source = SerialSource(stream, port=port, timeout=args.timeout,
                          baud=args.baud)
    print(f"listening on {port} at {args.baud} baud", flush=True)
    started = time.monotonic()
    shown = 0
    try:
        for frame in source:
            if shown < 3:
                shown += 1
                first = " ".join(f"{v.real:.0f}" for v in frame.values[:8])
                print(f"  t={frame.t:7.2f}s  {len(frame.values):3d} values  "
                      f"{'CSI' if source.doppler else 'channels'}  {first}",
                      flush=True)
            if args.seconds and time.monotonic() - started >= args.seconds:
                break
    except TimeoutError as error:
        print(f"\nerror: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        # A port that is open but speaking a different language than the
        # protocol should say so without a traceback, like every other source.
        print(f"\nerror: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        source.close()

    rate = source.measured_rate
    print(f"\nframes          : {source.frames}")
    print(f"kind            : {'CSI' if source.doppler else 'channel profile'}")
    print(f"measured rate   : {'n/a' if rate is None else f'{rate:.1f}/s'}")
    print(f"unreadable      : {source.rejected}"
          + (f" ({source.rejection})" if source.rejection else ""))
    if source.frames == 0:
        print("no samples were received. The sketch must be running and must "
              "print S, or C, lines; see README.md.")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
