#!/usr/bin/env python3
"""
Channel state information: what a WiFi receiver reports, and where frames come
from.

A CSI frame is one complex number per OFDM subcarrier, sampled once per received
frame. Each number is the sum of every path the signal took to get there:

    H[f] = sum over paths  a_p * exp(-j * 2*pi * f * delay_p)

Two frames a twentieth of a second apart differ because a person moved: a body
in the room is one of those paths, and when it moves by a few centimetres the
path gets longer or shorter, its delay changes, and its phase rotates by whole
radians. That is what makes this measurable at all — a person's movement is
enormous compared with the noise floor of the receiver, even though it is
invisible in RSSI.

Everything downstream sees only a stream of frames. That is deliberate: a
simulation, a recorded session and a receiver on the desk produce the same thing,
so the signal processing is written once and tested without hardware. The
simulation is also what gives the test suite ground truth to check against — the
tests know when a person is walking because they put them there.

The radio parameters are those of an ordinary 2.4 GHz link at 20 MHz: 64
subcarriers, 312.5 kHz apart. That is what an ESP32 reports and what most CSI
tooling defaults to, and the spacing is why this is a movement sensor rather
than a camera: range resolution is `c / (2 * bandwidth)`, so about 7.5 m at
20 MHz. Nothing here can measure distance. Motion is what it can see.

A recording is JSON Lines, one frame per line, which is the format the hardware
adapters are expected to write:

    {"t": 0.05, "csi": [[12, -3], [10, 4], ...]}

Integer pairs are what Espressif's CSI API reports, so a logger can dump them
without converting anything; `read_recording` turns them back into complex
numbers and the rest of the pipeline never learns they were integers.
"""

import json
import math
import random
import time

#: Speed of light, m/s. Used only to turn path lengths into delays.
C = 299_792_458.0

#: 20 MHz channel, 64 subcarriers, on WiFi channel 1.
CARRIER_FREQ = 2.412e9
SUBCARRIER_SPACING = 312_500.0
DEFAULT_SUBCARRIERS = 64

#: Wavelength at the carrier, the scale every Doppler number is measured in.
WAVELENGTH = C / CARRIER_FREQ

#: Fixed point scale for recorded frames.
#:
#: A real CSI logger reports integers (Espressif's API hands back signed bytes),
#: and the absolute scale of those integers is set by automatic gain control, so
#: only the *relative* size of the numbers carries information. Recording at a
#: fixed point rather than a float keeps that property rather than pretending a
#: receiver reports floats, and it is why the round-trip test is strict: 1/64 of
#: an amplitude unit is quantisation noise a real logger would also have.
RECORD_SCALE = 64.0


class Frame:
    """One CSI observation: a timestamp and a complex value per subcarrier."""

    __slots__ = ("t", "values")

    def __init__(self, t, values):
        self.t = t
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def amplitudes(self):
        """Magnitude per subcarrier. Phase is often unusable (see detect.py)."""
        return [abs(v) for v in self.values]

    @property
    def mean_amplitude(self):
        return sum(self.amplitudes()) / max(1, len(self.values))

    def as_record(self):
        """The JSON Lines shape, at the integer precision a logger reports."""
        return {
            "t": round(self.t, 6),
            "csi": [[round(v.real * RECORD_SCALE), round(v.imag * RECORD_SCALE)]
                    for v in self.values],
        }

    def __repr__(self):
        return f"Frame(t={self.t:.3f}, subcarriers={len(self.values)})"


# --------------------------------------------------------------- the room
# A static channel is the direct path plus a few surfaces that reflect. The
# amplitudes are relative and the delays are in nanoseconds of extra path, which
# is the scale that matters: 1 ns is 30 cm, and at 312.5 kHz spacing a 1 ns
# change rotates adjacent subcarriers by about a tenth of a radian.
DEFAULT_STATIC_PATHS = (
    (1.00, 0.0),      # direct
    (0.55, 12.0),     # a wall behind the receiver
    (0.35, 27.0),     # a doorway
    (0.22, 41.0),     # the far wall
)


class Environment:
    """
    A simulated room: a fixed multipath channel, a person who may be moving in
    it, and the impairments real receivers have.

    The impairments matter more than they look. Real CSI frames each carry an
    unknown phase offset (carrier and sampling frequency offset rotate the whole
    frame) and an unknown gain (automatic gain control re-scales it), so a
    detector that used phase, or used absolute amplitude, would report motion
    that is not there. Both are simulated here so the detector can be held to
    surviving them.
    """

    def __init__(self, subcarriers=DEFAULT_SUBCARRIERS, noise=0.02, seed=0,
                 static_paths=DEFAULT_STATIC_PATHS, phase_drift=0.25,
                 gain_drift=0.004, person_gain=0.45):
        self.subcarriers = subcarriers
        self.noise = noise
        self.static_paths = tuple(static_paths)
        self.phase_drift = phase_drift      # radians per frame, a random walk
        self.gain_drift = gain_drift        # fraction per frame, an AGC wander
        self.person_gain = person_gain      # how strongly a body reflects
        self.rng = random.Random(seed)
        self._phase = 0.0
        self._gain = 1.0

    def sample(self, t, displacement=None):
        """
        One frame at time `t`.

        `displacement` is how far the person's reflection path has lengthened,
        in metres, or None when nobody is in the room. Called with a varying
        displacement it produces the Doppler shift for free: the phase of that
        path is recomputed from the current path length every frame, so a path
        that is getting longer rotates with time exactly as physics would.

        The phase of each path is taken at the subcarrier's *absolute*
        frequency, carrier included. That matters more than it looks. At 2.4 GHz
        a path length change of one centimetre rotates the phase by about half a
        radian, which is why WiFi can feel a person move at all; modelling only
        the subcarrier offsets from the carrier — the shape across the band —
        would leave the channel almost motionless and produce a detector that
        works perfectly on a signal with no Doppler in it.
        """
        person_delay = None if displacement is None else displacement / C

        values = []
        for k in range(self.subcarriers):
            freq = CARRIER_FREQ + k * SUBCARRIER_SPACING
            acc = 0j
            for amplitude, delay_ns in self.static_paths:
                acc += amplitude * _exp(-2 * math.pi * freq * delay_ns * 1e-9)
            if person_delay is not None:
                acc += self.person_gain * _exp(-2 * math.pi * freq * person_delay)
            acc += _complex_gauss(self.rng, self.noise)
            values.append(acc)

        # Impairments, applied to the whole frame: an unknown phase rotation and
        # an unknown gain, both wandering.
        self._phase += self.rng.gauss(0.0, self.phase_drift)
        self._gain *= 1.0 + self.rng.gauss(0.0, self.gain_drift)
        rotate = _exp(self._phase) * self._gain
        return Frame(t, [v * rotate for v in values])


def _exp(theta):
    return complex(math.cos(theta), math.sin(theta))


def _complex_gauss(rng, sigma):
    return complex(rng.gauss(0.0, sigma), rng.gauss(0.0, sigma))


# ------------------------------------------------------------ scenarios
# A scenario answers one question: where is the person at time t? Returning None
# means nobody is there. Everything else about a scenario (how long they walk
# for, how fast) falls out of that answer.

def quiet(t):
    """Nobody in the room."""
    return None


def walk_cadence(t, hz=1.8, metres=0.02):
    """A body swaying as its owner takes steps: the cadence to look for."""
    return metres * math.sin(2 * math.pi * hz * t)


def breathe(t, hz=0.3, metres=0.008):
    """
    A person sitting completely still: 8 mm of chest movement at 0.3 Hz.

    This is the hard case, and it is here to be hard. At 20 frames a second a
    breath that size moves the reflection path at about 1.5 cm/s against a
    wavelength of 12.4 cm — a fraction of a radian per frame, comparable with
    the receiver's own noise. Real systems read it from *phase*, which needs
    synchronisation this project does not have, so the honest answer from this
    pipeline is that a motionless person is close to invisible. The tests say so
    rather than pretending otherwise.
    """
    return metres * math.sin(2 * math.pi * hz * t)


def seated(t, metres=0.12):
    """
    Someone sitting and using their hands: small, slow, irregular movement.

    The realistic version of "a person in the room who is not walking" — people
    do not sit like statues, and this is what a still-but-present person
    actually looks like to the channel.
    """
    return (metres * math.sin(2 * math.pi * 0.06 * t)
            + 0.03 * math.sin(2 * math.pi * 0.9 * t + 1.0)
            + 0.02 * math.sin(2 * math.pi * 1.7 * t))


def person_crossing(t0, t1, metres=5.0, cadence_hz=1.8, cadence_metres=0.02):
    """
    Someone walking across the link between t0 and t1: the reflection path
    lengthens smoothly by `metres`, with a step cadence on top of it.

    The two amplitudes are in a ratio a real body has. Five metres in ten
    seconds is a leisurely 0.5 m/s, and a 2 cm step cadence at 1.8 Hz swings the
    path at 0.23 m/s — faster than that and the sway would occasionally cancel
    the walking, so the channel would stand still for a moment mid-stride and a
    detector would be right to report nothing. A simulated walker whose sway
    outweighs their stride is not a person, and would make the tests agree with a
    signal no one produces.
    """
    def scenario(t):
        if t < t0 or t > t1:
            return None
        progress = (t - t0) / max(1e-9, t1 - t0)
        # Smoothstep: people do not start and stop instantly, and the corner in
        # a linear ramp would put a spike in the Doppler spectrum that no real
        # body produces.
        eased = progress * progress * (3 - 2 * progress)
        return metres * eased + cadence_metres * math.sin(2 * math.pi * cadence_hz * t)
    return scenario


def wander(t, metres=0.6, hz=0.07, seed=1):
    """Slow aimless movement: fidgeting, shifting weight."""
    return metres * math.sin(2 * math.pi * hz * t) * math.cos(2 * math.pi * 0.11 * t) \
        + 0.05 * math.sin(2 * math.pi * 1.3 * t)


def constant_velocity(metres_per_second, from_metres=0.0):
    """
    A body whose reflection path changes length at a fixed rate.

    Not a realistic person — nobody walks at one speed — but it is the case with
    a known answer: a path lengthening at `v` metres per second shifts the
    signal by `v / wavelength` hertz, so a test can check the Doppler axis
    against arithmetic instead of against itself. At 2.4 GHz the wavelength is
    about 12.4 cm, so a tenth of a metre per second is about 0.8 Hz.
    """
    def scenario(t):
        return from_metres + metres_per_second * t
    return scenario


def looping(scenario, period):
    """
    Repeat a scenario every `period` seconds, forever.

    A demo that shows someone walking once and then an empty room until the tab
    is closed is a demo nobody sees the interesting part of. Real sessions are
    not periodic, but a demonstration should be.
    """
    def repeated(t):
        return scenario(t % period)
    return repeated


def scripted(segments):
    """
    A timeline of scenarios: `scripted([(0, 5, quiet), (5, 15, walk)])`.

    Later segments win, and a scenario that returns None means nobody is there.
    """
    def scenario(t):
        chosen = quiet
        for start, end, fn in segments:
            if start <= t < end:
                chosen = fn
        if callable(chosen) and chosen is not quiet:
            return chosen(t)
        return None
    return scenario


#: The default walk. It starts with five seconds of empty room on purpose: the
#: detector calibrates its noise floor from the beginning of a session, so a
#: scenario that has someone moving from the first frame would be measuring a
#: room it never saw empty. Real sessions have the same requirement, which is
#: why the live view says "stay still" while it calibrates.
WALK = scripted([(0.0, 5.0, quiet), (5.0, 15.0, person_crossing(5.0, 15.0)),
                 (15.0, 600.0, quiet)])

SCENARIOS = {
    "quiet": quiet,
    "walk": WALK,
    "seated": seated,
    "breathe": breathe,
    "wander": wander,
    # A demo worth watching, on a 40 second loop: an empty room, someone walks
    # through it, sits down and uses their hands for a while, then leaves.
    "demo": looping(scripted([(0.0, 5.0, quiet),
                              (5.0, 15.0, person_crossing(5.0, 15.0)),
                              (15.0, 25.0, seated),
                              (25.0, 40.0, quiet)]), 40.0),
}


# ------------------------------------------------------------- sources

class Source:
    """A stream of frames. Sources are iterators, and may be infinite."""

    rate = 20.0
    length = None          # frames, when the source is finite

    def describe(self):
        return self.__class__.__name__

    def close(self):
        """Release anything held open. Sources that hold nothing ignore it."""


class SyntheticSource(Source):
    """
    Simulated frames from an `Environment` and a scenario.

    The clock is the frame index divided by the rate, so a run is reproducible:
    same seed, same scenario, same frames, every time. Tests depend on that.
    """

    def __init__(self, scenario="quiet", rate=20.0, seconds=30.0, seed=0,
                 subcarriers=DEFAULT_SUBCARRIERS, noise=0.02, environment=None):
        self.scenario_name = scenario if isinstance(scenario, str) else "custom"
        self.scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        self.rate = float(rate)
        self.seconds = seconds
        self.length = None if seconds is None else int(round(seconds * rate))
        self.environment = environment or Environment(
            subcarriers=subcarriers, noise=noise, seed=seed)
        self._i = 0

    def describe(self):
        return (f"simulated {self.scenario_name} room, {self.rate:.0f} frames/s, "
                f"{self.environment.subcarriers} subcarriers")

    def __iter__(self):
        return self

    def __next__(self):
        if self.length is not None and self._i >= self.length:
            raise StopIteration
        t = self._i / self.rate
        self._i += 1
        return self.environment.sample(t, self.scenario(t))


class PacedSource(Source):
    """
    Wraps a source so it runs at wall-clock speed, which is what a live view
    needs. Frames that are already due are not delayed, so a slow consumer falls
    behind rather than the clock drifting.
    """

    def __init__(self, inner, realtime=True, clock=time.monotonic, sleep=time.sleep):
        self.inner = inner
        self.realtime = realtime
        self._clock = clock
        self._sleep = sleep
        self.rate = inner.rate
        self.length = inner.length
        self._started = None
        self._yielded = 0

    def describe(self):
        pacing = "paced to real time" if self.realtime else "as fast as possible"
        return f"{self.inner.describe()} ({pacing})"

    def close(self):
        """Release the inner source's port, which a view may need to reopen."""
        close = getattr(self.inner, "close", None)
        if close:
            close()

    @property
    def doppler(self):
        """
        Whether this source can report speed at all, asked each time.

        A serial source cannot know until its first line whether the thing on
        the other end is sending CSI or sensor channels, so the answer is read
        from the inner source rather than copied at construction.
        """
        return getattr(self.inner, "doppler", True)

    def __iter__(self):
        return self

    def __next__(self):
        # Pull first, then pace. The other order waits for a frame that is not
        # coming and holds a finished session open for one frame's worth of
        # time, which is exactly what a recording that has run out looks like.
        frame = next(self.inner)
        if not self.realtime:
            return frame
        if self._started is None:
            self._started = self._clock()
        self._yielded += 1
        due = self._started + (self._yielded - 1) / self.rate
        now = self._clock()
        if due > now:
            self._sleep(due - now)
        return frame


class RecordingSource(Source):
    """
    Frames replayed from a JSON Lines recording — the same path real hardware
    will take, exercised without any hardware.

    A source is a stream, one pass only, like the synthetic one: rewind by
    building another. That is the contract the live views rely on, and a wrapper
    that paces a source has to be able to call `next` on it.
    """

    def __init__(self, path, rate=None):
        self.path = path
        self._frames = read_recording(path)
        self.length = len(self._frames)
        self.rate = rate or _rate_from(self._frames)
        self._i = 0

    def describe(self):
        return (f"recorded session {self.path}, {self.length} frames, "
                f"{self.rate:.0f} frames/s")

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self._frames):
            raise StopIteration
        frame = self._frames[self._i]
        self._i += 1
        return frame


def _rate_from(frames):
    """Infer the frame rate from a recording's own timestamps."""
    if len(frames) < 2:
        return 20.0
    span = frames[-1].t - frames[0].t
    return (len(frames) - 1) / span if span > 0 else 20.0


def write_recording(frames, path, limit=None):
    """Write frames as JSON Lines. Returns how many were written."""
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for frame in frames:
            if limit is not None and written >= limit:
                break
            f.write(json.dumps(frame.as_record()) + "\n")
            written += 1
    return written


def read_recording(path):
    """Read frames back, undoing the fixed-point scale of `as_record`."""
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            values = [complex(re / RECORD_SCALE, im / RECORD_SCALE)
                      for re, im in record["csi"]]
            frames.append(Frame(record.get("t", len(frames)), values))
    return frames


def open_source(spec, **kwargs):
    """
    Build a source from a one-line spec, which is how the tools choose one:

        sim:demo             a simulated room (any name in SCENARIOS)
        file:session.jsonl   a recorded session
        udp:5500             CSI packets off the wire (nexmon_csi on a Pi)
        udp:10.0.0.5:5500    ... on a specific address
        serial:COM3          lines on a USB serial port (see serial_node.py)
        serial:COM3:9600     ... at a baud rate other than 115200
        serial:COM3:auto     ... at whatever rate the firmware is using

    CSI does not come out of a laptop's own WiFi card: Windows exposes RSSI and
    nothing else, and a Linux monitor-mode capture needs a specific chip and a
    patched driver. The two practical receivers are an ESP32, which streams
    frames over USB, and a Raspberry Pi running nexmon_csi, which broadcasts them
    as UDP and is what `udp:` speaks. An ESP32 on USB speaks `serial:` as long as
    it prints one `C,<millis>,<re>,<im>,...` line per frame.

    `serial:` also accepts a board that has no radio at all — an Arduino — as
    long as that board is honest about what it measures: its `S,` lines carry a
    profile of sensor channels rather than CSI, and `serial_node.SerialSource`
    marks that stream as having no Doppler axis. Pretending an ATmega's ADC is a
    radio would produce a speed number with nothing behind it.
    """
    kind, _, rest = spec.partition(":")
    if kind == "sim":
        return SyntheticSource(scenario=rest or "demo", **kwargs)
    if kind in ("file", "rec"):
        # A recording carries its own timing, so the frame rate and length
        # arguments are deliberately not forwarded: taking them from the file's
        # timestamps is the only reading that cannot disagree with the file, and
        # a rate flag that silently contradicted a recording would be a quiet
        # way to get every frequency in the view wrong.
        return RecordingSource(rest)
    if kind in ("udp", "nexmon"):
        # Imported here so csi.py stays importable on its own: nexmon.py imports
        # this module for Frame, and the cycle only works at call time.
        from wifi_radar import nexmon

        host, port = _udp_spec(spec, rest)
        # `seconds` is the caller's business — watch.py stops on frame time —
        # but the rate is the detector's Doppler scale, so it is honoured.
        return nexmon.NexmonSource(host=host, port=port,
                                   rate=kwargs.get("rate", 20.0))
    if kind == "serial":
        # Imported at call time for the same reason as nexmon, and because
        # pyserial is only needed once a real port is opened.
        from wifi_radar import serial_node

        port, baud = serial_node.parse_spec(rest)
        auto_port = port.strip().lower() == "auto"
        if auto_port:
            # The board is whatever is plugged in, and asking a person which COM
            # port it got is the step where a first run usually stops.
            chosen = serial_node.choose_port(serial_node.available_ports())
            if chosen is None:
                raise OSError(
                    "serial:auto found no serial port. Plug the board in and "
                    "check that a COM port appears (Windows needs the CH340 "
                    "driver for Elegoo boards); "
                    "`python -m wifi_radar.serial_node --list` shows what was "
                    "found.")
            port = chosen.device
        found = False
        if baud is None:
            # `serial:...:auto`: discover the rate once, through the doctor's
            # scan, and open the session at the answer.
            baud, _attempts = serial_node.discover_baud(port)
            found = True
        stream = serial_node.open_serial(port, baud)
        source = serial_node.SerialSource(
            stream, port=port, baud=baud, rate=kwargs.get("rate", 20.0),
            timeout=kwargs.get("timeout") or serial_node.DEFAULT_TIMEOUT,
            baud_found=found)
        if not kwargs.get("reconnect", True):
            return source

        def reopen():
            """
            Open the same board again, re-choosing the port when it was auto:
            a replugged adapter does not always come back under the same name.

            A *source*, not a stream: the reconnection replaces the inner
            object, and everything downstream asks it for frames and for a
            description. Returning the port itself used to make the first
            reconnect the end of the session.
            """
            target = port
            if auto_port:
                chosen = serial_node.choose_port(serial_node.available_ports())
                if chosen is None:
                    raise OSError("serial:auto found no serial port while "
                                  "reconnecting; plug the board back in")
                target = chosen.device
            stream = serial_node.open_serial(target, baud)
            return serial_node.SerialSource(
                stream, port=target, baud=baud, rate=kwargs.get("rate", 20.0),
                timeout=kwargs.get("timeout") or serial_node.DEFAULT_TIMEOUT,
                baud_found=found)

        return serial_node.ReconnectingSource(
            source, reopen, port=port, rate=kwargs.get("rate", 20.0),
            timeout=kwargs.get("timeout") or serial_node.DEFAULT_TIMEOUT,
            baud=baud,
            give_up_after=kwargs.get("reconnect_seconds",
                                     serial_node.DEFAULT_RECONNECT_SECONDS),
            notify=kwargs.get("notify"))
    raise ValueError(
        f"unknown source {spec!r}: expected sim:<scenario>, file:<path>, "
        f"udp:[host:]port or serial:<port>. "
        f"Scenarios: {', '.join(sorted(SCENARIOS))}"
    )


def _udp_spec(spec, rest):
    """Split the tail of a `udp:` spec into host and port."""
    host = "0.0.0.0"
    if not rest:
        port = 5500
    elif ":" in rest:
        host, _, port = rest.rpartition(":")
        host = host or "0.0.0.0"
    else:
        port = rest
    try:
        port = int(port)
    except ValueError:
        raise ValueError(
            f"source {spec!r} needs a numeric port, got {port!r}; write "
            f"udp:5500 or udp:<host>:5500") from None
    if not 0 < port < 65536:
        raise ValueError(f"source {spec!r} has port {port}, outside 1-65535")
    return host, port
