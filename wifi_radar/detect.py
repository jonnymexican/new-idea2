#!/usr/bin/env python3
"""
Turning a stream of CSI frames into "is anything moving in here".

The whole detector rests on one quantity. A person walking changes the length of
one of the paths the signal takes, which rotates that path's phase against the
static ones, so the interference between them — and therefore the amplitude on
every subcarrier — moves far more than the receiver's noise does:

    a[t]  = |H[t]|                       magnitude per subcarrier
    m[t]  = mean over subcarriers | a[t]/mean(a[t]) - a[t-1]/mean(a[t-1]) |

Both halves of that formula are forced by real hardware, and both are testable:

- **magnitude, not phase.** Every frame arrives with an unknown phase rotation
  (carrier and sampling frequency offset), so the phase difference between two
  frames is noise until something synchronises them. Dropping phase is what makes
  a later frame comparable to an earlier one at all. The cost is honest and worth
  stating: with phase gone, the direction a person is moving cannot be recovered,
  and neither can distance — this sensor reports that something moved, on which
  subcarriers, and how fast, not where or which way;
- **divided by the frame's own mean magnitude.** Automatic gain control re-scales
  each frame, which is a change in amplitude, not a change in channel; without
  this division a quiet room full of AGC wander reports movement. With it, a
  frame scaled by any factor measures exactly zero movement against its
  predecessor, which is a property the tests check directly rather than
  statistically.

Everything else is about making that number mean something over time:

- a calibration window, taken while the room is still, measures the noise floor,
  and every threshold is a multiple of it rather than a fixed number. This is why
  the first couple of seconds of any session are the detector telling the user to
  stand still;
- a short moving average stops the instantaneous flag following single noisy
  frames;
- *presence* means "something moved here recently": it comes on with the first
  flagged frame and stays on for a couple of seconds after the last one, so it
  does not flicker between frames and it clears when the room does. It is
  deliberately not a claim that someone is standing there without moving. An
  amplitude-only metric cannot make that claim — a body holding still stops
  changing the amplitudes it is measured by, and the signal that would catch it
  is phase, which needs the synchronisation this pipeline does without. The
  README states the limit rather than letting the flag imply more;
- a windowed FFT of the metric gives the Doppler band — how much of the movement
  sits in the 0.3-8 Hz a human body occupies, and the frequency it peaks at.
  That peak is the interesting number: a step cadence near 2 Hz is someone
  walking, movement near 0.3 Hz is someone breathing.

What is deliberately absent is any claim about *where* the movement is. Range
resolution is `c / (2 * bandwidth)`: about 7.5 m for a 20 MHz channel, which is
wider than most rooms, so distance is not measurable at this bandwidth no matter
how good the signal processing is. The estimator that could tell walking from
breathing from a door closing is also absent — it would need labelled data that
does not exist yet, and guessing at it would make the honest numbers here worse.
"""

import math

#: Human movement lives in this band; 0.3 Hz is a breath, 8 Hz is a fast arm.
DEFAULT_BAND = (0.3, 8.0)

#: Defaults for the detector, all in units of the calibrated noise floor.
DEFAULT_ON_SIGMA = 4.0
DEFAULT_OFF_SIGMA = 2.0

#: How long presence survives after the last movement, in seconds. Two seconds
#: is about how long a person crossing a room is out of the signal between
#: steps, and short enough that an empty room reads as empty promptly.
DEFAULT_PRESENCE_HOLD = 2.0


# ------------------------------------------------------------------- FFT
def fft(values):
    """
    Iterative radix-2 Cooley-Tukey FFT.

    Written here rather than imported because the project it lives in has no
    numerical dependencies and this is 25 lines. It demands a power of two,
    which the detector's window size is configured to always be.
    """
    n = len(values)
    if n == 0 or (n & (n - 1)):
        raise ValueError(f"fft length must be a power of two, got {n}")
    a = list(values)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        angle = -2.0 * math.pi / length
        step = complex(math.cos(angle), math.sin(angle))
        half = length >> 1
        for start in range(0, n, length):
            w = 1 + 0j
            for k in range(start, start + half):
                u = a[k]
                v = a[k + half] * w
                a[k] = u + v
                a[k + half] = u - v
                w *= step
        length <<= 1
    return a


def hamming(n):
    """Window that keeps a burst of movement from leaking across the spectrum."""
    if n == 1:
        return [1.0]
    return [0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]


def power_spectrum(series, rate):
    """
    One-sided power spectrum of a real series, mean removed and Hann-windowed.

    Returns (frequencies, powers) as plain lists, both length n // 2 + 1.

    The window is applied to the samples that exist and *then* the result is
    zero-padded to a power of two. That order matters: windowing the padding as
    well tapers the taper into a step, so a perfectly constant input — the case
    that has to read as no movement at all — leaks a large low-frequency spike
    into the band. Zero padding after windowing only interpolates the spectrum.
    """
    count = len(series)
    if count == 0:
        raise ValueError("cannot take the spectrum of nothing")
    size = _next_power_of_two(count)
    mean = sum(series) / count
    window = hamming(count)
    # Normalising by the window's own weight keeps power comparable between
    # window sizes, which matters because the live view and the tests use
    # different ones.
    weight = sum(w * w for w in window) or 1.0
    tapered = [(v - mean) * w for v, w in zip(series, window)]
    spectrum = fft(tapered + [0.0] * (size - count))
    powers = [abs(spectrum[0]) ** 2 / weight]
    for k in range(1, size // 2):
        powers.append(2 * abs(spectrum[k]) ** 2 / weight)
    powers.append(abs(spectrum[size // 2]) ** 2 / weight)
    freqs = [k * rate / size for k in range(size // 2 + 1)]
    return freqs, powers


def _next_power_of_two(n):
    size = 1
    while size < n:
        size <<= 1
    return size


def dominant_frequency(freqs, powers, band=DEFAULT_BAND):
    """(frequency, in-band power) of the strongest bin inside `band`."""
    lo, hi = band
    best_f, best_p = None, 0.0
    total = 0.0
    for f, p in zip(freqs, powers):
        if lo <= f <= hi:
            total += p
            if p > best_p:
                best_f, best_p = f, p
    return best_f, total


# --------------------------------------------------------------- detection
class Detection:
    """One frame's worth of conclusions, and the numbers behind them."""

    __slots__ = ("t", "metric", "z", "motion", "presence", "activity",
                 "peak_hz", "band_power", "agreement", "amplitude",
                 "calibrating", "frames", "changes")

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw.get(name))

    def as_dict(self):
        return {
            "t": round(self.t, 4),
            "metric": round(self.metric, 6),
            "z": round(self.z, 3) if self.z is not None else None,
            "motion": bool(self.motion),
            "presence": bool(self.presence),
            "activity": round(self.activity, 3),
            "peak_hz": round(self.peak_hz, 3) if self.peak_hz is not None else None,
            "band_power_db": (round(self.band_power, 2)
                              if self.band_power is not None else None),
            "agreement": round(self.agreement, 3),
            "amplitude": round(self.amplitude, 4),
            "calibrating": bool(self.calibrating),
            "frames": self.frames,
            "changes": [round(c, 5) for c in (self.changes or [])],
        }


class MotionDetector:
    """
    An online detector: feed it frames in order, read back a conclusion per
    frame. It holds only what it needs — the previous frame, a few seconds of
    the metric, and its calibration — so it runs unchanged on a live stream.
    """

    def __init__(self, rate=20.0, window=128, calibrate_frames=None,
                 band=DEFAULT_BAND, on_sigma=DEFAULT_ON_SIGMA,
                 off_sigma=DEFAULT_OFF_SIGMA, presence_hold=DEFAULT_PRESENCE_HOLD,
                 smoothing=3, calibrate_seconds=2.0):
        self.rate = float(rate)
        self.window = window
        self.band = band
        self.on_sigma = on_sigma
        self.off_sigma = off_sigma
        self.presence_hold = float(presence_hold)
        self.presence_window = max(4, int(round(2.0 * self.rate)))
        self.smoothing = max(1, smoothing)
        self.calibrate_frames = (calibrate_frames if calibrate_frames is not None
                                 else int(round(calibrate_seconds * self.rate)))
        self.reset()

    def reset(self):
        self._prev_shape = None
        self._recent = []            # smoothed metric, latest last
        self._raw = []               # for the smoothing filter only
        self._calibration = []       # smoothed metric during calibration
        self._cal_changes = []       # per-subcarrier change during calibration
        self._cal_samples = 0
        self.frames = 0
        self._motion = False
        self._presence = False
        self._last_motion_t = None
        self._flags = []             # recent motion flags, for activity

    # -------------------------------------------------------------- state
    @property
    def calibrating(self):
        return self._cal_samples < self.calibrate_frames

    @property
    def calibration_progress(self):
        return min(1.0, self._cal_samples / max(1, self.calibrate_frames))

    @property
    def metrics(self):
        """The recent metric, oldest first: what the views draw as a trace."""
        return list(self._recent)

    @property
    def floor(self):
        """Median of the calibrated noise floor, or None before calibration."""
        return _median(self._calibration) if self._calibration else None

    @property
    def scale(self):
        """
        Spread of the floor, estimated with the median absolute deviation so a
        single noisy frame during calibration cannot inflate the thresholds.
        The ratio to `floor` is a floor of its own: in a very quiet room the
        deviation can round to nearly zero, and dividing by it would make every
        later frame look like an event.
        """
        if len(self._calibration) < 3:
            return None
        median = _median(self._calibration)
        mad = _median([abs(v - median) for v in self._calibration])
        return max(1.4826 * mad, 0.25 * median)

    # ------------------------------------------------------------- update
    def update(self, frame):
        """Consume one frame and return the `Detection` for it."""
        self.frames += 1
        if len(frame.values) == 0:
            raise ValueError("a frame with no subcarriers cannot be measured")

        magnitudes = frame.amplitudes()
        mean_magnitude = sum(magnitudes) / len(magnitudes)
        if mean_magnitude <= 0:
            # A receiver that heard nothing cannot be normalised; report no
            # movement rather than dividing by zero.
            shape = [0.0] * len(magnitudes)
        else:
            shape = [m / mean_magnitude for m in magnitudes]

        previous = self._prev_shape
        self._prev_shape = shape
        if previous is None or len(previous) != len(shape):
            # First frame (or a change in the number of subcarriers) has nothing
            # to compare against.
            return self._result(frame, metric=0.0, changes=[0.0] * len(shape))

        changes = [abs(a - b) for a, b in zip(shape, previous)]
        metric = sum(changes) / len(changes)

        self._raw.append(metric)
        if len(self._raw) > self.smoothing:
            self._raw.pop(0)
        smoothed = sum(self._raw) / len(self._raw)
        self._recent.append(smoothed)
        if len(self._recent) > self.window:
            self._recent.pop(0)

        if self.calibrating:
            self._cal_samples += 1
            if len(self._raw) == self.smoothing:
                self._calibration.append(smoothed)
                self._cal_changes.append(changes)
            return self._result(frame, metric=smoothed, changes=changes)

        floor, scale = self.floor, self.scale
        z = (smoothed - floor) / scale if scale else None
        if z is not None:
            # Hysteresis on the instantaneous flag too, so a value hovering at
            # the threshold cannot chatter.
            if self._motion:
                self._motion = z > self.off_sigma
            else:
                self._motion = z > self.on_sigma
        self._flags.append(1 if self._motion else 0)
        if len(self._flags) > self.presence_window:
            self._flags.pop(0)

        if self._motion:
            self._last_motion_t = frame.t
        self._presence = (self._last_motion_t is not None
                          and frame.t - self._last_motion_t < self.presence_hold)

        peak_hz, band_power = self._doppler()
        return self._result(frame, metric=smoothed, changes=changes, z=z,
                            peak_hz=peak_hz, band_power=band_power)

    # -------------------------------------------------------------- parts
    def _doppler(self):
        """
        Peak frequency and in-band power of the metric's own spectrum.

        Skipped until the window is full: a spectrum of five samples is not a
        spectrum, and reporting one would put a confident-looking number next to
        no evidence.
        """
        if len(self._recent) < self.window:
            return None, None
        freqs, powers = power_spectrum(self._recent, self.rate)
        peak, band_power = dominant_frequency(freqs, powers, self.band)
        total = sum(powers) or 1e-12
        return peak, 10 * math.log10(max(band_power / total, 1e-12))

    def _agreement(self, changes):
        """
        Fraction of subcarriers disturbed more than the floor's own change.

        Average change alone cannot tell a person from a single subcarrier
        glitching; a body in the room disturbs many subcarriers at once.
        """
        if not self._cal_changes:
            return 0.0
        baseline = [sum(c[i] for c in self._cal_changes) / len(self._cal_changes)
                    for i in range(len(changes))]
        moved = sum(1 for c, b in zip(changes, baseline) if b > 0 and c > 2 * b)
        return moved / max(1, len(changes))

    def _result(self, frame, metric, changes, z=None, peak_hz=None,
                band_power=None):
        activity = (sum(self._flags) / len(self._flags)) if self._flags else 0.0
        return Detection(
            t=frame.t, metric=metric, z=z, motion=self._motion,
            presence=self._presence, activity=activity, peak_hz=peak_hz,
            band_power=band_power, agreement=self._agreement(changes),
            amplitude=frame.mean_amplitude, calibrating=self.calibrating,
            frames=self.frames, changes=changes,
        )


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])



