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

#: How long to measure the empty-room floor before deciding anything, in seconds.
#:
#: Two seconds is enough for a radio, whose noise is flat and fast. It is not
#: enough for an input that drifts: a wire on an ADC pin picks up a slow mains
#: envelope, and a two-second sample of it is a sample of one part of that
#: envelope — calibrate on the quiet half-second and the other half of every
#: cycle reads as movement for the rest of the session. So `calibrate_seconds`
#: is a *minimum*, and the detector keeps measuring while the input still shows
#: slow change; passing a number fixes the window at that length instead.
DEFAULT_CALIBRATE_SECONDS = 2.0

#: Longest the automatic calibration will wait for a drifting input to settle.
#:
#: The waiting has a point — a window that covers whole cycles of a slow
#: envelope estimates the floor the rest of the session will be judged against —
#: but an input that never repeats (a hand on the sensor, a slowly emptying
#: battery) would otherwise calibrate forever. 40 s covers three cycles of the
#: 0.1 Hz mains envelope a wire picks up, which is the case this was built for.
DEFAULT_CALIBRATE_LIMIT_SECONDS = 40.0

#: How often automatic calibration asks whether it has measured enough, in
#: seconds. Once a cycle is cheap: the check is a median and an FFT over the
#: samples so far, and repeated checks would only make the decision noisier.
CALIBRATE_PROBE_SECONDS = 1.0

#: How many cycles of the slowest visible component to cover before stopping.
#: Fewer than this and the window is a sample of part of the envelope, which is
#: the mistake the automatic length exists to prevent.
CALIBRATE_CYCLES = 3

#: How much of the frame-to-frame change the high-pass has to remove before the
#: input counts as drifting, as a ratio of the filtered to unfiltered metric.
#: Measured on a simulated ADC drift the ratio is 0.2-0.4; on a quiet radio, a
#: walking person, and breathing it is 0.94-0.95 — the filter either takes most
#: of the change away or almost none of it, so anywhere in the gap would do.
CALIBRATE_DRIFT_RATIO = 0.6

#: Time constant of the baseline the metric is compared against, in seconds.
#:
#: A change slower than this is treated as the input moving under the detector
#: rather than as something happening in front of it: mains hum on a wire, a
#: supply sagging, a sensor warming up. 0.5 s is below the 0.3 Hz bottom of the
#: motion band, so a walk passes through unchanged (measured: 97.1% of the walk
#: detected with any value from 0 to 1 s), while the ten-second envelope a wire
#: on an ADC pin picks up is attenuated. 0 turns it off.
DEFAULT_HIGHPASS_SECONDS = 0.5

def parse_calibrate(text):
    """
    `--calibrate` for argparse: seconds, or "auto" for the automatic length.

    Auto is the default because the length that works depends on what the input
    is doing, not on what the user knows about it, and the wrong guess is silent:
    a too-short window does not fail, it just calls the rest of the session
    movement.
    """
    if isinstance(text, str) and text.strip().lower() == "auto":
        return None
    return float(text)


#: Smallest spread the threshold may be a multiple of.
#:
#: A transmitter's noise keeps `scale` comfortably away from zero, but an ADC
#: channel that does not move at all — an unconnected pin, a sensor reading a
#: constant — gives a calibration of exact zeros, so floor and spread are both
#: zero and `z` would be undefined. The floor/scale pair must never be allowed to
#: make a real change unmeasurable: an unplugged sensor whose first sample reads
#: exactly 0 and whose next reads 512 is the largest event in the room.
MIN_SCALE = 1e-9


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
                 smoothing=3, calibrate_seconds=DEFAULT_CALIBRATE_SECONDS,
                 calibrate_max_seconds=DEFAULT_CALIBRATE_LIMIT_SECONDS,
                 highpass_seconds=DEFAULT_HIGHPASS_SECONDS, doppler=True):
        self.rate = float(rate)
        # A channel profile has no speed to report: the metric's spectrum says
        # how fast the profile changed, not how fast anything moved through the
        # room, and printing it next to the word "Doppler" would invent a
        # quantity. Sources that are not CSI set this False (serial_node does).
        self.doppler = bool(doppler)
        self.window = window
        self.band = band
        self.on_sigma = on_sigma
        self.off_sigma = off_sigma
        self.presence_hold = float(presence_hold)
        self.presence_window = max(4, int(round(2.0 * self.rate)))
        self.smoothing = max(1, smoothing)
        # `calibrate_seconds=None` means automatic: measure at least the default
        # minimum, and keep going while the input still drifts.
        self.calibrate_auto = calibrate_seconds is None
        minimum = DEFAULT_CALIBRATE_SECONDS if calibrate_seconds is None else calibrate_seconds
        self.calibrate_seconds = float(minimum)
        self.calibrate_frames = (calibrate_frames if calibrate_frames is not None
                                 else int(round(self.calibrate_seconds * self.rate)))
        self.calibrate_limit_frames = int(round(calibrate_max_seconds * self.rate))
        self.calibrate_probe_frames = max(1, int(round(CALIBRATE_PROBE_SECONDS * self.rate)))
        self.highpass_seconds = float(highpass_seconds)
        self.reset()

    def reset(self):
        self._prev_shape = None
        self._slow = None            # per-channel baseline for the high-pass
        self._prev_high = None       # previous high-passed shape
        self._recent = []            # smoothed metric, latest last
        self._raw = []               # for the smoothing filter only
        self._calibration = []       # smoothed metric during calibration
        self._cal_changes = []       # per-subcarrier change during calibration
        self._cal_raw = []           # unfiltered metric during calibration
        self._cal_filtered = []      # high-passed metric during calibration
        self._cal_samples = 0
        self._cal_done = self.calibrate_frames <= 0
        self._cal_reason = "fixed" if self._cal_done else None
        # While the length is automatic the honest share of the work done is
        # unknown, so progress is measured against the longest it may take; the
        # browser shows seconds elapsed for exactly that reason.
        self._cal_target = max(1, self.calibrate_limit_frames
                               if self.calibrate_auto else self.calibrate_frames)
        self._cal_next_probe = self.calibrate_frames
        self.frames = 0
        self._motion = False
        self._presence = False
        self._last_motion_t = None
        self._flags = []             # recent motion flags, for activity

    # -------------------------------------------------------------- state
    @property
    def calibrating(self):
        return not self._cal_done

    @property
    def calibration_progress(self):
        if self._cal_done:
            return 1.0
        return min(0.99, self._cal_samples / max(1, self._cal_target))

    @property
    def calibration_reason(self):
        """
        Why the window closed, or None while it is still open.

        "fixed" (a length was given), "settled" (the input is not drifting),
        "cycles" (a drifting input has been watched for whole cycles), or
        "limit" (it never settled and the maximum was reached). The views print
        it, because "calibrating for 16 seconds" is otherwise indistinguishable
        from a detector that is stuck.
        """
        return self._cal_reason

    @property
    def calibration_seconds(self):
        return self._cal_samples / self.rate if self.rate else 0.0

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
        return max(1.4826 * mad, 0.25 * median, MIN_SCALE)

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
            # to compare against. The slow baseline starts from this frame so the
            # high-pass has somewhere to follow from.
            self._slow = list(shape)
            self._prev_high = None
            return self._result(frame, metric=0.0, changes=[0.0] * len(shape))

        changes = [abs(a - b) for a, b in zip(shape, previous)]
        raw_metric = sum(changes) / len(changes)
        metric = raw_metric
        if self.highpass_seconds > 0:
            filtered = self._highpassed(shape)
            if filtered is not None:
                metric = filtered

        self._raw.append(metric)
        if len(self._raw) > self.smoothing:
            self._raw.pop(0)
        smoothed = sum(self._raw) / len(self._raw)
        self._recent.append(smoothed)
        if len(self._recent) > self.window:
            self._recent.pop(0)

        if not self._cal_done:
            self._cal_samples += 1
            # Both versions are kept for the automatic length: how much of the
            # change the high-pass removed is what separates a drifting input
            # from a moving one.
            self._cal_raw.append(raw_metric)
            self._cal_filtered.append(metric)
            if len(self._raw) == self.smoothing:
                self._calibration.append(smoothed)
                self._cal_changes.append(changes)
            self._advance_calibration()
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

    def _highpassed(self, shape):
        """
        The change between frames with the slowly moving baseline taken out.

        Gain normalisation already cancels a drift that reaches every channel
        equally. What survives is drift that reaches them unequally — the case a
        wire on an ADC pin produces, because the first conversion after the mux
        moves carries the most of it and each later pin carries less. That drift
        is slow: a hand arriving changes the input in under a second, the mains
        envelope under the wire takes ten.

        So the shape is compared with a baseline that follows it with a time
        constant of `highpass_seconds`, and what is differenced is the deviation
        from that baseline rather than the value itself. A component slower than
        the time constant contributes almost nothing to the difference; anything
        faster passes through with its amplitude intact. `None` means there is
        not yet a previous deviation to compare against.
        """
        step = 1.0 / self.rate
        alpha = min(1.0, step / self.highpass_seconds)
        if self._slow is None or len(self._slow) != len(shape):
            self._slow = list(shape)
            self._prev_high = None
        else:
            self._slow = [slow + alpha * (value - slow)
                          for slow, value in zip(self._slow, shape)]
        high = [value - slow for value, slow in zip(shape, self._slow)]
        previous, self._prev_high = self._prev_high, high
        if previous is None or len(previous) != len(high):
            return None
        return sum(abs(a - b) for a, b in zip(high, previous)) / len(high)

    # -------------------------------------------------------- calibration
    def _advance_calibration(self):
        """
        Decide whether the window can close yet.

        A fixed window closes when its frame count arrives. An automatic one is
        re-asked once a second, and needs two answers: is the input drifting at
        all, and if so has it been watched for whole cycles of the drift?

        "Drifting" is measured rather than assumed: the high-pass is compared
        with no high-pass on the same frames. Movement survives that filter — a
        walk is a 1-2 Hz change and passes almost untouched, ratio 0.94 here —
        while a 0.1 Hz envelope is most of what the filter removes, ratio 0.2-0.4.
        So a ratio near 1 means there is nothing slow to wait for and the minimum
        is enough.
        """
        if not self.calibrate_auto:
            if self._cal_samples >= self.calibrate_frames:
                self._finish_calibration("fixed")
            return
        if self._cal_samples < self._cal_next_probe:
            return
        self._cal_next_probe += self.calibrate_probe_frames
        if self._cal_samples >= self.calibrate_limit_frames:
            self._finish_calibration("limit")
            return

        raw = _median(self._cal_raw)
        filtered = _median(self._cal_filtered)
        if raw <= 0 or filtered / raw >= CALIBRATE_DRIFT_RATIO:
            self._finish_calibration("settled")
            return

        needed = self._cycles_needed()
        self._cal_target = max(self._cal_target, needed)
        if self._cal_samples >= needed:
            self._finish_calibration("cycles")

    def _cycles_needed(self):
        """Frames needed to cover CALIBRATE_CYCLES of the slowest visible part.

        The spectrum of the metric says how slow the drift is, and the window
        grows until it holds whole cycles of it: a floor measured over part of
        an envelope is a floor for that part only. When the window is still too
        short to resolve anything below the motion band, the answer is the
        smallest useful extension rather than a guess at a period.
        """
        shortest = CALIBRATE_CYCLES * self.calibrate_seconds
        if len(self._calibration) < 4:
            return int(round(shortest * self.rate))
        freqs, powers = power_spectrum(self._calibration, self.rate)
        peak, power = None, 0.0
        for f, p in zip(freqs, powers):
            if 0 < f < self.band[0] and p > power:
                peak, power = f, p
        if peak is None:
            return int(round(shortest * self.rate))
        return max(1, int(round(CALIBRATE_CYCLES / peak * self.rate)))

    def _finish_calibration(self, reason):
        self._cal_done = True
        self._cal_reason = reason
        self._cal_target = max(1, self._cal_samples)
        # Nothing after this point reads them, and they are the two longest
        # lists the detector holds.
        self._cal_raw = []
        self._cal_filtered = []

    # -------------------------------------------------------------- parts
    def _doppler(self):
        """
        Peak frequency and in-band power of the metric's own spectrum.

        Skipped until the window is full: a spectrum of five samples is not a
        spectrum, and reporting one would put a confident-looking number next to
        no evidence.
        """
        if not self.doppler or len(self._recent) < self.window:
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



