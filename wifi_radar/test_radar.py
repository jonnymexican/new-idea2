#!/usr/bin/env python3
"""
The radar's invariants, checked against a simulator that knows the answer.

These tests can be strict because the frames are simulated: the suite knows
exactly when a person is walking, at what cadence, and when the room is empty.
Each test pins something that either is easy to get wrong or that a claim in the
README depends on:

- the hand-written FFT agrees with a naive DFT, because every Doppler number
  downstream is computed from it and a subtly wrong transform would still look
  plausible on a chart;
- the spectrum of a known sinusoid peaks where the sinusoid is, for the same
  reason;
- an empty room produces no detections: the detector's whole usefulness is that
  its noise floor is calibrated, not assumed;
- a person walking is detected for essentially all of the time they are walking
  and essentially none of the time they are not — the two halves of the same
  claim;
- presence holds while someone is in the room moving (walking, or sitting and
  using their hands) and clears a couple of seconds after they leave, without
  flickering in between;
- the Doppler peak matches the speed of the path that produced it, against a
  frequency worked out from the wavelength rather than from the code;
- a person sitting *completely* still is close to invisible, which is a limit
  this pipeline has by construction and the tests state rather than hide;
- the detector still works when every frame carries the unknown phase rotation
  and gain change real receivers produce, which is the reason the metric is a
  magnitude ratio rather than a phase difference;
- a session survives the round trip through the JSON Lines recording format at
  the integer precision a real logger would use, because that format is the
  path real hardware will take;
- a serial port's line protocol decodes into frames — sensor channels as a
  profile, CSI as complex pairs — with junk counted, a wrapped 32-bit clock
  undone, and a silent port failing with a message that names what to check,
  because that is the first thing that goes wrong when a board is plugged in and
  nothing appears;
- the port the tool opens on its own is the board rather than the modem, and is
  the same one on the next run, because "COM3 or COM4?" is the question that
  stops a first bring-up before the sketch is even suspected;
- the pin panel names the pin that actually moved and stays silent when several
  move together, because pointing at the largest of four equal numbers would
  make a still room look wired, and it carries sixteen pins without assuming
  eight;
- the page's own script parses, checked by handing it to node: a duplicate
  `const` in one draw function is invisible to every Python test and blanks the
  whole view;
- a nexmon packet is decoded from the layout the firmware documents, at all
  three channel widths and with both header revisions, and frames pushed through
  a real UDP socket reach the detector as the same walk a simulated source
  produces — the hardware path, exercised without the hardware.

Run with:  .venv/bin/python -m unittest wifi_radar.test_radar
"""

import contextlib
import io
import os
import re
import shutil
import socket
import subprocess
import tempfile
import unittest

from wifi_radar import csi, detect, live, nexmon, serial_node


def collect(source, detector, limit=None):
    """Feed a whole source through a detector, as a live session would."""
    out = []
    for i, frame in enumerate(source):
        if limit is not None and i >= limit:
            break
        out.append(detector.update(frame))
    return out


def detection_rate(detections, lo, hi, attribute="motion"):
    """Fraction of frames between two times where `attribute` was true."""
    window = [d for d in detections if lo <= d.t <= hi]
    if not window:
        return 0.0
    return sum(1 for d in window if getattr(d, attribute)) / len(window)


def transitioning(source, detector):
    """Detections plus the times presence changed state."""
    detections = collect(source, detector)
    edges = []
    was = False
    for d in detections:
        if d.presence != was:
            edges.append((d.t, d.presence))
            was = d.presence
    return detections, edges


class TestFourier(unittest.TestCase):
    def test_fft_matches_naive_dft(self):
        """A wrong transform still draws a plausible chart, so check the numbers."""
        import cmath
        import random

        rng = random.Random(7)
        n = 64
        values = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]
        fast = detect.fft(values)
        naive = []
        for k in range(n):
            acc = 0j
            for i, v in enumerate(values):
                acc += v * cmath.exp(-2j * cmath.pi * k * i / n)
            naive.append(acc)
        worst = max(abs(a - b) for a, b in zip(fast, naive))
        self.assertLess(worst, 1e-9, f"FFT disagrees with the DFT by {worst:g}")

    def test_fft_rejects_lengths_it_cannot_do(self):
        with self.assertRaises(ValueError):
            detect.fft([1 + 0j] * 6)

    def test_spectrum_finds_a_known_frequency(self):
        """The Doppler peak is only evidence if the transform locates a known tone."""
        import math

        rate, hz, seconds = 20.0, 1.8, 12.0
        series = [math.sin(2 * math.pi * hz * i / rate) for i in range(int(seconds * rate))]
        freqs, powers = detect.power_spectrum(series, rate)
        peak, _ = detect.dominant_frequency(freqs, powers, band=(0.3, 8.0))
        self.assertAlmostEqual(peak, hz, delta=0.25)

    def test_spectrum_ignores_a_constant_offset(self):
        """
        A gain step is not movement, so it must not appear as one.

        Compared in absolute power, not as a fraction of the total: with a truly
        constant input every bin holds only floating point dust, and a ratio of
        two dust values is a random number between 0 and 1. The sine wave beside
        it is what stops this test from passing vacuously.
        """
        import math

        rate = 20.0
        freqs, powers = detect.power_spectrum([5.0] * 320, rate)
        _, flat_power = detect.dominant_frequency(freqs, powers, band=(0.3, 8.0))
        self.assertLess(flat_power, 1e-9)

        sine = [5.0 + math.sin(2 * math.pi * 1.8 * i / rate) for i in range(320)]
        freqs, powers = detect.power_spectrum(sine, rate)
        _, sine_power = detect.dominant_frequency(freqs, powers, band=(0.3, 8.0))
        self.assertGreater(sine_power, 1e-3)


class TestEnvironment(unittest.TestCase):
    def test_a_seed_reproduces_a_session_exactly(self):
        """Every number in this project is reproducible or it is not evidence."""
        first = list(csi.SyntheticSource(scenario="walk", seed=3, seconds=4.0))
        second = list(csi.SyntheticSource(scenario="walk", seed=3, seconds=4.0))
        self.assertEqual([f.values for f in first], [f.values for f in second])

    def test_a_different_seed_is_a_different_session(self):
        first = list(csi.SyntheticSource(scenario="walk", seed=3, seconds=1.0))
        second = list(csi.SyntheticSource(scenario="walk", seed=4, seconds=1.0))
        self.assertNotEqual(first[0].values, second[0].values)

    def test_the_frame_rate_sets_the_timestamps(self):
        frames = list(csi.SyntheticSource(scenario="quiet", rate=25.0, seconds=1.0))
        self.assertEqual(len(frames), 25)
        self.assertAlmostEqual(frames[1].t, 0.04)

    def test_nobody_is_in_the_room_during_quiet(self):
        source = csi.SyntheticSource(scenario="quiet", seconds=5.0)
        self.assertTrue(all(source.scenario(t) is None for t in (0.0, 1.0, 4.9)))

    def test_the_demo_repeats_so_a_viewer_sees_the_interesting_part(self):
        """
        The demo is the one scenario that loops: a walk that happens once and
        then never again is a walk whoever opened the page missed. The loop has
        to land in the same segments it does on the first pass.
        """
        demo = csi.SCENARIOS["demo"]
        self.assertIsNone(demo(1.0))
        self.assertIsNone(demo(41.0), "the loop restarts empty, as it opens")
        self.assertAlmostEqual(demo(10.0), demo(50.0))
        self.assertAlmostEqual(demo(20.0), demo(60.0))
        self.assertIsNotNone(demo(45.0), "five seconds into the loop is a walk")

    def test_the_walk_scenario_has_a_person_only_while_they_walk(self):
        source = csi.SyntheticSource(scenario="walk", seconds=20.0)
        self.assertIsNone(source.scenario(1.0))
        self.assertIsNotNone(source.scenario(10.0))
        self.assertIsNone(source.scenario(16.0))

    def test_a_moving_person_changes_the_channel_far_more_than_noise(self):
        """
        The premise of the whole detector. If this ever fails, the detector is
        measuring noise and any number it reports is meaningless.

        The person walks from the first frame here rather than from t=5, and both
        runs share a seed, so the two sources carry the same noise sequence: the
        difference measured is the person, not two random draws.
        """
        def amplitude_change(frames):
            """The quantity the detector actually uses: magnitude per subcarrier."""
            return sum(abs(a.amplitudes()[i] - b.amplitudes()[i])
                       for a, b in zip(frames, frames[1:])
                       for i in range(len(a.values)))

        quiet = list(csi.SyntheticSource(scenario="quiet", seed=5, seconds=3.0))
        walking = list(csi.SyntheticSource(
            scenario=csi.person_crossing(0.0, 8.0), seed=5, seconds=3.0))
        self.assertGreater(amplitude_change(walking), 5 * amplitude_change(quiet))


class TestDetector(unittest.TestCase):
    def detector(self, **kw):
        kw.setdefault("rate", 20.0)
        return detect.MotionDetector(**kw)

    def test_a_quiet_room_reports_nothing(self):
        """The false-positive rate is what makes a detection mean anything."""
        detections = collect(
            csi.SyntheticSource(scenario="quiet", seed=11, seconds=25.0),
            self.detector())
        after = [d for d in detections if not d.calibrating]
        self.assertEqual(sum(1 for d in after if d.motion), 0)
        self.assertFalse(any(d.presence for d in after))

    def test_a_walking_person_is_detected_while_they_walk_and_not_after(self):
        """
        The walk scenario starts with five seconds of empty room, so the floor
        being compared against was measured in a room nobody was in. That is the
        only version of this test that means anything: calibrating with a person
        present folds their movement into the baseline, which is a real way to
        get a detector that never fires and never looks broken.
        """
        detections = collect(
            csi.SyntheticSource(scenario="walk", seed=11, seconds=25.0),
            self.detector())
        during = detection_rate(detections, 6.0, 14.5)
        before = detection_rate(detections, 2.0, 4.5)
        after = detection_rate(detections, 17.0, 24.5)
        self.assertGreaterEqual(during, 0.85, f"only {during:.0%} of the walk detected")
        self.assertLessEqual(before, 0.02, f"{before:.0%} false positives before")
        self.assertLessEqual(after, 0.02, f"{after:.0%} false positives after")

    def test_a_seated_person_holds_presence(self):
        """
        Someone sitting and using their hands is a person in the room, and a
        presence flag that only knew about walking would call the room empty.
        """
        detections = collect(
            csi.SyntheticSource(scenario=csi.scripted(
                [(0.0, 5.0, csi.quiet), (5.0, 25.0, csi.seated)]), seed=13,
                seconds=25.0),
            self.detector())
        self.assertGreaterEqual(detection_rate(detections, 8.0, 23.0, "presence"), 0.8)
        self.assertGreaterEqual(detection_rate(detections, 8.0, 23.0), 0.25,
                                "a seated person moving their hands should flag frames")
        self.assertLess(detection_rate(detections, 2.0, 4.5, "presence"), 0.1)

    def test_presence_clears_shortly_after_the_person_leaves(self):
        """
        The hold has to release, or the flag is permanently on and says nothing.
        Two seconds is the setting; a person who has left the room must be gone
        from the view within about that, not held by hysteresis.
        """
        detections, edges = transitioning(
            csi.SyntheticSource(scenario="walk", seed=13, seconds=30.0),
            self.detector())
        on_times = [t for t, state in edges if state]
        off_times = [t for t, state in edges if not state]
        self.assertTrue(on_times, "presence never turned on")
        self.assertTrue(off_times, "presence never turned off again")
        self.assertLess(off_times[-1], 17.5, "took too long to release after they left")
        self.assertLess(detection_rate(detections, 18.0, 29.0, "presence"), 0.05)

    def test_presence_does_not_flicker(self):
        """A flag that toggles every frame is not a state anyone can act on."""
        _, edges = transitioning(
            csi.SyntheticSource(scenario="demo", seed=17, seconds=40.0),
            self.detector())
        self.assertLessEqual(len(edges), 6, f"presence toggled {len(edges)} times")

    def test_the_doppler_peak_matches_the_speed_of_the_path(self):
        """
        The one test here with an answer worked out independently of the code: a
        path lengthening at `v` metres per second shifts the signal by `v` over
        the wavelength, so 2.4 GHz turns 15 cm/s into about 1.2 Hz. If the FFT,
        the window or the frame rate were wrong, the peak would land somewhere
        else — and a chart of it would look just as convincing.
        """
        peaks = {}
        for speed in (0.15, 0.35):
            detections = collect(
                csi.SyntheticSource(scenario=csi.constant_velocity(speed),
                                    seed=19, seconds=14.0),
                self.detector())
            seen = [d.peak_hz for d in detections if d.peak_hz is not None]
            self.assertTrue(seen, f"no Doppler estimates at {speed} m/s")
            peaks[speed] = sum(seen) / len(seen)
            self.assertAlmostEqual(peaks[speed], speed / csi.WAVELENGTH, delta=0.3)
        self.assertGreater(peaks[0.35], peaks[0.15],
                           "a faster path must read as a higher frequency")

    def test_a_seated_person_reads_slower_than_a_walker(self):
        """
        Both are people, and the difference between them is the whole point of
        having a Doppler axis: the flag says something moved, and the peak says
        how fast. A seated person's hands move the path at a few centimetres a
        second and must not read like someone crossing the room.
        """
        def mean_peak(scenario, lo, hi, band=detect.DEFAULT_BAND):
            """
            Average peak during a window there is movement in. Averaging over an
            empty room as well would mostly measure where the noise happened to
            peak, and would drag the walk's number down towards it.
            """
            detections = collect(
                csi.SyntheticSource(scenario=scenario, seed=23, seconds=25.0),
                self.detector(band=band))
            seen = [d.peak_hz for d in detections
                    if d.peak_hz is not None and lo <= d.t <= hi]
            self.assertTrue(seen, "no Doppler estimates in the movement window")
            return sum(seen) / len(seen)

        seated = mean_peak(csi.scripted([(0.0, 5.0, csi.quiet),
                                        (5.0, 25.0, csi.seated)]), 8.0, 23.0)
        walking = mean_peak("walk", 7.0, 14.0)
        self.assertLess(seated, 1.2, "a seated person should read slowly")
        self.assertGreater(walking, 1.5, "a walk should read faster than sitting")

    def test_a_motionless_person_is_close_to_invisible(self):
        """
        The documented limit, held to a number rather than left to a sentence.
        An 8 mm breath at 0.3 Hz moves the path at about 1.5 cm/s, which is a
        fraction of the receiver's own frame-to-frame noise once phase has been
        discarded, and phase is discarded deliberately (see detect.py). Real
        systems read breathing from phase with a synchronised receiver. This one
        does not, and the test says so: if a later change made the detector
        claim a motionless person, this test would fail and the claim would have
        to be justified rather than quietly believed.
        """
        detections = collect(
            csi.SyntheticSource(scenario=csi.scripted(
                [(0.0, 5.0, csi.quiet), (5.0, 25.0, csi.breathe)]), seed=29,
                seconds=25.0),
            self.detector())
        self.assertLess(detection_rate(detections, 8.0, 24.0), 0.2)
        self.assertLess(detection_rate(detections, 8.0, 24.0, "presence"), 0.2)

    def test_an_unknown_phase_and_gain_per_frame_are_not_movement(self):
        """
        Real receivers rotate and re-scale every frame. This is the test that
        justifies using a magnitude ratio instead of a phase difference.
        """
        hostile = csi.Environment(seed=29, phase_drift=2.0, gain_drift=0.05)
        quiet = collect(
            csi.SyntheticSource(scenario="quiet", seconds=25.0, environment=hostile),
            self.detector())
        after = [d for d in quiet if not d.calibrating]
        self.assertEqual(sum(1 for d in after if d.motion), 0,
                         "phase and gain wander were mistaken for movement")

        walking = collect(
            csi.SyntheticSource(scenario="walk", seconds=25.0,
                                environment=csi.Environment(seed=29, phase_drift=2.0,
                                                            gain_drift=0.05)),
            self.detector())
        self.assertGreaterEqual(detection_rate(walking, 5.5, 15.0), 0.9)

    def test_agreement_separates_a_body_from_a_single_bad_subcarrier(self):
        """A body disturbs many subcarriers; a glitch disturbs one."""
        detections = collect(
            csi.SyntheticSource(scenario="walk", seed=31, seconds=25.0),
            self.detector())
        during = [d.agreement for d in detections if 7.0 <= d.t <= 14.0]
        quiet = [d.agreement for d in detections if 18.0 <= d.t <= 24.0]
        self.assertGreater(sum(during) / len(during), 0.6)
        self.assertLess(sum(quiet) / len(quiet), 0.25)

    def test_calibration_is_reported_rather_than_hidden(self):
        """The user is being asked to stand still; the view has to say so."""
        detector = self.detector(calibrate_seconds=1.0)
        frames = list(csi.SyntheticSource(scenario="walk", seconds=3.0))
        first = detector.update(frames[0])
        self.assertTrue(first.calibrating)
        self.assertFalse(first.presence)
        self.assertFalse(first.motion)
        for frame in frames[1:25]:
            last = detector.update(frame)
        self.assertFalse(last.calibrating)
        self.assertEqual(detector.frames, 25)

    def test_a_frame_of_the_wrong_length_does_not_crash_the_session(self):
        detector = self.detector()
        detector.update(csi.Frame(0.0, [1 + 0j] * 64))
        odd = detector.update(csi.Frame(0.05, [1 + 0j] * 32))
        self.assertEqual(len(odd.changes), 32)
        self.assertFalse(odd.motion)

    def test_an_empty_frame_is_refused(self):
        with self.assertRaises(ValueError):
            self.detector().update(csi.Frame(0.0, []))

    def test_the_metric_is_invariant_to_a_frame_wide_gain(self):
        """
        Scaling a frame is what AGC does; it must not look like movement. Checked
        directly rather than statistically, so a failure points at the formula.
        """
        base = csi.Frame(0.0, [complex(1.0, 0.3), complex(0.7, -0.2), complex(1.4, 0.1)])
        scaled = csi.Frame(0.05, [v * 3.7 for v in base.values])
        detector = self.detector()
        detector.update(base)
        result = detector.update(scaled)
        self.assertAlmostEqual(result.metric, 0.0, places=9)


class TestRecording(unittest.TestCase):
    def test_a_session_survives_the_round_trip_to_json_lines(self):
        """
        This format is how real hardware will arrive, at integer precision, so
        the pipeline is held to it rather than to floats in memory.
        """
        import tempfile

        source = csi.SyntheticSource(scenario="walk", seed=37, seconds=20.0)
        frames = list(source)
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            path = handle.name
        written = csi.write_recording(frames, path)
        self.assertEqual(written, len(frames))

        replayed = csi.RecordingSource(path)
        self.assertEqual(replayed.length, len(frames))
        self.assertAlmostEqual(replayed.rate, 20.0, delta=0.1)

        from_source = collect(csi.SyntheticSource(scenario="walk", seed=37, seconds=20.0),
                              detect.MotionDetector(rate=20.0))
        from_file = collect(replayed, detect.MotionDetector(rate=replayed.rate))
        # Not an exact match: a frame or two near the threshold can fall on
        # either side of it once amplitudes are rounded to integers. The claim
        # being made is that the recording carries the motion, not that the last
        # decimal place survives, so the tolerance is stated rather than the
        # equality pretended.
        agree = sum(1 for a, b in zip(from_file, from_source) if a.motion == b.motion)
        self.assertGreaterEqual(agree / len(from_source), 0.95,
                                f"only {agree / len(from_source):.0%} of frames agreed")
        self.assertGreaterEqual(detection_rate(from_file, 6.0, 14.5), 0.85)

    def test_quantisation_to_integers_keeps_the_motion_detectable(self):
        """A logger that reports integers must not silently break detection."""
        import tempfile

        frames = list(csi.SyntheticSource(scenario="walk", seed=41, seconds=20.0))
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            path = handle.name
        csi.write_recording(frames, path)
        replayed = collect(csi.RecordingSource(path), detect.MotionDetector(rate=20.0))
        self.assertGreaterEqual(detection_rate(replayed, 5.5, 15.0), 0.9)


class TestPacing(unittest.TestCase):
    """
    The wrapper every live view runs through, which the tests missed at first: a
    source that could be iterated but not wrapped left `watch.py` and `live.py`
    broken on the recording path while every unit test still passed. Pacing is
    checked with a fake clock so no test ever actually sleeps.
    """

    def test_pacing_wraps_a_simulated_source(self):
        paced = csi.PacedSource(csi.SyntheticSource(scenario="quiet", seconds=1.0),
                                clock=lambda: 0.0, sleep=lambda _: None)
        frames = list(paced)
        self.assertEqual(len(frames), 20)
        self.assertEqual([f.t for f in frames],
                         [f.t for f in csi.SyntheticSource(scenario="quiet",
                                                           seconds=1.0)])

    def test_pacing_wraps_a_recording(self):
        """The path that was broken: `--source file:` through `PacedSource`."""
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            path = handle.name
        csi.write_recording(csi.SyntheticSource(seconds=1.0), path)
        paced = csi.PacedSource(csi.open_source(f"file:{path}"), clock=lambda: 0.0,
                                sleep=lambda _: None)
        frames = list(paced)
        self.assertEqual(len(frames), 20)
        self.assertEqual(len(frames[0].values), 64)

    def test_pacing_waits_when_it_is_behind(self):
        """A paced source sleeps until each frame is due, and keeps its rate."""
        slept = []
        now = [0.0]

        def clock():
            return now[0]

        def sleep(seconds):
            slept.append(round(seconds, 4))
            now[0] += seconds

        paced = csi.PacedSource(csi.SyntheticSource(seconds=0.2, rate=10.0),
                                clock=clock, sleep=sleep)
        list(paced)
        self.assertEqual(len(slept), 1, "only the first frame is already due")
        self.assertAlmostEqual(slept[0], 0.1, places=3)

    def test_an_unpaced_source_yields_the_same_frames_immediately(self):
        frames = list(csi.PacedSource(csi.SyntheticSource(seconds=0.5),
                                      realtime=False))
        self.assertEqual(len(frames), 10)


class TestNexmonPackets(unittest.TestCase):
    """
    The parser, held to the firmware's documented layout.

    The payload is built here from the byte layout rather than copied from an
    implementation, so this is a check against the format and not a check that
    the parser agrees with itself. The two header revisions are the trap: the
    fields sit at the same offset *relative to the CSI*, which is why one parser
    can read both, and a mistake there still yields a frame — just the wrong one.
    """

    def tones(self, count, scale=1.0):
        """Distinct, exactly representable values, so any shift is visible."""
        return [complex(scale * (i + 1), scale * -(i + 1)) for i in range(count)]

    def test_every_width_decodes_exactly_with_either_header(self):
        for count in (64, 128, 256):
            for header in (18, 20):
                values = self.tones(count)
                packet = nexmon.parse_payload(
                    nexmon.build_payload(values, header=header))
                self.assertEqual(packet.values, values, f"{count} tones, {header}B")
                self.assertEqual(packet.header, header)
                self.assertEqual(packet.bandwidth_mhz, nexmon.BANDWIDTH_MHZ[count])

    def test_header_fields_are_read_where_the_header_puts_them(self):
        """A header-length mistake shifts every field, so pin each one."""
        for header in (18, 20):
            packet = nexmon.parse_payload(nexmon.build_payload(
                self.tones(64), sequence=513, core=1, chanspec=0x1006, chip=0x2b55,
                header=header))
            self.assertEqual(packet.sequence, 513, f"{header}B header")
            self.assertEqual(packet.core, 1, f"{header}B header")
            self.assertEqual(packet.chanspec, 0x1006, f"{header}B header")
            self.assertEqual(packet.chip, 0x2b55, f"{header}B header")

    def test_int16_values_are_not_rescaled_on_the_way_in(self):
        """
        Real CSI is the receiver's own uncalibrated scale, thousands not ones,
        and the detector normalises per frame. Clipping or normalising here
        would throw that away before the code that relies on it sees it.
        """
        values = self.tones(64, scale=400.0)
        packet = nexmon.parse_payload(nexmon.build_payload(values))
        self.assertEqual(packet.values, values)

    def test_the_rssi_fork_is_recognised_and_stock_is_not(self):
        stock = nexmon.parse_payload(nexmon.build_payload(self.tones(64)))
        self.assertIsNone(stock.rssi)
        forked = nexmon.parse_payload(
            nexmon.build_payload(self.tones(64), rssi=-63))
        self.assertEqual(forked.rssi, -63)
        self.assertEqual(forked.values, self.tones(64), "RSSI byte must not shift the CSI")

    def test_an_unknown_length_is_refused_with_the_accepted_sizes(self):
        """Half-parsing the wrong 256 bytes would report nonsense all session."""
        with self.assertRaises(ValueError) as caught:
            nexmon.parse_payload(b"\x11" * 200)
        message = str(caught.exception)
        self.assertIn("200", message)
        self.assertIn("274", message)
        self.assertIn("1044", message)

    def test_a_hand_written_packet_decodes_the_way_the_documentation_reads(self):
        """
        The one check that is not circular: this payload is written out byte by
        byte from the documented offsets, not through `build_payload`. If the
        builder and the parser shared a mistaken offset, every other test here
        would still pass; this one would not.

            0-3   magic            11 11 11 11
            4-5   extra magic      11 11          (20-byte header only)
            6-11  source MAC       02 00 00 00 00 01
            12-13 sequence         02 01          = 258
            14-15 core             01 00          = 1
            16-17 chanspec         06 10          = 0x1006
            18-19 chip             45 43          = 0x4345
            20-  CSI               64 * (int16 real, int16 imaginary)
        """
        header = bytes([0x11] * 6 + [0x02, 0x00, 0x00, 0x00, 0x00, 0x01,
                                     0x02, 0x01, 0x01, 0x00, 0x06, 0x10,
                                     0x45, 0x43])
        import struct
        payload = header + b"".join(struct.pack("<hh", i, -i) for i in range(64))
        self.assertEqual(len(payload), 276)

        packet = nexmon.parse_payload(payload)
        self.assertEqual(packet.header, 20)
        self.assertEqual(packet.sequence, 258)
        self.assertEqual(packet.core, 1)
        self.assertEqual(packet.chanspec, 0x1006)
        self.assertEqual(packet.chip, 0x4345)
        self.assertIsNone(packet.rssi)
        self.assertEqual(packet.values[0], 0j)
        self.assertEqual(packet.values[1], complex(1, -1))
        self.assertEqual(packet.values[63], complex(63, -63))

    def test_a_payload_without_the_magic_is_refused(self):
        payload = b"\x00\x00" + nexmon.build_payload(self.tones(64))[2:]
        with self.assertRaises(ValueError) as caught:
            nexmon.parse_payload(payload)
        self.assertIn("magic", str(caught.exception))


class TestNexmonSource(unittest.TestCase):
    """
    The socket, the timestamps and the refusals — on a real local socket.

    A mocked socket would prove the parser and nothing about the thing that
    actually goes wrong on a desk: packets that are not CSI, a second core, or
    silence. So this binds a real port on loopback and sends real datagrams.
    """

    def loopback(self, **kwargs):
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        address = receiver.getsockname()
        kwargs.setdefault("timeout", 2.0)
        source = nexmon.NexmonSource(sock=receiver, **kwargs)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(sender.close)
        self.addCleanup(source.close)
        return sender, address, source

    def test_a_frame_arrives_over_the_socket_with_its_arrival_time(self):
        sender, address, source = self.loopback(rate=25.0)
        sender.sendto(nexmon.build_payload(TestNexmonPackets().tones(64),
                                           sequence=7), address)
        frame = next(source)
        self.assertEqual(len(frame.values), 64)
        self.assertAlmostEqual(frame.t, 0.0, places=6)
        self.assertEqual(source.frames, 1)
        self.assertIn("64 tones", source.describe())
        self.assertIn("20 MHz", source.describe())

    def test_a_walk_detected_through_the_socket_matches_the_simulator(self):
        """
        The claim the whole hardware seam rests on: frames encoded the way the
        firmware encodes them, sent over UDP, reach the detector as the same
        walk. Scaled to the receiver's int16 range so quantisation is the real
        kind, not the simulator's.
        """
        simulated = list(csi.SyntheticSource(scenario="walk", seed=37, seconds=20.0))
        # A clock that ticks one frame per packet: loopback delivers 400 frames
        # in a few milliseconds, and a detector timestamped by real arrivals
        # would think the whole walk happened at t=0.
        ticks = [0.0]

        def clock():
            now = ticks[0]
            ticks[0] += 0.05
            return now

        sender, address, source = self.loopback(clock=clock)
        detector = detect.MotionDetector(rate=20.0)
        detections = []
        for i, frame in enumerate(simulated):
            # One at a time: a burst of 400 datagrams can overflow the receive
            # buffer and drop frames, which would be the test failing on the
            # operating system rather than on the code.
            sender.sendto(nexmon.build_payload([v * 512 for v in frame.values],
                                              sequence=i), address)
            detections.append(detector.update(next(source)))
        self.assertGreaterEqual(detection_rate(detections, 6.0, 14.5), 0.85)
        self.assertLessEqual(detection_rate(detections, 0.0, 3.0), 0.15)

    def test_silence_says_what_to_check_instead_of_hanging(self):
        _, _, source = self.loopback(timeout=0.05)
        with self.assertRaises(TimeoutError) as caught:
            next(source)
        message = str(caught.exception)
        self.assertIn("nexmon_csi", message)
        self.assertIn(str(source.port), message)

    def test_other_traffic_on_the_port_is_skipped_and_counted(self):
        sender, address, source = self.loopback()
        sender.sendto(b"hello", address)
        sender.sendto(b"\x11" * 300, address)
        sender.sendto(nexmon.build_payload(TestNexmonPackets().tones(64)), address)
        frame = next(source)
        self.assertEqual(len(frame.values), 64)
        self.assertEqual(source.rejected, 2)
        # The first refusal is kept, because that is the one worth showing a
        # user: later junk is usually the same mistake repeated.
        self.assertIn("not a nexmon CSI frame", source.rejection or "")

    def test_a_stream_of_nothing_but_junk_fails_loudly(self):
        """Skipping forever would look exactly like a working radar."""
        sender, address, source = self.loopback()
        for _ in range(nexmon.MAX_REJECTED):
            sender.sendto(b"not csi at all", address)
        with self.assertRaises(ValueError) as caught:
            next(source)
        self.assertIn("none was a nexmon CSI frame", str(caught.exception))

    def test_a_second_core_is_ignored_because_it_is_a_second_channel(self):
        """
        Interleaving two cores' frames is not one channel sampled twice — it is
        two — and it reads as violent motion. The extractor usually runs one
        core, so the first seen wins and the rest are counted.
        """
        sender, address, source = self.loopback()
        values = TestNexmonPackets().tones(64)
        sender.sendto(nexmon.build_payload(values, core=0, sequence=1), address)
        sender.sendto(nexmon.build_payload(values, core=1, sequence=2), address)
        sender.sendto(nexmon.build_payload(values, core=0, sequence=3), address)
        first = next(source)
        second = next(source)
        self.assertEqual(source.frames, 2)
        self.assertEqual(source.other_cores, 1)
        self.assertIn("other cores", source.describe())
        self.assertLess(first.t, second.t)


class TestSourceSpec(unittest.TestCase):
    def test_each_spec_kind_builds_a_source(self):
        import tempfile

        simulated = csi.open_source("sim:walk", seconds=1.0)
        self.assertEqual(simulated.length, 20)

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            path = handle.name
        csi.write_recording(csi.SyntheticSource(seconds=1.0), path)
        self.assertEqual(len(list(csi.open_source(f"file:{path}"))), 20)

    def test_a_recording_keeps_its_own_timing(self):
        """"
        A recording's frame rate comes from its timestamps, and the rate and
        length flags a live view passes are deliberately ignored for it: a rate
        that disagreed with the file would put every frequency in the view
        quietly in the wrong place.
        """
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            path = handle.name
        csi.write_recording(csi.SyntheticSource(seconds=2.0, rate=25.0), path)
        source = csi.open_source(f"file:{path}", rate=20.0, seconds=1.0)
        self.assertEqual(source.length, 50)
        self.assertAlmostEqual(source.rate, 25.0, delta=0.1)

    def test_a_serial_spec_opens_the_port_it_names(self):
        """
        A port that cannot be opened has to say which port and why, rather than
        returning a source that will never produce a frame. The name is one no
        machine has, so this checks the error path on every platform; a board
        that is present is covered by the stream tests below.
        """
        with self.assertRaises(OSError) as caught:
            csi.open_source("serial:COM250")
        message = str(caught.exception)
        self.assertIn("COM250", message)
        self.assertTrue("pyserial" in message or "baud" in message, message)

    def test_serial_auto_opens_the_board_it_found(self):
        """`--source serial:auto` has to mean the same thing as the probe does."""
        ports = [FakePort("COM1", "Communications Port"),
                 FakePort("COM4", "USB-SERIAL CH340", vid=0x1A86)]
        opened = []

        def fake_open(port, baud=serial_node.DEFAULT_BAUD):
            opened.append(port)
            return io.StringIO("S,0,10,20\nS,50,11,19\n")

        original_ports = serial_node.available_ports
        original_open = serial_node.open_serial
        serial_node.available_ports = lambda: ports
        serial_node.open_serial = fake_open
        self.addCleanup(setattr, serial_node, "available_ports", original_ports)
        self.addCleanup(setattr, serial_node, "open_serial", original_open)

        source = csi.open_source("serial:auto")
        self.addCleanup(source.close)
        self.assertEqual(opened, ["COM4"])
        self.assertEqual(source.port, "COM4")
        self.assertEqual(len(list(source)), 2)

    def test_serial_auto_with_nothing_plugged_in_says_what_to_do(self):
        original = serial_node.available_ports
        serial_node.available_ports = lambda: []
        self.addCleanup(setattr, serial_node, "available_ports", original)

        with self.assertRaises(OSError) as caught:
            csi.open_source("serial:auto")
        message = str(caught.exception)
        self.assertIn("no serial port", message)
        self.assertIn("--list", message)

    def test_an_unknown_spec_lists_what_is_available(self):
        with self.assertRaises(ValueError) as caught:
            csi.open_source("radar:now")
        message = str(caught.exception)
        self.assertIn("sim:", message)
        self.assertIn("demo", message)

    def test_a_udp_spec_binds_the_port_it_names(self):
        """A free port is borrowed and released, so CI cannot collide on 5500."""
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        source = csi.open_source(f"udp:127.0.0.1:{port}", rate=33.0)
        self.addCleanup(source.close)
        self.assertIsInstance(source, nexmon.NexmonSource)
        self.assertEqual(source.port, port)
        self.assertEqual(source.host, "127.0.0.1")
        self.assertEqual(source.rate, 33.0)

    def test_a_udp_spec_without_a_port_uses_nexmons_default(self):
        # The split is checked directly rather than by binding 5500, which a
        # real listener on the same machine may already hold.
        self.assertEqual(csi._udp_spec("udp:", ""), ("0.0.0.0", nexmon.DEFAULT_PORT))
        self.assertEqual(csi._udp_spec("nexmon:", ""), ("0.0.0.0", nexmon.DEFAULT_PORT))
        self.assertEqual(csi._udp_spec("udp:", "5501"), ("0.0.0.0", 5501))

    def test_a_udp_spec_with_a_bad_port_says_so(self):
        with self.assertRaises(ValueError) as caught:
            csi.open_source("udp:notaport")
        self.assertIn("numeric port", str(caught.exception))

    def test_the_named_scenarios_all_can_be_built(self):
        """Every name in the menu has to work, not just the ones the tests use."""
        for name in sorted(csi.SCENARIOS):
            source = csi.open_source(f"sim:{name}", seconds=1.0)
            self.assertEqual(len(list(source)), 20, f"scenario {name}")
            self.assertEqual(source.length, 20, f"scenario {name}")


# --------------------------------------------------------------- serial
# A board on a USB port is tested here as a string, which is the only way to
# separate "the firmware is wrong" from "the detector is wrong" before there is
# a board to plug in.

PROFILE = (512, 498, 733, 501, 455, 610, 521, 487)
JITTER = (0, 1, -1, 0, 1, 0, -1, 0)


def channel_lines(seconds=8.0, rate=20.0, move_at=4.0, hz=1.5):
    """
    A Mega's output: eight ADC channels, plus a hand waving from `move_at`.

    The jitter is not decoration. A profile that is bit-identical every frame
    calibrates to a floor and a spread of exactly zero, which is the degenerate
    case the detector has to survive; the real board never produces one, and the
    test below is the only place that can.
    """
    import math

    lines = ["# mega_channel_node 8 channels 20 Hz"]
    for i in range(int(seconds * rate)):
        t = i / rate
        values = [PROFILE[c] + JITTER[(i + c) % len(JITTER)]
                  for c in range(len(PROFILE))]
        if t >= move_at:
            swing = math.sin(2 * math.pi * hz * (t - move_at))
            for c in (2, 3, 4, 5):          # the channels the hand is nearest
                values[c] += int(round(90 * swing))
        lines.append(f"S,{round(t * 1000)},{','.join(str(v) for v in values)}")
    return "\n".join(lines) + "\n"


class QuietPort:
    """A port that is open and delivering nothing, which is the usual failure."""

    def __init__(self):
        self.closed = False

    def readline(self):
        return b""

    def close(self):
        self.closed = True


class TestSerialProtocol(unittest.TestCase):
    def test_a_sample_line_is_a_list_of_channel_values(self):
        kind, millis, values = serial_node.parse_line("S,1204,512,498,733\r\n")
        self.assertEqual(kind, "channels")
        self.assertEqual(millis, 1204)
        self.assertEqual([v.real for v in values], [512, 498, 733])
        self.assertEqual([v.imag for v in values], [0, 0, 0])

    def test_a_csi_line_is_real_imaginary_pairs(self):
        kind, _, values = serial_node.parse_line("C,40,-12,3,10,4")
        self.assertEqual(kind, "csi")
        self.assertEqual(values, [complex(-12, 3), complex(10, 4)])

    def test_comments_and_blank_lines_are_not_frames(self):
        """The sketch prints a banner; a banner must not count as junk."""
        for line in ("", "   \n", "# mega_channel_node 8 channels 20 Hz"):
            self.assertIsNone(serial_node.parse_line(line))

    def test_malformed_lines_say_what_was_expected(self):
        """A wrong baud rate or a stray println must be legible, not silent."""
        cases = {
            "hello": "neither a sensor sample",
            "S,12": "at least one value",
            "S,now,1,2": "not an integer",
            "S,12,1,x": "not an integer",
            "C,12,1,2,3": "odd number",
        }
        for line, expected in cases.items():
            with self.assertRaises(ValueError, msg=line) as caught:
                serial_node.parse_line(line)
            self.assertIn(expected, str(caught.exception), line)

    def test_a_spec_separates_port_from_baud(self):
        self.assertEqual(serial_node.parse_spec("COM3"),
                         ("COM3", serial_node.DEFAULT_BAUD))
        self.assertEqual(serial_node.parse_spec("/dev/ttyACM0:9600"),
                         ("/dev/ttyACM0", 9600))
        self.assertEqual(serial_node.parse_spec("/dev/serial/by-id/x:115200"),
                         ("/dev/serial/by-id/x", 115200))
        for bad in ("", ":9600"):
            with self.assertRaises(ValueError, msg=bad):
                serial_node.parse_spec(bad)


class FakePort:
    """The attributes pyserial's ListPortInfo gives us, and nothing else."""

    def __init__(self, device, description="", hwid="", vid=None, pid=None):
        self.device = device
        self.description = description
        self.hwid = hwid
        self.vid = vid
        self.pid = pid


class TestPortDiscovery(unittest.TestCase):
    def test_a_ch340_counts_as_a_likely_board(self):
        """An Elegoo Mega does not identify as an Arduino; its CH340 does."""
        confidence, reason = serial_node.identify_port(
            FakePort("COM4", "USB-SERIAL CH340", "USB VID:PID=1A86:7523", 0x1A86, 0x7523))
        self.assertEqual(confidence, 3)
        self.assertIn("CH340", reason)

    def test_an_unrelated_usb_adapter_is_not_called_a_board(self):
        confidence, reason = serial_node.identify_port(
            FakePort("COM7", "USB Serial Port", "USB VID:PID=0403:6001", 0x0403, 0x6001))
        self.assertEqual(confidence, 1)
        self.assertIn("not a known board chip", reason)

    def test_the_built_in_port_is_the_last_resort(self):
        confidence, _ = serial_node.identify_port(FakePort("COM1", "Communications Port"))
        self.assertEqual(confidence, 0)

    def test_auto_prefers_the_board_over_the_modem(self):
        ports = [FakePort("COM1", "Communications Port"),
                 FakePort("COM7", "USB Serial Port", vid=0x0403),
                 FakePort("COM4", "USB-SERIAL CH340", vid=0x1A86)]
        self.assertEqual(serial_node.choose_port(ports).device, "COM4")

    def test_auto_is_stable_when_two_boards_are_equally_plausible(self):
        """Two Megas must not be chosen between at random on every run."""
        ports = [FakePort("COM9", vid=0x1A86), FakePort("COM4", vid=0x1A86)]
        self.assertEqual(serial_node.choose_port(ports).device, "COM4")
        self.assertEqual(serial_node.choose_port(list(reversed(ports))).device, "COM4")

    def test_no_ports_is_not_a_port(self):
        self.assertIsNone(serial_node.choose_port([]))

    def test_list_shows_every_port_and_the_one_auto_would_open(self):
        ports = [FakePort("COM1", "Communications Port"),
                 FakePort("COM5", "USB-SERIAL CH340", vid=0x1A86)]
        original = serial_node.available_ports
        serial_node.available_ports = lambda: ports
        self.addCleanup(setattr, serial_node, "available_ports", original)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = serial_node.main(["--list"])
        self.assertEqual(status, 0)
        text = out.getvalue()
        self.assertIn("COM1", text)
        self.assertIn("COM5", text)
        self.assertIn("*", text)
        self.assertIn("auto would open", text)

    def test_list_with_nothing_plugged_in_explains_the_driver(self):
        original = serial_node.available_ports
        serial_node.available_ports = lambda: []
        self.addCleanup(setattr, serial_node, "available_ports", original)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = serial_node.main(["--list"])
        self.assertEqual(status, 0)
        self.assertIn("CH340 driver", out.getvalue())

    def test_auto_with_no_ports_exits_with_a_reason(self):
        original = serial_node.available_ports
        serial_node.available_ports = lambda: []
        self.addCleanup(setattr, serial_node, "available_ports", original)

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            status = serial_node.main(["--port", "auto"])
        self.assertEqual(status, 2)
        self.assertIn("no serial port", err.getvalue())

    def test_auto_opens_the_board_it_chose_and_reads_it(self):
        """The whole command the README gives, minus the cable."""
        ports = [FakePort("COM1", "Communications Port"),
                 FakePort("COM4", "USB-SERIAL CH340", vid=0x1A86)]
        opened = []

        def fake_open(port, baud=serial_node.DEFAULT_BAUD):
            opened.append(port)
            return io.StringIO("# banner\nS,0,100,200,300\nS,50,101,205,299\n")

        original_ports = serial_node.available_ports
        original_open = serial_node.open_serial
        serial_node.available_ports = lambda: ports
        serial_node.open_serial = fake_open
        self.addCleanup(setattr, serial_node, "available_ports", original_ports)
        self.addCleanup(setattr, serial_node, "open_serial", original_open)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = serial_node.main(["--port", "auto", "--seconds", "0.1"])
        self.assertEqual(status, 0)
        self.assertEqual(opened, ["COM4"])
        text = out.getvalue()
        self.assertIn("opening COM4", text)
        self.assertIn("channel profile", text)


class TestLiveView(unittest.TestCase):
    """The per-pin panel, which is the wiring view."""

    def test_the_pin_that_moved_is_the_one_named(self):
        self.assertEqual(live.busiest_channel([0.01, 0.02, 0.4, 0.01]), 2)

    def test_a_still_room_names_no_pin(self):
        """Every channel jitters; naming the largest of four equal numbers
        would make an empty room look wired."""
        self.assertIsNone(live.busiest_channel([0.020, 0.021, 0.019]))

    def test_no_change_at_all_names_no_pin(self):
        self.assertIsNone(live.busiest_channel([]))
        self.assertIsNone(live.busiest_channel([0.0, 0.0, 0.0]))

    def test_one_pin_waking_from_dead_flat_stands_out(self):
        """A pin being plugged in is the case the panel exists for."""
        self.assertEqual(live.busiest_channel([0.0, 0.0, 0.002]), 2)

    def test_the_snapshot_carries_the_level_of_every_pin(self):
        stream = io.StringIO("S,0,100,200,300,400,500,600,700,800\n"
                             "S,50,110,210,310,410,510,610,710,810\n")
        session = live.RadarSession(serial_node.SerialSource(stream),
                                    realtime=False)
        session._run()
        state = session.snapshot()
        self.assertEqual(state["levels"],
                         [110, 210, 310, 410, 510, 610, 710, 810])
        self.assertEqual(state["subcarriers"], 8)
        self.assertFalse(state["doppler"])

    def test_before_the_first_frame_there_are_no_levels(self):
        session = live.RadarSession(serial_node.SerialSource(io.StringIO("")),
                                    realtime=False)
        state = session.snapshot()
        self.assertEqual(state["levels"], [])
        self.assertIsNone(state["busiest"])

    def test_a_sixteen_pin_board_reaches_the_panel_intact(self):
        """A Mega has A0–A15; nothing between the line and the panel may assume 8."""
        lines = ["# mega_channel_node 16 channels 20 Hz"]
        for i in range(60):
            values = [512] * 16
            values[15] = 512 + (80 if i % 2 else 0)   # only A15 is wired
            lines.append("S,%d,%s" % (i * 50, ",".join(str(v) for v in values)))
        session = live.RadarSession(
            serial_node.SerialSource(io.StringIO("\n".join(lines) + "\n")),
            realtime=False)
        session._run()
        state = session.snapshot()
        self.assertEqual(len(state["levels"]), 16)
        self.assertEqual(state["busiest"], 15)

    def test_a_csi_source_does_not_name_a_pin(self):
        """Its rows are tones; the waterfall is where those are read."""
        session = live.RadarSession(csi.open_source("sim:walk", seconds=1.0),
                                    realtime=False)
        session._run()
        state = session.snapshot()
        self.assertTrue(state["doppler"])
        self.assertIsNone(state["busiest"])
        self.assertEqual(len(state["levels"]), 64)


class TestLivePage(unittest.TestCase):
    """
    The page is one long Python string, so nothing else checks that the script
    in it is even valid JavaScript. A duplicate `const` in a draw function is
    invisible to every Python test and blanks the entire view, so the parser is
    asked directly.
    """

    def test_the_page_script_parses(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        match = re.search(r"<script>(.*?)</script>", live.PAGE, re.S)
        self.assertIsNotNone(match, "the page has no <script> block")
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                         encoding="utf-8") as handle:
            handle.write(match.group(1))
            path = handle.name
        self.addCleanup(os.unlink, path)
        result = subprocess.run([node, "--check", path],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class TestBringUpCommands(unittest.TestCase):
    """The two commands a first run types, failing without a traceback."""

    def _fail_open(self):
        def boom(*args, **kwargs):
            raise OSError("serial:auto found no serial port. Plug the board in")

        original = csi.open_source
        csi.open_source = boom
        self.addCleanup(setattr, csi, "open_source", original)

    def test_watch_says_the_board_is_missing_rather_than_crashing(self):
        from wifi_radar import watch

        self._fail_open()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            status = watch.main(["--source", "serial:auto"])
        self.assertEqual(status, 2)
        self.assertIn("no serial port", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_live_says_the_board_is_missing_rather_than_crashing(self):
        from wifi_radar import live

        self._fail_open()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = live.main(["--source", "serial:auto"])
        self.assertEqual(status, 2)
        self.assertIn("no serial port", out.getvalue())


class TestSerialSource(unittest.TestCase):
    def test_samples_become_frames_on_the_devices_own_clock(self):
        stream = io.StringIO("# banner\nS,5000,1,2\nS,5050,3,4\n")
        source = serial_node.SerialSource(stream, port="COM3")
        frames = list(source)
        self.assertEqual(len(frames), 2)
        # Relative to the first sample, so a session never starts at 5000s.
        self.assertEqual([f.t for f in frames], [0.0, 0.05])
        self.assertEqual(source.frames, 2)
        self.assertFalse(source.doppler)
        self.assertIn("2 sensor channels", source.describe())
        self.assertIn("no Doppler", source.describe())

    def test_the_32_bit_clock_is_undone_rather_than_jumping_backwards(self):
        """`millis()` wraps after 49.7 days; a session must not restart there."""
        stream = io.StringIO("S,4294966000,1,2\nS,4294967000,1,2\n"
                             "S,728,1,2\nS,1728,1,2\n")
        times = [f.t for f in serial_node.SerialSource(stream)]
        self.assertEqual(times, [0.0, 1.0, 2.024, 3.024])

    def test_a_csi_line_switches_the_source_to_csi(self):
        stream = io.StringIO("C,0,1,2,3,4\nC,50,1,2,3,4\n")
        source = serial_node.SerialSource(stream)
        next(source)
        self.assertTrue(source.doppler)
        self.assertIn("CSI", source.describe())

    def test_unreadable_lines_are_counted_and_skipped(self):
        stream = io.StringIO("junk\nS,0,10,20\nmore junk\nS,50,12,21\n")
        source = serial_node.SerialSource(stream)
        frames = list(source)
        self.assertEqual(len(frames), 2)
        self.assertEqual(source.rejected, 2)
        self.assertIn("neither a sensor sample", source.rejection)

    def test_a_port_that_says_nothing_but_noise_fails_loudly(self):
        """Otherwise a misconfigured port looks like a perfectly quiet room."""
        stream = io.StringIO("\n".join(["nonsense"] * serial_node.MAX_REJECTED) + "\n")
        with self.assertRaises(ValueError) as caught:
            list(serial_node.SerialSource(stream, port="COM3"))
        message = str(caught.exception)
        self.assertIn("COM3", message)
        self.assertIn("baud", message)

    def test_silence_on_an_open_port_names_what_to_check(self):
        ticks = [0.0]

        def clock():
            ticks[0] += 1.0
            return ticks[0]

        source = serial_node.SerialSource(QuietPort(), port="COM7",
                                          timeout=10.0, clock=clock)
        with self.assertRaises(TimeoutError) as caught:
            next(source)
        message = str(caught.exception)
        self.assertIn("COM7", message)
        self.assertIn("sketch", message)
        self.assertIn("serial monitor", message)

    def test_a_saved_capture_ends_instead_of_timing_out(self):
        """A replay of a captured log is finite; a quiet board is an error."""
        source = serial_node.SerialSource(io.StringIO("S,0,1,2\n"))
        self.assertEqual(len(list(source)), 1)

    def test_an_empty_stream_is_an_empty_session(self):
        self.assertEqual(list(serial_node.SerialSource(io.StringIO(""))), [])

    def test_a_channel_node_feeds_the_detector_without_inventing_speed(self):
        """
        The whole point of the serial path: a board with no radio produces a
        frame the existing detector can measure, without a frequency axis it
        cannot justify.
        """
        source = serial_node.SerialSource(io.StringIO(channel_lines()))
        detector = detect.MotionDetector(rate=20.0, doppler=False)
        detections = collect(source, detector)
        self.assertGreater(len(detections), 100)
        self.assertEqual(detection_rate(detections, 1.0, 3.9), 0.0)
        self.assertGreater(detection_rate(detections, 4.2, 7.9), 0.9)
        for detection in detections:
            self.assertIsNone(detection.peak_hz)
            self.assertIsNone(detection.band_power)


if __name__ == "__main__":
    unittest.main(verbosity=2)
