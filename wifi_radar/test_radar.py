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
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from wifi_radar import (csi, detect, doctor, fakeboard, handtrial, live, nexmon,
                        selftest, serial_node)


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

    def test_the_high_pass_is_on_by_default(self):
        """A filter that has to be remembered is one that is not there."""
        self.assertGreater(detect.DEFAULT_HIGHPASS_SECONDS, 0)
        self.assertEqual(self.detector().highpass_seconds,
                         detect.DEFAULT_HIGHPASS_SECONDS)

    def test_highpass_zero_turns_the_filter_off(self):
        from wifi_radar import watch

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = watch.main(["--source", "sim:walk", "--seconds", "1",
                                 "--highpass", "0", "--fast", "--every", "10"])
        self.assertEqual(status, 0)

    def test_a_high_pass_follows_slow_drift_instead_of_reporting_it(self):
        """
        A wire on an ADC pin swings on a slow envelope, and the first pin after
        the mux move carries most of it. The filter follows that baseline, so the
        drift stops arriving as change; a fast hand does not get followed.
        """
        import math

        def frames(seconds=20.0):
            for i in range(int(seconds * 20)):
                t = i / 20.0
                swing = 10.0 * math.sin(2 * math.pi * 0.1 * t)
                values = [complex(300.0, 0)] * 8
                values[0] = complex(300.0 + swing, 0)   # A0 leads the swing
                yield csi.Frame(t, values)

        def spread(**kw):
            det = self.detector(calibrate_seconds=2.0, **kw)
            metrics = [d.metric for d in (det.update(f) for f in frames())
                       if not d.calibrating]
            return statistics.pstdev(metrics), statistics.mean(metrics)

        plain_sd, plain_mean = spread(highpass_seconds=0.0)
        filtered_sd, filtered_mean = spread(highpass_seconds=0.5)
        self.assertLess(filtered_sd, plain_sd)
        self.assertLess(filtered_mean, plain_mean)

    def test_a_fast_change_on_one_channel_survives_the_high_pass(self):
        """The filter must not trade drift rejection for latency."""
        def frames():
            for i in range(60):
                noise = 0.3 if i % 2 else -0.3
                values = [complex(300 + noise, 0)] * 8
                if 40 <= i < 43:            # a 0.15 s edge, like a hand arriving
                    values[0] = complex(340 + noise, 0)
                yield i, csi.Frame(i / 20.0, values)

        det = self.detector(calibrate_seconds=1.0, highpass_seconds=0.5)
        quiet, edge = [], []
        for i, frame in frames():
            m = det.update(frame).metric
            if 20 <= i < 40:            # settled, before the hand
                quiet.append(m)
            elif 40 <= i < 48:          # the edge up and the edge back down
                edge.append(m)
        self.assertGreater(max(edge), 10 * max(quiet))

    @staticmethod
    def adc_drift(seconds=30.0, hz=0.1, amplitude=10.0):
        """A wire on A0, swinging on the slow envelope it picks up from the air."""
        import math

        for i in range(int(seconds * 20)):
            t = i / 20.0
            values = [complex(300.0, 0)] * 8
            values[0] = complex(300.0 + amplitude * math.sin(2 * math.pi * hz * t), 0)
            yield csi.Frame(t, values)

    def test_automatic_calibration_stops_at_the_minimum_when_nothing_drifts(self):
        """A radio's floor is flat: there is nothing to wait for."""
        detector = self.detector(calibrate_seconds=None)
        for frame in csi.SyntheticSource(scenario="quiet", seconds=10.0, seed=5):
            detector.update(frame)
        self.assertEqual(detector.calibration_reason, "settled")
        self.assertAlmostEqual(detector.calibration_seconds, 2.0, places=1)

    def test_automatic_calibration_follows_a_drifting_floor_for_whole_cycles(self):
        """
        The point of the longer window, measured rather than asserted: its floor
        is the session's floor, where a two-second sample of a ten-second
        envelope lands on whichever part of it was sampled.
        """
        automatic = self.detector(calibrate_seconds=None)
        for frame in self.adc_drift(30.0):
            automatic.update(frame)
        self.assertEqual(automatic.calibration_reason, "cycles")
        self.assertGreater(automatic.calibration_seconds, 8.0)

        reference = self.detector(calibrate_seconds=40.0)
        for frame in self.adc_drift(40.0):
            reference.update(frame)
        self.assertLess(abs(automatic.floor / reference.floor - 1), 0.10,
                        f"auto floor {automatic.floor} against {reference.floor}")

        minimum = self.detector(calibrate_seconds=2.0)
        for frame in self.adc_drift(40.0):
            minimum.update(frame)
        self.assertGreater(abs(minimum.floor / reference.floor - 1), 0.10,
                           "the minimum window was already representative, so "
                           "the longer one proves nothing")

    def test_automatic_calibration_stops_at_its_limit(self):
        detector = self.detector(calibrate_seconds=None, calibrate_max_seconds=5.0)
        for frame in self.adc_drift(10.0):
            detector.update(frame)
        self.assertEqual(detector.calibration_reason, "limit")
        self.assertAlmostEqual(detector.calibration_seconds, 5.0, places=1)

    def test_a_number_still_fixes_the_window_exactly(self):
        detector = self.detector(calibrate_seconds=1.0)
        for frame in self.adc_drift(5.0):
            detector.update(frame)
        self.assertEqual(detector.calibration_reason, "fixed")
        self.assertAlmostEqual(detector.calibration_seconds, 1.0, places=1)

    def test_without_the_high_pass_there_is_no_drift_to_wait_for(self):
        """With the filter off the two series are identical, and the detector "
        says so rather than looking at a number that cannot mean anything."""
        detector = self.detector(calibrate_seconds=None, highpass_seconds=0.0)
        for frame in self.adc_drift(10.0):
            detector.update(frame)
        self.assertEqual(detector.calibration_reason, "settled")

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


class FakeBootloader:
    """
    A port that answers stk500v2 GET_SYNC the way a Mega 2560 does.

    The reply is the bytes a real board sent, kept verbatim: a test written from
    the datasheet would pass against a frame no bootloader sends.
    """

    REPLY = bytes.fromhex("1b0000020e30c0e7")

    def __init__(self, reply=REPLY):
        self.reply = reply
        self.writes = 0
        self.dtr = None
        self.closed = False

    def write(self, data):
        self.writes += 1

    def flush(self):
        pass

    def read(self, size=64):
        if self.reply and self.writes:
            data, self.reply = self.reply, b""
            return data
        return b""

    def close(self):
        self.closed = True


class TestDoctor(unittest.TestCase):
    """The bring-up doctor: identity, bootloader, and is anything printing."""

    def test_the_bytes_the_mega_sent_are_read_as_a_sync(self):
        replies, status = doctor.parse_sync_replies(FakeBootloader.REPLY * 3)
        self.assertEqual(replies, 3)
        self.assertEqual(status, 0xC0)
        self.assertTrue(doctor.status_ok(status))

    def test_noise_is_not_a_sync(self):
        self.assertEqual(doctor.parse_sync_replies(b"hello, world\r\n" * 4),
                         (0, None))

    def test_a_reply_without_the_token_is_not_a_sync(self):
        self.assertEqual(doctor.parse_sync_replies(
            bytes.fromhex("1b0000020030c0e7")), (0, None))

    def test_the_probe_resets_the_board_and_closes_the_port(self):
        port = FakeBootloader()
        result = doctor.probe_bootloader(
            "COM5", sync_seconds=0.05, interval=0.0,
            open_port=lambda device, baud=115200: port, sleep=lambda s: None)
        self.assertTrue(result["answered"])
        self.assertEqual(result["replies"], 1)
        self.assertIs(port.dtr, True)      # the reset edge was driven
        self.assertTrue(port.closed)

    def test_a_port_that_cannot_be_written_to_is_an_error_not_a_hang(self):
        """A wedged USB chip blocks writes; the probe must surface that."""
        class Frozen(FakeBootloader):
            def write(self, data):
                raise OSError("write timeout")

        port = Frozen()
        result = doctor.probe_bootloader(
            "COM5", sync_seconds=0.05, interval=0.0,
            open_port=lambda device, baud=115200: port, sleep=lambda s: None)
        self.assertFalse(result["answered"])
        self.assertIn("write failed", result["error"])
        self.assertTrue(port.closed)

    def test_a_board_that_never_answers_is_not_called_alive(self):
        port = FakeBootloader(reply=b"")
        result = doctor.probe_bootloader(
            "COM5", sync_seconds=0.05, interval=0.0,
            open_port=lambda device, baud=115200: port, sleep=lambda s: None)
        self.assertFalse(result["answered"])
        self.assertEqual(result["replies"], 0)

    def test_listening_reads_samples_at_the_boards_own_rate(self):
        heard = doctor.listen(
            "COM5", seconds=0.05, sleep=lambda s: None,
            open_port=lambda device, baud=115200: io.StringIO(
                "# mega_channel_node 8 channels 20 Hz\n"
                "S,100,1,2\nS,150,3,4\nS,200,5,6\n"))
        self.assertEqual(heard["samples"], 3)
        self.assertIn("mega_channel_node", heard["banner"])
        self.assertEqual(heard["first_sample"], "S,100,1,2")
        self.assertAlmostEqual(heard["rate"], 20.0)

    def test_listening_tells_noise_apart_from_silence(self):
        heard = doctor.listen(
            "COM5", seconds=0.05, sleep=lambda s: None,
            open_port=lambda device, baud=115200: io.StringIO("hello\nhello\n"))
        self.assertEqual(heard["samples"], 0)
        self.assertEqual(heard["rejected"], 2)
        self.assertIsNotNone(heard["rejection"])
        self.assertGreater(heard["bytes"], 0)

    def test_json_payload_for_a_printing_board(self):
        boot = {"answered": True, "replies": 18, "status": 0xC0,
                "error": None, "duration": 2.1}
        heard = {"error": None, "samples": 61, "lines": 62, "bytes": 2400,
                 "banner": "# mega_channel_node 8 channels 20 Hz",
                 "first_sample": "S,50,1,2", "rejected": 0, "rejection": None,
                 "rate": 20.0123, "duration": 4.1}
        verdict, status = doctor.classify(boot, heard)
        doc = doctor.as_json(
            "COM5", FakePort("COM5", "Arduino Mega 2560 (COM5)", vid=0x2341,
                             pid=0x0042), boot, heard, verdict, status)
        self.assertEqual(verdict, "printing")
        self.assertEqual(doc["exit_status"], 0)
        self.assertEqual(doc["port"], "COM5")
        self.assertEqual(doc["board"]["vid"], 0x2341)
        self.assertEqual(doc["board"]["pid"], 0x0042)
        self.assertEqual(doc["sketch"]["rate_hz"], 20.01)
        self.assertTrue(doc["sketch"]["printing"])
        self.assertTrue(doc["bootloader"]["status_ok"])
        self.assertEqual(doc["bootloader"]["status"], 0xC0)
        json.dumps(doc)                      # must be serialisable as it stands

    def test_json_payload_for_silent_firmware(self):
        boot = {"answered": True, "replies": 18, "status": 0xC0,
                "error": None, "duration": 2.1}
        heard = {"error": None, "samples": 0, "lines": 0, "bytes": 0,
                 "banner": None, "first_sample": None, "rejected": 0,
                 "rejection": None, "rate": None, "duration": 4.0}
        verdict, status = doctor.classify(boot, heard)
        doc = doctor.as_json("COM5", None, boot, heard, verdict, status)
        self.assertEqual((verdict, status), ("silent_firmware", 1))
        self.assertEqual(doc["sketch"]["samples"], 0)
        self.assertFalse(doc["sketch"]["printing"])
        self.assertIsNone(doc["sketch"]["rate_hz"])
        self.assertIn("sketch", doc["message"])

    def test_json_mode_prints_one_parseable_object(self):
        """Scripts get a document and nothing else, with the same exit status."""
        ports = [FakePort("COM1", "Communications Port"),
                 FakePort("COM5", "Arduino Mega 2560 (COM5)", vid=0x2341)]
        streams = [FakeBootloader(),
                   io.StringIO("# banner\nS,0,10,20\nS,50,11,19\n")]

        original_ports = serial_node.available_ports
        original_open = serial_node.open_serial
        serial_node.available_ports = lambda: ports
        serial_node.open_serial = lambda device, baud=115200: streams.pop(0)
        self.addCleanup(setattr, serial_node, "available_ports", original_ports)
        self.addCleanup(setattr, serial_node, "open_serial", original_open)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = doctor.main(["--json", "--sync-seconds", "0.05",
                                  "--seconds", "0.05"])
        self.assertEqual(status, 0)
        doc = json.loads(out.getvalue())
        self.assertEqual(doc["verdict"], "printing")
        self.assertEqual(doc["exit_status"], 0)
        self.assertEqual(doc["port"], "COM5")
        self.assertEqual(doc["sketch"]["samples"], 2)

    def test_json_mode_with_no_ports_exits_2_without_stderr(self):
        original = serial_node.available_ports
        serial_node.available_ports = lambda: []
        self.addCleanup(setattr, serial_node, "available_ports", original)

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = doctor.main(["--json"])
        self.assertEqual(status, 2)
        doc = json.loads(out.getvalue())
        self.assertEqual(doc["verdict"], "no_port")
        self.assertEqual(doc["exit_status"], 2)
        self.assertIsNone(doc["board"])
        self.assertEqual(err.getvalue(), "")

    def test_a_printing_board_is_ok(self):
        text, status = doctor.report(
            "COM5",
            FakePort("COM5", "Arduino Mega 2560 (COM5)", vid=0x2341, pid=0x0042),
            {"answered": True, "replies": 24, "status": 0xC0,
             "error": None, "duration": 2.0},
            {"error": None, "samples": 62, "lines": 63, "bytes": 1600,
             "banner": "# mega_channel_node 8 channels 20 Hz",
             "first_sample": "S,50,1,2", "rejected": 0, "rejection": None,
             "rate": 20.0, "duration": 4.1})
        self.assertEqual(status, 0)
        self.assertIn("COM5", text)
        self.assertIn("20.0/s", text)
        self.assertIn("watch --source serial:COM5", text)

    def test_the_scan_finds_the_rate_the_firmware_uses(self):
        """A silent 115200 and a talking 9600 must end at 9600, and stop."""
        def opener(device, baud=115200):
            if baud == 9600:
                return io.StringIO("# banner\nS,0,10,20\nS,50,11,19\n")
            return io.StringIO("")

        heard, tried = doctor.find_baud(
            "COM5", 115200, seconds=0.05, scan_seconds=0.05,
            open_port=opener, sleep=lambda s: None)
        self.assertEqual(heard["samples"], 2)
        self.assertEqual(heard["baud"], 9600)
        self.assertEqual(heard["baud_requested"], 115200)
        self.assertEqual([entry["baud"] for entry in tried], [115200, 9600])

    def test_a_scan_that_finds_nothing_reports_every_rate_it_tried(self):
        heard, tried = doctor.find_baud(
            "COM5", 115200, seconds=0.02, scan_seconds=0.02,
            open_port=lambda device, baud=115200: io.StringIO("hello\n"),
            sleep=lambda s: None)
        self.assertEqual(heard["samples"], 0)
        self.assertGreater(len(tried), 1)
        self.assertIn(9600, [entry["baud"] for entry in tried])

    def test_scanning_can_be_turned_off(self):
        heard, tried = doctor.find_baud(
            "COM5", 115200, seconds=0.02, scan_seconds=0.0,
            open_port=lambda device, baud=115200: io.StringIO(""),
            sleep=lambda s: None)
        self.assertEqual([entry["baud"] for entry in tried], [115200])
        self.assertEqual(heard["baud"], 115200)

    def test_a_printing_board_at_another_rate_is_named_in_the_report(self):
        boot = {"answered": True, "replies": 18, "status": 0xC0,
                "error": None, "duration": 2.1}
        heard = {"error": None, "samples": 2, "lines": 2, "bytes": 60,
                 "banner": None, "first_sample": "S,0,10,20",
                 "rejected": 0, "rejection": None, "rate": 20.0,
                 "duration": 0.1, "baud": 9600, "baud_requested": 115200,
                 "tried": [{"baud": 115200, "bytes": 0, "samples": 0,
                            "rejected": 0},
                           {"baud": 9600, "bytes": 60, "samples": 2,
                            "rejected": 0}]}
        verdict, status = doctor.classify(boot, heard)
        text, status = doctor.report("COM5", FakePort("COM5", vid=0x2341),
                                     boot, heard)
        self.assertEqual((verdict, status), ("printing", 0))
        self.assertIn("--baud 9600", text)
        self.assertIn("9600", text)
        doc = doctor.as_json("COM5", FakePort("COM5", vid=0x2341),
                             boot, heard, verdict, status)
        self.assertEqual(doc["sketch"]["baud"], 9600)
        self.assertEqual(doc["sketch"]["baud_requested"], 115200)
        self.assertEqual(doc["bauds_tried"], [115200, 9600])
        self.assertIn("9600", doc["message"])

    def test_the_command_scans_when_the_requested_rate_is_silent(self):
        ports = [FakePort("COM5", "Arduino Mega 2560 (COM5)", vid=0x2341)]
        opened = []

        def opener(device, baud=115200):
            opened.append(baud)
            if len(opened) == 1:            # the bootloader probe
                return FakeBootloader()
            if baud == 9600:
                return io.StringIO("S,0,10,20\nS,50,11,19\n")
            return io.StringIO("")

        original_ports = serial_node.available_ports
        original_open = serial_node.open_serial
        serial_node.available_ports = lambda: ports
        serial_node.open_serial = opener
        self.addCleanup(setattr, serial_node, "available_ports", original_ports)
        self.addCleanup(setattr, serial_node, "open_serial", original_open)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = doctor.main(["--json", "--sync-seconds", "0.05",
                                  "--seconds", "0.05", "--scan-seconds", "0.05"])
        self.assertEqual(status, 0)
        doc = json.loads(out.getvalue())
        self.assertEqual(doc["sketch"]["baud"], 9600)
        self.assertEqual(doc["verdict"], "printing")
        self.assertEqual(doc["bauds_tried"][:2], [115200, 9600])

    def test_a_silent_board_that_answers_its_bootloader_is_a_sketch_problem(self):
        text, status = doctor.report(
            "COM5", FakePort("COM5", vid=0x2341),
            {"answered": True, "replies": 24, "status": 0xC0,
             "error": None, "duration": 2.0},
            {"error": None, "samples": 0, "lines": 0, "bytes": 0,
             "banner": None, "first_sample": None, "rejected": 0,
             "rejection": None, "rate": None, "duration": 4.0})
        self.assertEqual(status, 1)
        self.assertIn("not printing", text)
        self.assertIn("mega_channel_node", text)

    def test_something_else_printing_is_named_as_the_wrong_protocol(self):
        text, status = doctor.report(
            "COM5", FakePort("COM5", vid=0x2341),
            {"answered": False, "replies": 0, "status": None,
             "error": None, "duration": 2.0},
            {"error": None, "samples": 0, "lines": 33, "bytes": 400,
             "banner": None, "first_sample": None, "rejected": 33,
             "rejection": "a line starting 'hello' is neither a sensor sample",
             "rate": None, "duration": 4.0})
        self.assertEqual(status, 1)
        self.assertIn("wrong protocol", text)
        self.assertIn("--baud", text)

    def test_a_port_that_cannot_be_opened_exits_2(self):
        busy = "cannot open COM5 at 115200 baud: access denied"
        text, status = doctor.report(
            "COM5", FakePort("COM5", vid=0x2341),
            {"answered": False, "replies": 0, "status": None,
             "error": busy, "duration": 0.0},
            {"error": busy, "samples": 0, "lines": 0, "bytes": 0,
             "banner": None, "first_sample": None, "rejected": 0,
             "rejection": None, "rate": None, "duration": 0.0})
        self.assertEqual(status, 2)
        self.assertIn("access denied", text)

    def test_the_command_runs_end_to_end_against_a_fake_board(self):
        ports = [FakePort("COM1", "Communications Port"),
                 FakePort("COM5", "Arduino Mega 2560 (COM5)", vid=0x2341)]
        streams = [FakeBootloader(),
                   io.StringIO("# banner\nS,0,10,20\nS,50,11,19\n")]

        original_ports = serial_node.available_ports
        original_open = serial_node.open_serial
        serial_node.available_ports = lambda: ports
        serial_node.open_serial = lambda device, baud=115200: streams.pop(0)
        self.addCleanup(setattr, serial_node, "available_ports", original_ports)
        self.addCleanup(setattr, serial_node, "open_serial", original_open)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = doctor.main(["--sync-seconds", "0.05", "--seconds", "0.05"])
        self.assertEqual(status, 0)
        text = out.getvalue()
        self.assertIn("COM5", text)
        self.assertIn("2 samples", text)

    def test_no_ports_exits_2_and_names_the_driver(self):
        original = serial_node.available_ports
        serial_node.available_ports = lambda: []
        self.addCleanup(setattr, serial_node, "available_ports", original)

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            status = doctor.main([])
        self.assertEqual(status, 2)
        self.assertIn("CH340", err.getvalue())


class PortLikeFile(io.RawIOBase):
    """
    pyserial's `Serial` is an `io.RawIOBase` with `in_waiting` and `baudrate`.

    That shape — an IOBase that is a port, not a file — is what used to be read
    as a finished capture, so it is the shape a regression test has to have.
    """

    baudrate = 115200
    in_waiting = 0

    def readable(self):
        return True

    def readline(self, size=-1):
        return b""


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

    def test_auto_baud_in_a_spec_is_none_not_a_number(self):
        self.assertEqual(serial_node.parse_spec("COM5:auto"), ("COM5", None))
        self.assertEqual(serial_node.parse_spec("/dev/ttyACM0:AUTO"),
                         ("/dev/ttyACM0", None))
        self.assertIsNone(serial_node.parse_baud("auto"))
        self.assertEqual(serial_node.parse_baud("9600"), 9600)


class TestAutoBaud(unittest.TestCase):
    """The rate is found the way the doctor finds it, once, at open time."""

    def test_discovery_asks_the_doctor_which_rate_works(self):
        seen = {}

        def fake_find(port, baud=serial_node.DEFAULT_BAUD, **kwargs):
            seen["port"] = port
            seen["baud"] = baud
            return ({"samples": 40, "baud": 9600},
                    [{"baud": 115200}, {"baud": 9600}])

        with mock.patch.object(doctor, "find_baud", fake_find):
            rate, tried = serial_node.discover_baud("COM5")
        self.assertEqual((rate, seen["port"]), (9600, "COM5"))
        self.assertEqual([item["baud"] for item in tried], [115200, 9600])

    def test_discovery_that_finds_nothing_names_the_rates_and_the_doctor(self):
        with mock.patch.object(doctor, "find_baud",
                               lambda port, **kw: ({"samples": 0},
                                                   [{"baud": 115200},
                                                    {"baud": 9600}])):
            with self.assertRaises(OSError) as caught:
                serial_node.discover_baud("COM5")
        message = str(caught.exception)
        self.assertIn("115200, 9600", message)
        self.assertIn("wifi_radar.doctor", message)

    def test_a_source_says_when_its_rate_was_found(self):
        found = serial_node.SerialSource(io.StringIO(""), port="COM5", baud=9600,
                                         baud_found=True)
        self.assertIn("9600 baud (found by scanning)", found.describe())
        asked = serial_node.SerialSource(io.StringIO(""), port="COM5", baud=9600)
        self.assertIn("9600 baud,", asked.describe())
        self.assertNotIn("found by scanning", asked.describe())

    def test_the_page_reads_the_rate_the_stream_is_actually_using(self):
        class Stream(io.StringIO):
            baudrate = 57600

        source = serial_node.SerialSource(Stream(""), port="COM5", baud=115200)
        self.assertEqual(source.baud_in_use, 57600)
        self.assertIn("57600", source.describe())

    def test_the_idle_timeout_reaches_a_serial_source(self):
        """A board being reset by hand needs longer than the ten-second default."""
        with mock.patch.object(serial_node, "open_serial",
                               lambda port, baud, timeout=0.2: io.StringIO("")):
            source = csi.open_source("serial:COM5", timeout=45.0)
        self.assertEqual(source.timeout, 45.0)

    def test_a_serial_spec_with_auto_discovers_before_opening(self):
        opened = {}

        def fake_open(port, baud=serial_node.DEFAULT_BAUD, timeout=0.2):
            opened["port"], opened["baud"] = port, baud
            return io.StringIO("")

        with mock.patch.object(serial_node, "discover_baud",
                               lambda port, **kw: (9600, [{"baud": 9600}])), \
                mock.patch.object(serial_node, "open_serial", fake_open):
            source = csi.open_source("serial:COM5:auto")
        self.assertEqual(opened, {"port": "COM5", "baud": 9600})
        self.assertTrue(source.baud_found)


class TestHandTrial(unittest.TestCase):
    """The guided antenna trial: schedule, recording, and per-segment z."""

    def test_the_schedule_is_a_contiguous_timeline(self):
        from wifi_radar import handtrial

        marks = handtrial.schedule(10, 5, [20, 30])
        self.assertEqual([mark["name"] for mark in marks],
                         ["idle", "20 cm", "30 cm"])
        self.assertEqual(marks[0], {"name": "idle", "start": 0.0, "end": 10.0})
        self.assertEqual(marks[-1]["end"], 20.0)
        for before, after in zip(marks, marks[1:]):
            self.assertEqual(before["end"], after["start"])

    def test_a_hand_that_arrives_shows_up_in_the_segment_it_arrived_in(self):
        from wifi_radar import handtrial

        def frames():
            for i in range(8 * 20):
                t = i / 20.0
                values = [complex(300.0, 0)] * 8
                values[0] = complex(340.0 if t >= 4.0 else 300.0, 0)
                yield csi.Frame(t, values)

        rows, _ = handtrial.analyse(list(frames()), handtrial.schedule(4, 4, [20]),
                                    detect.DEFAULT_HIGHPASS_SECONDS, 20.0,
                                    calibration=2.0)
        self.assertGreaterEqual(rows["idle"]["frames"], 38)
        self.assertLess(rows["idle"]["max"], 5.0)
        self.assertGreater(rows["20 cm"]["max"], 20.0)

    def test_the_placement_control_passes_only_when_a_channel_moved(self):
        from wifi_radar import handtrial

        self.assertTrue(handtrial.control_passed(
            {"idle": {"worst": 2.1}, "2 cm": {"worst": 4.4, "channel": 7}}))
        self.assertFalse(handtrial.control_passed(
            {"idle": {"worst": 2.1}, "2 cm": {"worst": 1.8, "channel": 7}}))
        self.assertFalse(handtrial.control_passed({}))

    def test_the_control_is_two_equal_stretches(self):
        from wifi_radar import handtrial

        marks = handtrial.control_marks(10)
        self.assertEqual([mark["name"] for mark in marks], ["idle", "2 cm"])
        self.assertEqual(marks[0]["end"], marks[1]["start"])
        self.assertEqual(marks[1]["end"], 20.0)

    def test_recording_stops_at_the_end_of_the_schedule(self):
        from wifi_radar import handtrial

        source = csi.SyntheticSource(scenario="quiet", seconds=20.0, seed=1)
        marks = handtrial.schedule(1, 1, [20])
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "trial.jsonl")
            written = handtrial.record(source, marks, path, say=lambda *a: None)
            with open(path, encoding="utf-8") as handle:
                lines = sum(1 for _ in handle)
        self.assertEqual(written, lines)
        self.assertGreaterEqual(written, 38)
        self.assertLess(written, 50)


class TestThresholdsFromARecording(unittest.TestCase):
    """Deriving the calibration length and threshold from what was recorded."""

    def write(self, root, marks=None, seconds=12.0, hand_at=None):
        """A recording of a quiet room, optionally with a hand arriving at t."""
        path = os.path.join(root, "trial.jsonl")
        with open(path, "w", encoding="utf-8") as out:
            for i in range(int(seconds * 20)):
                t = i / 20.0
                level = 300.0 + (40.0 if hand_at is not None and t >= hand_at else 0.0)
                values = [complex(level, 0)] * 8
                out.write(json.dumps(csi.Frame(t, values).as_record()) + "\n")
        if marks is not None:
            with open(path + ".marks.json", "w", encoding="utf-8") as out:
                json.dump({"marks": marks, "frames": int(seconds * 20)}, out)
        return path

    def run_main(self, argv):
        from wifi_radar import thresholds

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = thresholds.main(argv)
        return status, out.getvalue()

    def test_the_lowest_window_that_holds_its_idle_rate_is_the_one_chosen(self):
        from wifi_radar import thresholds

        rows = [{"seconds": 2.0, "rate": 0.20},
                {"seconds": 5.0, "rate": 0.01},
                {"seconds": 10.0, "rate": 0.0}]
        self.assertEqual(thresholds.pick_calibration(rows), 5.0)
        # Asked for no false alarms at all, only the window measured at zero
        # qualifies; the others are not promoted just because none does.
        self.assertEqual(thresholds.pick_calibration(rows, max_idle=0.0), 10.0)
        self.assertIsNone(thresholds.pick_calibration(rows[:2], max_idle=0.0))

    def test_a_window_and_a_threshold_are_derived_from_a_marked_recording(self):
        marks = [{"name": "idle", "start": 0.0, "end": 6.0},
                 {"name": "20 cm", "start": 6.0, "end": 12.0}]
        with tempfile.TemporaryDirectory() as root:
            path = self.write(root, marks, hand_at=6.0)
            status, text = self.run_main(["--record", path, "--sigmas", "2,4"])
        self.assertEqual(status, 0)
        self.assertIn("calibration window", text)
        self.assertIn("20 cm", text)
        self.assertIn("--calibrate", text)
        self.assertIn("--on-sigma", text)

    def test_an_unmarked_recording_is_taken_as_one_idle_stretch(self):
        """A plain recording of an empty room still sizes the window."""
        with tempfile.TemporaryDirectory() as root:
            path = self.write(root, marks=None)
            status, text = self.run_main(["--record", path, "--sigmas", "4"])
        self.assertEqual(status, 0)
        self.assertIn("none marked", text)
        self.assertIn("calibration window", text)

    def test_an_idle_stretch_too_short_to_judge_a_window_is_refused(self):
        marks = [{"name": "idle", "start": 0.0, "end": 3.0},
                 {"name": "20 cm", "start": 3.0, "end": 6.0}]
        with tempfile.TemporaryDirectory() as root:
            path = self.write(root, marks, seconds=6.0)
            from wifi_radar import thresholds
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                status = thresholds.main(["--record", path])
        self.assertEqual(status, 2)
        self.assertIn("--idle", err.getvalue())


class TestCues(unittest.TestCase):
    """A trial announced out loud, so it can be run without watching a screen."""

    def cue(self, mode="both", **kwargs):
        from wifi_radar import cues

        said, beeped = [], []
        kwargs.setdefault("background", False)
        kwargs.setdefault("speak", lambda text: said.append(text) or True)
        kwargs.setdefault("beep", lambda **kw: beeped.append(kw) or True)
        return cues.Cue(mode, **kwargs), said, beeped

    def test_off_says_nothing_at_all(self):
        cue, said, beeped = self.cue("off")
        cue.announce("hand at 20 centimetres")
        self.assertEqual((said, beeped), ([], []))

    def test_both_beeps_and_speaks_the_change(self):
        cue, said, beeped = self.cue("both")
        cue.announce("hand at 20 centimetres")
        self.assertEqual(said, ["hand at 20 centimetres"])
        self.assertEqual(len(beeped), 1)

    def test_a_voice_that_fails_still_leaves_a_tone(self):
        """Silence is the one outcome a segment change cannot have."""
        cue, _, beeped = self.cue("both", speak=lambda text: False)
        cue.announce("hand at 30 centimetres")
        self.assertEqual(len(beeped), 2)

    def test_a_broken_cue_never_ends_the_trial(self):
        def boom(*args, **kwargs):
            raise OSError("no audio device")

        cue, _, _ = self.cue("both", speak=boom, beep=boom)
        cue.announce("hand at 40 centimetres")      # must not raise

    def test_the_trial_cues_every_segment_in_order(self):
        from wifi_radar import handtrial

        class Counting:
            def __init__(self):
                self.texts = []

            def announce(self, text):
                self.texts.append(text)

        cue = Counting()
        source = csi.SyntheticSource(scenario="quiet", seconds=6.0, seed=1)
        marks = handtrial.schedule(2, 2, [20, 30])
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "trial.jsonl")
            handtrial.record(source, marks, path, say=lambda *a: None, cue=cue)
        self.assertEqual(cue.texts[:3],
                         ["empty room, please step back",
                          "hand at 20 centimetres",
                          "hand at 30 centimetres"])

    def test_the_cue_arrives_at_the_start_of_its_segment_not_a_segment_late(self):
        """The hand moves on the cue, so the cue has to be on time."""
        from wifi_radar import handtrial

        seen = {"n": 0}
        cues_at = {}

        class Counting:
            def announce(self, text):
                cues_at[text] = seen["n"]

        def frames():
            for i in range(120):       # 6 s at 20 fps
                seen["n"] = i
                yield csi.Frame(i / 20.0, [complex(300.0, 0)] * 8)

        marks = handtrial.schedule(2, 2, [20, 30])
        with tempfile.TemporaryDirectory() as root:
            handtrial.record(frames(), marks,
                             os.path.join(root, "trial.jsonl"),
                             say=lambda *a: None, cue=Counting())
        # 20 cm starts at 2 s, which is frame 40; the old comparison used the
        # segment's end and fired at frame 80 instead.
        self.assertLessEqual(abs(cues_at["hand at 20 centimetres"] - 40), 1,
                             cues_at)
        self.assertLessEqual(abs(cues_at["hand at 30 centimetres"] - 80), 1,
                             cues_at)

    def test_a_trial_that_lost_the_board_names_the_stretches_it_lost(self):
        """`n/a` reads as "no hand there"; a dead board is not a negative result."""
        from wifi_radar import handtrial

        marks = handtrial.schedule(20, 15, [2, 20, 30, 40])
        frames = [csi.Frame(i / 20.0, [complex(300.0, 0)] * 8) for i in range(120)]
        covered, wanted, missing = handtrial.missing_segments(frames, marks)
        self.assertAlmostEqual(covered, 5.95, places=2)
        self.assertEqual(wanted, 80.0)
        self.assertEqual(missing, ["2 cm", "20 cm", "30 cm", "40 cm"])

    def test_a_complete_trial_has_nothing_missing(self):
        from wifi_radar import handtrial

        marks = handtrial.schedule(2, 2, [20, 30])
        frames = [csi.Frame(i / 20.0, [complex(300.0, 0)] * 8) for i in range(121)]
        covered, wanted, missing = handtrial.missing_segments(frames, marks)
        self.assertEqual(missing, [])
        self.assertGreaterEqual(covered, wanted - 0.5)

    def test_a_hand_stretch_with_no_frames_is_refused(self):
        """Timestamps 80 s apart can hide a recording that holds two frames."""
        from wifi_radar import handtrial

        rows = {"idle": {"frames": 0, "over_4": None},
                "2 cm": {"frames": 0, "over_4": None},
                "5 cm": {"frames": 300, "over_4": 0.0}}
        self.assertEqual(handtrial.empty_segments(rows), ["2 cm"])

    def test_a_slow_hand_shows_in_the_level_table_and_not_in_the_motion_one(self):
        """The control run at 2 cm moved every channel 36-58% and z stayed at 1."""
        from wifi_radar import handtrial

        def frames():
            for i in range(12 * 20):
                t = i / 20.0
                # A hand arriving over two seconds, then held: no frame-to-frame
                # change big enough to look like motion.
                factor = 1.0 + 0.5 * min(1.0, max(0.0, (t - 4.0) / 2.0))
                values = [complex(300.0 * factor, 0)] * 8
                yield csi.Frame(t, values)

        marks = handtrial.schedule(4, 8, [2])
        rows = handtrial.level_table(list(frames()), marks)
        self.assertLess(abs(rows["idle"]["worst"]), 3.0)
        self.assertGreater(rows["2 cm"]["worst"], 3.0)

    def test_a_steady_input_shows_no_level_shift(self):
        from wifi_radar import handtrial

        frames = [csi.Frame(i / 20.0, [complex(300.0, 0)] * 8) for i in range(160)]
        rows = handtrial.level_table(frames, handtrial.schedule(4, 4, [2]))
        self.assertLess(abs(rows["2 cm"]["worst"]), 1.0)

    def test_a_run_with_no_hand_in_it_is_called_a_baseline(self):
        """Every segment at the idle rate means the hand was not there."""
        from wifi_radar import handtrial

        rows = {"idle": {"frames": 300, "over_4": 0.03},
                "2 cm": {"frames": 300, "over_4": 0.02},
                "5 cm": {"frames": 300, "over_4": 0.01}}
        self.assertTrue(handtrial.baseline_only(rows))

    def test_one_stretch_above_the_idle_rate_is_a_measurement(self):
        from wifi_radar import handtrial

        rows = {"idle": {"frames": 300, "over_4": 0.03},
                "2 cm": {"frames": 300, "over_4": 0.25},
                "5 cm": {"frames": 300, "over_4": 0.01}}
        self.assertFalse(handtrial.baseline_only(rows))

    def test_a_stretch_with_no_frames_is_not_evidence_either_way(self):
        from wifi_radar import handtrial

        rows = {"idle": {"frames": 0, "over_4": 0.0},
                "2 cm": {"frames": 0, "over_4": 0.0}}
        self.assertFalse(handtrial.baseline_only(rows))

    def test_a_prompt_names_the_distance_rather_than_the_mark(self):
        from wifi_radar import handtrial

        self.assertEqual(handtrial.prompt_for("2 cm"), "hand at 2 centimetres")
        self.assertIn("step back", handtrial.prompt_for("idle"))

    def test_the_speaker_takes_one_prompt_per_line(self):
        """The engine stays open, so a prompt is a write rather than a startup."""
        import sys

        from wifi_radar import cues

        with tempfile.TemporaryDirectory() as root:
            heard = os.path.join(root, "heard.txt")
            script = ("import sys\n"
                      f"out = open({heard!r}, 'a', encoding='utf-8')\n"
                      "for line in sys.stdin:\n"
                      "    out.write(line)\n"
                      "    out.flush()\n")
            speaker = cues.Speaker([sys.executable, "-u", "-c", script])
            self.assertTrue(speaker.start())
            self.assertTrue(speaker.say("hand at 20 centimetres"))
            self.assertTrue(speaker.say("hand at 30\ncentimetres"))
            speaker.close()
            with open(heard, encoding="utf-8") as handle:
                spoken = handle.read().split("\n")
        self.assertEqual(spoken[0], "hand at 20 centimetres")
        # A line break in a prompt would become a second, earlier-sounding cue.
        self.assertEqual(spoken[1], "hand at 30 centimetres")

    def test_a_dead_speaker_falls_back_to_a_short_lived_engine(self):
        from wifi_radar import cues

        class Dead:
            def say(self, text):
                return False

            def start(self):
                return True

            def close(self):
                pass

        said = []
        cue = cues.Cue("voice", speaker=Dead(), background=False,
                       speak=lambda text: said.append(text) or True,
                       beep=lambda **kw: True)
        cue.announce("hand at 40 centimetres")
        self.assertEqual(said, ["hand at 40 centimetres"])

    def test_the_speech_command_is_usable_or_absent(self):
        from wifi_radar import cues

        command = cues.speech_command("hand at 20 centimetres")
        if command is None:
            self.skipTest("no speech engine on this machine")
        self.assertTrue(all(isinstance(part, str) for part in command), command)
        # The sentence is handed over in the environment, never spliced into a
        # command line, so punctuation cannot be read as shell syntax.
        self.assertTrue(cues.SPEECH_ENV in " ".join(command)
                        or "hand at 20 centimetres" in command, command)


class TestBoardReset(unittest.TestCase):
    """A Mega resets without its USB bridge going away: millis() restarts."""

    def test_the_session_clock_keeps_going_forward_across_a_reset(self):
        lines = "S,1000,1,2\nS,1050,2,3\nS,0,3,4\nS,50,4,5\n"
        source = serial_node.SerialSource(io.StringIO(lines), port="COM5")
        times = [frame.t for frame in source]
        self.assertEqual(source.reboots, 1)
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])), times)
        self.assertIn("board reset 1x", source.describe())

    def test_a_reset_does_not_look_like_the_counter_wrapping(self):
        """The wrap moves the clock forwards by 49.7 days; a reset by a second."""
        lines = "S,4294967290,1,2\nS,10,3,4\n"
        source = serial_node.SerialSource(io.StringIO(lines), port="COM5")
        times = [frame.t for frame in source]
        self.assertGreater(times[1], times[0])
        self.assertLess(times[1], 1.0)
        self.assertEqual(source.reboots, 0)


class TestOpeningAPort(unittest.TestCase):
    """Opening the port must leave the board running its sketch, not its loader."""

    def test_opening_pulses_dtr(self):
        class Fake:
            def __init__(self):
                self.edges = []

            @property
            def dtr(self):
                return True

            @dtr.setter
            def dtr(self, value):
                self.edges.append(value)

        stream = Fake()
        slept = []
        self.assertTrue(serial_node.pulse_reset(stream, sleep=slept.append))
        self.assertEqual(stream.edges, [False, True])
        self.assertEqual(slept, [0.3, 0.1])

    def test_a_stream_with_no_dtr_is_left_alone(self):
        self.assertFalse(serial_node.pulse_reset(io.StringIO()))

    def test_open_serial_resets_after_opening(self):
        """The board that stayed silent for twenty seconds until it was reset."""
        reset = []

        class FakePort:
            def __init__(self):
                self.flushed = 0

            @property
            def dtr(self):
                return True

            @dtr.setter
            def dtr(self, value):
                reset.append(value)

            def reset_input_buffer(self):
                self.flushed += 1

        # A stand-in module rather than the installed pyserial: the suite is
        # meant to run on an interpreter with only the standard library, and
        # patching the real module made this test need pyserial to exist just
        # to be replaced. What is under test is open_serial's own behaviour, not
        # whether the driver is present.
        fake_serial = mock.Mock(Serial=lambda *a, **kw: FakePort(),
                                SerialException=OSError)
        with mock.patch.dict(sys.modules, {"serial": fake_serial}):
            stream = serial_node.open_serial("COM5", 115200)
        self.assertEqual(reset[-2:], [False, True], reset)
        self.assertIsInstance(stream, FakePort)
        # A stale line left in the driver buffer would become the session's
        # zero, putting every live frame tens of seconds into the schedule.
        self.assertEqual(stream.flushed, 1)


class TestReconnect(unittest.TestCase):
    """A cable pulled mid-session must not end the session."""

    class Clock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            return self.t

        def sleep(self, seconds):
            self.t += seconds

    class Unplugging(io.StringIO):
        """Lines, then an OSError: the adapter vanishing mid-read."""

        def __init__(self, text, after):
            super().__init__(text)
            self.left = after

        def readline(self):
            if self.left <= 0:
                raise OSError("device disconnected")
            self.left -= 1
            return super().readline()

    def inner(self, text, after=2, **kwargs):
        stream = self.Unplugging(text, after)
        return serial_node.SerialSource(stream, port="COM5", **kwargs)

    def test_frames_continue_after_the_board_resets(self):
        clock = self.Clock()
        opened = []
        second = io.StringIO("S,0,1,2,3\nS,50,4,5,6\nS,100,7,8,9\n")

        def opener():
            opened.append(1)
            return serial_node.SerialSource(second, port="COM5")

        inner = self.inner("# banner\nS,0,10,11,12\nS,50,13,14,15\n", after=3)
        source = serial_node.ReconnectingSource(
            inner, opener, port="COM5", clock=clock, sleep=clock.sleep,
            give_up_after=10.0, retry_seconds=1.0)
        frames = [next(source) for _ in range(4)]
        self.assertEqual(len(opened), 1)
        self.assertEqual(source.reconnects, 1)
        self.assertEqual(source.frames, 4)
        self.assertTrue(all(b.t > a.t for a, b in zip(frames, frames[1:])), frames)

    def test_a_reconnect_that_hands_back_a_bare_stream_still_makes_frames(self):
        """A pyserial object iterates lines of bytes, not frames.

        csi.open_source used to return the port itself from its reopen, so the
        first reconnect replaced the working source with a raw stream: the next
        frame failed and `describe()` raised. Found on the wire, not here.
        """
        clock = self.Clock()
        first = self.inner("S,0,1,2\nS,50,3,4\n", after=2)
        stream = io.StringIO("S,0,5,6\nS,50,7,8\n")
        source = serial_node.ReconnectingSource(
            first, lambda: stream, port="COM5", clock=clock,
            sleep=clock.sleep, retry_seconds=1.0)
        frames = [next(source) for _ in range(4)]
        self.assertEqual(source.reconnects, 1)
        self.assertEqual([f.values for f in frames],
                         [[1, 2], [3, 4], [5, 6], [7, 8]])
        self.assertIn("COM5", source.describe())
        self.assertEqual(source.channels, 2)

    def test_a_reconnect_opens_a_source_rather_than_a_port(self):
        """The opener given to ReconnectingSource must return a source."""
        def fake_open(port, baud=serial_node.DEFAULT_BAUD, timeout=0.2):
            return io.StringIO("S,0,1,2\n")

        with mock.patch.object(serial_node, "discover_baud",
                               lambda port, **kw: (9600, [])), \
                mock.patch.object(serial_node, "open_serial", fake_open), \
                mock.patch.object(serial_node, "available_ports", lambda: []), \
                mock.patch.object(serial_node, "choose_port",
                                  lambda ports: mock.Mock(device="COM5")):
            source = csi.open_source("serial:COM5:auto")
            again = source._open()
        self.assertTrue(hasattr(again, "describe"), type(again))
        self.assertEqual(again.port, "COM5")

    def test_the_session_clock_continues_instead_of_restarting(self):
        """The board's millis() restarts at zero; the session's must not."""
        clock = self.Clock()
        second = io.StringIO("S,0,1,2,3\nS,50,4,5,6\nS,100,7,8,9\n")
        inner = self.inner("S,0,10,11,12\nS,1000,13,14,15\n", after=2)
        source = serial_node.ReconnectingSource(
            inner, lambda: serial_node.SerialSource(second, port="COM5"),
            port="COM5", clock=clock, sleep=clock.sleep, retry_seconds=2.0)
        last_before = next(source)
        next(source)
        after = next(source)
        self.assertGreater(after.t, 1.0)
        self.assertGreater(after.t, 0.0)
        self.assertGreater(after.t, last_before.t)

    def test_a_board_that_never_comes_back_ends_the_session(self):
        clock = self.Clock()

        def opener():
            raise OSError("no port")

        source = serial_node.ReconnectingSource(
            self.inner("S,0,1\nS,50,2\nS,100,3\n", after=2), opener,
            port="COM5", clock=clock, sleep=clock.sleep,
            give_up_after=3.0, retry_seconds=1.0)
        with self.assertRaises(OSError):
            while True:
                next(source)
        self.assertTrue(source.offline)
        self.assertEqual(source.reconnects, 0)
        self.assertIn("no port", source.describe())

    def test_a_replay_that_ends_is_not_reconnected(self):
        clock = self.Clock()
        opened = []
        source = serial_node.ReconnectingSource(
            serial_node.SerialSource(io.StringIO("S,0,1\n"), port="file"),
            lambda: opened.append(1), clock=clock, sleep=clock.sleep)
        self.assertEqual(len(list(source)), 1)
        self.assertEqual((source.reconnects, opened), (0, []))

    def test_describe_counts_the_reconnects(self):
        clock = self.Clock()
        second = io.StringIO("S,0,1,2,3\nS,50,4,5,6\nS,100,7,8,9\n")
        source = serial_node.ReconnectingSource(
            self.inner("S,0,10,11,12\nS,50,13,14,15\n", after=2),
            lambda: serial_node.SerialSource(second, port="COM5"),
            port="COM5", clock=clock, sleep=clock.sleep, retry_seconds=1.0)
        self.assertNotIn("reconnected", source.describe())
        for _ in range(3):
            next(source)
        self.assertIn("reconnected 1x", source.describe())

    def test_reconnects_are_announced(self):
        clock = self.Clock()
        seen = []
        second = io.StringIO("S,0,1,2,3\nS,50,4,5,6\nS,100,7,8,9\n")
        source = serial_node.ReconnectingSource(
            self.inner("S,0,10,11,12\nS,50,13,14,15\n", after=2),
            lambda: serial_node.SerialSource(second, port="COM5"),
            port="COM5", clock=clock, sleep=clock.sleep, retry_seconds=1.0,
            notify=seen.append)
        for _ in range(3):
            next(source)
        self.assertTrue(any("stopped" in message for message in seen), seen)
        self.assertTrue(any("reconnected" in message for message in seen), seen)

    def test_open_source_reconnects_by_default_and_can_be_told_not_to(self):
        with mock.patch.object(serial_node, "open_serial",
                               lambda port, baud: io.StringIO("")):
            wrapped = csi.open_source("serial:COM5")
            bare = csi.open_source("serial:COM5", reconnect=False)
        self.assertIsInstance(wrapped, serial_node.ReconnectingSource)
        self.assertIsInstance(bare, serial_node.SerialSource)


class TestThresholdSweep(unittest.TestCase):
    """Choosing `on_sigma` by measurement, which is the only way to choose it."""

    def test_a_quiet_room_clears_a_high_threshold(self):
        from wifi_radar import thresholds

        frames = list(csi.SyntheticSource(scenario="quiet", seconds=20.0, seed=9))
        rows = thresholds.sweep(frames, [1.0, 10.0], calibration=None)
        self.assertEqual(rows[1]["flagged"], 0)
        self.assertGreaterEqual(rows[0]["flagged"], rows[1]["flagged"])

    def test_the_pick_is_the_loosest_threshold_under_the_budget(self):
        from wifi_radar import thresholds

        rows = [{"sigma": 2.0, "rate": 0.2}, {"sigma": 4.0, "rate": 0.001},
                {"sigma": 6.0, "rate": 0.0}]
        self.assertEqual(thresholds.pick(rows, 0.01), 4.0)

    def test_no_threshold_under_the_budget_says_so(self):
        from wifi_radar import thresholds

        self.assertIsNone(thresholds.pick([{"sigma": 2.0, "rate": 0.5}], 0.01))


class TestDoctorContract(unittest.TestCase):
    """What CI asserts about the doctor, tested where a pty may not exist."""

    def test_a_printing_document_passes(self):
        document = {"verdict": "printing",
                    "sketch": {"samples": 39, "baud": 115200}}
        self.assertEqual(fakeboard.check(document, 0), [])

    def test_a_board_that_is_not_printing_fails_on_all_three_counts(self):
        problems = fakeboard.check(
            {"verdict": "silent_firmware", "sketch": {"samples": 0}}, 1)
        self.assertTrue(any("exited 1" in problem for problem in problems), problems)
        self.assertTrue(any("silent_firmware" in problem for problem in problems),
                        problems)
        self.assertTrue(any("no samples" in problem for problem in problems),
                        problems)

    def test_a_document_with_no_sketch_section_is_not_a_pass(self):
        problems = fakeboard.check({"verdict": "printing"}, 0)
        self.assertTrue(any("no samples" in problem for problem in problems), problems)


class TestDoctorOnTheLivePage(unittest.TestCase):
    """A board that is attached and silent says why on the page."""

    def test_a_silent_serial_source_puts_the_verdict_on_the_page(self):
        class Silent(csi.Source):
            port = "COM5"
            rate = 20.0
            length = None

            def describe(self):
                return "serial COM5 at 115200 baud, waiting for lines"

            def __next__(self):
                raise TimeoutError("nothing readable on COM5 for 10s")

        with mock.patch.object(live, "doctor_check",
                               lambda port: ("silent_firmware",
                                             "not printing — upload "
                                             "arduino/mega_channel_node")):
            session = live.RadarSession(Silent(), realtime=False)
            session.start()
            session._thread.join(timeout=10)
            session.stop()
            snapshot = session.snapshot()
        self.assertIn("TimeoutError", snapshot["error"])
        self.assertEqual(snapshot["diagnosis"]["verdict"], "silent_firmware")
        self.assertIn("mega_channel_node", snapshot["diagnosis"]["message"])

    def test_a_failing_source_without_a_port_does_not_run_the_doctor(self):
        class Broken(csi.Source):
            rate = 20.0
            length = None

            def __next__(self):
                raise ValueError("not serial")

        calls = []
        with mock.patch.object(live, "doctor_check",
                               lambda port: calls.append(port)):
            session = live.RadarSession(Broken(), realtime=False)
            session.start()
            session._thread.join(timeout=10)
            session.stop()
        self.assertEqual(calls, [])
        self.assertIsNone(session.snapshot()["diagnosis"])

    def test_the_doctor_check_reads_the_json_verdict(self):
        result = subprocess.CompletedProcess(
            [], 1, json.dumps({"verdict": "silent_firmware",
                               "message": "not printing"}), "")
        with mock.patch.object(live.subprocess, "run", lambda *a, **k: result):
            verdict, message = live.doctor_check("COM5")
        self.assertEqual((verdict, message), ("silent_firmware", "not printing"))

    def test_a_doctor_that_prints_nothing_does_not_crash_the_page(self):
        result = subprocess.CompletedProcess([], 2, "", "boom")
        with mock.patch.object(live.subprocess, "run", lambda *a, **k: result):
            verdict, message = live.doctor_check("COM5")
        self.assertEqual(verdict, "unknown")
        self.assertIn("boom", message)


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


class TestCalibrationOption(unittest.TestCase):
    """`--calibrate` decides what the detector is willing to call an empty room."""

    def test_watch_calibration_window_reaches_the_detector(self):
        """A 2 s session with 0.5 s of calibration has measured frames; the 2 s
        default would spend the whole session calibrating."""
        from wifi_radar import watch

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = watch.main(["--source", "sim:walk", "--seconds", "2",
                                 "--calibrate", "0.5", "--fast",
                                 "--every", "10"])
        self.assertEqual(status, 0)
        text = out.getvalue()
        self.assertIn("calibrated floor", text)
        match = re.search(r"of (\d+) measured", text)
        self.assertIsNotNone(match)
        self.assertGreater(int(match.group(1)), 0)

    def test_watch_defaults_to_an_automatic_window(self):
        """No --calibrate, no guess: the run says how long it measured and why."""
        from wifi_radar import watch

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = watch.main(["--source", "sim:quiet", "--seconds", "3",
                                 "--fast", "--every", "10"])
        self.assertEqual(status, 0)
        self.assertIn("calibration     : 2.0s (auto, the floor had settled)",
                      out.getvalue())

    def test_watch_accepts_auto_by_name(self):
        from wifi_radar import watch

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = watch.main(["--source", "sim:quiet", "--seconds", "3",
                                 "--calibrate", "auto", "--fast",
                                 "--every", "10"])
        self.assertEqual(status, 0)
        self.assertIn("auto, the floor had settled", out.getvalue())

    def test_live_session_defaults_to_an_automatic_window(self):
        from wifi_radar import live

        session = live.RadarSession(
            csi.SyntheticSource(scenario="quiet", seconds=1.0, seed=2),
            realtime=False)
        self.assertTrue(session.detector.calibrate_auto)

    def test_watch_refuses_a_calibration_window_of_zero(self):
        from wifi_radar import watch

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            status = watch.main(["--calibrate", "0", "--source", "sim:walk"])
        self.assertEqual(status, 2)
        self.assertIn("--calibrate", err.getvalue())

    def test_watch_refuses_a_negative_highpass(self):
        from wifi_radar import watch

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            status = watch.main(["--highpass", "-1", "--source", "sim:walk"])
        self.assertEqual(status, 2)
        self.assertIn("--highpass", err.getvalue())

    def test_both_views_refuse_a_threshold_of_zero(self):
        """Zero sigmas would call every frame movement, for ever."""
        from wifi_radar import live, watch

        for module in (watch, live):
            # watch reports on stderr and live on stdout; both are the user's
            # screen, so both are read here.
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(err):
                status = module.main(["--on-sigma", "0", "--source", "sim:walk"])
            self.assertEqual(status, 2, module.__name__)
            self.assertIn("--on-sigma", out.getvalue() + err.getvalue(),
                          module.__name__)

    def test_a_measured_threshold_reaches_the_detector(self):
        from wifi_radar import live

        session = live.RadarSession(
            csi.SyntheticSource(scenario="quiet", seconds=1.0, seed=2),
            realtime=False, on_sigma=6.5)
        self.assertEqual(session.detector.on_sigma, 6.5)

    def test_a_run_shorter_than_its_calibration_says_so(self):
        """Otherwise a session that never left calibration reads as a quiet room."""
        from wifi_radar import watch

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = watch.main(["--source", "sim:walk", "--seconds", "1",
                                 "--calibrate", "30", "--fast", "--every", "10"])
        self.assertEqual(status, 0)
        text = out.getvalue()
        self.assertIn("0 of 0 measured", text)
        self.assertIn("no frames measured", text)

    def test_live_refuses_a_calibration_window_of_zero(self):
        from wifi_radar import live

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = live.main(["--calibrate", "0"])
        self.assertEqual(status, 2)
        self.assertIn("--calibrate", out.getvalue())

    def test_live_main_passes_the_window_to_the_session(self):
        """The flag has to survive the trip from the command line, not just the
        constructor test below."""
        from wifi_radar import live

        captured = {}

        def fake_build(session, host, port):
            captured["session"] = session

            class Server:
                def serve_forever(self):
                    raise KeyboardInterrupt

                def server_close(self):
                    pass

            return Server()

        original = live.build_server
        live.build_server = fake_build
        self.addCleanup(setattr, live, "build_server", original)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = live.main(["--source", "sim:walk", "--seconds", "1",
                                "--calibrate", "25", "--highpass", "0.5",
                                "--port", "0"])
        self.assertEqual(status, 0)
        self.assertEqual(captured["session"].detector.calibrate_frames, 500)
        self.assertEqual(captured["session"].detector.highpass_seconds, 0.5)

    def test_a_live_session_carries_the_calibration_window_to_the_detector(self):
        """A wire's slow envelope needs 20-30 s of floor, which is 400-600 frames."""
        session = live.RadarSession(serial_node.SerialSource(io.StringIO("")),
                                    realtime=False, calibrate_seconds=25.0)
        self.assertEqual(session.detector.calibrate_frames, 500)


SILENT_REPORT = (
    "bootloader : answered 18 syncs in 2.1s (status 0xc0 = OK)\n"
    "sketch     : nothing arrived in 4.1s (0 bytes)\n\n"
    "not printing — the cable, the port and the chip are all fine. Upload "
    "arduino/mega_channel_node to the board, then run this again.\n")

PRINTING_REPORT = (
    "bootloader : answered 18 syncs in 2.1s (status 0xc0 = OK)\n"
    "sketch     : 62 lines, 61 samples in 4.1s, 20.0/s on the board's own clock\n\n"
    "ok — the board is identified, alive, and printing samples.\n")


class FakeRunner:
    """subprocess.run, scripted: uploads and doctor calls answered in order."""

    def __init__(self, uploads, doctors):
        self.uploads = list(uploads)
        self.doctors = list(doctors)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append(list(command))
        if "--upload" in command:
            code, out, err = self.uploads.pop(0)
        else:
            code, out = self.doctors.pop(0)
            err = ""
        return subprocess.CompletedProcess(command, code, out, err)


class TestHardwareSelfTest(unittest.TestCase):
    """The scripted hardware check: no board needed to test the script itself."""

    def run_it(self, uploads, doctors, force=False):
        runner = FakeRunner(uploads, doctors)
        with tempfile.TemporaryDirectory() as root:
            silent = selftest.write_silent_sketch(root)
            firmware = root + "/mega_channel_node"
            checks = selftest.run_selftest("COM5", "cli", silent, firmware,
                                           run=runner, say=lambda *a, **k: None,
                                           force=force)
        return checks, runner

    def test_a_whole_pass_reports_every_check_green(self):
        checks, runner = self.run_it(
            [(0, "ok", ""), (0, "ok", "")],
            [(0, PRINTING_REPORT), (1, SILENT_REPORT), (0, PRINTING_REPORT)])
        self.assertEqual(len(checks), 9, checks)
        self.assertTrue(all(check["ok"] for check in checks), checks)
        uploads = [call for call in runner.calls if "--upload" in call]
        self.assertEqual(len(uploads), 2)
        self.assertTrue(uploads[0][-1].endswith("silent"))
        self.assertTrue(uploads[1][-1].endswith("mega_channel_node"))
        # `arduino-cli upload` does not compile, and on a fresh sketch there is
        # no cached build for it to find; this was a real failure on the bench.
        self.assertEqual(uploads[0][1:3], ["compile", "--upload"])
        self.assertEqual(uploads[1][1:3], ["compile", "--upload"])

    def test_a_wrong_verdict_fails_and_the_firmware_is_still_restored(self):
        checks, runner = self.run_it(
            [(0, "ok", ""), (0, "ok", "")],
            [(0, PRINTING_REPORT), (0, PRINTING_REPORT),
             (0, PRINTING_REPORT)])
        failed = [check["name"] for check in checks if not check["ok"]]
        self.assertIn("the doctor exits 1 for a silent sketch", failed)
        self.assertIn("the doctor says it is not printing", failed)
        self.assertEqual(len([c for c in runner.calls if "--upload" in c]), 2)
        self.assertTrue(any(c["name"] == "samples are arriving again"
                            for c in checks))

    def test_a_failed_restore_is_reported_and_not_papered_over(self):
        checks, runner = self.run_it(
            [(0, "ok", ""), (1, "", "boom")],
            [(0, PRINTING_REPORT), (1, SILENT_REPORT)])
        failed = [check["name"] for check in checks if not check["ok"]]
        self.assertEqual(failed, ["mega_channel_node is restored"])
        self.assertEqual(len([c for c in runner.calls if "--upload" in c]), 2)

    def test_an_upload_that_fails_stops_the_verdict_checks(self):
        """Nothing was flashed, so nothing about the doctor is meaningful."""
        checks, runner = self.run_it([(1, "", "no board"), (0, "ok", "")],
                                     [(0, PRINTING_REPORT), (0, PRINTING_REPORT)])
        failed = [check["name"] for check in checks if not check["ok"]]
        self.assertEqual(failed, ["the silent sketch uploads"])
        self.assertFalse(any("not printing" in check["name"] for check in checks))

    def test_a_board_that_is_not_printing_is_not_flashed_at_all(self):
        """No working firmware to restore means no licence to overwrite it."""
        checks, runner = self.run_it([], [(2, "error: cannot open COM5")])
        self.assertEqual([check["name"] for check in checks],
                         ["the board prints before the test starts"])
        self.assertFalse(any("--upload" in call for call in runner.calls))

    @staticmethod
    def alive(pid):
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                 capture_output=True, text=True).stdout
            return str(pid) in out
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def test_a_timeout_kills_the_children_the_tool_started(self):
        """The bug that wedged the board: avrdude outliving its parent."""
        parent = (
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(60)'])\n"
            "open(sys.argv[1], 'w').write(str(child.pid))\n"
            "time.sleep(60)\n")
        with tempfile.TemporaryDirectory() as root:
            pidfile = os.path.join(root, "child.pid")
            result = selftest.run_bounded(
                [sys.executable, "-c", parent, pidfile], timeout=2.0)
            with open(pidfile, encoding="utf-8") as handle:
                grandchild = int(handle.read())
        self.assertEqual(result.returncode, 1)
        self.assertIn("timed out", result.stderr)
        deadline = time.time() + 5.0
        while time.time() < deadline and self.alive(grandchild):
            time.sleep(0.1)
        self.assertFalse(self.alive(grandchild),
                         "the grandchild outlived the timeout and still holds "
                         "whatever port it opened")

    def test_a_default_run_flashes_under_the_tree_killing_runner(self):
        """The wiring, not just the helper: flash() must not fall back to run()."""
        uploads = []

        def fake_bounded(command, **kwargs):
            uploads.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        reports = [(0, PRINTING_REPORT), (1, SILENT_REPORT),
                   (0, PRINTING_REPORT)]

        def fake_doctor(port, run=None, **kwargs):
            return reports.pop(0)

        with mock.patch.object(selftest, "run_bounded", fake_bounded), \
                mock.patch.object(selftest, "ask_doctor", fake_doctor):
            with tempfile.TemporaryDirectory() as root:
                silent = selftest.write_silent_sketch(root)
                checks = selftest.run_selftest(
                    "COM5", "cli", silent, root + "/mega_channel_node",
                    say=lambda *a, **k: None)
        self.assertTrue(all(check["ok"] for check in checks), checks)
        self.assertEqual(len(uploads), 2)
        self.assertTrue(all("--upload" in command for command in uploads))

    def test_the_failure_detail_starts_on_a_line_boundary(self):
        """The one line that names the fault must not be inside a cut word."""
        result = subprocess.CompletedProcess([], 1, "", "\n".join(
            ["Error: timeout"] * 40 + ["Error during Upload: failed"]))
        tail = selftest._tail(result, limit=120)
        self.assertTrue(tail.startswith("Error:"), tail)
        self.assertTrue(tail.endswith("Error during Upload: failed"), tail)

    def test_force_overwrites_a_board_that_is_merely_missing_its_sketch(self):
        """The one case where refusing is wrong: there is nothing to preserve."""
        checks, runner = self.run_it(
            [(0, "ok", ""), (0, "ok", "")],
            [(1, SILENT_REPORT), (1, SILENT_REPORT), (0, PRINTING_REPORT)],
            force=True)
        self.assertTrue(all(check["ok"] for check in checks), checks)
        self.assertTrue(checks[0]["name"].startswith("--force:"), checks[0])
        self.assertNotIn("the board prints before the test starts",
                         [check["name"] for check in checks])
        self.assertEqual(len([c for c in runner.calls if "--upload" in c]), 2)

    def test_force_does_not_override_a_port_that_will_not_open(self):
        """A forced run over an unreachable port spends a timeout to learn nothing."""
        checks, runner = self.run_it([], [(2, "could not open port 'COM5'")],
                                     force=True)
        self.assertEqual([check["name"] for check in checks],
                         ["the board prints before the test starts"])
        self.assertIn("will not open", checks[0]["detail"])
        self.assertFalse(any("--upload" in call for call in runner.calls))

    def test_force_still_restores_and_still_reports_a_failed_restore(self):
        checks, _ = self.run_it(
            [(0, "ok", ""), (1, "", "no board")],
            [(1, SILENT_REPORT), (1, SILENT_REPORT)], force=True)
        self.assertEqual(selftest.summary(checks)[-1][:7], "WARNING")

    def test_the_json_result_carries_every_check(self):
        checks, _ = self.run_it(
            [(0, "ok", ""), (0, "ok", "")],
            [(0, PRINTING_REPORT), (1, SILENT_REPORT), (0, PRINTING_REPORT)])
        document = selftest.as_json(checks, "COM5", "cli", "fw")
        self.assertEqual((document["passed"], document["total"]), (9, 9))
        self.assertEqual(document["failed"], [])
        self.assertEqual(document["exit_status"], 0)
        self.assertEqual(len(document["checks"]), 9)
        self.assertEqual(document["port"], "COM5")

    def test_the_json_result_names_what_failed_and_why(self):
        checks, _ = self.run_it([], [(2, "cannot open COM5")])
        document = selftest.as_json(checks)
        self.assertEqual(document["failed"],
                         ["the board prints before the test starts"])
        self.assertEqual(document["exit_status"], 1)
        self.assertIn("exit 2", document["checks"][0]["detail"])
        self.assertIn("will not open", document["checks"][0]["detail"])

    def test_the_closing_lines_do_not_claim_a_restore_that_never_happened(self):
        """A refusal that says "mega_channel_node is back" is the bug it tests for."""
        refused, _ = self.run_it([], [(2, "error: cannot open COM5")])
        self.assertEqual(selftest.summary(refused), [
            "0/1 checks passed",
            "failed: the board prints before the test starts",
            "nothing was flashed: the board was not touched"])

    def test_a_passing_run_ends_on_the_count_alone(self):
        checks, _ = self.run_it(
            [(0, "ok", ""), (0, "ok", "")],
            [(0, PRINTING_REPORT), (1, SILENT_REPORT), (0, PRINTING_REPORT)])
        self.assertEqual(selftest.summary(checks), ["9/9 checks passed"])

    def test_a_failed_restore_warns_instead_of_reassuring(self):
        checks, _ = self.run_it(
            [(0, "ok", ""), (1, "", "no board")],
            [(0, PRINTING_REPORT), (1, SILENT_REPORT)])
        lines = selftest.summary(checks)
        self.assertIn("WARNING: the restore failed", lines[-1])
        self.assertIn("may still be silent", lines[-1])
        self.assertFalse(any("was left with mega_channel_node" in line
                             for line in lines))

    def test_a_restore_that_failed_before_anything_was_flashed_says_only_that(self):
        """"Compiled sketch not found" never touched the board; don't imply it did."""
        checks, _ = self.run_it(
            [(1, "", "Compiled sketch not found"), (1, "", "timed out")],
            [(0, PRINTING_REPORT)])
        lines = selftest.summary(checks)
        self.assertIn("by hand", lines[-1])
        self.assertNotIn("may still be silent", lines[-1])

    def test_the_silent_sketch_is_named_the_way_arduino_expects(self):
        with tempfile.TemporaryDirectory() as root:
            sketch = selftest.write_silent_sketch(root)
            self.assertEqual(sketch.name, "silent")
            source = (sketch / "silent.ino").read_text(encoding="utf-8")
            self.assertIn("void setup()", source)
            self.assertNotIn("Serial", source)

    def test_the_cli_can_be_named_explicitly_or_reported_missing(self):
        self.assertEqual(selftest.find_arduino_cli("C:/tools/arduino-cli.exe"),
                         "C:/tools/arduino-cli.exe")
        with self.assertRaises(FileNotFoundError) as caught:
            selftest.find_arduino_cli(None, env={}, which=lambda name: None,
                                      candidates=[])
        self.assertIn("arduino-cli", str(caught.exception))

    def test_no_port_exits_2_before_anything_is_flashed(self):
        original = serial_node.available_ports
        serial_node.available_ports = lambda: []
        self.addCleanup(setattr, serial_node, "available_ports", original)

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            status = selftest.main([])
        self.assertEqual(status, 2)
        self.assertIn("CH340", err.getvalue())


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

    def test_a_silent_port_shaped_like_an_iobase_still_times_out(self):
        """
        A real pyserial port is an IOBase, and it goes silent for the ~1.5 s
        after opening it resets the board. Reading that silence as EOF reported
        a printing Mega as `frames: 0`.
        """
        ticks = [0.0]

        def clock():
            ticks[0] += 1.0
            return ticks[0]

        source = serial_node.SerialSource(PortLikeFile(), port="COM5",
                                          timeout=10.0, clock=clock)
        with self.assertRaises(TimeoutError) as caught:
            next(source)
        self.assertIn("COM5", str(caught.exception))

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
