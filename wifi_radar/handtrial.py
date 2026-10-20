#!/usr/bin/env python3
"""
A guided antenna trial: record, then measure what the hand did at each distance.

The questions this answers are physical, so the procedure has to be too. It
records an idle stretch and then one stretch per distance, printing when to
change position, and writes both the frames and the marks that say where each
stretch is:

    python -m wifi_radar.handtrial --source serial:COM5 --record out/wire.jsonl

Afterwards it replays the recording twice — once through the high-pass, once
without — and reports each segment's z. That pairing is the point: the filter is
supposed to remove drift without removing the hand, and the only way to know is
to compare the two runs on the same frames.

The idle stretch is also what the calibration is measured on, so it reports the
window the detector chose and why, and how different the floor would have been
from a two-second sample. That is the number behind `--calibrate auto`.
"""

import argparse
import json
import statistics
import sys

from wifi_radar import csi, cues, detect

#: How far a channel's level has to move, in idle standard deviations, before it
#: is a hand rather than the room. Three is where the noise stops being a
#: plausible explanation and the arrival of something is — measured on the wire,
#: where the empty room's own excursions reached 2.1σ and a hand at 2 cm 4.4σ.
LEVEL_SIGMAS = 3.0


def schedule(idle_seconds, seconds, distances):
    """[(name, start, end)] for the idle stretch and one per distance."""
    segments = ([("idle", idle_seconds)]
                + [(f"{float(d):g} cm", seconds) for d in distances])
    marks, at = [], 0.0
    for name, length in segments:
        marks.append({"name": name, "start": round(at, 2), "end": round(at + length, 2)})
        at += length
    return marks


def prompt_for(name):
    """What to say when the segment called `name` begins."""
    if name == "idle":
        return "empty room, please step back"
    if name in ("0 cm", "0cm"):
        return "touch the wire with your finger"
    return "hand at " + name.replace(" cm", " centimetres")


def record(source, marks, path, say=print, cue=None):
    """
    Write frames to `path` while walking the source through the schedule.

    Segment changes are driven by the frames' own timestamps rather than a wall
    clock, so the marks describe the data rather than the machine that recorded
    it — a session that pauses on a reconnect keeps its segments aligned.

    Each change is said and, if a `cue` is given, also announced out loud. The
    announcement is what makes the trial work without eyes on the terminal, and
    it is driven from here rather than from a timer so that a pause, a reconnect
    or a slow frame moves the cue with the data.
    """
    written = 0
    index = -1
    started = None

    def announce(at):
        say(f">>> {marks[at]['name']}: "
            f"{marks[at]['end'] - marks[at]['start']:.0f}s")
        if cue is not None:
            cue.announce(prompt_for(marks[at]["name"]))

    with open(path, "w", encoding="utf-8") as out:
        for frame in source:
            if started is None:
                started = frame.t
            elapsed = frame.t - started
            # Against the next segment's *start*, which is the moment the hand
            # is meant to move. Comparing with its end announced every distance
            # one segment late — a quarter of a minute after the hand had
            # already moved on, in a trial with 15 s stretches.
            while index + 1 < len(marks) and elapsed >= marks[index + 1]["start"]:
                index += 1
                announce(index)
            if index < 0:
                index = 0
                announce(index)
            out.write(json.dumps(frame.as_record()) + "\n")
            written += 1
            if elapsed >= marks[-1]["end"]:
                break
    return written


def missing_segments(frames, marks, tolerance=0.5):
    """
    How far the recording got, and which stretches have no frames in them.

    A source that dies mid-trial does not fail: it ends. The recording then
    holds a calibrating stretch and nothing else, and a table of `n/a` reads as
    "no hand was detected" when the truth is "no hand was recorded". Catching it
    here is the difference between a negative result and a broken run.
    """
    covered = frames[-1].t - frames[0].t if frames else 0.0
    wanted = marks[-1]["end"] if marks else 0.0
    if covered >= wanted - tolerance:
        return covered, wanted, []
    return covered, wanted, [mark["name"] for mark in marks
                             if mark["start"] >= covered]


def empty_segments(rows, first="idle"):
    """
    Hand stretches with no measured frames in them.

    A span of timestamps is not coverage: a source that reconnects can hand back
    two frames 80 s apart, and the recording then looks a minute and a half long
    while holding two samples. The segments the analysis actually saw are the
    only evidence that the stretches were recorded at all.
    """
    return [name for name, stats in rows.items()
            if name != first and not stats["frames"]]


def level_table(frames, marks, window_seconds=1.0):
    """
    What each stretch did to the channel levels, in idle sigmas.

    The metric is a change between consecutive frames, so a hand that comes in
    slowly registers as nothing even when it moves a channel by half — which is
    exactly what a control run at 2 cm did: all eight channels rose 36-58% and
    the motion table stayed at zero. Measuring the level itself, against the
    spread the empty room had, answers "was there a hand" separately from "was
    it moving", and on this hardware the first is the stronger question.
    """
    if not frames:
        return {}
    origin = frames[0].t
    magnitudes = [[abs(value) for value in frame.values] for frame in frames]
    stamps = [frame.t - origin for frame in frames]
    idle = [values for frame, values in zip(frames, magnitudes)
            if frame.t - origin < marks[0]["end"]]
    if not idle:
        return {}
    channels = len(idle[0])
    base = [statistics.median([values[c] for values in idle]) for c in range(channels)]
    spread = [max(1.0, statistics.pstdev([values[c] for values in idle]))
              for c in range(channels)]

    def peak(shifts_by_channel):
        """(largest |shift|, its channel, its size) over the channels."""
        channel = max(range(channels),
                      key=lambda c: abs(shifts_by_channel[c]))
        return shifts_by_channel[channel], channel, shifts_by_channel

    rows = {}
    for mark in marks:
        window = [(stamp, values) for stamp, values in zip(stamps, magnitudes)
                  if mark["start"] <= stamp < mark["end"]]
        if not window:
            continue
        # Windows, not the whole stretch: a hand that stays for four seconds of
        # a fifteen-second stretch is a presence, and a median over the whole
        # stretch would call it nothing. This is what the control run at 2 cm
        # did — 36-58% on every channel for four seconds, and 0.6σ by median.
        worst, channel, shifts, at, fractions = 0.0, 0, [], None, []
        step = max(1, int(round(window_seconds * 20)))
        for start in range(0, len(window), step):
            piece = [values for _, values in window[start:start + step]]
            if not piece:
                continue
            medians = [statistics.median([values[c] for values in piece])
                       for c in range(channels)]
            here = [(value - base[c]) / spread[c]
                    for c, value in enumerate(medians)]
            shift, c, _ = peak(here)
            if abs(shift) > abs(worst):
                worst, channel, shifts = shift, c, here
                # The fractional change at the same moment: sigma-normalised
                # numbers say how surprising a change is, and for "which pin is
                # the wire in" the size of the change is the answer.
                fractions = [(value - base[i]) / base[i]
                             for i, value in enumerate(medians)]
                at = window[start][0]
        all_medians = [statistics.median([values[c] for _, values in window])
                       for c in range(channels)]
        rows[mark["name"]] = {
            "frames": len(window),
            "shifts": shifts,
            "fractions": fractions,
            "worst": worst,
            "channel": channel,
            "at": at,
            "median": all_medians[0],
        }
    return rows


def control_passed(levels, name="2 cm", threshold=LEVEL_SIGMAS):
    """Whether the placement control saw a hand at all."""
    row = levels.get(name)
    return bool(row) and abs(row["worst"]) >= threshold


def control_name(distance):
    """What the control's hand stretch is called, and how it is announced."""
    return f"{float(distance):g} cm"


def control_marks(seconds, distance=2.0):
    """
    The placement control: `seconds` empty, then `seconds` with a hand close by.

    Two centimetres is the default because it is what an antenna trial asks for;
    zero means touching the wire, which is the one thing that still works if the
    wire has stopped behaving like an antenna to a nearby hand.
    """
    name = control_name(distance)
    return [{"name": "idle", "start": 0.0, "end": float(seconds)},
            {"name": name, "start": float(seconds), "end": 2.0 * float(seconds)}]


def baseline_only(rows, idle="idle", margin=0.0):
    """
    True when no stretch rose above what the empty room does.

    A trial with the hand in the wrong place and a trial with an antenna that
    cannot see it produce the same table: every segment at the idle rate. They
    do not mean the same thing, and the second one is a broken run rather than a
    measurement — the person may not have heard the prompt, or may have been
    across the room. Named here so the summary can say it instead of leaving a
    row of zeros to be read as reach.
    """
    if idle not in rows or rows[idle]["frames"] == 0:
        return False
    floor = rows[idle]["over_4"]
    others = [stats for name, stats in rows.items()
              if name != idle and stats["frames"]]
    return bool(others) and all(stats["over_4"] <= floor + margin
                                for stats in others)


def summarise(zs):
    if not zs:
        return {"frames": 0, "median": None, "max": None, "over_4": None}
    ordered = sorted(zs)
    return {
        "frames": len(zs),
        "median": statistics.median(zs),
        "max": ordered[-1],
        "over_4": sum(1 for z in zs if z > detect.DEFAULT_ON_SIGMA) / len(zs),
    }


def analyse(frames, marks, highpass_seconds, rate,
            calibration=None, **detector_kw):
    """
    (per-segment z statistics, the detector) for one high-pass setting.

    The detector runs over the whole recording in order, so its calibration is
    the idle stretch and every later segment is judged against the room as it
    was before the hand arrived.
    """
    detector = detect.MotionDetector(
        rate=rate, calibrate_seconds=calibration,
        highpass_seconds=highpass_seconds, **detector_kw)
    buckets = {mark["name"]: [] for mark in marks}
    for frame in frames:
        detection = detector.update(frame)
        if detection.calibrating or detection.z is None:
            continue
        for mark in marks:
            if mark["start"] <= frame.t - frames[0].t < mark["end"]:
                buckets[mark["name"]].append(detection.z)
                break
    return {name: summarise(zs) for name, zs in buckets.items()}, detector


def _level_table(title, rows):
    print(f"\n{title}")
    print(f"  {'segment':>10}  {'frames':>6}  {'channel':>7}  {'level':>7}  "
          f"{'shift':>8}  {'at':>6}")
    for name, row in rows.items():
        at = "     -" if row.get("at") is None else f"{row['at']:5.0f}s"
        print(f"  {name:>10}  {row['frames']:6d}  {row['channel']:7d}  "
              f"{row['median']:7.0f}  {row['worst']:+7.1f}σ  {at:>6}")


def _table(title, rows):
    print(f"\n{title}")
    print(f"  {'segment':>10}  {'frames':>6}  {'median z':>8}  {'max z':>7}  {'z>4':>6}")
    for name, stats in rows.items():
        if stats["frames"]:
            print(f"  {name:>10}  {stats['frames']:6d}  {stats['median']:8.1f}  "
                  f"{stats['max']:7.1f}  {100 * stats['over_4']:5.1f}%")
        else:
            print(f"  {name:>10}  {0:6d}  {'n/a':>8}  {'n/a':>7}  {'n/a':>6}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Record an antenna trial with a hand at several distances, "
                    "then report z per segment with and without the high-pass.")
    parser.add_argument("--source", default="serial:auto",
                        help="the input to record (default %(default)s)")
    parser.add_argument("--record", required=True,
                        help="where to write the frames, e.g. out/wire.jsonl")
    parser.add_argument("--idle", type=float, default=20.0,
                        help="seconds of empty room first (default %(default)s)")
    parser.add_argument("--seconds", type=float, default=15.0,
                        help="seconds per distance (default %(default)s)")
    parser.add_argument("--distances", default="20,30,40",
                        help="hand distances to step through, in cm (default %(default)s)")
    parser.add_argument("--calibration", default="auto", metavar="SECONDS|auto")
    parser.add_argument("--cue", default="both", choices=cues.MODES,
                        help="how to announce each distance change so it can "
                             "be followed without watching the terminal: tones, "
                             "speech, both (the default) or off")
    parser.add_argument("--timeout", type=float, default=None,
                        help="a serial source's idle timeout, in seconds")
    parser.add_argument("--no-reconnect", dest="reconnect", action="store_false",
                        help="end the trial rather than wait for a board that "
                             "stops mid-run; a reconnect keeps the schedule "
                             "aligned but nobody is being guided during the gap")
    parser.add_argument("--control-distance", type=float, default=2.0,
                        metavar="CM",
                        help="where the hand goes during the control, in cm; "
                             "0 makes it touch the wire (default %(default)s)")
    parser.add_argument("--control", type=float, default=10.0, metavar="SECONDS",
                        help="length of the placement control that opens every "
                             "trial: this many seconds of empty room, then the "
                             "same with a hand at 2 cm, abandoned if the hand "
                             "never appears (default %(default)s); 0 turns it off")
    parser.add_argument("--json", action="store_true",
                        help="print the measurements as one object")
    args = parser.parse_args(argv)

    try:
        distances = [float(part) for part in args.distances.split(",") if part.strip()]
    except ValueError:
        print("error: --distances takes comma-separated numbers, e.g. 20,30,40",
              file=sys.stderr)
        return 2
    marks = schedule(args.idle, args.seconds, distances)

    def open_source(seconds):
        options = {"seconds": seconds}
        if args.source.startswith("serial:"):
            options["reconnect"] = args.reconnect
            options["notify"] = lambda message: print(f"note  : {message}",
                                                      flush=True)
            if args.timeout is not None:
                options["timeout"] = args.timeout
        return csi.open_source(args.source, **options)

    def opened(seconds):
        """A source, or None with the reason already printed."""
        try:
            return open_source(seconds)
        except (ValueError, OSError) as error:
            print(f"error: {error}", file=sys.stderr)
            return None

    cue = cues.Cue(args.cue)
    print(f"recording {args.source} -> {args.record}")
    print(f"cue     : {cue.describe()}", flush=True)
    print("keep still and stay back from the wire during the idle stretch; "
          "then hold your hand at each distance when it says so.", flush=True)
    # The engine is started before the first frame, because starting it takes
    # seconds and the first prompt is due immediately.
    cue.start()
    try:
        if args.control > 0:
            # A hand that is not where the wire is looks exactly like an antenna
            # that sees nothing, and the difference is twenty seconds, not eighty.
            control = control_marks(args.control, args.control_distance)
            name = control_name(args.control_distance)
            control_path = args.record + ".control.jsonl"
            where = ("touching the wire" if args.control_distance == 0
                     else f"with a hand {name} from the wire")
            print(f"control : {args.control:g}s empty, then "
                  f"{args.control:g}s {where}", flush=True)
            source = opened(control[-1]["end"] + 5.0)
            if source is None:
                return 2
            try:
                record(source, control, control_path, cue=cue)
            finally:
                # A serial port can only be held by one source at a time, and
                # the trial needs it back for its own schedule.
                getattr(source, "close", lambda: None)()
            seen = level_table(csi.read_recording(control_path), control)
            if not control_passed(seen, name):
                worst = (seen.get(name) or {}).get("worst") or 0.0
                print(f"error: the {name} control saw nothing ({worst:+.1f}σ, "
                      f"needs {LEVEL_SIGMAS:g}σ), so the rest of the trial "
                      f"would measure an empty room. Check that the wire is the "
                      f"one in A0 and that the hand goes {name} from its free "
                      f"end, then run this again.", file=sys.stderr)
                return 4
            # Every channel, because the shape of the response is the answer to
            # "is the wire in the pin I think it is": the sketch reads A0..A7 in
            # order, so A0 is the first number and the peak should be there.
            sizes = " ".join(f"{100 * value:+.0f}%"
                             for value in seen[name]["fractions"])
            print(f"control : hand seen, {seen[name]['worst']:+.1f}σ worst; "
                  f"A0..A7 level {sizes}", flush=True)

        source = opened(marks[-1]["end"] + 5.0)
        if source is None:
            return 2
        try:
            written = record(source, marks, args.record, cue=cue)
        finally:
            getattr(source, "close", lambda: None)()
        cue.announce("trial complete, you can step away")
    finally:
        cue.close()
    marks_path = args.record + ".marks.json"
    with open(marks_path, "w", encoding="utf-8") as out:
        json.dump({"marks": marks, "frames": written}, out, indent=2)

    frames = csi.read_recording(args.record)
    if not frames:
        # The schedule is driven by frames, so a board that never spoke records
        # nothing at all — and an empty trial must say that rather than division
        # by zero on a floor that was never measured.
        print(f"error: no frames arrived from {args.source}. If the board is "
              f"plugged in, its adapter has probably stopped transmitting; "
              f"unplug and replug it, then run this again.", file=sys.stderr)
        return 2
    covered, wanted, missing = missing_segments(frames, marks)
    if missing:
        were = "was" if len(missing) == 1 else "were"
        print(f"error: the trial stopped {covered:.0f}s into {wanted:.0f}s, so "
              f"{', '.join(missing)} {were} never recorded. The board stopped "
              f"mid-run — unplug it and plug it back in, then run this again.",
              file=sys.stderr)
        return 2

    rate = csi._rate_from(frames) if len(frames) > 1 else 20.0
    calibration = detect.parse_calibrate(args.calibration)
    on, detector = analyse(frames, marks, detect.DEFAULT_HIGHPASS_SECONDS,
                           rate, calibration)
    off, unfiltered = analyse(frames, marks, 0.0, rate, calibration)

    levels = level_table(frames, marks)

    empty = empty_segments(on)
    if empty:
        print(f"error: {', '.join(empty)} recorded no frames, so this trial "
              f"measured nothing. The board stopped producing lines — unplug "
              f"it and plug it back in, then run this again.", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"record": args.record, "marks": marks,
                          "frames": written, "rate": rate,
                          "calibration": {"reason": detector.calibration_reason,
                                          "seconds": round(detector.calibration_seconds, 1)},
                          "highpass_on": on, "highpass_off": off}, indent=2))
        return 0

    print(f"\nrecorded {written} frames to {args.record} "
          f"(marks in {marks_path}, {rate:.1f} frames/s)")
    _table("high-pass on (0.5 s)", on)
    _table("high-pass off", off)
    print(f"\ncalibration : {detector.calibration_reason} after "
          f"{detector.calibration_seconds:.1f}s (auto)")
    floor = ("n/a" if detector.floor is None else f"{detector.floor:.6f}")
    plain = ("n/a" if unfiltered.floor is None else f"{unfiltered.floor:.6f}")
    print(f"floor       : {floor} with the filter, {plain} without")
    _level_table("levels (the channel that moved most, and by how much)", levels)
    if cue.mode != "off":
        # Delivered, not heard: the engine accepting a prompt is as far as a
        # program can check. A run with no hand in it is the place to look.
        print(f"cues        : {cue.beeped} sounded, {cue.spoken} spoken")

    moved = [(name, row) for name, row in levels.items()
             if name != marks[0]["name"] and abs(row["worst"]) >= LEVEL_SIGMAS]
    if baseline_only(on) and not moved:
        print(f"\nwarning: no stretch rose above the idle rate and no channel "
              f"level moved by {LEVEL_SIGMAS:g}σ, so this is a baseline, not a "
              f"measurement — no hand was in front of the wire during any of "
              f"the segments. Check that the prompts were heard, then run the "
              f"trial again with the hand in place when it says so.")
        return 3
    if baseline_only(on) and moved:
        # The hand was there and did not move: the motion table is honestly
        # empty and the level table is the one that answers the question.
        print(f"\nnote: no stretch cleared the motion threshold, but "
              f"{', '.join(name for name, _ in moved)} moved a channel level — "
              f"a hand that arrives slowly is presence, not motion, and the "
              f"level table above is the measurement.")

    print("\nread it sideways: a hand that shows up with the filter on and "
          "disappears with it off means the filter is eating the signal; the "
          "reverse means the filter is doing its job.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
