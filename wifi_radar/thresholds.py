#!/usr/bin/env python3
"""
How many false alarms does each threshold actually produce?

`on_sigma` is a multiple of the calibrated spread, and the right multiple
depends on the input's noise in a way no amount of theory settles: a radio's
noise is flat and fast, but a wire on an ADC pin carries hum, drift and whatever
the mains is doing, and the tail of that distribution is what a threshold has to
clear. So it is measured, on the input in front of you, rather than guessed:

    python -m wifi_radar.thresholds --source serial:COM5 --seconds 120
    python -m wifi_radar.thresholds --source sim:quiet --hand sim:walk --seconds 60

The first run says how often a still room trips each threshold; the optional
second says what each threshold would still catch. A threshold worth using is
the lowest one whose idle rate is under `--max-idle` (1% by default) — lower
buys nothing but false alarms, higher starts missing the thing the detector is
for.

Idle numbers are only numbers if the room was actually idle: leave the room, or
at least keep still, for the whole run. The detector says so while it
calibrates, and this says so if the run ended before calibration did.

A recording with marks — what `handtrial` writes — answers a second question
without going near the hardware again: how long the calibration window has to be
before its floor is worth trusting. The evidence for a window is the rest of the
idle stretch, so the window is chosen by measuring exactly that:

    python -m wifi_radar.thresholds --record out/wire-new.jsonl

That prints the idle rate each window length produces, picks the shortest one
under `--max-idle`, then sweeps the threshold with that window fixed and prints
the `--calibrate` and `--on-sigma` the recording supports. Nothing here is
assumed about the shape of the noise: a wire's is not Gaussian, and the tail is
the only part that a threshold has to clear.
"""

import argparse
import json
import statistics
import sys

from wifi_radar import csi, detect

DEFAULT_SIGMAS = (2.0, 3.0, 4.0, 5.0, 6.0)

#: Calibration windows to try against a marked recording, in seconds. The
#: shortest that holds its idle rate is the one to use: a longer window only
#: costs the person standing still at the start of every session.
CALIBRATION_CANDIDATES = (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0)

#: How much of the idle stretch has to be left to judge a window by. A window
#: that consumes all of its own evidence cannot be chosen from it.
MIN_PROBE_SECONDS = 4.0

#: The idle rate above which a threshold is not worth using.
DEFAULT_MAX_IDLE = 0.01

#: What counts as "something happened" when measuring what a threshold misses.
#: Working out where a hand was is not the job here; the timeline of a recording
#: is not labelled, so the frames where the metric rose at all are the only
#: available stand-in for "the hand was there", and a low bar keeps that set
#: honest about what a higher threshold would drop.
ACTIVE_Z = 1.0


def measure(frames, sigma, rate=20.0,
            highpass_seconds=detect.DEFAULT_HIGHPASS_SECONDS,
            calibration=None, **detector_kw):
    """
    One threshold's numbers: how often it fires, and against what.

    `median_z` is the idle run's typical distance above the floor, which is the
    number that says whether a threshold is far away or nearly touching: a
    median of 0 and a threshold of 4 is a wide margin, a median of 3 is not.
    """
    detector = detect.MotionDetector(
        rate=rate, calibrate_seconds=calibration, on_sigma=sigma,
        highpass_seconds=highpass_seconds, **detector_kw)
    measured = flagged = 0
    active = caught = 0
    zs = []
    for frame in frames:
        detection = detector.update(frame)
        if detection.calibrating:
            continue
        measured += 1
        flagged += 1 if detection.motion else 0
        if detection.z is not None:
            zs.append(detection.z)
            active += 1 if detection.z > ACTIVE_Z else 0
            caught += 1 if detection.z > sigma else 0
    return {
        "sigma": sigma,
        "measured": measured,
        "flagged": flagged,
        "rate": flagged / measured if measured else None,
        "median_z": statistics.median(zs) if zs else None,
        # Of the frames where anything rose at all, how many clear this
        # threshold — the "what would it miss" half of the measurement.
        "active": active,
        "caught": caught,
        "recall": caught / active if active else None,
        "calibration": detector.calibration_reason,
        "seconds": round(detector.calibration_seconds, 1),
    }


def sweep(frames, sigmas=DEFAULT_SIGMAS, **kwargs):
    """`measure` for each threshold, in the order given."""
    return [measure(frames, sigma, **kwargs) for sigma in sigmas]


def pick(rows, max_idle=DEFAULT_MAX_IDLE):
    """
    The lowest threshold whose idle rate stays under `max_idle`.

    Falls back to the strictest row when every one of them is too noisy: the
    honest answer then is that this input is not quiet enough for any of them,
    and the table says so rather than naming a number that does not work.
    """
    usable = [row for row in rows
              if row["rate"] is not None and row["rate"] <= max_idle]
    return usable[0]["sigma"] if usable else None


def _series(frames, rate, calibration, sigma, highpass_seconds):
    """(session time, z) for every frame, and the detector that measured them."""
    detector = detect.MotionDetector(
        rate=rate, calibrate_seconds=calibration, on_sigma=sigma,
        highpass_seconds=highpass_seconds)
    origin = frames[0].t
    out = []
    for frame in frames:
        detection = detector.update(frame)
        out.append((frame.t - origin,
                    None if detection.calibrating else detection.z))
    return out, detector


def _above(values, sigma):
    """The fraction of `values` over the threshold, or None with nothing to judge."""
    return sum(1 for z in values if z > sigma) / len(values) if values else None


def load_marks(record):
    """
    The segment marks written beside a recording.

    They are what makes a recording answer questions rather than just hold data:
    without them an idle frame and a frame with a hand in front of the wire are
    the same frame. `handtrial` writes them, and nothing else needs to.
    """
    with open(record + ".marks.json", encoding="utf-8") as handle:
        return json.load(handle)["marks"]


def calibration_table(frames, marks, rate,
                      candidates=CALIBRATION_CANDIDATES,
                      sigma=detect.DEFAULT_ON_SIGMA,
                      highpass_seconds=detect.DEFAULT_HIGHPASS_SECONDS):
    """
    Each candidate window, judged on the idle frames that follow it.

    The window is the only part of a session the floor is measured on, so the
    only honest test of a window is what it does to the frames after it: a
    window that leaves the idle stretch quiet has measured the room, one that
    leaves it tripping a sixth of the time has measured a lucky second.
    """
    idle_end = marks[0]["end"]
    rows = []
    for seconds in candidates:
        if idle_end - seconds < MIN_PROBE_SECONDS:
            continue
        series, detector = _series(frames, rate, seconds, sigma, highpass_seconds)
        rest = [z for t, z in series
                if z is not None and seconds <= t < idle_end]
        if not rest:
            continue
        rows.append({"seconds": seconds, "floor": detector.floor,
                     "scale": detector.scale, "frames": len(rest),
                     "rate": _above(rest, sigma)})
    return rows


def pick_calibration(rows, max_idle=DEFAULT_MAX_IDLE):
    """The shortest window whose idle rate is acceptable, or None."""
    usable = [row for row in rows
              if row["rate"] is not None and row["rate"] <= max_idle]
    return usable[0]["seconds"] if usable else None


def segment_table(frames, marks, rate, calibration, sigmas=DEFAULT_SIGMAS,
                  highpass_seconds=detect.DEFAULT_HIGHPASS_SECONDS):
    """
    Per threshold, the share of each marked stretch above it.

    The first mark is the idle stretch and is the false-alarm column; the ones
    after it are where the hand was. Reading them together is the whole point: a
    threshold that clears the idle but never reaches the hand has bought
    silence rather than detection.
    """
    rows = []
    for sigma in sigmas:
        series, detector = _series(frames, rate, calibration, sigma,
                                   highpass_seconds)
        segments = {}
        for mark in marks:
            values = [z for t, z in series
                      if z is not None and mark["start"] <= t < mark["end"]]
            segments[mark["name"]] = {
                "frames": len(values), "rate": _above(values, sigma),
                "median_z": statistics.median(values) if values else None,
            }
        rows.append({"sigma": sigma, "segments": segments,
                     "rate": segments[marks[0]["name"]]["rate"],
                     "calibration": detector.calibration_reason})
    return rows


def choose_sigma(rows, max_idle=DEFAULT_MAX_IDLE):
    """The lowest threshold that keeps the idle stretch under `max_idle`."""
    usable = [row for row in rows
              if row["rate"] is not None and row["rate"] <= max_idle]
    return usable[0]["sigma"] if usable else None


def _collect(spec, seconds, rate):
    """Frames from a source spec, as a list, ending at `seconds` of session time."""
    source = csi.open_source(spec, seconds=seconds, rate=rate)
    frames = []
    for frame in source:
        if seconds and frame.t > seconds:
            break
        frames.append(frame)
    return frames


def _percent(value):
    return "  n/a" if value is None else f"{100 * value:5.1f}%"


def _z(value):
    return "     n/a" if value is None else f"{value:8.1f}"


def _report_record(args, sigmas, max_idle, highpass):
    """Derive the calibration length and threshold from a marked recording."""
    frames = csi.read_recording(args.record)
    if len(frames) < 2:
        print(f"error: {args.record} holds {len(frames)} frames; there is "
              f"nothing to measure.", file=sys.stderr)
        return 2
    try:
        marks = load_marks(args.record)
        marked = True
    except OSError:
        # A plain recording of an empty room is still the right input for the
        # calibration question, and it is the easier one to make: it needs a
        # minute of standing still, not a hand moved on cue. Taken as one long
        # idle stretch, which is what it is, it can size the window — but there
        # is nothing in it to test a threshold against.
        marks = [{"name": "idle", "start": 0.0,
                  "end": round(frames[-1].t - frames[0].t, 2)}]
        marked = False

    rate = csi._rate_from(frames)
    idle = marks[0]
    windows = calibration_table(frames, marks, rate, highpass_seconds=highpass)
    if not windows:
        print(f"error: the idle stretch is {idle['end'] - idle['start']:g}s, "
              f"too short to judge a window by — after the window there would "
              f"be less than {MIN_PROBE_SECONDS:g}s left to check it with. "
              f"Record with `--idle 60`.", file=sys.stderr)
        return 2

    seconds = pick_calibration(windows, max_idle)
    if seconds is None:
        # Every window tripped the idle stretch more than asked. The longest is
        # still the most evidence available, and saying so is better than
        # naming a number that was measured to be too small.
        seconds = windows[-1]["seconds"]
        best = min(windows, key=lambda row: row["rate"])["rate"]
        # Two decimals, because the near miss is the usual case: 1.01% against
        # a 1% target reads as "1.0%, under the target" at one decimal.
        chosen_why = (f"no window here keeps the idle under "
                      f"{100 * max_idle:.0f}%: the best is {seconds:g}s at "
                      f"{100 * best:.2f}%")
    else:
        chosen_why = (f"pick {seconds:g}s: the shortest window that keeps the "
                      f"idle under {100 * max_idle:.0f}%")

    rows = segment_table(frames, marks, rate, seconds, sigmas,
                         highpass_seconds=highpass)
    sigma = choose_sigma(rows, max_idle)

    if args.json:
        print(json.dumps({
            "record": args.record, "frames": len(frames), "rate": rate,
            "segments": [mark["name"] for mark in marks],
            "max_idle": max_idle, "calibration": windows,
            "chosen_calibration": seconds, "thresholds": rows,
            "chosen_sigma": sigma,
        }, indent=2))
        return 0

    print(f"record   : {args.record}, {len(frames)} frames at {rate:.0f}/s")
    print(f"idle     : {idle['start']:g}s-{idle['end']:g}s")
    if marked:
        print(f"segments : {', '.join(mark['name'] for mark in marks)}")
    else:
        print("segments : none marked — the whole recording is taken as idle, "
              "so only the calibration window can be measured from it")
    print()

    print(f"calibration window — threshold {detect.DEFAULT_ON_SIGMA:g}, judged on "
          f"the idle frames after it")
    print(f"  {'window':>7}  {'floor':>9}  {'scale':>9}  {'frames':>7}  "
          f"{'idle flagged':>13}")
    for row in windows:
        print(f"  {row['seconds']:6.1f}s  {row['floor']:9.5f}  {row['scale']:9.5f}  "
              f"{row['frames']:7d}  {_percent(row['rate']):>13}")
    print(f"  {chosen_why}\n")

    print(f"threshold — calibration {seconds:g}s")
    names = [mark["name"] for mark in marks]
    print(f"  {'sigma':>6}  " + "  ".join(f"{name:>8}" for name in names))
    for row in rows:
        cells = "  ".join(f"{_percent(row['segments'][name]['rate']):>8}"
                          for name in names)
        print(f"  {row['sigma']:6.1f}  {cells}")

    print()
    if sigma is None:
        print(f"no threshold here keeps the idle under {100 * max_idle:.0f}%: "
              f"this input is too noisy for any of them. Find what is moving "
              f"in the room, or raise --highpass.")
        return 0
    print(f"pick {sigma:g}: the lowest that keeps the idle under "
          f"{100 * max_idle:.0f}%")
    print(f"\nfor this recording: --calibrate {seconds:g} --on-sigma {sigma:g}")

    chosen = rows[[row["sigma"] for row in rows].index(sigma)]
    idle_rate = chosen["segments"][names[0]]["rate"]
    quiet = [name for name in names[1:]
             if chosen["segments"][name]["rate"] <= idle_rate]
    if quiet:
        # Said out loud, because a table whose columns all look alike is
        # otherwise read as a detector that works on everything.
        print(f"note : at {sigma:g}, {', '.join(quiet)} sit at or below the idle "
              f"rate — this recording does not show a hand there beyond what "
              f"the empty room does.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure a source's false-alarm rate at several detection "
                    "thresholds, and say which one to use.")
    parser.add_argument("--source", default=None,
                        help="the input to measure, e.g. serial:COM5 or sim:quiet")
    parser.add_argument("--record", default=None,
                        help="a marked recording (handtrial writes the marks "
                             "beside it), to derive the calibration length and "
                             "the threshold from what was actually recorded")
    parser.add_argument("--hand", default=None,
                        help="optional input with a person moving, to measure "
                             "what each threshold would still catch")
    parser.add_argument("--seconds", type=float, default=120.0,
                        help="how much of the session to measure (default %(default)s)")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--sigmas", default=",".join(str(s) for s in DEFAULT_SIGMAS),
                        help="comma-separated thresholds to try (default %(default)s)")
    parser.add_argument("--calibration", default="auto", metavar="SECONDS|auto",
                        help="calibration window for the measurement (default auto)")
    parser.add_argument("--highpass", type=float,
                        default=detect.DEFAULT_HIGHPASS_SECONDS,
                        help="high-pass time constant (default %(default)s)")
    parser.add_argument("--max-idle", type=float, default=DEFAULT_MAX_IDLE,
                        help="the highest idle rate worth accepting, as a "
                             "fraction (default %(default)s = 1%%)")
    parser.add_argument("--json", action="store_true",
                        help="print the measurements as one object")
    args = parser.parse_args(argv)

    try:
        sigmas = [float(part) for part in args.sigmas.split(",") if part.strip()]
    except ValueError:
        print("error: --sigmas takes comma-separated numbers, e.g. 3,4,5",
              file=sys.stderr)
        return 2
    if not sigmas:
        print("error: --sigmas needs at least one threshold.", file=sys.stderr)
        return 2
    if args.record and args.source:
        print("error: --record measures a file, --source measures a live input; "
              "pass one of them.", file=sys.stderr)
        return 2
    if not args.record and not args.source:
        print("error: pass --source to measure a live input, or --record to "
              "derive the numbers from a recording.", file=sys.stderr)
        return 2
    if args.record:
        return _report_record(args, sigmas, args.max_idle, args.highpass)

    calibration = detect.parse_calibrate(args.calibration)

    try:
        idle = _collect(args.source, args.seconds, args.rate)
        hand = _collect(args.hand, args.seconds, args.rate) if args.hand else None
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    kwargs = {"rate": args.rate, "calibration": calibration,
              "highpass_seconds": args.highpass}
    idle_rows = sweep(idle, sigmas, **kwargs)
    hand_rows = sweep(hand, sigmas, **kwargs) if hand is not None else None
    chosen = pick(idle_rows, args.max_idle)

    if args.json:
        print(json.dumps({
            "source": args.source, "hand": args.hand,
            "seconds": args.seconds, "max_idle": args.max_idle,
            "idle": idle_rows, "with_hand": hand_rows, "chosen": chosen,
        }, indent=2))
        return 0

    print(f"idle source : {args.source}, {len(idle)} frames")
    if hand is not None:
        print(f"hand source : {args.hand}, {len(hand)} frames")
    print()
    header = f"{'sigma':>6}  {'idle flagged':>12}  {'median z':>8}"
    if hand_rows:
        header += f"  {'hand caught':>13}"
    print(header)
    for index, row in enumerate(idle_rows):
        line = (f"{row['sigma']:6.1f}  {_percent(row['rate']):>12}  "
                f"{_z(row['median_z'])}")
        if hand_rows:
            line += f"  {_percent(hand_rows[index]['recall']):>13}"
        print(line)

    print()
    if chosen is None:
        print(f"no threshold here keeps the idle rate under "
              f"{100 * args.max_idle:.0f}%: this input is too noisy for any of "
              f"them. Find what is moving in the room, or raise --highpass.")
    else:
        print(f"pick {chosen:g}: the lowest that stays under "
              f"{100 * args.max_idle:.0f}% false alarms while idle.")
    if any(row["calibration"] is None for row in idle_rows):
        print("warning: the idle run ended before calibration did — nothing "
              "was measured. Use a longer --seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
