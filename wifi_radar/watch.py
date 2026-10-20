#!/usr/bin/env python3
"""
A live motion trace in the terminal.

    .venv/bin/python -m wifi_radar.watch                      # the demo scenario
    .venv/bin/python -m wifi_radar.watch --source sim:walk --seconds 25
    .venv/bin/python -m wifi_radar.watch --record session.jsonl
    .venv/bin/python -m wifi_radar.watch --source file:session.jsonl

The trace is one line per update: presence, the motion metric as a bar, how far
above the calibrated floor it is, and where the Doppler peak sits. While the
detector is calibrating it says so, because the thing it is asking for — an
empty, still room for two seconds — is something only the person watching it can
provide.

Piping the output (or running in a terminal that is not interactive) turns the
in-place redraw off and prints one line at a time instead, so a session can be
logged and read afterwards.
"""

import argparse
import json
import sys
import time

from wifi_radar import csi, detect

#: Blocks used for the metric bar, low to high.
BAR = "▁▂▃▄▅▆▇█"


def sparkline(history, floor, columns=34):
    """
    The last `columns` samples as column heights.

    A single bar would only ever show the current frame, which is unreadable —
    the point of a trace is that a person walking is a shape that rises and
    falls over a second or two, not a number that changes between lines. The
    scale is a few times the calibrated floor, so the same display works in a
    quiet room and a noisy one.
    """
    full = max(6 * floor, 1e-9)
    recent = history[-columns:]
    bar = "".join(BAR[min(len(BAR) - 1, int(min(1.0, m / full) * (len(BAR) - 1)))]
                  for m in recent)
    return bar + " " * (columns - len(bar))


def draw_line(detection, floor, history, columns=34):
    """One trace line: presence, the sparkline, and the numbers behind it."""
    bar = sparkline(history, floor or 0.0, columns)
    if detection.calibrating:
        state = "calibrating — keep still"
    elif detection.presence:
        state = "PRESENT"
    else:
        state = "       "
    z = "  n/a" if detection.z is None else f"{detection.z:5.1f}"
    peak = "  n/a" if detection.peak_hz is None else f"{detection.peak_hz:5.2f}"
    return (f"{detection.t:7.2f}s {state} {bar:<{columns}} "
            f"m={detection.metric:7.5f} z={z} peak={peak}Hz "
            f"band={_db(detection.band_power)}")


def _db(value):
    return "  n/a" if value is None else f"{value:5.1f}"


def calibration_note(detector):
    """
    How long the window was and why it closed.

    An automatic window that ran for sixteen seconds looks like a hang unless
    the reason is printed next to it, and the reason is the interesting part:
    the input was still drifting, so the detector waited for whole cycles of it.
    """
    seconds = detector.calibration_seconds
    reason = detector.calibration_reason
    if reason is None:
        return f"{seconds:.1f}s so far, still measuring"
    words = {"fixed": "fixed window",
             "settled": "auto, the floor had settled",
             "cycles": "auto, covered whole cycles of the drift",
             "limit": "auto, stopped at the limit"}
    return f"{seconds:.1f}s ({words.get(reason, reason)})"


def summarise(detections, floor, took, frames=None, calibration=None):
    """What the session amounted to, so a run can be read after it has gone."""
    measured = [d for d in detections if not d.calibrating]
    flagged = [d for d in measured if d.motion]
    present = [d for d in measured if d.presence]
    peaks = [d.peak_hz for d in measured if d.peak_hz is not None]
    lines = [
        "",
        f"frames          : {len(detections)} ({frames or len(detections)} read)",
        f"calibrated floor: {floor:.6f}" if floor else "calibrated floor: n/a",
        *([f"calibration     : {calibration}"] if calibration else []),
        f"wall clock      : {took:.1f}s",
        f"flagged frames  : {len(flagged)} of {len(measured)} measured "
        f"({100 * len(flagged) / max(1, len(measured)):.1f}%)",
        f"presence held   : {100 * len(present) / max(1, len(measured)):.1f}% of the session",
    ]
    if detections and not measured:
        # Not an error, but not a measurement either: a session shorter than its
        # own calibration window would otherwise read as a quiet room.
        lines.append(f"no frames measured: the run ended during the "
                     f"{len(detections)}-frame calibration. Run longer than "
                     f"--calibrate, or lower it.")
    if peaks:
        ordered = sorted(peaks)
        lines.append(f"doppler peak    : median {ordered[len(ordered) // 2]:.2f} Hz "
                     f"over {len(peaks)} estimates")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="sim:demo",
                        help="sim:<scenario>, file:<path>, udp:[host:]port "
                             "(nexmon CSI) or serial:<port> (a CSI receiver or "
                             "a sensor node — see serial_node.py)")
    parser.add_argument("--seconds", type=float, default=40.0,
                        help="how long to watch, in seconds of session time")
    parser.add_argument("--rate", type=float, default=20.0, help="frames per second")
    parser.add_argument("--fast", action="store_true",
                        help="run as fast as the CPU allows instead of in real time")
    parser.add_argument("--record", default=None,
                        help="write the frames to this JSON Lines file as they arrive")
    parser.add_argument("--every", type=float, default=0.25,
                        help="seconds between trace lines")
    parser.add_argument("--timeout", type=float, default=None, metavar="SECONDS",
                        help="a serial source gives up after this long with "
                             "nothing readable (default 10); raise it while a "
                             "board is being reset or plugged back in by hand")
    parser.add_argument("--no-reconnect", dest="reconnect", action="store_false",
                        help="end the session when a serial board stops instead "
                             "of waiting for it to come back")
    parser.add_argument("--reconnect-seconds", type=float, default=None,
                        metavar="SECONDS",
                        help="how long to keep trying after a serial board goes "
                             "away (default 120)")
    parser.add_argument("--calibrate", type=detect.parse_calibrate,
                        default=None, metavar="SECONDS|auto",
                        help="how long to measure the still floor before "
                             "deciding anything; auto (the default) keeps "
                             "measuring while the input drifts and stops once "
                             "the floor has settled, a number fixes the window "
                             "at that many seconds")
    parser.add_argument("--highpass", type=float,
                        default=detect.DEFAULT_HIGHPASS_SECONDS,
                        metavar="SECONDS",
                        help="subtract a baseline that follows the input with "
                             "this time constant, so drift slower than it is "
                             "ignored (default %(default)s); 0 turns it off")
    parser.add_argument("--on-sigma", type=float,
                        default=detect.DEFAULT_ON_SIGMA, metavar="SIGMA",
                        help="how many calibrated sigmas of change count as "
                             "motion (default %(default)s); a wire on an ADC "
                             "pin can need more, and thresholds.py measures "
                             "the number on the input in front of you")
    args = parser.parse_args(argv)

    if args.highpass < 0:
        print("error: --highpass cannot be negative; 0 turns the filter off.",
              file=sys.stderr)
        return 2

    if args.on_sigma <= 0:
        print("error: --on-sigma must be more than 0; it multiplies the "
              "calibrated spread, and a threshold of zero would call every "
              "frame movement.", file=sys.stderr)
        return 2

    if args.calibrate is not None and args.calibrate <= 0:
        print("error: --calibrate must be more than 0 seconds (or auto). The "
              "detector needs a still window to measure the noise floor from; "
              "without one every frame would be judged against nothing.",
              file=sys.stderr)
        return 2

    try:
        options = {"seconds": args.seconds, "rate": args.rate}
        if args.source.startswith("serial:"):
            # Only a serial source has an idle timeout or a cable to lose; the
            # simulator and a recording end on their own.
            if args.timeout is not None:
                options["timeout"] = args.timeout
            options["reconnect"] = args.reconnect
            if args.reconnect_seconds is not None:
                options["reconnect_seconds"] = args.reconnect_seconds
            options["notify"] = lambda message: print(f"note  : {message}",
                                                      flush=True)
        source = csi.open_source(args.source, **options)
    except (ValueError, OSError) as error:
        # A serial port that is not there is an OSError, and its message already
        # names the port and what to check; a traceback would bury that.
        print(f"error: {error}", file=sys.stderr)
        return 2

    paced = csi.PacedSource(source, realtime=not args.fast)
    detector = detect.MotionDetector(rate=paced.rate,
                                     calibrate_seconds=args.calibrate,
                                     highpass_seconds=args.highpass,
                                     on_sigma=args.on_sigma)
    interactive = sys.stdout.isatty()
    recorded = open(args.record, "w", encoding="utf-8") if args.record else None

    # Flushed throughout: this is a live trace, and a tracer that appears one
    # buffer-full late is not one. Piping the output is the common case here.
    print(f"source: {paced.describe()}", flush=True)
    print(f"frames: {paced.length if paced.length is not None else 'unbounded'}"
          f" at {paced.rate:.0f}/s"
          + (f"; recording to {args.record}" if recorded else ""), flush=True)
    print("-" * 96, flush=True)

    detections = []
    last_line = 0.0
    noted = False
    started = time.time()
    read = 0
    failed = None
    try:
        for frame in paced:
            if args.seconds and frame.t > args.seconds:
                break
            read += 1
            if recorded:
                # One place defines the recording format: `Frame.as_record`.
                recorded.write(json.dumps(frame.as_record()) + "\n")
            # A serial source can switch from sensor channels to CSI once it
            # sees the first line, so the detector is told each frame.
            detector.doppler = paced.doppler
            detection = detector.update(frame)
            detections.append(detection)
            if detection.t - last_line >= args.every:
                last_line = detection.t
                if not paced.doppler and not noted:
                    # Said once, at the first line rather than at startup: a
                    # serial source only learns from that line whether it is
                    # talking to a CSI receiver or to a board with no radio, and
                    # a reader who expects a speed and finds "peak=n/a" will
                    # otherwise read it as a fault.
                    noted = True
                    print("note  : this source measures channel changes, not "
                          "CSI — there is no speed to report", flush=True)
                line = draw_line(detection, detector.floor, detector.metrics)
                if interactive:
                    sys.stdout.write("\r" + line[:96])
                    sys.stdout.flush()
                else:
                    print(line, flush=True)
    except KeyboardInterrupt:
        print("\nstopped")
    except (TimeoutError, OSError, ValueError) as error:
        # A hardware source fails in ways a simulated one cannot: nothing
        # arriving at all, or a packet that is not CSI. Printing the traceback
        # buries the one line that says which, and the message is written to be
        # read (see nexmon.NexmonSource).
        failed = error
        print(f"\nerror: {error}", file=sys.stderr)
    finally:
        if recorded:
            recorded.close()

    if interactive:
        sys.stdout.write("\r" + " " * 96 + "\r")
    print(summarise(detections, detector.floor, time.time() - started, read,
                    calibration=calibration_note(detector)))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
