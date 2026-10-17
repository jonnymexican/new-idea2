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


def summarise(detections, floor, took, frames=None):
    """What the session amounted to, so a run can be read after it has gone."""
    measured = [d for d in detections if not d.calibrating]
    flagged = [d for d in measured if d.motion]
    present = [d for d in measured if d.presence]
    peaks = [d.peak_hz for d in measured if d.peak_hz is not None]
    lines = [
        "",
        f"frames          : {len(detections)} ({frames or len(detections)} read)",
        f"calibrated floor: {floor:.6f}" if floor else "calibrated floor: n/a",
        f"wall clock      : {took:.1f}s",
        f"flagged frames  : {len(flagged)} of {len(measured)} measured "
        f"({100 * len(flagged) / max(1, len(measured)):.1f}%)",
        f"presence held   : {100 * len(present) / max(1, len(measured)):.1f}% of the session",
    ]
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
                             "(nexmon CSI) or serial:<port>")
    parser.add_argument("--seconds", type=float, default=40.0,
                        help="how long to watch, in seconds of session time")
    parser.add_argument("--rate", type=float, default=20.0, help="frames per second")
    parser.add_argument("--fast", action="store_true",
                        help="run as fast as the CPU allows instead of in real time")
    parser.add_argument("--record", default=None,
                        help="write the frames to this JSON Lines file as they arrive")
    parser.add_argument("--every", type=float, default=0.25,
                        help="seconds between trace lines")
    args = parser.parse_args(argv)

    try:
        source = csi.open_source(args.source, seconds=args.seconds, rate=args.rate)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    paced = csi.PacedSource(source, realtime=not args.fast)
    detector = detect.MotionDetector(rate=paced.rate)
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
            detection = detector.update(frame)
            detections.append(detection)
            if detection.t - last_line >= args.every:
                last_line = detection.t
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
    print(summarise(detections, detector.floor, time.time() - started, read))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
