#!/usr/bin/env python3
"""
One command that answers the three questions a silent board asks.

A board on the end of a cable fails in ways that look identical from the host:

1. no port appears at all — missing driver, charge-only cable, powered-down hub;
2. a port appears and nothing is on the other end — the wrong port, a USB-serial
   adapter that is not the board, a board with no power;
3. the board is alive and the sketch is not running, or is running at another baud
   rate and printing something this package cannot parse.

`--list` distinguishes the first from the rest, and then stops. This module goes
through all three in one run, using the only signals that exist: the USB identity
the port exposes, the AVR bootloader's own protocol, and the line protocol the
firmware speaks.

The bootloader probe is what separates "dead board" from "silent sketch". Every
Arduino resets when its port is opened — DTR is wired to the reset pin — and for
about a second the bootloader listens for stk500v2 commands before jumping to the
application. A `GET_SYNC` inside that second is answered by the MCU itself, so a
reply proves the chip is powered, clocked, wired to the USB serial chip, and
running its bootloader. It proves nothing about the sketch, which is why the
second half resets the board again and listens for sample lines.

The frame below is not from a datasheet: it is the one that made a Mega 2560 on
this desk answer, twenty-four times, immediately before avrdude flashed it.
"""

import json
import time

from wifi_radar import serial_node

#: stk500v2 framing. `SIZE` counts the token and the command, and the checksum is
#: the XOR of every byte after the start byte — the rule this frame was accepted
#: with, so it is the rule written here.
MESSAGE_START = 0x1B
TOKEN = 0x0E
STK_GET_SYNC = 0x30
SYNC_FRAME = bytes([MESSAGE_START, 0x00, 0x00, 0x02, TOKEN, STK_GET_SYNC, 0x3C])

#: How long to keep asking for a sync. The bootloader's window is about a second;
#: asking for two covers a board that resets slower than the one this was
#: measured on, and a board that never answers costs two seconds, once.
DEFAULT_SYNC_SECONDS = 2.0

#: How long to listen for sample lines after the reset. The bootloader spends
#: roughly a second of that window saying nothing, so four seconds is about three
#: seconds of samples — enough for sixty frames at 20 Hz and a measured rate.
DEFAULT_LISTEN_SECONDS = 4.0

#: Rates to try when the requested one produces no samples, most likely first.
#: 9600 is the historical Arduino default and what most tutorials write; 115200
#: is what the sketch here uses. The rest turn up in libraries and older boards.
COMMON_BAUDS = (9600, 57600, 38400, 19200, 230400, 250000, 500000)

#: How long each candidate rate is given while scanning. Opening the port resets
#: the board and its bootloader is silent for about a second, so two seconds is
#: about one second of samples — a line or two at any sane rate.
DEFAULT_SCAN_SECONDS = 2.0

#: What to say when there is no port to open. This is the failure mode with the
#: least diagnostic value from the host side and the most common cause, so it
#: names the cause.
NO_PORT = (
    "no serial port found. On Windows a board appears as a COM port only after "
    "the CH340 driver is installed (Elegoo and most clones use that chip, not an "
    "Arduino one); on Linux check `dmesg | tail` after plugging it in, and that "
    "your user is in the dialout group."
)


def parse_sync_replies(data):
    """
    (replies, last status byte) from whatever the bootloader answered.

    A reply is eight bytes from an ATmega2560: start, sequence, size, token, the
    echoed command, a status byte and a checksum. Only the fields that are stable
    across bootloader versions are required here — start byte, token, echoed
    command — because a doctor that insists on the exact checksum of one
    bootloader build would call a working board dead.

    Scanning forward rather than searching for the start byte keeps a 0x1B inside
    a payload from being counted as the beginning of another reply.
    """
    replies = 0
    status = None
    index = 0
    while index < len(data):
        if data[index] != MESSAGE_START:
            index += 1
            continue
        if (len(data) - index < 6 or data[index + 4] != TOKEN
                or data[index + 5] != STK_GET_SYNC):
            index += 1
            continue
        replies += 1
        if len(data) - index >= 7:
            status = data[index + 6]
        index += 8
    return replies, status


def status_ok(status):
    """Whether an stk500v2 status byte reports success, per the low nibble."""
    return status is not None and (status & 0x0F) == 0


def read_available(stream, limit=4096):
    """
    Bytes the port already holds, without waiting for any more.

    A CH340 on Windows can block indefinitely inside ReadFile after a framing
    error — and a wrong baud rate is precisely a stream of framing errors, so
    this is the state the doctor exists to diagnose. Measured here: reading at
    9600 from a board printing at 115200 returned two reads and then hung even
    with a 0.2 s timeout set. So nothing here asks a port for bytes it does not
    have; `in_waiting` is the only thing trusted, and the caller's loop does the
    waiting. A stream with no `in_waiting` is a replay or a test double, and
    `readline()` on one returns at end of file rather than blocking.
    """
    waiting = getattr(stream, "in_waiting", None)
    if waiting is None:
        if hasattr(stream, "readline"):
            return stream.readline()
        return stream.read(limit)
    if not waiting:
        return b""
    return stream.read(min(waiting, limit))


def reset_board(stream, sleep=time.sleep):
    """
    Pull DTR low and let it go: the same edge the Arduino auto-reset makes.

    pyserial asserts DTR when the port opens, so the board has already reset by
    the time this runs. Driving the line explicitly makes the timing the host's
    own rather than a race with whatever the driver did on open. A stream with no
    `dtr` is a replay or a test double, and needs nothing.
    """
    if not hasattr(stream, "dtr"):
        return
    try:
        stream.dtr = False
        sleep(0.05)
        stream.dtr = True
    except (OSError, ValueError):
        # Some drivers refuse the modem-line ioctl: a pseudo-terminal has no
        # DTR at all, and a few USB bridges report ENOTTY. The open itself
        # already reset a board that auto-resets, so this is a timing detail,
        # and a diagnostic that crashes on a timing detail is worse than one
        # that skips it.
        return


def probe_bootloader(port, baud=serial_node.DEFAULT_BAUD,
                     sync_seconds=DEFAULT_SYNC_SECONDS, interval=0.05,
                     open_port=None, sleep=time.sleep, clock=time.monotonic):
    """
    Ask the AVR bootloader for a sync, repeatedly, just after a reset.

    The result says only what the hardware said: whether it answered, how many
    times, and the last status byte. Whether that matters is the caller's job —
    a board with a different bootloader says nothing here and can still be
    printing perfectly good samples.
    """
    opener = open_port or serial_node.open_serial
    result = {"answered": False, "replies": 0, "status": None,
              "error": None, "duration": 0.0}
    started = clock()
    try:
        stream = opener(port, baud)
    except OSError as error:
        result["error"] = str(error)
        result["duration"] = round(clock() - started, 2)
        return result

    # A wedged USB serial chip can block a write indefinitely, and a diagnostic
    # tool that hangs on the fault it is meant to name is worse than useless.
    # Seven bytes never need long; a timeout turns a hang into an error.
    try:
        stream.write_timeout = 0.5
    except (AttributeError, ValueError):
        pass

    data = bytearray()
    try:
        reset_board(stream, sleep)
        deadline = clock() + sync_seconds
        while clock() < deadline:
            try:
                stream.write(SYNC_FRAME)
                flush = getattr(stream, "flush", None)
                if flush is not None:
                    flush()
            except OSError as error:
                result["error"] = f"write failed: {error}"
                break
            chunk = read_available(stream, 256)
            if chunk:
                data += chunk
            sleep(interval)
    finally:
        try:
            stream.close()
        except (AttributeError, OSError):
            pass

    result["duration"] = round(clock() - started, 2)
    result["replies"], result["status"] = parse_sync_replies(bytes(data))
    result["answered"] = result["replies"] > 0 and status_ok(result["status"])
    return result


def listen(port, baud=serial_node.DEFAULT_BAUD,
           seconds=DEFAULT_LISTEN_SECONDS, open_port=None,
           sleep=time.sleep, clock=time.monotonic):
    """
    Listen for sample lines and report what arrived, byte by byte.

    Three outcomes have to stay distinguishable, because they have three
    different fixes: nothing at all (the sketch is not running), bytes that are
    not lines in this protocol (usually the wrong baud rate), and samples. So the
    result counts bytes and lines as well as frames, and keeps the first thing
    that arrived so it can be shown.
    """
    opener = open_port or serial_node.open_serial
    heard = {"bytes": 0, "lines": 0, "samples": 0, "rejected": 0,
             "banner": None, "first_sample": None, "sample": None,
             "rejection": None, "rate": None, "duration": 0.0, "error": None}
    started = clock()
    try:
        stream = opener(port, baud)
    except OSError as error:
        heard["error"] = str(error)
        heard["duration"] = round(clock() - started, 2)
        return heard

    first_ms = None
    last_ms = None
    pending = bytearray()

    def take(raw):
        """One line from the port, counted as a line, a sample, or a rejection."""
        nonlocal first_ms, last_ms
        stripped = raw.strip()
        if not stripped:
            return
        heard["lines"] += 1
        if stripped.startswith("#") or stripped.startswith("//"):
            if heard["banner"] is None:
                heard["banner"] = stripped
            return
        try:
            _, millis, _ = serial_node.parse_line(stripped)
        except ValueError as error:
            heard["rejected"] += 1
            if heard["rejection"] is None:
                heard["rejection"] = str(error)
            return
        heard["samples"] += 1
        if heard["first_sample"] is None:
            heard["first_sample"] = stripped
        heard["sample"] = stripped
        if first_ms is None:
            first_ms = millis
        last_ms = millis

    try:
        reset_board(stream, sleep)
        deadline = clock() + seconds
        while clock() < deadline:
            chunk = read_available(stream)
            if not chunk:
                sleep(0.01)
                continue
            if isinstance(chunk, str):
                # A text stream hands over whole lines; nothing to reassemble.
                heard["bytes"] += len(chunk)
                for line in chunk.splitlines():
                    take(line)
                continue
            heard["bytes"] += len(chunk)
            pending += chunk
            while b"\n" in pending:
                raw, _, pending = pending.partition(b"\n")
                take(raw.decode("utf-8", "replace"))
    finally:
        try:
            stream.close()
        except (AttributeError, OSError):
            pass

    heard["duration"] = round(clock() - started, 2)
    if (heard["samples"] >= 2 and first_ms is not None and last_ms is not None
            and last_ms > first_ms):
        heard["rate"] = (heard["samples"] - 1) / ((last_ms - first_ms) / 1000.0)
    return heard


def find_baud(port, baud=serial_node.DEFAULT_BAUD,
              seconds=DEFAULT_LISTEN_SECONDS,
              scan_seconds=DEFAULT_SCAN_SECONDS, open_port=None,
              sleep=time.sleep, clock=time.monotonic):
    """
    The rate the firmware is actually using, or the requested one if none works.

    A wrong baud fails in two disguises that both look like a broken board: no
    bytes at all when the mismatch is gross, and a stream of lines that never
    parse when it is close. So the requested rate is tried first, and only when
    it yields no samples are the common rates tried. The first rate that produces
    samples wins and the scan stops there — reporting every rate that happened to
    produce bytes would name whichever rate best decodes the noise.

    `scan_seconds` of 0 disables the scan, which is what a script that wants one
    open-and-close should pass.
    """
    window = min(seconds, scan_seconds) if scan_seconds > 0 else 0.0
    tried = []

    def attempt(rate, duration):
        heard = listen(port, rate, duration, open_port=open_port,
                       sleep=sleep, clock=clock)
        heard["baud"] = rate
        heard["baud_requested"] = baud
        tried.append({"baud": rate, "bytes": heard["bytes"],
                      "samples": heard["samples"],
                      "rejected": heard["rejected"]})
        return heard

    heard = attempt(baud, seconds)
    if heard["samples"] or heard["error"] or window <= 0:
        heard["tried"] = tried
        return heard, tried

    for rate in COMMON_BAUDS:
        if rate == baud:
            continue
        other = attempt(rate, window)
        if other["samples"]:
            other["tried"] = tried
            return other, tried
        if other["error"]:
            # The port went away mid-scan; trying more rates cannot help.
            break

    heard["tried"] = tried
    return heard, tried


def board_line(port, info):
    """The USB identity of the port, in one line, or an honest admission."""
    if info is None:
        return f"board      : {port} (named by hand, not enumerated)"
    parts = [getattr(info, "device", port) or port]
    description = getattr(info, "description", "") or ""
    if description:
        parts.append(description)
    vid, pid = getattr(info, "vid", None), getattr(info, "pid", None)
    if vid is not None and pid is not None:
        parts.append(f"VID:PID={vid:04X}:{pid:04X}")
    serial_number = getattr(info, "serial_number", None)
    if serial_number:
        parts.append(f"SER={serial_number}")
    return "board      : " + "  ".join(parts)


def classify(boot, heard):
    """
    (verdict, exit status): the one place the three answers become a conclusion.

    The text report, the JSON report and the process exit code all come from
    here, so they cannot drift apart. The verdict names *which* fix applies, not
    just whether it works — "silent_firmware" and "silent_no_bootloader" are the
    same empty screen with completely different causes.
    """
    if boot["error"] and heard["error"]:
        return "unopenable", 2
    if heard["error"]:
        return "unopenable", 2
    if heard["samples"]:
        return "printing", 0
    if heard["bytes"]:
        return "wrong_protocol", 1
    if boot["answered"]:
        return "silent_firmware", 1
    return "silent_no_bootloader", 1


#: One sentence per verdict, for the JSON payload's `message`.
VERDICT_MESSAGE = {
    "printing": "the board is identified, alive, and printing samples",
    "silent_firmware": "the MCU answers its bootloader but nothing is printing: "
                       "the sketch is missing or not running",
    "silent_no_bootloader": "nothing is printing and the bootloader did not "
                            "answer: check power, the port and the cable",
    "wrong_protocol": "bytes are arriving but none parse as S, or C, lines: "
                      "wrong baud rate or wrong firmware",
    "unopenable": "the port could not be opened; something else may hold it",
    "no_port": NO_PORT,
}


def as_json(port, info, boot, heard, verdict, status):
    """
    The same answers as `report`, as one object a script can read.

    `verdict` and `exit_status` are duplicated deliberately: the exit code is
    what CI already branches on, and having it inside the document means a log
    file read months later still says what the run concluded. Numbers stay
    numbers (`rate_hz`, `vid`) rather than the formatted strings the human
    report prints, because a script that has to strip "20.0/s" has a bug.
    """
    board = None
    if info is not None:
        confidence, reason = serial_node.identify_port(info)
        board = {
            "device": getattr(info, "device", port) or port,
            "description": getattr(info, "description", "") or None,
            "manufacturer": getattr(info, "manufacturer", "") or None,
            "vid": getattr(info, "vid", None),
            "pid": getattr(info, "pid", None),
            "serial_number": getattr(info, "serial_number", None),
            "recognised": reason or None,
            "confidence": confidence,
        }
    elif port is not None:
        board = {"device": port, "description": None, "manufacturer": None,
                 "vid": None, "pid": None, "serial_number": None,
                 "recognised": None, "confidence": None}

    message = VERDICT_MESSAGE.get(verdict, "")
    if (verdict == "printing" and heard is not None and heard.get("baud")
            and heard.get("baud") != heard.get("baud_requested")):
        message = (f"the firmware is printing at {heard['baud']} baud, not "
                   f"{heard['baud_requested']}")

    return {
        "port": port,
        "verdict": verdict,
        "exit_status": status,
        "message": message,
        "board": board,
        "bootloader": None if boot is None else {
            "answered": bool(boot["answered"]),
            "replies": boot["replies"],
            "status": boot["status"],
            "status_ok": status_ok(boot["status"]),
            "seconds": boot["duration"],
            "error": boot["error"],
        },
        "sketch": None if heard is None else {
            "printing": heard["samples"] > 0,
            "bytes": heard["bytes"],
            "lines": heard["lines"],
            "samples": heard["samples"],
            "rejected": heard["rejected"],
            "rate_hz": None if heard["rate"] is None else round(heard["rate"], 2),
            "baud": heard.get("baud"),
            "baud_requested": heard.get("baud_requested"),
            "banner": heard["banner"],
            "first_sample": heard["first_sample"],
            "rejection": heard["rejection"],
            "seconds": heard["duration"],
            "error": heard["error"],
        },
        "bauds_tried": [entry["baud"] for entry in (heard or {}).get("tried", [])],
    }


def report(port, info, boot, heard):
    """
    (text, exit status) for the three answers.

    0 means a board that is identified, alive and printing samples — the only
    state in which the rest of the package has something to read. 1 means a board
    that answered or a port that opened but produced no samples, which is a
    firmware or baud problem. 2 means there was nothing to open, which is a
    driver or cable problem. The distinction is the point of the exit code: a
    shell can tell "no board" from "board, wrong firmware".
    """
    verdict, status = classify(boot, heard)
    lines = [board_line(port, info)]
    if info is not None:
        _, reason = serial_node.identify_port(info)
        if reason:
            lines.append(f"recognised : {reason}")
    lines.append("")

    if verdict == "unopenable" and boot["error"] and heard["error"]:
        lines.append(f"open       : failed — {heard['error']}")
        lines.append("")
        lines.append("nothing could be opened. If a serial monitor or another "
                     "program is holding the port, close it first.")
        return "\n".join(lines), status

    if boot["answered"]:
        lines.append(f"bootloader : answered {boot['replies']} sync"
                     f"{'' if boot['replies'] == 1 else 's'} in "
                     f"{boot['duration']:.1f}s (status 0x{boot['status']:02x} = OK)")
        lines.append("             the MCU is powered, clocked and running its "
                     "bootloader")
    elif boot["replies"]:
        lines.append(f"bootloader : answered {boot['replies']} sync"
                     f"{'' if boot['replies'] == 1 else 's'}, status "
                     f"0x{boot['status']:02x} is not OK")
    elif boot["error"]:
        lines.append(f"bootloader : could not open the port — {boot['error']}")
    else:
        lines.append(f"bootloader : silence for {boot['duration']:.1f}s after a "
                     f"reset")
        lines.append("             not fatal on its own: a board with a different "
                     "bootloader says nothing here")

    baud_used = heard.get("baud")
    baud_asked = heard.get("baud_requested")
    if heard["error"]:
        lines.append(f"sketch     : could not open the port — {heard['error']}")
    elif heard["samples"]:
        rate = "rate not measured" if heard["rate"] is None else f"{heard['rate']:.1f}/s on the board's own clock"
        lines.append(f"sketch     : {heard['lines']} lines, {heard['samples']} "
                     f"samples in {heard['duration']:.1f}s, {rate}")
        if baud_used and baud_asked and baud_used != baud_asked:
            lines.append(f"             (arrived at {baud_used} baud; "
                         f"{baud_asked} produced no samples)")
        if heard["banner"]:
            lines.append(f"             {heard['banner']}")
        lines.append(f"             {heard['first_sample']}")
    elif heard["bytes"]:
        summary = f"{heard['bytes']} bytes"
        if heard["lines"]:
            summary += f" in {heard['lines']} lines"
        lines.append(f"sketch     : {summary} in {heard['duration']:.1f}s, none of "
                     f"it a sample line ({heard['rejected']} unreadable)")
        if heard["rejection"]:
            lines.append(f"             first: {heard['rejection']}")
    else:
        lines.append(f"sketch     : nothing arrived in {heard['duration']:.1f}s "
                     f"(0 bytes)")

    tried = heard.get("tried") or []
    if len(tried) > 1 and not heard["samples"]:
        names = ", ".join(str(entry["baud"]) for entry in tried)
        lines.append(f"baud       : tried {names} — none produced sample lines")

    lines.append("")
    if verdict == "printing":
        lines.append("ok — the board is identified, alive, and printing samples.")
        if baud_used and baud_asked and baud_used != baud_asked:
            lines.append(f"   baud : the firmware is printing at {baud_used}, not "
                         f"{baud_asked} — pass --baud {baud_used}")
        lines.append(f"   next: python -m wifi_radar.watch --source serial:{port}")
    elif verdict == "unopenable":
        lines.append("the port could not be opened for reading. Close anything "
                     "else holding it — the Arduino IDE's serial monitor is the "
                     "usual one — and run this again.")
    elif verdict == "wrong_protocol":
        lines.append("wrong protocol — something is printing, but not S, or C, "
                     "lines that parse. Check --baud against the sketch's "
                     "Serial.begin(), and that the firmware is the one in "
                     "arduino/mega_channel_node. The bootloader was probed at "
                     "the same baud, so a wrong --baud also explains a silent "
                     "bootloader.")
    elif verdict == "silent_firmware":
        lines.append("not printing — the cable, the port and the chip are all "
                     "fine, so the sketch is the missing piece. Upload "
                     "arduino/mega_channel_node to the board, then run this "
                     "again. If you just uploaded it, close the Arduino IDE's "
                     "serial monitor: it holds the port open and the board stays "
                     "in the bootloader it was left in.")
    else:
        lines.append("nothing is printing and the bootloader did not answer. "
                     "Check that the board has power and that this is its port "
                     "(`--port` or `--list`), then try another cable — a "
                     "charge-only cable powers the board and carries no data.")
    return "\n".join(lines), status


def main(argv=None):
    """Identify the board, probe its bootloader, and say whether it is printing."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="One-step bring-up: which board is this, is its chip alive, "
                    "and is the sketch printing?",
        epilog="Exit status 0 means samples are arriving, 1 means a board or port "
               "was found but nothing usable is printing, 2 means nothing could "
               "be opened.")
    parser.add_argument("--port", default="auto",
                        help="COM3 on Windows, /dev/ttyACM0 on Linux; auto (the "
                             "default) picks the likely board")
    parser.add_argument("--baud", type=int, default=serial_node.DEFAULT_BAUD)
    parser.add_argument("--seconds", type=float, default=DEFAULT_LISTEN_SECONDS,
                        help="how long to listen for sample lines")
    parser.add_argument("--sync-seconds", type=float,
                        default=DEFAULT_SYNC_SECONDS,
                        help="how long to ask the bootloader for a sync")
    parser.add_argument("--json", action="store_true",
                        help="print one JSON object instead of the text report; "
                             "the exit status is the same either way")
    parser.add_argument("--scan-seconds", type=float,
                        default=DEFAULT_SCAN_SECONDS, metavar="SECONDS",
                        help="when --baud produces no samples, try the common "
                             "rates and give each this long (default "
                             "%(default)s); 0 disables the scan")
    args = parser.parse_args(argv)

    if args.scan_seconds < 0:
        print("error: --scan-seconds cannot be negative; 0 turns the scan off.",
              file=sys.stderr)
        return 2

    def emit(payload, status):
        """The JSON document, or the text report when `--json` was not asked for."""
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(payload, file=sys.stderr if status == 2 else sys.stdout)

    try:
        ports = serial_node.available_ports()
    except OSError as error:
        if args.json:
            payload = as_json(None, None, None, None, "unopenable", 2)
            payload["message"] = str(error)
            emit(payload, 2)
        else:
            print(f"error: {error}", file=sys.stderr)
        return 2

    port, info = args.port, None
    if port.strip().lower() == "auto":
        info = serial_node.choose_port(ports)
        if info is None:
            if args.json:
                emit(as_json(None, None, None, None, "no_port", 2), 2)
            else:
                print(NO_PORT, file=sys.stderr)
            return 2
        port = info.device
    else:
        info = next((p for p in ports if p.device.lower() == port.lower()), None)

    boot = probe_bootloader(port, args.baud, max(0.0, args.sync_seconds))
    heard, _ = find_baud(port, args.baud, max(0.1, args.seconds),
                         scan_seconds=args.scan_seconds)
    verdict, status = classify(boot, heard)
    if args.json:
        emit(as_json(port, info, boot, heard, verdict, status), status)
    else:
        text, _ = report(port, info, boot, heard)
        print(text, file=sys.stderr if status == 2 else sys.stdout)
    return status


if __name__ == "__main__":
    import sys
    sys.exit(main())
