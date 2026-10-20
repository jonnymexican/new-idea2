#!/usr/bin/env python3
"""
The one check that needs a board on the desk.

Everything else in this package runs without hardware. This does not, because the
claim it tests is about real bytes: that `wifi_radar.doctor` can tell "the chip is
alive and the sketch is missing" from "nothing is there". A unit test can assert
what a fake port replies, but not that an actual bootloader answers the same way.

So it flashes a sketch that does nothing, runs the doctor as a user would, asserts
the verdict, restores `arduino/mega_channel_node`, and asserts samples come back.
The firmware is restored whatever happens in between — a self-test that leaves the
board silent is worse than no self-test.

It is deliberately not part of the unittest suite: CI has no Mega on the end of a
cable, and a test that skips is a test that rots. Run it by hand:

    python -m wifi_radar.selftest
    python -m wifi_radar.selftest --port /dev/ttyACM0 --arduino-cli /usr/bin/arduino-cli
    python -m wifi_radar.selftest --force      # the board is silent because its sketch is

Needs `arduino-cli` with the `arduino:avr` core (the Arduino IDE bundles a
working copy), and the port free — the IDE's serial monitor holds it open. The
board is flashed twice and reset repeatedly, so anything else on that cable will
see it.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from wifi_radar import doctor, serial_node

#: The board the sketch in `arduino/` is written for.
DEFAULT_FQBN = "arduino:avr:mega"

#: Long enough for a first compile *and* a Mega's 250 KB flash at 115200; a
#: wedged adapter gives up rather than hanging the self-test forever.
UPLOAD_TIMEOUT = 300.0

#: The doctor spends a bootloader probe plus a listen window; two runs of that.
DOCTOR_TIMEOUT = 90.0

#: What a board with working firmware and no loop looks like: a sketch that
#: compiles, runs, and says nothing at all.
SILENT_SKETCH = """\
// Written by wifi_radar.selftest: a board that is alive and says nothing.
void setup() {}
void loop() {}
"""


def find_arduino_cli(explicit=None, env=None, which=shutil.which, candidates=None):
    """
    The arduino-cli to drive, from a flag, the environment, PATH, or the IDE.

    The IDE ships its own copy buried in application resources, which is why the
    last resort is a list of those paths rather than a note to go and install
    something the user probably already has.
    """
    if explicit:
        return explicit
    environment = os.environ if env is None else env
    if environment.get("ARDUINO_CLI"):
        return environment["ARDUINO_CLI"]
    found = which("arduino-cli")
    if found:
        return found
    if candidates is None:
        local = Path(environment.get("LOCALAPPDATA", "")) if environment.get("LOCALAPPDATA") else None
        candidates = [
            Path("/Applications/Arduino IDE.app/Contents/Resources/app/lib/"
                 "backend/resources/arduino-cli"),
            Path.home() / ".local/share/arduino-ide/resources/app/lib/backend/"
                          "resources/arduino-cli",
            Path("/opt/arduino-ide/resources/app/lib/backend/resources/arduino-cli"),
        ]
        if local is not None:
            candidates.insert(0, local / "Programs" / "Arduino IDE" / "resources"
                              / "app" / "lib" / "backend" / "resources"
                              / "arduino-cli.exe")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "arduino-cli not found. Install it, set ARDUINO_CLI, pass --arduino-cli, "
        "or use the copy the Arduino IDE bundles; without it there is no way to "
        "flash the test sketch.")


def write_silent_sketch(root):
    """
    A sketch directory with the silent sketch in it, named the way Arduino expects.

    The folder and the .ino have to share a name, which is why this is a function
    rather than a string written at the top level.
    """
    sketch = Path(root) / "silent"
    sketch.mkdir(parents=True, exist_ok=True)
    (sketch / "silent.ino").write_text(SILENT_SKETCH, encoding="utf-8")
    return sketch


def _completed(command, code, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, code, stdout, stderr)


def _tail(result, limit=400):
    """
    The end of the tool's output, cut on a line boundary.

    arduino-cli's upload failure repeats one, two-word status per line, so a
    blind character slice starts mid-word ("eout Error: unable to write…") and
    buries the one line that says what went wrong. Keep whole trailing lines.
    """
    text = (result.stderr or result.stdout or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    kept = []
    while lines and (not kept or len(" ".join([lines[-1]] + kept)) <= limit):
        kept.insert(0, lines.pop())
    return " ".join(kept)[:limit]


def _kill_tree(process):
    """Kill a tool and everything it started, so nothing keeps the port open."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                       capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.kill()
    except OSError:
        pass


def run_bounded(command, timeout=UPLOAD_TIMEOUT):
    """
    Run a tool, and on timeout kill the whole tree it started.

    `subprocess.run(timeout=)` kills only its direct child. arduino-cli uploads
    through avrdude, which is that child's child: on a timeout avrdude survives,
    still holding the serial port, and every later upload dies with "Access is
    denied" — which looks exactly like a dead USB adapter and sent this bench
    chasing the cable. So the child gets its own process group (a new session on
    POSIX) and the timeout takes the group with it.
    """
    kwargs = {"start_new_session": True} if os.name != "nt" else {}
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True,
                               encoding="utf-8", errors="replace", **kwargs)
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        out, err = process.communicate()
        note = f"timed out after {timeout:.0f}s; the tool and its children were killed"
        return _completed(command, 1, out or "", f"{err or ''}\n{note}".strip())
    return _completed(command, process.returncode, out, err)


def flash(cli, port, sketch, fqbn=DEFAULT_FQBN, run=None,
          timeout=UPLOAD_TIMEOUT):
    """
    Compile and upload a sketch, turning a missing/frozen tool into a failure.

    `arduino-cli upload` does not compile — it looks for a cached build and fails
    with "Compiled sketch not found" when there is none, which is every fresh
    checkout and every sketch this module writes. `compile --upload` is the one
    command that does both.
    """
    command = [cli, "compile", "--upload", "-p", port, "--fqbn", fqbn,
               str(sketch)]
    runner = run or run_bounded
    try:
        return runner(command, capture_output=True, text=True, encoding="utf-8",
                      errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return _completed(command, 1, "", str(error))


def ask_doctor(port, python=sys.executable, run=subprocess.run,
               timeout=DOCTOR_TIMEOUT):
    """
    (exit status, output) from the real command, run the way a user runs it.

    The scan is off: the silent sketch is silent at every rate, and waiting out
    seven of them to learn nothing is time the self-test does not need to spend.
    """
    command = [python, "-m", "wifi_radar.doctor", "--port", port,
               "--scan-seconds", "0"]
    try:
        result = run(command, capture_output=True, text=True, encoding="utf-8",
                     errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return 2, f"could not run the doctor: {error}"
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def run_selftest(port, cli, silent, firmware, fqbn=DEFAULT_FQBN,
                 run=None, say=print, force=False):
    """
    Flash, check, restore, check — and report every assertion either way.

    It refuses to start on a board that is not already printing: overwriting a
    board that is broken, and then promising to restore firmware that was never
    running, would turn a diagnosis into a second fault. `force` is for the case
    where that reasoning is backwards — the board is silent *because* the sketch
    is missing, which is the very thing the restore fixes. Then the preflight is
    recorded under its own name — a board being overwritten on purpose — rather
    than as a board that prints. The restore is in a `finally`, so whichever later
    assertion fails the board is left running the firmware it had. The checks are
    returned as a list rather than raised, so a failing run can print all of them
    at once.
    """
    checks = []
    # Uploads go through run_bounded unless a runner is supplied (the tests' fake),
    # because that is the one that takes a timed-out avrdude with it. The doctor
    # subprocess spawns nothing, so it is run plainly.
    runner = run or subprocess.run

    def check(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        mark = "ok  " if ok else "FAIL"
        say(f"{mark} {name}" + (f"  [{detail}]" if detail and not ok else ""))
        return ok

    status, text = ask_doctor(port, run=runner)
    healthy = status == 0 and re.search(r"\d+ samples", text) is not None
    if healthy:
        check("the board prints before the test starts", True)
    elif force and status == 1:
        check("--force: the board is silent, so it is being overwritten on "
              "purpose", True, f"exit {status}")
    else:
        # Exit 2 is "nothing could be opened", which no upload can fix — forcing
        # past it would spend a timeout and leave the same board behind.
        reason = (f"exit {status}; the self-test needs a working board to take "
                  f"away and give back")
        if status == 1:
            reason += " (--force overrides this on a board that is merely " \
                      "missing its sketch)"
        else:
            reason += " — and --force cannot help with a port that will not open"
        check("the board prints before the test starts", False, reason)
        return checks

    try:
        uploaded = flash(cli, port, silent, fqbn=fqbn, run=run)
        if not check("the silent sketch uploads", uploaded.returncode == 0,
                     _tail(uploaded)):
            return checks

        status, text = ask_doctor(port, run=runner)
        check("the doctor exits 1 for a silent sketch", status == 1,
              f"exit {status}")
        check("the doctor says it is not printing", "not printing" in text,
              text.strip()[-200:])
        check("the doctor names the sketch to upload", "mega_channel_node" in text,
              text.strip()[-200:])
        check("the doctor still proves the chip is alive",
              "bootloader : answered" in text, text.strip()[-200:])
    finally:
        restored = flash(cli, port, firmware, fqbn=fqbn, run=run)
        if check("mega_channel_node is restored", restored.returncode == 0,
                 _tail(restored)):
            status, text = ask_doctor(port, run=runner)
            check("the doctor exits 0 again", status == 0, f"exit {status}")
            found = re.search(r"(\d+) samples", text)
            check("samples are arriving again",
                  bool(found) and int(found.group(1)) > 0,
                  text.strip()[-200:])
    return checks


def summary(checks):
    """
    The closing lines: what failed, and what state the board was left in.

    The board-state line has to follow the checks rather than be assumed: a run
    that refused to start never touched the board, and a restore that failed
    leaves it silent. Saying "mega_channel_node is back" either way would be the
    same kind of lie this self-test exists to catch.
    """
    failed = [check for check in checks if not check["ok"]]
    lines = [f"{len(checks) - len(failed)}/{len(checks)} checks passed"]
    if not failed:
        return lines
    lines.append("failed: " + ", ".join(check["name"] for check in failed))
    restored = next((check for check in checks
                     if check["name"] == "mega_channel_node is restored"), None)
    if restored is None:
        lines.append("nothing was flashed: the board was not touched")
    elif restored["ok"]:
        lines.append("the board was left with mega_channel_node, whatever "
                     "happened above")
    else:
        # Whether the warning can mention the silent sketch depends on whether
        # it ever got there: an upload that failed at compile leaves the old
        # firmware alone, and saying otherwise sends the user looking for it.
        silent_ok = any(check["name"] == "the silent sketch uploads" and check["ok"]
                        for check in checks)
        lines.append(
            "WARNING: the restore failed and the silent sketch did get on — the "
            "board may still be silent; upload arduino/mega_channel_node to it "
            "by hand" if silent_ok else
            "WARNING: the restore failed — upload arduino/mega_channel_node to "
            "the board by hand before using it")
    return lines


def as_json(checks, port=None, cli=None, firmware=None):
    """
    The run as one object, for a job that records results rather than reads prose.

    Every check is kept with its detail, not just the failures: a pass that says
    which assertion passed and a failure that keeps the tool's own output are
    the two things needed to tell "it works" from "it ran nothing".
    """
    failed = [check for check in checks if not check["ok"]]
    return {
        "port": port,
        "cli": cli,
        "firmware": str(firmware) if firmware else None,
        "checks": checks,
        "passed": len(checks) - len(failed),
        "total": len(checks),
        "failed": [check["name"] for check in failed],
        "exit_status": 1 if failed else 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Flash a silent sketch, prove the doctor diagnoses it, then "
                    "put mega_channel_node back and prove samples return.")
    parser.add_argument("--port", default="auto",
                        help="COM3 on Windows, /dev/ttyACM0 on Linux; auto (the "
                             "default) picks the likely board")
    parser.add_argument("--arduino-cli", default=None,
                        help="path to arduino-cli; defaults to ARDUINO_CLI, PATH, "
                             "or the copy the Arduino IDE bundles")
    parser.add_argument("--fqbn", default=DEFAULT_FQBN,
                        help="board definition (default %(default)s)")
    parser.add_argument("--json", action="store_true",
                        help="print one JSON object with every check instead of "
                             "the running commentary, so a scheduled job can "
                             "record the result without parsing lines")
    parser.add_argument("--force", action="store_true",
                        help="run even when the board is not printing yet — for "
                             "a board whose sketch is missing, which is what the "
                             "restore puts back; the preflight is then reported "
                             "under its own name, not as a board that prints")
    args = parser.parse_args(argv)

    try:
        ports = serial_node.available_ports()
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    port = args.port
    if port.strip().lower() == "auto":
        chosen = serial_node.choose_port(ports)
        if chosen is None:
            print(doctor.NO_PORT, file=sys.stderr)
            return 2
        port = chosen.device

    try:
        cli = find_arduino_cli(args.arduino_cli)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    firmware = Path(__file__).with_name("arduino") / "mega_channel_node"
    if not args.json:
        print(f"port   : {port}")
        print(f"cli    : {cli}")
        print(f"firmware: {firmware}")
        print("-" * 70, flush=True)

    with tempfile.TemporaryDirectory(prefix="wifi_radar_selftest_") as root:
        silent = write_silent_sketch(root)
        checks = run_selftest(port, cli, silent, firmware, fqbn=args.fqbn,
                              force=args.force,
                              say=(lambda *a, **k: None) if args.json else print)

    failed = [check for check in checks if not check["ok"]]
    if args.json:
        print(json.dumps(as_json(checks, port, cli, firmware), indent=2))
        return 1 if failed else 0
    print("-" * 70)
    for line in summary(checks):
        print(line)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
