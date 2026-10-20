#!/usr/bin/env python3
"""
A board on a pseudo-terminal, so CI can run the doctor against a real port.

The doctor's contract is small and worth enforcing where it cannot rot: when
`S,` lines arrive, it must exit 0, say `printing`, and count the samples. The
unit tests pin every piece of that, but none of them runs the actual command
against the actual port-opening code — which is the part that has already broken
once, on a wedged adapter.

So this starts a writer on one end of a pty, hands the other end to
`python -m wifi_radar.doctor --json`, and fails when the verdict is not
`printing`. It is the same assertion the hardware self-test makes, without the
board:

    python -m wifi_radar.fakeboard
    python -m wifi_radar.fakeboard --doctor-seconds 1.5 --verbose

Linux and macOS only. A pty is how a serial port is simulated, and Windows has
no equivalent that pyserial can open — the hardware self-test exists for that
machine. The check is split out as `check()` so the contract can be tested
wherever the tests run, whether or not a pty is available.
"""

import argparse
import json
import os
import subprocess
import sys
import time

#: What the included sketch prints, so the lines the doctor reads are the lines
#: the firmware writes rather than a convenient invention.
BANNER = b"# mega_channel_node 8 channels 20 Hz\r\n"

#: A pty ignores this, but pyserial will not open a port without one.
FAKE_BAUD = 115200


def check(document, status):
    """
    What is wrong with a doctor result, as a list of sentences.

    Emptiness is success: the doctor said what it must say. Everything is
    phrased as the failure rather than the assertion so a red CI log reads as
    the problem, not as a restatement of the test.
    """
    problems = []
    if status != 0:
        problems.append(f"the doctor exited {status}, not 0")
    verdict = document.get("verdict")
    if verdict != "printing":
        problems.append(f"verdict was {verdict!r}, not 'printing'")
    sketch = document.get("sketch") or {}
    samples = sketch.get("samples") or 0
    if not samples:
        problems.append(f"no samples were counted (sketch.samples={samples})")
    if sketch.get("baud") not in (None, FAKE_BAUD):
        problems.append(f"sketch.baud was {sketch.get('baud')}, not {FAKE_BAUD}")
    return problems


def serve(rate=20.0, sleep=time.sleep, silent=False):
    """
    (device path, stop) — a pty printing `S,` lines at `rate`.

    The writer owns the master side and the doctor opens the slave; writes that
    fail because the reader has gone are expected and end the thread rather than
    raising into whatever is shutting down.
    """
    import pty
    import threading

    master, slave = pty.openpty()
    path = os.ttyname(slave)
    stop = threading.Event()

    def write():
        index = 0
        if silent:
            return
        try:
            os.write(master, BANNER)
            while not stop.is_set():
                values = ",".join(str(500 + (index + k) % 7) for k in range(8))
                os.write(master, f"S,{index * 50},{values}\r\n".encode("ascii"))
                index += 1
                sleep(1.0 / rate)
        except OSError:
            # The doctor closed its end: nothing left to say to anyone.
            pass

    thread = threading.Thread(target=write, daemon=True)
    thread.start()

    def close():
        stop.set()
        thread.join(timeout=2.0)
        for handle in (master, slave):
            try:
                os.close(handle)
            except OSError:
                pass

    return path, close


def run_doctor(device, seconds=2.0, timeout=30.0):
    """(parsed JSON, exit status, raw output) from the doctor against `device`."""
    command = [sys.executable, "-m", "wifi_radar.doctor", "--port", device,
               "--json", "--scan-seconds", "0",
               "--sync-seconds", "0.2", "--seconds", str(seconds)]
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
    output = (result.stdout or "") + (result.stderr or "")
    try:
        document = json.loads(result.stdout)
    except ValueError:
        document = {}
    return document, result.returncode, output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the doctor against a simulated serial port and fail "
                    "unless it says the board is printing.")
    parser.add_argument("--doctor-seconds", type=float, default=2.0,
                        help="how long the doctor listens (default %(default)s)")
    parser.add_argument("--rate", type=float, default=20.0,
                        help="how fast the simulated board prints")
    parser.add_argument("--verbose", action="store_true",
                        help="print the doctor's JSON even when it passes")
    parser.add_argument("--silent", action="store_true",
                        help="print nothing at all, to prove the check fails "
                             "on a board that is not printing")
    args = parser.parse_args(argv)

    if os.name == "nt":
        print("error: a simulated port needs a pty, which Windows does not "
              "have. Run the hardware self-test there: python -m "
              "wifi_radar.selftest", file=sys.stderr)
        return 2

    device, stop = serve(rate=args.rate, silent=args.silent)
    try:
        print(f"simulated board on {device}", flush=True)
        document, status, output = run_doctor(device, seconds=args.doctor_seconds)
    finally:
        stop()

    problems = check(document, status)
    if args.verbose or problems:
        print(json.dumps(document, indent=2) if document else output.strip())
    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        print(output.strip()[-800:], file=sys.stderr)
        return 1
    print(f"ok   the doctor said {document['verdict']} with "
          f"{document['sketch']['samples']} samples at "
          f"{document['sketch'].get('baud')} baud", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
