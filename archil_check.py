#!/usr/bin/env python3
"""
Does the disk outlive the command? Write a line in one exec, read it in another.

Two separate commands is the whole point: each one runs in a fresh container, so
text that comes back from the second one was not in memory, it was on the disk.
The check exits non-zero, with the reason, on a missing key, a failed command, or
a read-back that does not contain the line the first command wrote.

    .venv/bin/python archil_check.py

It reads ARCHIL_API_KEY, ARCHIL_REGION and ARCHIL_DISK_ID from the environment,
or from `.env` at the repo root (see `.env.example`). With no disk id it finds
the repo's disk by name and creates it if that is the first run — the one-time
mount token that comes back with a new disk is never printed, because a command
does not need it and a token in a log is a credential in a log. The id it used
is printed, to be put in `.env` as ARCHIL_DISK_ID.

Each run writes its own `checks/<utc>-<random>/note.txt` on the disk. A fresh
subdirectory per run is not decoration: the mount is shared between commands,
and creating the directory is what claims it for writing.
"""

import os
import secrets
import shlex
import sys
import time

import archil

import archil_store


def main():
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{stamp}-{secrets.token_hex(3)}"
    directory = f"checks/{run_id}"
    note = f"{directory}/note.txt"
    written = f"archil_check.py wrote this at {stamp} in exec one ({run_id})"

    try:
        archil_store.load_env_file()
        wanted = os.environ.get("ARCHIL_DISK_ID", "").strip()
        if wanted:
            disk = archil_store.disk(wanted)
            how = "from ARCHIL_DISK_ID"
        else:
            disk = archil_store.find_disk()
            if disk is not None:
                how = "found by name"
            else:
                disk = archil_store.create_disk()
                how = "created"

        # Exec one: make the run's own directory, write the line, show it. The
        # `cat` here proves the write command itself worked; the proof of
        # persistence is the second exec below.
        first = archil_store.run(
            f"mkdir -p {shlex.quote(directory)} && "
            f"printf '%s\\n' {shlex.quote(written)} > {shlex.quote(note)} && "
            f"cat {shlex.quote(note)}",
            disk_id=disk.id,
        )
        if first.exit_code != 0:
            print(
                f"archil check: the write failed (exit {first.exit_code}): "
                f"{first.stderr.strip() or 'no stderr'}",
                file=sys.stderr,
            )
            return 1
        if written not in first.stdout:
            print(
                f"archil check: exec one wrote {note} but read back "
                f"{first.stdout.strip()!r}",
                file=sys.stderr,
            )
            return 1

        # Exec two, a separate container: nothing is in memory this time.
        second = archil_store.run(f"cat {shlex.quote(note)}", disk_id=disk.id)
        if second.exit_code != 0:
            print(
                f"archil check: the read-back failed (exit {second.exit_code}): "
                f"{second.stderr.strip() or 'no stderr'}",
                file=sys.stderr,
            )
            return 1
        if written not in second.stdout:
            print(
                f"archil check: {note} came back as {second.stdout.strip()!r}, "
                f"which does not contain what exec one wrote",
                file=sys.stderr,
            )
            return 1
    except archil_store.NotConfigured as exc:
        print(f"archil check: {exc}", file=sys.stderr)
        return 2
    except archil.ArchilError as exc:
        hint = ""
        if "401" in str(exc) or "unauthor" in str(exc).lower():
            hint = (
                "\n  A 401 usually means the key is a disk/mount token (adt_…) "
                "rather than an API key, or it was made in another region than "
                f"{archil_store.region()}. Get a `key-` key for an AWS region "
                "from https://console.archil.com/api-keys."
            )
        print(f"archil check: the Archil API refused: {exc}{hint}", file=sys.stderr)
        return 1

    print(f"disk     {disk.name} ({disk.id}), {disk.region} [{how}]")
    print(f"wrote    {note} in exec one")
    print(f"read     {second.stdout.strip()!r} in exec two")
    print("ok       the second command read what the first one wrote")
    if not os.environ.get("ARCHIL_DISK_ID", "").strip():
        print(f"\nput this in .env:  ARCHIL_DISK_ID={disk.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
