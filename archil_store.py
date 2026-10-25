#!/usr/bin/env python3
"""
Files that outlive the run: a disk on Archil instead of this laptop's disk.

Everything this repo writes today is tied to the machine that wrote it. `runs/`
holds a checkpoint, a loss curve and a sample, and a run continued somewhere
else starts from nothing. Archil gives a directory that outlives the process
that filled it, so a file written by one command is still there for the next
one — from a different container, on a different machine.

The SDK is `archil` (pinned in requirements.txt). This module is the one place
it is configured and the one place a disk is reached for, so nothing else in the
repo touches a key or a client. It is deliberately small, and it imports without
a key: no environment is read until a disk is actually asked for.

Credentials come from the environment, or from `.env` at the repo root
(gitignored — `.env.example` is the template):

    ARCHIL_API_KEY   an API key starting `key-`, from
                     https://console.archil.com/api-keys. Not a disk or mount
                     token (`adt_…`): those are for mounting, and the API
                     answers 401 to one. Keys are region-scoped, so a key made
                     in aws-us-east-1 fails in aws-us-west-2.
    ARCHIL_REGION    defaults to aws-us-east-1. Commands (`run`) exist only in
                     aws-us-east-1, aws-us-west-2 and aws-eu-west-1.
    ARCHIL_DISK_ID   `dsk-…` — the disk these files live on.

Usage:

    import archil_store

    archil_store.write_file("runs/metrics.json", open("runs/metrics.json", "rb").read())
    archil_store.read_file("runs/metrics.json")
    result = archil_store.run("ls -l /mnt/data")
    print(result.stdout, result.exit_code)

A command given to `run()` executes on the disk: the filesystem is mounted at
/mnt/data, which is also the working directory, and every call is a fresh
container — the files survive, the process state and any installs do not. The
mount is shared, so a command that writes should work inside a fresh
subdirectory of its own (creating it claims it) rather than in the shared root,
or run `archil checkout <path>` first, otherwise the write meets a read-only
file system.

The key stays server-side: it is read from this process's environment and never
reaches a browser.

    .venv/bin/python archil_check.py    # proves a write survives to a second exec
"""

import os

import archil

ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(ROOT, ".env")
DEFAULT_REGION = "aws-us-east-1"

# One disk for the repo, found again by name when ARCHIL_DISK_ID is not set yet.
DISK_NAME = "new-idea2-persistence"

_configured = False


class NotConfigured(RuntimeError):
    """Archil is not set up yet: something is missing from the environment."""


def load_env_file(path=ENV_FILE):
    """Read `KEY=VALUE` lines from `.env` into os.environ, without overriding.

    The rest of the repo reads the environment directly; this is the only place
    that also reads the file, so `ARCHIL_*` can be set once for every command.
    A value already in the environment wins, so a one-off
    `ARCHIL_DISK_ID=… python archil_check.py` still behaves as asked. Returns
    the names that were set — never the values.
    """
    if not os.path.exists(path):
        return []
    loaded = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip()
            if name.startswith("export "):
                name = name[len("export "):].strip()
            value = value.strip().strip('"').strip("'")
            if name and value and not os.environ.get(name):
                os.environ[name] = value
                loaded.append(name)
    return loaded


def api_key():
    """The API key, or a NotConfigured that says what to do about it."""
    key = os.environ.get("ARCHIL_API_KEY", "").strip()
    if not key:
        raise NotConfigured(
            "ARCHIL_API_KEY is not set. Put an Archil API key in .env — it "
            "starts with `key-` and comes from "
            "https://console.archil.com/api-keys. A disk or mount token "
            "(`adt_…`) is a different credential and gets HTTP 401 here."
        )
    return key


def region():
    """The region the key belongs to; aws-us-east-1 unless told otherwise."""
    return os.environ.get("ARCHIL_REGION", "").strip() or DEFAULT_REGION


def configure():
    """Point the SDK at the key and region in the environment, once."""
    load_env_file()
    global _configured
    if _configured:
        return
    archil.configure(api_key=api_key(), region=region())
    _configured = True


def _env_disk_id():
    load_env_file()
    value = os.environ.get("ARCHIL_DISK_ID", "").strip()
    if not value:
        raise NotConfigured(
            "ARCHIL_DISK_ID is not set. Run `archil_check.py` once — it creates "
            "a disk when this is missing — and put the id it prints in .env."
        )
    return value


def disk(disk_id=None):
    """The disk these files live on: ARCHIL_DISK_ID unless one is named."""
    configure()
    return archil.get_disk(disk_id or _env_disk_id())


def find_disk(name=DISK_NAME):
    """The first disk called `name`, or None — so a check reuses its own disk."""
    configure()
    for candidate in archil.list_disks(name=name):
        if getattr(candidate, "name", None) == name:
            return candidate
    return None


def create_disk(name=DISK_NAME):
    """Create a disk and return it.

    The result also carries a one-time mount token. It is deliberately dropped
    rather than returned or printed: a command run through `run()` does not
    need it, and a token in a log is a credential in a log.
    """
    configure()
    result = archil.create_disk(name=name)
    return result.disk


def run(command, disk_id=None):
    """Run a shell command on the disk.

    Returns the SDK's result: `exit_code`, `stdout`, `stderr` and `timing`. The
    timeout is five minutes and each stream is capped at 128 KiB.
    """
    return disk(disk_id).exec(command)


def write_file(path, body, disk_id=None):
    """Write an object at `path` on the disk; `body` may be str or bytes."""
    if isinstance(body, str):
        body = body.encode("utf-8")
    return disk(disk_id).put_object(path, body)


def read_file(path, disk_id=None):
    """Read the object at `path` back, as text."""
    return disk(disk_id).get_object(path).decode("utf-8")
