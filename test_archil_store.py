"""The Archil store, without an Archil account, a network, or a key.

Three things are worth pinning about `archil_store`, and none of them needs the
service. The `.env` reader: it parses the file the way `.env.example` is
written, and a real environment variable wins over it, so a one-off override on
the command line still means what it says. The refusals: with no key, the
module names the credential to go and get instead of raising whatever the SDK
raises, and it does that *before* touching the SDK at all. And the wiring: the
key and region reach `archil.configure` once, a disk id reaches `get_disk`, and
a `str` becomes bytes on the way to `put_object`.

The SDK is replaced by a fake for all of it — a test that could reach Archil is
a test that fails on a train. What is real is the module under test; only the
network-shaped edge of it is a stand-in.

The repo's own `.env` is never read: every test that goes through the default
path points it at a file in a temporary directory, so a developer's real key
cannot quietly turn a "missing key" test green (and cannot be echoed by a
failure message either).
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

try:
    import archil  # noqa: F401  (the SDK is pinned in requirements-archil.txt)
    import archil_store
except ModuleNotFoundError as missing:  # pragma: no cover - CI installs it
    archil_store = None
    MISSING = str(missing)
else:
    MISSING = None


def setUpModule():
    """Skip the module cleanly when the SDK is absent, rather than at import.

    A `python -m unittest test_archil_store` on a checkout that has not
    installed requirements-archil.txt should say "skipped", not print a
    traceback from the loader. CI installs the wheel, so there it runs.
    """
    if archil_store is None:  # pragma: no cover - covered by the local run
        raise unittest.SkipTest(
            f"the Archil SDK is not installed (requirements-archil.txt): {MISSING}"
        )


ARCHIL_NAMES = ("ARCHIL_API_KEY", "ARCHIL_REGION", "ARCHIL_DISK_ID")


class Artifact:
    """Enough of an SDK object: attributes, and nothing that acts."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class FakeDisk:
    """A disk that remembers instead of storing."""

    def __init__(self, sdk, disk_id):
        self.sdk = sdk
        self.id = disk_id
        self.objects = {}
        self.commands = []

    def exec(self, command):
        self.commands.append(command)
        return Artifact(exit_code=0, stdout=f"ran: {command}", stderr="",
                        timing=Artifact(total_ms=1))

    def put_object(self, path, body):
        self.objects[path] = body
        return Artifact(path=path)

    def get_object(self, path):
        return self.objects[path]


class FakeSDK:
    """The SDK's surface, recorded. Nothing here opens a socket."""

    def __init__(self):
        self.configured = []
        self.disks = {}
        self.listed = []
        self.created_names = []

    def configure(self, **fields):
        self.configured.append(fields)

    def get_disk(self, disk_id):
        if disk_id not in self.disks:
            self.disks[disk_id] = FakeDisk(self, disk_id)
        return self.disks[disk_id]

    def list_disks(self, name=None):
        return [disk for disk in self.listed if name is None or disk.name == name]

    def create_disk(self, name=None):
        self.created_names.append(name)
        disk = Artifact(id=f"dsk-{len(self.created_names)}", name=name)
        # The real call also returns a one-time mount token; create_disk() is
        # supposed to hand back only the disk, and a test below checks that.
        return Artifact(disk=disk, mount_token="adt_must-not-escape")


class StoreCase(unittest.TestCase):
    """No Archil variables, no `.env`, a fake SDK, and a clean module state."""

    def setUp(self):
        self.saved = {name: os.environ.pop(name, None) for name in ARCHIL_NAMES}
        self.addCleanup(self.restore_environment)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # The default path, when nothing else is asked for.
        self.env_file = os.path.join(self.tmp.name, "absent.env")

        inner = archil_store.load_env_file
        self.load_patch = mock.patch.object(
            archil_store, "load_env_file",
            lambda path=None: inner(self.env_file if path is None else path),
        )
        self.load_patch.start()
        self.addCleanup(self.load_patch.stop)

        self.fake = FakeSDK()
        self.sdk_patch = mock.patch.object(archil_store, "archil", self.fake)
        self.sdk_patch.start()
        self.addCleanup(self.sdk_patch.stop)

        # configure() is once per process by design, which would make the
        # second test see the first test's key.
        archil_store._configured = False

    def restore_environment(self):
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def write_env(self, text):
        with open(self.env_file, "w", encoding="utf-8") as handle:
            handle.write(text)
        return self.env_file

    def temp_env(self, text):
        handle, path = tempfile.mkstemp(suffix=".env", dir=self.tmp.name)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        self.addCleanup(os.unlink, path)
        return path


class EnvFileTest(StoreCase):
    def test_parses_the_file_the_way_the_template_is_written(self):
        path = self.temp_env(
            "# a comment\n"
            "\n"
            'ARCHIL_API_KEY="key-abc"\n'
            "export ARCHIL_REGION=aws-eu-west-1\n"
            "ARCHIL_DISK_ID='dsk-123'\n"
            "NOT_A_SETTING\n"
            "ARCHIL_EMPTY=\n"
        )
        self.assertEqual(
            archil_store.load_env_file(path),
            ["ARCHIL_API_KEY", "ARCHIL_REGION", "ARCHIL_DISK_ID"],
        )
        self.assertEqual(os.environ["ARCHIL_API_KEY"], "key-abc")
        self.assertEqual(os.environ["ARCHIL_DISK_ID"], "dsk-123")
        # An `export` prefix is the shell's, not part of the name.
        self.assertEqual(archil_store.region(), "aws-eu-west-1")
        # A line with no `=` and a name with no value set nothing at all.
        self.assertNotIn("NOT_A_SETTING", os.environ)
        self.assertNotIn("ARCHIL_EMPTY", os.environ)

    def test_a_real_variable_wins_over_the_file(self):
        path = self.temp_env("ARCHIL_API_KEY=key-from-file\nARCHIL_REGION=aws-us-west-2\n")
        os.environ["ARCHIL_API_KEY"] = "key-from-the-environment"
        loaded = archil_store.load_env_file(path)
        self.assertEqual(os.environ["ARCHIL_API_KEY"], "key-from-the-environment")
        # Only the name the file actually supplied is reported.
        self.assertEqual(loaded, ["ARCHIL_REGION"])

    def test_a_missing_file_is_not_an_error(self):
        self.assertEqual(
            archil_store.load_env_file(os.path.join(self.tmp.name, "nope.env")), []
        )

    def test_names_come_back_not_values(self):
        path = self.temp_env("ARCHIL_API_KEY=key-do-not-log-me\n")
        loaded = archil_store.load_env_file(path)
        self.assertEqual(loaded, ["ARCHIL_API_KEY"])
        self.assertNotIn("key-do-not-log-me", repr(loaded))


class NoKeyTest(StoreCase):
    def test_the_key_is_required_and_the_error_says_where_to_get_one(self):
        with self.assertRaises(archil_store.NotConfigured) as caught:
            archil_store.api_key()
        message = str(caught.exception)
        self.assertIn("ARCHIL_API_KEY", message)
        self.assertIn("console.archil.com/api-keys", message)
        # The mistake people actually make: a mount token looks like a key.
        self.assertIn("adt_", message)

    def test_whitespace_is_not_a_key(self):
        os.environ["ARCHIL_API_KEY"] = "   \n"
        with self.assertRaises(archil_store.NotConfigured):
            archil_store.api_key()

    def test_a_key_is_returned_without_its_surrounding_space(self):
        os.environ["ARCHIL_API_KEY"] = "  key-abc\n"
        self.assertEqual(archil_store.api_key(), "key-abc")

    def test_no_key_refuses_before_the_sdk_is_touched(self):
        with self.assertRaises(archil_store.NotConfigured):
            archil_store.disk()
        self.assertEqual(self.fake.configured, [])
        self.assertEqual(self.fake.disks, {})
        self.assertFalse(archil_store._configured)

    def test_a_key_without_a_disk_id_points_at_the_check(self):
        os.environ["ARCHIL_API_KEY"] = "key-abc"
        with self.assertRaises(archil_store.NotConfigured) as caught:
            archil_store.disk()
        self.assertIn("archil_check.py", str(caught.exception))
        # The key is fine, so configuring it was right; asking for a disk by an
        # id that does not exist yet is what must not happen.
        self.assertEqual(len(self.fake.configured), 1)
        self.assertEqual(self.fake.disks, {})

    def test_an_explicit_disk_id_still_needs_a_key(self):
        with self.assertRaises(archil_store.NotConfigured):
            archil_store.disk("dsk-explicit")
        self.assertEqual(self.fake.disks, {})


class RegionTest(StoreCase):
    def test_the_default_is_aws_us_east_1(self):
        self.assertEqual(archil_store.region(), "aws-us-east-1")
        self.assertEqual(archil_store.DEFAULT_REGION, "aws-us-east-1")

    def test_the_environment_wins(self):
        os.environ["ARCHIL_REGION"] = " aws-us-west-2 "
        self.assertEqual(archil_store.region(), "aws-us-west-2")

    def test_a_blank_region_falls_back_rather_than_breaking(self):
        os.environ["ARCHIL_REGION"] = "  "
        self.assertEqual(archil_store.region(), archil_store.DEFAULT_REGION)


class WiringTest(StoreCase):
    def test_the_key_and_region_reach_the_sdk(self):
        os.environ["ARCHIL_API_KEY"] = "key-abc"
        archil_store.configure()
        self.assertEqual(
            self.fake.configured, [{"api_key": "key-abc", "region": "aws-us-east-1"}]
        )

    def test_configure_happens_once_per_process(self):
        os.environ["ARCHIL_API_KEY"] = "key-abc"
        archil_store.configure()
        archil_store.configure()
        self.assertEqual(len(self.fake.configured), 1)
        self.assertTrue(archil_store._configured)

    def test_configure_reads_the_dotenv(self):
        self.write_env("ARCHIL_API_KEY=key-from-file\nARCHIL_REGION=aws-eu-west-1\n")
        archil_store.configure()
        self.assertEqual(
            self.fake.configured, [{"api_key": "key-from-file", "region": "aws-eu-west-1"}]
        )

    def test_the_disk_id_reaches_get_disk(self):
        os.environ["ARCHIL_API_KEY"] = "key-abc"
        os.environ["ARCHIL_DISK_ID"] = "dsk-from-the-environment"
        disk = archil_store.disk()
        self.assertEqual(disk.id, "dsk-from-the-environment")
        self.assertEqual(list(self.fake.disks), ["dsk-from-the-environment"])

    def test_an_explicit_id_wins_over_the_environment(self):
        os.environ["ARCHIL_API_KEY"] = "key-abc"
        os.environ["ARCHIL_DISK_ID"] = "dsk-from-the-environment"
        self.assertEqual(archil_store.disk("dsk-named").id, "dsk-named")

    def test_a_string_is_written_as_bytes(self):
        os.environ["ARCHIL_API_KEY"] = "key-abc"
        os.environ["ARCHIL_DISK_ID"] = "dsk-1"
        archil_store.write_file("checks/run.json", '{"ok": true}')
        self.assertEqual(self.fake.disks["dsk-1"].objects["checks/run.json"],
                         b'{"ok": true}')

    def test_bytes_are_written_unchanged(self):
        os.environ["ARCHIL_API_KEY"] = "key-abc"
        os.environ["ARCHIL_DISK_ID"] = "dsk-1"
        archil_store.write_file("curve.png", b"\x89PNG\r\n")
        self.assertEqual(self.fake.disks["dsk-1"].objects["curve.png"], b"\x89PNG\r\n")

    def test_a_file_reads_back_as_text(self):
        os.environ["ARCHIL_API_KEY"] = "key-abc"
        os.environ["ARCHIL_DISK_ID"] = "dsk-1"
        self.fake.get_disk("dsk-1").objects["note.txt"] = "hello".encode("utf-8")
        self.assertEqual(archil_store.read_file("note.txt"), "hello")

    def test_a_command_runs_on_the_disk_and_its_result_comes_back(self):
        os.environ["ARCHIL_API_KEY"] = "key-abc"
        result = archil_store.run("ls -l", disk_id="dsk-2")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("ls -l", result.stdout)
        self.assertEqual(self.fake.disks["dsk-2"].commands, ["ls -l"])

    def test_create_disk_hands_back_the_disk_not_the_mount_token(self):
        os.environ["ARCHIL_API_KEY"] = "key-abc"
        disk = archil_store.create_disk()
        self.assertEqual(disk.name, archil_store.DISK_NAME)
        self.assertFalse(hasattr(disk, "mount_token"))
        self.assertEqual(self.fake.created_names, [archil_store.DISK_NAME])

    def test_find_disk_matches_by_name_and_forgives_a_miss(self):
        os.environ["ARCHIL_API_KEY"] = "key-abc"
        self.assertIsNone(archil_store.find_disk())
        self.fake.listed = [
            Artifact(id="dsk-other", name="some-other-disk"),
            Artifact(id="dsk-ours", name=archil_store.DISK_NAME),
        ]
        self.assertEqual(archil_store.find_disk().id, "dsk-ours")


class ImportTest(unittest.TestCase):
    """The module's headline promise: importing it needs no key and no `.env`."""

    def test_importing_reads_no_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            shutil.copy(archil_store.__file__, directory)
            with open(os.path.join(directory, ".env"), "w", encoding="utf-8") as handle:
                handle.write("ARCHIL_API_KEY=key-that-must-not-be-read\n")
            environment = {
                name: value for name, value in os.environ.items()
                if not name.startswith("ARCHIL_")
            }
            environment["PYTHONPATH"] = directory
            done = subprocess.run(
                [sys.executable, "-c",
                 "import archil_store; "
                 "print(sorted(n for n in __import__('os').environ "
                 "if n.startswith('ARCHIL_')))"],
                cwd=directory, env=environment, capture_output=True, text=True,
                timeout=60,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(done.stdout.strip(), "[]", done.stdout)


if __name__ == "__main__":
    unittest.main()
