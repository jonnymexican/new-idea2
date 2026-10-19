"""Tests for the provenance survival lab.

Run with:  .venv/bin/python -m unittest -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import lab


class LabCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.certs = lab.make_certs(cls.root / "certs")
        cls.signer = lab.load_signer(cls.certs["chain"], cls.certs["key"])
        cls.ca = (cls.root / "certs" / "ca.cert.pem").read_text()
        cls.fixture = lab.make_fixture(cls.root / "source.jpg")
        cls.signed = cls.root / "signed.jpg"
        lab.sign_created(cls.fixture, cls.signed, cls.signer)
        with Image.open(cls.fixture) as fh:
            cls.src = fh.convert("RGB")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def reencode(self, name="re.jpg", quality=75):
        dest = self.root / name
        self.src.save(dest, format="JPEG", quality=quality, subsampling=2)
        return dest

    def same_pixels(self, a, b):
        with Image.open(a) as ia, Image.open(b) as ib:
            return ia.convert("RGB").tobytes() == ib.convert("RGB").tobytes()


class TestCredentials(LabCase):
    def test_control_is_trusted_and_claims_ai(self):
        row = lab.verify(self.signed, self.ca)
        self.assertTrue(row["present"])
        self.assertEqual(row["state"], "Trusted")
        self.assertTrue(row["signature_ok"])
        self.assertTrue(row["data_hash_ok"])
        self.assertTrue(row["trusted"])
        self.assertTrue(row["ai_claim"])
        self.assertIn("c2pa.actions.v2", row["assertions"])

    def test_untrusted_without_the_test_anchor(self):
        row = lab.verify(self.signed)
        self.assertEqual(row["state"], "Valid")
        self.assertTrue(row["data_hash_ok"])
        self.assertFalse(row["trusted"])
        self.assertIn("signingCredential.untrusted", row["failures"])

    def test_certs_are_a_chain_not_self_signed(self):
        chain = (self.root / "certs" / "chain.pem").read_text()
        self.assertIn("BEGIN CERTIFICATE", chain)
        self.assertEqual(chain.count("BEGIN CERTIFICATE"), 2)


class TestSegmentSurgery(LabCase):
    def test_roundtrip_preserves_manifest_bytes(self):
        segments = lab.c2pa_segments(self.signed.read_bytes())
        self.assertGreaterEqual(len(segments), 1)
        other = self.reencode()
        injected = lab.inject_segments(other.read_bytes(), segments)
        dest = self.root / "injected.jpg"
        dest.write_bytes(injected)
        self.assertEqual(lab.c2pa_segments(injected), segments)

    def test_injection_does_not_touch_pixels(self):
        other = self.reencode("pixels_a.jpg")
        dest = self.root / "pixels_b.jpg"
        dest.write_bytes(
            lab.inject_segments(
                other.read_bytes(), lab.c2pa_segments(self.signed.read_bytes())
            )
        )
        self.assertTrue(self.same_pixels(other, dest))

    def test_strip_removes_only_app11(self):
        data = self.signed.read_bytes()
        stripped = lab.strip_segments(data)
        self.assertEqual(lab.c2pa_segments(stripped), [])
        dest = self.root / "stripped.jpg"
        dest.write_bytes(stripped)
        self.assertTrue(self.same_pixels(self.signed, dest))

    def test_non_jpeg_is_rejected(self):
        with self.assertRaises(ValueError):
            lab.c2pa_segments(b"\x89PNG\r\n\x1a\n")


class TestArms(LabCase):
    def test_naive_reencode_drops_the_manifest(self):
        row = lab.verify(self.reencode("naive.jpg"), self.ca)
        self.assertFalse(row["present"])
        self.assertEqual(lab.verdict(row), "absent")

    def test_preserved_manifest_is_present_but_invalid(self):
        base = self.reencode("preserve_base.jpg")
        dest = self.root / "preserve.jpg"
        dest.write_bytes(
            lab.inject_segments(
                base.read_bytes(), lab.c2pa_segments(self.signed.read_bytes())
            )
        )
        row = lab.verify(dest, self.ca)
        self.assertTrue(row["present"])
        self.assertTrue(row["claim_readable"], "the claim must still parse")
        self.assertTrue(row["ai_claim"], "an invalid credential still says AI")
        self.assertFalse(row["data_hash_ok"], "the binding must be broken")
        self.assertEqual(lab.verdict(row), "invalid")

    def test_crop_preserve_breaks_the_binding(self):
        crop = lab.BY_NAME["crop_10pct"]
        img, spec = crop.fn(self.src)
        base = self.root / "crop.jpg"
        img.save(base, format="JPEG", **spec)
        dest = self.root / "crop_preserve.jpg"
        dest.write_bytes(
            lab.inject_segments(
                base.read_bytes(), lab.c2pa_segments(self.signed.read_bytes())
            )
        )
        row = lab.verify(dest, self.ca)
        self.assertTrue(row["present"])
        self.assertEqual(lab.verdict(row), "invalid")
        with Image.open(dest) as cropped:
            self.assertNotEqual(cropped.size, self.src.size)

    def test_rebind_validates_with_parent_ingredient(self):
        base = self.reencode("rebind_base.jpg")
        dest = self.root / "rebind.jpg"
        lab.sign_edited(base, dest, self.signed, self.signer)
        row = lab.verify(dest, self.ca)
        self.assertEqual(row["state"], "Trusted")
        self.assertTrue(row["data_hash_ok"])
        self.assertEqual(row["ingredients"], 1)
        self.assertEqual(lab.verdict(row), "valid")
        self.assertNotEqual(row["manifest_id"], lab.verify(self.signed, self.ca)["manifest_id"])

    def test_strip_rewrite_keeps_every_pixel(self):
        t = lab.BY_NAME["strip_rewrite"]
        img, spec = t.fn(self.src)
        dest = self.root / "rewrite.png"
        img.save(dest, format="PNG", **spec)
        self.assertTrue(self.same_pixels(self.fixture, dest))
        self.assertFalse(lab.verify(dest, self.ca)["present"])


class TestVerdict(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(lab.verdict({"present": False, "state": None}), "absent")
        self.assertEqual(lab.verdict({"present": True, "state": "Trusted"}), "valid")
        self.assertEqual(lab.verdict({"present": True, "state": "Valid"}), "valid")
        self.assertEqual(lab.verdict({"present": True, "state": "Invalid"}), "invalid")


class TestSummary(unittest.TestCase):
    def test_counts_and_flags(self):
        rows = [
            {"mode": "as_is", "verdict": "absent", "claim_readable": False,
             "ai_claim": False, "trusted": False, "verify_ms": 1.0},
            {"mode": "preserve", "verdict": "invalid", "claim_readable": True,
             "ai_claim": True, "trusted": False, "verify_ms": 3.0},
            {"mode": "preserve", "verdict": "invalid", "claim_readable": True,
             "ai_claim": False, "trusted": False, "verify_ms": 5.0},
            {"mode": "rebind", "verdict": "valid", "claim_readable": True,
             "ai_claim": True, "trusted": True, "verify_ms": 7.0},
        ]
        summary = lab.summarise(rows)
        self.assertEqual(summary["rows"], 4)
        self.assertEqual(summary["by_mode"]["as_is"]["absent"], 1)
        self.assertEqual(summary["by_mode"]["preserve"]["invalid"], 2)
        self.assertEqual(summary["by_mode"]["preserve"]["invalid_but_claim_readable"], 2)
        self.assertEqual(summary["by_mode"]["preserve"]["invalid_but_ai_claim"], 1)
        self.assertEqual(summary["by_mode"]["rebind"]["valid"], 1)
        self.assertEqual(summary["by_mode"]["rebind"]["trusted"], 1)


class TestMatrix(LabCase):
    def test_matrix_writes_report_and_expected_rows(self):
        out = self.root / "matrix"
        matrix = lab.run_matrix(
            out, self.certs, names=["jpeg_q75", "strip_rewrite"]
        )
        self.assertTrue((out / "report.json").exists())
        self.assertTrue((out / "report.md").exists())
        stored = json.loads((out / "report.json").read_text())
        self.assertEqual(stored["meta"]["transforms"], ["jpeg_q75", "strip_rewrite"])
        # control + (as_is, preserve, rebind) for jpeg + (as_is, rebind) for png
        self.assertEqual(len(stored["rows"]), 6)
        modes = stored["summary"]["by_mode"]
        self.assertEqual(modes["as_is"]["absent"], 2)
        self.assertEqual(modes["preserve"]["invalid"], 1)
        self.assertEqual(modes["rebind"]["valid"], 2)
        text = (out / "report.md").read_text()
        self.assertIn("| `jpeg_q75` |", text)
        self.assertIn("INVALID", text)

    def test_matrix_respects_mode_filter(self):
        out = self.root / "matrix_modes"
        matrix = lab.run_matrix(
            out, self.certs, names=["jpeg_q75"], modes=("rebind",)
        )
        self.assertEqual(sorted(r["mode"] for r in matrix["rows"]), ["-", "rebind"])
        self.assertEqual(matrix["summary"]["by_mode"]["rebind"]["valid"], 1)
        self.assertNotIn("as_is", matrix["summary"]["by_mode"])


if __name__ == "__main__":
    unittest.main()
