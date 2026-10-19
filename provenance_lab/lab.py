"""Provenance survival lab.

Measures what happens to a real C2PA credential when a file is edited and
re-shared, across three arms:

    as_is     the editing tool re-encodes and drops metadata (Pillow default)
    preserve  the tool carries the original manifest through unchanged
              (the interesting failure: present, but the binding is stale)
    rebind    a C2PA-aware editor re-signs the result with the original as an
              ingredient (the only arm that can still validate)

Everything is measured with c2pa-python (SDK 0.91) against a throwaway test CA.
No platform accounts, no network.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import c2pa
from PIL import Image, ImageDraw
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

VERSION = "0.1.0"
AI_SOURCE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
FIXTURE_SIZE = (960, 640)
MODES = ("as_is", "preserve", "rebind")


# --------------------------------------------------------------------------
# test credentials
# --------------------------------------------------------------------------


def make_certs(dest: Path, days: int = 3650) -> dict[str, Path]:
    """Write a CA, a leaf signed by it, the leaf's PKCS#8 key and a chain.

    c2pa-rs refuses a self-signed signing certificate, so the leaf has to be
    issued by a CA even when both are throwaway.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "Provenance Lab Test CA"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Provenance Lab"),
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        ]
    )
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf_name = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "Provenance Lab Test Signer"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Provenance Lab"),
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        ]
    )
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(leaf_name)
        .issuer_name(ca_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]),
            critical=False,
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.RFC822Name("signer@provenance-lab.test")]),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    paths = {
        "ca": dest / "ca.cert.pem",
        "leaf": dest / "leaf.cert.pem",
        "chain": dest / "chain.pem",
        "key": dest / "leaf.key.pem",
    }
    ca_pem = ca_cert.public_bytes(serialization.Encoding.PEM).decode("ascii")
    leaf_pem = leaf_cert.public_bytes(serialization.Encoding.PEM).decode("ascii")
    paths["ca"].write_text(ca_pem)
    paths["leaf"].write_text(leaf_pem)
    paths["chain"].write_text(leaf_pem + ca_pem)
    paths["key"].write_bytes(
        leaf_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return paths


def load_signer(chain_path: Path, key_path: Path) -> c2pa.Signer:
    info = c2pa.C2paSignerInfo(
        b"es256",
        Path(chain_path).read_bytes(),
        Path(key_path).read_bytes(),
        None,
    )
    signer = c2pa.Signer.from_info(info)
    if not signer.is_valid:
        raise RuntimeError("signer failed to initialise")
    return signer


def reader_context(ca_pem: str | None):
    if not ca_pem:
        return None
    settings = c2pa.Settings.from_dict(
        {"trust": {"trust_anchors": ca_pem}}
    )
    return c2pa.Context(settings)


# --------------------------------------------------------------------------
# signing
# --------------------------------------------------------------------------


def _manifest(title: str) -> dict:
    return {"claim_generator": f"provenance-lab/{VERSION}", "title": title}


def sign_created(src: Path, dest: Path, signer: c2pa.Signer) -> None:
    """Sign a file as AI-generated (create intent, trained algorithmic media)."""
    builder = c2pa.Builder.from_json(_manifest(Path(src).name))
    builder.set_intent(
        c2pa.C2paBuilderIntent.CREATE,
        c2pa.C2paDigitalSourceType.TRAINED_ALGORITHMIC_MEDIA,
    )
    builder.sign_file(src, dest, signer)


def sign_edited(src: Path, dest: Path, parent: Path, signer: c2pa.Signer) -> None:
    """Re-sign an edit, recording the signed original as a parent ingredient."""
    builder = c2pa.Builder.from_json(_manifest(Path(src).name))
    builder.set_intent(c2pa.C2paBuilderIntent.EDIT)
    fmt = "image/png" if Path(parent).suffix.lower() == ".png" else "image/jpeg"
    with open(parent, "rb") as f:
        builder.add_ingredient_from_stream(
            {"title": Path(parent).name, "relationship": "parentOf"}, fmt, f
        )
    builder.sign_file(src, dest, signer)


# --------------------------------------------------------------------------
# JPEG segment surgery
# --------------------------------------------------------------------------

SOI = b"\xff\xd8"


def jpeg_segments(data: bytes) -> list[tuple[int, bytes]]:
    """Return (marker, raw segment) pairs before the scan data."""
    if not data.startswith(SOI):
        raise ValueError("not a JPEG")
    out: list[tuple[int, bytes]] = []
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            raise ValueError(f"bad marker at offset {i}")
        marker = data[i + 1]
        while marker == 0xFF:  # fill bytes
            i += 1
            marker = data[i + 1]
        if marker in (0xD9, 0xDA):  # EOI, SOS
            break
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        if length < 2:
            raise ValueError(f"bad segment length at offset {i}")
        out.append((marker, data[i : i + 2 + length]))
        i += 2 + length
    return out


def c2pa_segments(data: bytes) -> list[bytes]:
    """The APP11 segments that carry a C2PA/JUMBF manifest."""
    return [seg for marker, seg in jpeg_segments(data) if marker == 0xEB]


def inject_segments(jpeg: bytes, segments: Iterable[bytes]) -> bytes:
    """Put segments back immediately after SOI, as a metadata-preserving tool would."""
    if not jpeg.startswith(SOI):
        raise ValueError("not a JPEG")
    return SOI + b"".join(segments) + jpeg[2:]


def strip_segments(jpeg: bytes, marker: int = 0xEB) -> bytes:
    """Remove every segment with the given marker (for round-trip tests)."""
    if not jpeg.startswith(SOI):
        raise ValueError("not a JPEG")
    i = 2
    kept: list[bytes] = []
    while i + 4 <= len(jpeg):
        m = jpeg[i + 1]
        while m == 0xFF:  # fill bytes
            i += 1
            m = jpeg[i + 1]
        if m in (0xD9, 0xDA):  # EOI, SOS: keep the rest untouched
            break
        length = int.from_bytes(jpeg[i + 2 : i + 4], "big")
        if m != marker:
            kept.append(jpeg[i : i + 2 + length])
        i += 2 + length
    return SOI + b"".join(kept) + jpeg[i:]


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Transform:
    name: str
    kind: str  # "jpeg" or "png"
    fn: Callable[[Image.Image], tuple[Image.Image, dict]]
    note: str


def _q(quality: int, **extra) -> dict:
    return {"quality": quality, "subsampling": 2, **extra}


def _double_share(img: Image.Image) -> tuple[Image.Image, dict]:
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75, subsampling=2)
    buf.seek(0)
    again = Image.open(buf).convert("RGB")
    return again, _q(75)


def _screenshot(img: Image.Image) -> tuple[Image.Image, dict]:
    W, H = 1920, 1080
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    scaled = img.copy()
    scaled.thumbnail((W, H), Image.LANCZOS)
    canvas.paste(scaled, ((W - scaled.width) // 2, (H - scaled.height) // 2))
    return canvas, _q(85)


TRANSFORMS: tuple[Transform, ...] = (
    Transform("jpeg_q95", "jpeg", lambda i: (i, _q(95)), "mild re-encode"),
    Transform("jpeg_q75", "jpeg", lambda i: (i, _q(75)), "typical messaging re-encode"),
    Transform("jpeg_q50", "jpeg", lambda i: (i, _q(50)), "aggressive re-encode"),
    Transform(
        "resize_50",
        "jpeg",
        lambda i: (
            i.resize((i.width // 2, i.height // 2), Image.LANCZOS),
            _q(90),
        ),
        "halve the dimensions",
    ),
    Transform(
        "crop_10pct",
        "jpeg",
        lambda i: (
            i.crop(
                (
                    int(i.width * 0.1),
                    int(i.height * 0.1),
                    i.width - int(i.width * 0.1),
                    i.height - int(i.height * 0.1),
                )
            ),
            _q(90),
        ),
        "crop 10% off every edge",
    ),
    Transform(
        "rotate_90",
        "jpeg",
        lambda i: (i.transpose(Image.Transpose.ROTATE_270), _q(90)),
        "rotate a quarter turn",
    ),
    Transform("screenshot_1080p", "jpeg", _screenshot, "re-photograph as a screenshot"),
    Transform(
        "cdn_w800_q70",
        "jpeg",
        lambda i: (
            i.resize((800, round(i.height * 800 / i.width)), Image.LANCZOS),
            _q(70),
        ),
        "CDN-style width=800&q=70 transform",
    ),
    Transform("double_share", "jpeg", _double_share, "shared twice at q75"),
    Transform(
        "strip_rewrite",
        "png",
        lambda i: (i, {}),
        "same pixels, metadata silently dropped",
    ),
)

BY_NAME = {t.name: t for t in TRANSFORMS}


# --------------------------------------------------------------------------
# fixture
# --------------------------------------------------------------------------


def make_fixture(path: Path) -> Path:
    """A deterministic image with gradients, blocks and text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    W, H = FIXTURE_SIZE
    img = Image.new("RGB", (W, H), (18, 24, 38))
    d = ImageDraw.Draw(img)
    for x in range(0, W, 8):
        shade = int(18 + 90 * x / W)
        d.rectangle([x, 0, x + 8, H], fill=(shade // 3, shade // 2, shade))
    d.rectangle([80, 90, 400, 300], fill=(232, 178, 44))
    d.ellipse([520, 120, 780, 380], fill=(40, 170, 150))
    d.polygon([(300, 520), (520, 380), (700, 540)], fill=(200, 70, 90))
    d.text((84, 320), "provenance lab fixture", fill=(10, 10, 10))
    d.text((84, 344), "synthetic / AI-generated", fill=(10, 10, 10))
    img.save(path, format="JPEG", quality=92, subsampling=2)
    return path


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def verify(path: Path, ca_pem: str | None = None) -> dict:
    """Read one file and report credential presence, validity and trust."""
    path = Path(path)
    started = time.perf_counter()
    row = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "present": False,
        "claim_readable": False,
        "state": None,
        "signature_ok": False,
        "data_hash_ok": False,
        "trusted": False,
        "ai_claim": False,
        "assertions": [],
        "ingredients": 0,
        "manifest_id": None,
        "failures": [],
        "verify_ms": 0.0,
    }
    try:
        reader = c2pa.Reader.try_create(path, context=reader_context(ca_pem))
        if reader is None:
            row["verify_ms"] = (time.perf_counter() - started) * 1000
            return row
        row["present"] = True
        row["state"] = reader.get_validation_state()
        try:
            claim = json.loads(reader.json())
            detail = json.loads(reader.detailed_json())
        except Exception:
            row["verify_ms"] = (time.perf_counter() - started) * 1000
            return row
        row["claim_readable"] = True
        row["manifest_id"] = claim.get("active_manifest")
        manifest = claim.get("manifests", {}).get(row["manifest_id"], {})
        assertions = manifest.get("assertions", [])
        row["assertions"] = [a.get("label") for a in assertions]
        row["ingredients"] = len(manifest.get("ingredients", []))
        for assertion in assertions:
            if assertion.get("label", "").startswith("c2pa.actions"):
                for action in assertion.get("data", {}).get("actions", []):
                    if AI_SOURCE in (action.get("digitalSourceType") or ""):
                        row["ai_claim"] = True
        results = detail.get("validation_results", {}).get("activeManifest", {})
        success = {x.get("code") for x in results.get("success", [])}
        failure = {x.get("code") for x in results.get("failure", [])}
        row["signature_ok"] = "claimSignature.validated" in success
        row["data_hash_ok"] = "assertion.dataHash.match" in success
        row["trusted"] = "signingCredential.trusted" in success
        row["failures"] = sorted(failure)
    except Exception as exc:  # unreadable manifest is a result, not a crash
        row["failures"] = [f"reader_error:{type(exc).__name__}"]
    row["verify_ms"] = (time.perf_counter() - started) * 1000
    return row


def verdict(row: dict) -> str:
    if not row["present"]:
        return "absent"
    if row["state"] in ("Valid", "Trusted"):
        return "valid"
    return "invalid"


# --------------------------------------------------------------------------
# matrix
# --------------------------------------------------------------------------


def _pixels_equal(a: Image.Image, b: Image.Image) -> bool | None:
    a = a.convert("RGB")
    b = b.convert("RGB")
    if a.size != b.size:
        return None
    return a.tobytes() == b.tobytes()


def run_matrix(
    out_dir: Path,
    certs: dict[str, Path],
    names: Iterable[str] | None = None,
    modes: Iterable[str] = MODES,
) -> dict:
    out_dir = Path(out_dir)
    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    signer = load_signer(certs["chain"], certs["key"])
    ca_pem = Path(certs["ca"]).read_text()

    fixture = make_fixture(assets / "source.jpg")
    signed = assets / "signed.jpg"
    sign_created(fixture, signed, signer)
    segments = c2pa_segments(signed.read_bytes())
    overhead = signed.stat().st_size - fixture.stat().st_size
    with Image.open(fixture) as fh:
        source_img = fh.convert("RGB")

    started = time.perf_counter()
    rows: list[dict] = []

    def record(transform: str, mode: str, path: Path, pixels_same):
        row = verify(path, ca_pem)
        row.update(
            {
                "transform": transform,
                "mode": mode,
                "pixels_unchanged": pixels_same,
                "verdict": verdict(row),
            }
        )
        rows.append(row)
        return row

    control = record("original", "-", signed, True)
    control["bytes_baseline"] = control["bytes"]

    chosen = [BY_NAME[n] for n in names] if names else list(TRANSFORMS)
    for t in chosen:
        base_name = f"{t.name}__as_is"
        base_path = assets / f"{base_name}.{'png' if t.kind == 'png' else 'jpg'}"
        img, spec = t.fn(source_img)
        fmt = "PNG" if t.kind == "png" else "JPEG"
        img.save(base_path, format=fmt, **spec)
        # Compare the decoded result, not the in-memory image: a JPEG round trip
        # is lossy even at high quality, and that is part of what we measure.
        with Image.open(base_path) as saved_img:
            same_pixels = _pixels_equal(saved_img, source_img)
        if "as_is" in modes:
            record(t.name, "as_is", base_path, same_pixels)

        if "preserve" in modes and t.kind == "jpeg":
            preserved = assets / f"{t.name}__preserve.jpg"
            preserved.write_bytes(inject_segments(base_path.read_bytes(), segments))
            record(t.name, "preserve", preserved, same_pixels)

        if "rebind" in modes:
            suffix = "png" if t.kind == "png" else "jpg"
            rebound = assets / f"{t.name}__rebind.{suffix}"
            sign_edited(base_path, rebound, signed, signer)
            with Image.open(rebound) as a, Image.open(base_path) as b:
                rebound_same = _pixels_equal(a, b)
            record(t.name, "rebind", rebound, rebound_same)

    elapsed = time.perf_counter() - started
    matrix = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "c2pa_python": getattr(c2pa, "__version__", "?"),
            "c2pa_sdk": c2pa.sdk_version(),
            "pillow": Image.__version__,
            "fixture": f"{FIXTURE_SIZE[0]}x{FIXTURE_SIZE[1]}",
            "elapsed_s": round(elapsed, 2),
            "credential_overhead_bytes": overhead,
            "signing_ca": "self-issued test CA (not a public trust anchor)",
            "transforms": [t.name for t in chosen],
        },
        "control": control,
        "rows": rows,
    }
    matrix["summary"] = summarise(rows)
    (out_dir / "report.json").write_text(json.dumps(matrix, indent=2) + "\n")
    (out_dir / "report.md").write_text(render_markdown(matrix) + "\n")
    return matrix


def summarise(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    modes = sorted({r["mode"] for r in rows if r["mode"] != "-"})
    by_mode = {}
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        counts = {"absent": 0, "invalid": 0, "valid": 0}
        for r in subset:
            counts[r["verdict"]] += 1
        measures = [r["verify_ms"] for r in subset]
        by_mode[mode] = {
            "files": len(subset),
            **counts,
            "invalid_but_claim_readable": sum(
                1 for r in subset if r["verdict"] == "invalid" and r["claim_readable"]
            ),
            "invalid_but_ai_claim": sum(
                1 for r in subset if r["verdict"] == "invalid" and r["ai_claim"]
            ),
            "trusted": sum(1 for r in subset if r["trusted"]),
            "median_verify_ms": round(sorted(measures)[len(measures) // 2], 2)
            if measures
            else 0.0,
        }
    return {"rows": len(rows), "by_mode": by_mode}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

SYMBOL = {"absent": "gone", "invalid": "INVALID", "valid": "valid"}


def render_markdown(matrix: dict) -> str:
    meta = matrix["meta"]
    control = matrix["control"]
    summary = matrix["summary"]
    rows = matrix["rows"]
    lines: list[str] = []
    lines.append("# Provenance survival report")
    lines.append("")
    lines.append(f"Generated {meta['generated']} · c2pa-python {meta['c2pa_python']} "
                 f"(SDK {meta['c2pa_sdk']}) · Pillow {meta['pillow']}")
    lines.append("")
    lines.append(f"Control (`signed.jpg`, untouched): **{control['state']}**, "
                 f"{control['bytes']} bytes, assertions "
                 f"{', '.join(control['assertions']) or 'none'}. "
                 f"The credential itself adds {meta['credential_overhead_bytes']} bytes "
                 f"to the {meta['fixture']} fixture.")
    lines.append("")
    lines.append("| transform | pixels unchanged | as_is | preserve | rebind |")
    lines.append("|---|---|---|---|---|")

    def cell(values: dict[str, str], name: str) -> str:
        if name not in values:
            return "n/a"
        return SYMBOL[values[name]]

    by_transform: dict[str, dict[str, str]] = {}
    pixels: dict[str, str] = {}
    for r in rows:
        if r["mode"] == "-":
            continue
        by_transform.setdefault(r["transform"], {})[r["mode"]] = r["verdict"]
        if r["mode"] == "as_is":  # the other arms re-use the same pixels
            if r["pixels_unchanged"] is None:
                pixels[r["transform"]] = "n/a"
            else:
                pixels[r["transform"]] = "yes" if r["pixels_unchanged"] else "no"
    for transform in meta["transforms"]:
        values = by_transform.get(transform, {})
        lines.append(
            f"| `{transform}` | {pixels.get(transform, 'n/a')} "
            f"| {cell(values, 'as_is')} | {cell(values, 'preserve')} "
            f"| {cell(values, 'rebind')} |"
        )

    lines.append("")
    lines.append("## What provenance costs (bytes)")
    lines.append("")
    lines.append("| transform | as_is | preserve | rebind |")
    lines.append("|---|---|---|---|")
    sizes: dict[str, dict[str, int]] = {}
    for r in rows:
        if r["mode"] == "-":
            continue
        sizes.setdefault(r["transform"], {})[r["mode"]] = r["bytes"]
    for transform in meta["transforms"]:
        row_sizes = sizes.get(transform, {})
        cells = [f"{row_sizes[m]:,}" if m in row_sizes else "n/a"
                 for m in ("as_is", "preserve", "rebind")]
        lines.append(f"| `{transform}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## What the modes mean")
    lines.append("")
    lines.append("- **as_is** — the tool re-encodes and drops metadata. "
                 "A missing credential is not evidence of anything.")
    lines.append("- **preserve** — the tool carries the original manifest through "
                 "unchanged. The credential is still there and still readable, but "
                 "its hash no longer matches the pixels.")
    lines.append("- **rebind** — a C2PA-aware editor signs a new manifest with the "
                 "original as a parent ingredient. Only this arm can validate.")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("| mode | files | absent | invalid | valid | invalid but still readable | invalid but still claims AI | median verify |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for mode, stats in summary["by_mode"].items():
        lines.append(
            f"| {mode} | {stats['files']} | {stats['absent']} | {stats['invalid']} "
            f"| {stats['valid']} | {stats['invalid_but_claim_readable']} "
            f"| {stats['invalid_but_ai_claim']} | {stats['median_verify_ms']} ms |"
        )
    lines.append("")
    lines.append(f"Whole matrix took {meta['elapsed_s']} s.")
    lines.append("")
    lines.append("## Limits")
    lines.append("")
    lines.append("- The signing CA is a throwaway test anchor, so `trusted` means "
                 "trusted against that CA — not against a public trust list.")
    lines.append("- The screenshot arm re-renders the image onto a 1920×1080 canvas; "
                 "it models the metadata behaviour of a screenshot, not a real "
                 "compositor or display pipeline.")
    lines.append("- One transform per file. `double_share` is the only chained case.")
    lines.append("- JPEG and PNG only: no video, PDF, HEIF, or HTML embedding.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd")

    p_certs = sub.add_parser("certs", help="create a throwaway signing CA and leaf")
    p_certs.add_argument("--out", default="out/certs")

    p_run = sub.add_parser("run", help="sign, transform, verify, report")
    p_run.add_argument("--out", default="out/run")
    p_run.add_argument("--certs", default=None)
    p_run.add_argument("--transforms", default=None,
                       help="comma-separated subset of transform names")
    p_run.add_argument("--modes", default=",".join(MODES))

    p_verify = sub.add_parser("verify", help="verify a single file")
    p_verify.add_argument("file")
    p_verify.add_argument("--ca", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "certs":
        paths = make_certs(Path(args.out))
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
        return 0
    if args.cmd == "verify":
        ca = Path(args.ca).read_text() if args.ca else None
        row = verify(Path(args.file), ca)
        row["verdict"] = verdict(row)
        print(json.dumps(row, indent=2))
        return 0
    if args.cmd == "run" or args.cmd is None:
        out = Path(args.out if args.cmd else "out/run")
        certs_dir = Path(args.certs) if getattr(args, "certs", None) else out / "certs"
        certs = {
            "ca": certs_dir / "ca.cert.pem",
            "chain": certs_dir / "chain.pem",
            "key": certs_dir / "leaf.key.pem",
        }
        if not all(p.exists() for p in certs.values()):
            make_certs(certs_dir)
        names = args.transforms.split(",") if getattr(args, "transforms", None) else None
        modes = tuple(args.modes.split(",")) if getattr(args, "modes", None) else MODES
        unknown = set(names or []) - set(BY_NAME)
        if unknown:
            parser.error(f"unknown transforms: {', '.join(sorted(unknown))}")
        matrix = run_matrix(out, certs, names, modes)
        print(render_markdown(matrix))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
