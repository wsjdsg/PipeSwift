#!/usr/bin/env python3
"""Repack vLLM's bundled FlashAttention into a standalone installable wheel.

PipeSwift's FP8 KV cache needs vLLM's FlashAttention fork, not upstream FA3, which
lacks the SM90 two-level FP8 accumulation fix (vllm-project/flash-attention#104);
on an H100 its attention output drifts -67% at 128k context.

vLLM ships that fork inside its own wheel as ``vllm/vllm_flash_attn``, and in the
releases we target it imports nothing from vLLM -- only torch. So we lift the
subpackage out and re-emit it as a top-level ``vllm_flash_attn`` wheel.

The wheel is rebuilt rather than installed from the extracted directory on
purpose: an extracted archive is untrusted input, and putting it on PYTHONPATH
would let the interpreter load code out of it. This only reads bytes from the zip
and writes a fresh, well-formed wheel.

Usage: python repack_vllm_flash_attn.py <vllm-wheel> -o <out-dir> [--include-fa2]
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import sys
import zipfile

SRC_PREFIX = "vllm/vllm_flash_attn/"
PKG = "vllm_flash_attn"
TAG = "cp38-abi3-linux_x86_64"


def wheel_hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return "sha256=" + digest.decode("ascii").rstrip("=")


def read_version(zf: zipfile.ZipFile) -> str:
    """__version__ from the subpackage __init__, without importing it."""
    src = zf.read(SRC_PREFIX + "__init__.py").decode("utf-8")
    for line in src.splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("no __version__ in vllm_flash_attn/__init__.py")


def select_members(zf: zipfile.ZipFile, include_fa2: bool) -> list[str]:
    names = [
        n for n in zf.namelist() if n.startswith(SRC_PREFIX) and not n.endswith("/")
    ]
    if not include_fa2:
        # PipeSwift only ever asks for fa_version=3 and the FA2 import is already
        # guarded, so skipping it saves ~374MB.
        names = [n for n in names if "_vllm_fa2_C" not in n]
    return sorted(names)


def build_wheel(zf, names, version, vllm_ver) -> bytes:
    dist_info = f"{PKG}-{version}.dist-info"
    records: list[tuple[str, str, int]] = []
    buf = io.BytesIO()
    out = zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED)

    for name in names:
        rel = PKG + "/" + name[len(SRC_PREFIX) :]
        data = zf.read(name)
        out.writestr(rel, data)
        records.append((rel, wheel_hash(data), len(data)))
        mb = len(data) / 2**20
        print(f"  + {rel} ({mb:.1f} MB)" if mb > 1 else f"  + {rel}")

    metadata = (
        "Metadata-Version: 2.1\n"
        f"Name: {PKG}\n"
        f"Version: {version}\n"
        "Summary: vLLM's FlashAttention fork, repacked standalone for PipeSwift "
        "FP8 KV cache\n"
        "Requires-Python: >=3.9\n"
        "\n"
        f"Lifted from the vllm {vllm_ver} wheel. Carries the SM90 FP8 two-level\n"
        "accumulation fix (vllm-project/flash-attention#104) that upstream\n"
        "Dao-AILab FA3 does not have.\n"
    ).encode()
    wheel_meta = (
        "Wheel-Version: 1.0\n"
        "Generator: pipeswift-repack-vllm-fa\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {TAG}\n"
    ).encode()

    for rel, data in (
        (f"{dist_info}/METADATA", metadata),
        (f"{dist_info}/WHEEL", wheel_meta),
    ):
        out.writestr(rel, data)
        records.append((rel, wheel_hash(data), len(data)))

    rec = io.StringIO()
    writer = csv.writer(rec, lineterminator="\n")
    for row in records:
        writer.writerow(row)
    writer.writerow([f"{dist_info}/RECORD", "", ""])
    out.writestr(f"{dist_info}/RECORD", rec.getvalue())

    out.close()
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel", help="path to a downloaded vllm-*.whl")
    ap.add_argument("-o", "--out-dir", required=True)
    ap.add_argument("--include-fa2", action="store_true", help="also ship _vllm_fa2_C")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    zf = zipfile.ZipFile(args.wheel)
    names = select_members(zf, args.include_fa2)
    if not names:
        print(f"error: no {SRC_PREFIX} in {args.wheel}", file=sys.stderr)
        return 1

    version = read_version(zf)
    vllm_ver = os.path.basename(args.wheel).split("-")[1]
    payload = build_wheel(zf, names, version, vllm_ver)
    zf.close()

    out_path = os.path.join(args.out_dir, f"{PKG}-{version}-{TAG}.whl")
    with open(out_path, "wb") as f:
        f.write(payload)

    print(f"\nwrote {out_path} ({len(payload) / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
