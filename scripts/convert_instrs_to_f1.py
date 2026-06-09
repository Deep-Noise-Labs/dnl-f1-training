#!/usr/bin/env python3
"""
Convert AISynth instrs_C3_midasheng dataset to Foundation-1 WAV+JSON sidecar format.

Source layout (already split):
    /data/aisynth_datasets/training_datasets/instrs_C3_midasheng/
        train/*.wav + *.json
        valid/*.wav + *.json
        test/*.wav + *.json

Target layout (F1 / GCSDataset compatible):
    OUTPUT_ROOT/
        train/<id>.wav + <id>.json
        valid/...
        test/...

Each output JSON contains:
    text            — CLAP prompt: "{Instrument}, {keywords}, {description}, {key}"
    seconds_start   — 0
    seconds_total   — ceil(duration) (default 3)
    (original AISynth fields preserved for traceability)

Usage:
    python3 scripts/convert_instrs_to_f1.py \\
        --input /data/aisynth_datasets/training_datasets/instrs_C3_midasheng \\
        --output /data/aisynth_datasets/training_datasets/f1_instrs_C3_midasheng \\
        --symlink-wav

    # Validate a few samples:
    python3 scripts/convert_instrs_to_f1.py --input ... --output ... --dry-run --limit 20

    # Upload to GCS after conversion:
    gsutil -m rsync -r OUTPUT_ROOT/train gs://BUCKET/training_datasets/f1_instrs_C3_midasheng/train/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SPLITS = ("train", "valid", "test")


def _meta_str(value: object, default: str = "") -> str:
    """Coerce JSON sidecar values (str, float, int, None, …) to a stripped string."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return str(value).strip()
    if isinstance(value, (int, float)):
        # Avoid "3.0" noise for whole-number floats used as text
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip()
    if isinstance(value, (list, tuple)):
        return ", ".join(_meta_str(v) for v in value if _meta_str(v))
    return str(value).strip()


def build_f1_text(meta: dict) -> str:
    """Foundation-1 prompt schema: Instrument, timbre_tags, description, key."""
    instrument = _meta_str(meta.get("instrument"))
    keywords = _meta_str(meta.get("keywords"))
    description = _meta_str(meta.get("description"))
    key = _meta_str(meta.get("key"))
    parts = [p for p in (instrument, keywords, description, key) if p]
    if not parts:
        title = _meta_str(meta.get("title")) or _meta_str(meta.get("name")) or "audio"
        return title
    return ", ".join(parts)


def convert_sidecar(src_meta: dict) -> dict:
    duration = float(src_meta.get("duration", 3.0))
    seconds_total = max(1, int(round(duration)))
    out = dict(src_meta)
    out["text"] = build_f1_text(src_meta)
    out["seconds_start"] = 0
    out["seconds_total"] = seconds_total
    out["source_format"] = "instrs_C3_midasheng"
    return out


def process_split(
    input_root: Path,
    output_root: Path,
    split: str,
    symlink_wav: bool,
    dry_run: bool,
    limit: int | None,
) -> tuple[int, int, list[str]]:
    in_dir = input_root / split
    out_dir = output_root / split
    errors: list[str] = []
    ok = 0
    skipped = 0

    if not in_dir.is_dir():
        errors.append(f"missing split directory: {in_dir}")
        return ok, skipped, errors

    wav_files = sorted(in_dir.glob("*.wav"))
    if limit is not None:
        wav_files = wav_files[:limit]

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for wav_path in wav_files:
        stem = wav_path.stem
        json_in = in_dir / f"{stem}.json"
        if not json_in.exists():
            errors.append(f"{split}/{stem}: missing JSON sidecar")
            skipped += 1
            continue

        try:
            src_meta = json.loads(json_in.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{split}/{stem}: invalid JSON ({exc})")
            skipped += 1
            continue

        f1_meta = convert_sidecar(src_meta)
        if not f1_meta.get("text"):
            errors.append(f"{split}/{stem}: empty text after conversion")
            skipped += 1
            continue

        if dry_run:
            ok += 1
            continue

        out_wav = out_dir / wav_path.name
        out_json = out_dir / f"{stem}.json"

        if symlink_wav:
            if out_wav.exists() or out_wav.is_symlink():
                out_wav.unlink()
            os.symlink(wav_path.resolve(), out_wav)
        else:
            if not out_wav.exists():
                import shutil
                shutil.copy2(wav_path, out_wav)

        out_json.write_text(json.dumps(f1_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        ok += 1

    return ok, skipped, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/data/aisynth_datasets/training_datasets/instrs_C3_midasheng"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/aisynth_datasets/training_datasets/f1_instrs_C3_midasheng"),
    )
    parser.add_argument(
        "--symlink-wav",
        action="store_true",
        help="Symlink WAV files instead of copying (saves ~300GB disk).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not write files.")
    parser.add_argument("--limit", type=int, default=None, help="Max files per split (for testing).")
    args = parser.parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.is_dir():
        print(f"ERROR: input not found: {input_root}", file=sys.stderr)
        return 1

    total_ok = 0
    total_err = 0
    all_errors: list[str] = []

    for split in SPLITS:
        ok, skipped, errs = process_split(
            input_root,
            output_root,
            split,
            args.symlink_wav,
            args.dry_run,
            args.limit,
        )
        total_ok += ok
        total_err += len(errs)
        all_errors.extend(errs)
        print(f"{split}: converted={ok} skipped={skipped} errors={len(errs)}")

    if all_errors:
        print("\nFirst 20 errors:", file=sys.stderr)
        for e in all_errors[:20]:
            print(f"  {e}", file=sys.stderr)
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more", file=sys.stderr)

    mode = "dry-run" if args.dry_run else ("symlink" if args.symlink_wav else "copy")
    print(f"\nDone ({mode}). Total OK={total_ok} errors={total_err} output={output_root}")
    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main())
