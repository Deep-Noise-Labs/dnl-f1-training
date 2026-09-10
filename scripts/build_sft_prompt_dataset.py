#!/usr/bin/env python3
"""
Build a full-sentence-prompt SFT dataset for Foundation-1 fine-tuning.

Foundation-1 is currently trained on structured, templated prompts
("{Instrument}, {timbre_tags}, {description}, {key}" - see convert_instrs_to_f1.py
and README_DNL.md § Data Format). That schema can't represent a full natural-
language request like "vintage synth with loud buzz" or "whiny high-pitched
violin" - there's no slot for a qualifier that spans two attributes at once.

This script builds the supervised fine-tuning (SFT) dataset for that gap:
a human ("good with language and good with sounds") reverse-engineers a
full-sentence prompt for each of a batch of *existing* sounds, tagged with a
curriculum difficulty tier and a descriptor category, so the resulting
(full_prompt, sound) pairs can (a) fine-tune the model directly on natural
language and (b) be reinforced in difficulty order once RLHF is wired up on
top. It does not train anything or fabricate any prompts itself - it is the
authoring/QA/export tooling around a dataset that people still have to write.

Workflow:

    1. init-queue   - scan a directory of existing sounds and write an
                       authoring queue (one JSON line per sound, empty prompt
                       fields) for a human annotator to fill in by hand.
    2. report       - show progress toward the pair-count target, curriculum
                       tier distribution, and descriptor coverage (timbre,
                       envelope, articulation, noise, era, instrument) so a
                       team can see where the dataset is thin.
    3. export       - take the annotated ("done") queue entries and emit
                       WAV+JSON sidecars in the exact Foundation-1 format
                       (text/seconds_start/seconds_total), split
                       deterministically into train/valid/test so the held-out
                       set stays stable across re-exports, ready to feed
                       straight into pre_encode.py / train.py.

Usage:
    python3 scripts/build_sft_prompt_dataset.py init-queue \\
        --sounds-dir /data/aisynth_datasets/sft_source_sounds \\
        --queue queue.jsonl

    # Hand queue.jsonl to an annotator. Each line's full_prompt,
    # curriculum_tier, and qualifiers fields start empty - fill them in.

    python3 scripts/build_sft_prompt_dataset.py report --queue queue.jsonl

    python3 scripts/build_sft_prompt_dataset.py export \\
        --queue queue.jsonl \\
        --output /data/aisynth_datasets/training_datasets/f1_sft_v1 \\
        --version sft-v1 --symlink-wav
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import wave
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

CURRICULUM_TIERS = ("kindergarten", "first_grade", "second_grade_plus")
"""Ben's difficulty ladder: single qualifier -> two qualifiers -> three or more."""

DESCRIPTOR_CATEGORIES = ("timbre", "envelope", "articulation", "noise", "era", "instrument")
"""Coverage dimensions to track per the data-ops notes, so gaps are visible."""

SPLITS = ("train", "valid", "test")
DEFAULT_SPLIT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}


@dataclass
class QueueEntry:
    id: str
    wav_path: str
    reference_structured_text: str | None
    full_prompt: str = ""
    curriculum_tier: str = ""
    qualifiers: dict[str, str] = field(default_factory=lambda: {c: "" for c in DESCRIPTOR_CATEGORIES})
    annotator: str = ""
    annotated_at: str = ""
    status: str = "pending"

    def is_done(self) -> bool:
        return self.status == "done" and bool(self.full_prompt.strip())


def _sound_id(wav_path: Path, sounds_root: Path) -> str:
    return wav_path.relative_to(sounds_root).with_suffix("").as_posix()


def _reference_text(wav_path: Path) -> str | None:
    sidecar = wav_path.with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return meta.get("text") or meta.get("description") or None


def load_queue(queue_path: Path) -> list[QueueEntry]:
    if not queue_path.exists():
        return []
    entries = []
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        raw.setdefault("qualifiers", {c: "" for c in DESCRIPTOR_CATEGORIES})
        entries.append(QueueEntry(**raw))
    return entries


def write_queue(queue_path: Path, entries: list[QueueEntry]) -> None:
    with queue_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def cmd_init_queue(args: argparse.Namespace) -> int:
    sounds_root = args.sounds_dir.resolve()
    if not sounds_root.is_dir():
        print(f"ERROR: sounds dir not found: {sounds_root}", file=sys.stderr)
        return 1

    queue_path = args.queue
    existing = {e.id: e for e in load_queue(queue_path)}

    wav_files = sorted(sounds_root.rglob("*.wav"))
    if args.limit is not None:
        wav_files = wav_files[: args.limit]

    added = 0
    for wav_path in wav_files:
        sound_id = _sound_id(wav_path, sounds_root)
        if sound_id in existing:
            continue
        existing[sound_id] = QueueEntry(
            id=sound_id,
            wav_path=str(wav_path.relative_to(sounds_root)),
            reference_structured_text=_reference_text(wav_path),
        )
        added += 1

    all_entries = sorted(existing.values(), key=lambda e: e.id)
    write_queue(queue_path, all_entries)
    print(f"Queue now has {len(all_entries)} entries ({added} newly added) -> {queue_path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    entries = load_queue(args.queue)
    if not entries:
        print(f"No entries found in {args.queue}. Run init-queue first.")
        return 0

    done = [e for e in entries if e.is_done()]
    tier_counts = Counter(e.curriculum_tier for e in done)
    coverage = {c: sum(1 for e in done if e.qualifiers.get(c, "").strip()) for c in DESCRIPTOR_CATEGORIES}
    prompt_counts = Counter(e.full_prompt.strip().lower() for e in done)
    duplicate_prompts = {p: n for p, n in prompt_counts.items() if n > 1}
    copies_reference = [
        e.id for e in done if e.reference_structured_text and e.full_prompt.strip() == e.reference_structured_text.strip()
    ]

    print(f"Queue: {args.queue}")
    print(f"  total entries:      {len(entries)}")
    print(f"  annotated (done):   {len(done)}")
    print(f"  target:             {args.target}")
    print(f"  progress:           {len(done)}/{args.target} ({100 * len(done) / args.target:.1f}%)")
    print()
    print("  curriculum tier distribution:")
    for tier in CURRICULUM_TIERS:
        print(f"    {tier:<18} {tier_counts.get(tier, 0)}")
    untiered = len(done) - sum(tier_counts.values())
    if untiered:
        print(f"    (missing/unknown)  {untiered}")
    print()
    print("  descriptor coverage (entries with that qualifier filled in):")
    for category in DESCRIPTOR_CATEGORIES:
        print(f"    {category:<14} {coverage[category]}/{len(done)}")
    print()
    if duplicate_prompts:
        print(f"  WARNING: {len(duplicate_prompts)} duplicate full_prompt value(s) across different sounds")
    if copies_reference:
        print(
            f"  WARNING: {len(copies_reference)} entries reuse the templated reference text verbatim "
            "(defeats the point of writing a natural-language prompt) - flag for re-annotation"
        )
    return 0


def _wav_duration_seconds(wav_path: Path) -> float:
    try:
        with wave.open(str(wav_path), "rb") as f:
            return f.getnframes() / float(f.getframerate())
    except (wave.Error, EOFError, FileNotFoundError):
        return 3.0  # Foundation-1's one-shot default; export still records this explicitly.


def _split_for_id(sound_id: str, ratios: dict[str, float], seed: str) -> str:
    """Deterministic split assignment so re-exporting the same queue with the
    same seed never moves a sound between train/valid/test - the whole point
    of keeping a held-out evaluation set stable across dataset versions."""
    digest = hashlib.sha256(f"{seed}:{sound_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF

    cumulative = 0.0
    for split in SPLITS:
        cumulative += ratios.get(split, 0.0)
        if bucket < cumulative:
            return split
    return SPLITS[-1]


def _parse_split_ratios(raw: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for part in raw.split(","):
        name, _, value = part.partition(":")
        ratios[name.strip()] = float(value)
    total = sum(ratios.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"split ratios must sum to 1.0, got {total} ({raw})")
    return ratios


def cmd_export(args: argparse.Namespace) -> int:
    entries = [e for e in load_queue(args.queue) if e.is_done()]
    if not entries:
        print("No annotated (status=done) entries to export.", file=sys.stderr)
        return 1

    sounds_root = args.sounds_dir.resolve()
    output_root = args.output.resolve()
    ratios = _parse_split_ratios(args.split)

    counts = Counter()
    for entry in entries:
        split = _split_for_id(entry.id, ratios, args.seed)
        wav_src = sounds_root / entry.wav_path
        if not wav_src.exists():
            print(f"SKIP {entry.id}: source wav not found at {wav_src}", file=sys.stderr)
            continue

        out_dir = output_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = entry.id.replace("/", "__")
        out_wav = out_dir / f"{stem}.wav"
        out_json = out_dir / f"{stem}.json"

        if not args.dry_run:
            if args.symlink_wav:
                if out_wav.exists() or out_wav.is_symlink():
                    out_wav.unlink()
                out_wav.symlink_to(wav_src.resolve())
            else:
                import shutil

                shutil.copy2(wav_src, out_wav)

            duration = _wav_duration_seconds(wav_src)
            sidecar = {
                "text": entry.full_prompt.strip(),
                "seconds_start": 0,
                "seconds_total": max(1, round(duration)),
                "source_format": "sft_prompt_dataset",
                "dataset_version": args.version,
                "curriculum_tier": entry.curriculum_tier,
                "qualifiers": entry.qualifiers,
                "annotator": entry.annotator,
                "annotated_at": entry.annotated_at,
                "reference_structured_text": entry.reference_structured_text,
            }
            out_json.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        counts[split] += 1

    mode = "dry-run" if args.dry_run else ("symlink" if args.symlink_wav else "copy")
    for split in SPLITS:
        print(f"{split}: {counts.get(split, 0)}")
    print(f"\nDone ({mode}). Total exported={sum(counts.values())} version={args.version} output={output_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-queue", help="Scan existing sounds and create/extend the annotation queue.")
    p_init.add_argument("--sounds-dir", type=Path, required=True)
    p_init.add_argument("--queue", type=Path, required=True)
    p_init.add_argument("--limit", type=int, default=None, help="Max sounds to add (for a first small batch).")
    p_init.set_defaults(func=cmd_init_queue)

    p_report = sub.add_parser("report", help="Show annotation progress and descriptor coverage.")
    p_report.add_argument("--queue", type=Path, required=True)
    p_report.add_argument("--target", type=int, default=1000, help="Target pair count (a few hundred to ~2000).")
    p_report.set_defaults(func=cmd_report)

    p_export = sub.add_parser("export", help="Export annotated pairs as Foundation-1 WAV+JSON sidecars.")
    p_export.add_argument("--queue", type=Path, required=True)
    p_export.add_argument("--sounds-dir", type=Path, required=True, help="Same directory passed to init-queue.")
    p_export.add_argument("--output", type=Path, required=True)
    p_export.add_argument("--version", type=str, required=True, help='Dataset version tag, e.g. "sft-v1".')
    p_export.add_argument("--split", type=str, default="train:0.8,valid:0.1,test:0.1")
    p_export.add_argument("--seed", type=str, default="dnl-sft-v1", help="Split-assignment seed (keep stable).")
    p_export.add_argument("--symlink-wav", action="store_true")
    p_export.add_argument("--dry-run", action="store_true")
    p_export.set_defaults(func=cmd_export)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
