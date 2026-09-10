"""Tests for scripts/build_sft_prompt_dataset.py (SFT prompt-authoring/export tooling)."""

from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path

import pytest

from tests.conftest import get_repo_root

REPO_ROOT = get_repo_root()


def _load_module():
    path = REPO_ROOT / "scripts" / "build_sft_prompt_dataset.py"
    spec = importlib.util.spec_from_file_location("build_sft_prompt_dataset", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # dataclass field resolution needs the module registered in sys.modules
    # (it looks itself up via cls.__module__) before exec_module runs.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sft():
    return _load_module()


def _write_tiny_wav(path: Path, seconds: float = 1.0, framerate: int = 8000) -> None:
    n_frames = int(seconds * framerate)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(framerate)
        f.writeframes(b"\x00\x00" * n_frames)


@pytest.fixture
def sounds_dir(tmp_path: Path) -> Path:
    root = tmp_path / "sounds"
    root.mkdir()
    _write_tiny_wav(root / "bass_001.wav", seconds=3.0)
    (root / "bass_001.json").write_text(
        json.dumps({"text": "Synth Bass, distorted, layered, dynamic, fat bass., C3"}), encoding="utf-8"
    )
    _write_tiny_wav(root / "violin_002.wav", seconds=2.0)
    return root


class TestInitQueue:
    def test_creates_one_entry_per_wav(self, sft, sounds_dir, tmp_path):
        queue_path = tmp_path / "queue.jsonl"
        args = sft.build_parser().parse_args(
            ["init-queue", "--sounds-dir", str(sounds_dir), "--queue", str(queue_path)]
        )
        assert args.func(args) == 0

        entries = sft.load_queue(queue_path)
        assert {e.id for e in entries} == {"bass_001", "violin_002"}
        bass = next(e for e in entries if e.id == "bass_001")
        assert bass.reference_structured_text == "Synth Bass, distorted, layered, dynamic, fat bass., C3"
        assert bass.status == "pending"
        assert bass.full_prompt == ""

    def test_is_idempotent_and_preserves_annotations(self, sft, sounds_dir, tmp_path):
        queue_path = tmp_path / "queue.jsonl"
        args = sft.build_parser().parse_args(
            ["init-queue", "--sounds-dir", str(sounds_dir), "--queue", str(queue_path)]
        )
        args.func(args)

        entries = {e.id: e for e in sft.load_queue(queue_path)}
        entries["bass_001"].full_prompt = "Vintage synth bass with a loud buzz"
        entries["bass_001"].curriculum_tier = "first_grade"
        entries["bass_001"].status = "done"
        sft.write_queue(queue_path, list(entries.values()))

        args.func(args)  # re-run init-queue over the same directory

        reloaded = {e.id: e for e in sft.load_queue(queue_path)}
        assert len(reloaded) == 2
        assert reloaded["bass_001"].full_prompt == "Vintage synth bass with a loud buzz"
        assert reloaded["bass_001"].status == "done"


class TestReport:
    def test_counts_progress_and_coverage(self, sft, tmp_path, capsys):
        queue_path = tmp_path / "queue.jsonl"
        entries = [
            sft.QueueEntry(
                id="a",
                wav_path="a.wav",
                reference_structured_text="Bass, deep, C1",
                full_prompt="A deep vintage synth bass",
                curriculum_tier="kindergarten",
                qualifiers={"timbre": "deep", "envelope": "", "articulation": "", "noise": "", "era": "vintage", "instrument": "synth bass"},
                status="done",
            ),
            sft.QueueEntry(
                id="b",
                wav_path="b.wav",
                reference_structured_text="Violin, whiny, D3",
                full_prompt="Violin, whiny, D3",  # copies the reference verbatim - should be flagged
                curriculum_tier="first_grade",
                status="done",
            ),
            sft.QueueEntry(id="c", wav_path="c.wav", reference_structured_text=None),  # still pending
        ]
        sft.write_queue(queue_path, entries)

        args = sft.build_parser().parse_args(["report", "--queue", str(queue_path), "--target", "10"])
        assert args.func(args) == 0

        out = capsys.readouterr().out
        assert "annotated (done):   2" in out
        assert "kindergarten       1" in out
        assert "reuse the templated reference text verbatim" in out

    def test_empty_queue_does_not_crash(self, sft, tmp_path, capsys):
        queue_path = tmp_path / "missing.jsonl"
        args = sft.build_parser().parse_args(["report", "--queue", str(queue_path)])
        assert args.func(args) == 0
        assert "Run init-queue first" in capsys.readouterr().out


class TestExport:
    def test_exports_only_done_entries_with_f1_schema(self, sft, sounds_dir, tmp_path):
        queue_path = tmp_path / "queue.jsonl"
        entries = [
            sft.QueueEntry(
                id="bass_001",
                wav_path="bass_001.wav",
                reference_structured_text="Synth Bass, distorted, layered, dynamic, fat bass., C3",
                full_prompt="A gritty vintage synth bass with heavy distortion",
                curriculum_tier="first_grade",
                annotator="patrik@deepnoise.ai",
                status="done",
            ),
            sft.QueueEntry(id="violin_002", wav_path="violin_002.wav", reference_structured_text=None),  # pending
        ]
        sft.write_queue(queue_path, entries)

        out_root = tmp_path / "out"
        args = sft.build_parser().parse_args(
            [
                "export",
                "--queue", str(queue_path),
                "--sounds-dir", str(sounds_dir),
                "--output", str(out_root),
                "--version", "sft-v1-test",
                "--symlink-wav",
            ]
        )
        assert args.func(args) == 0

        exported = list(out_root.rglob("bass_001.json"))
        assert len(exported) == 1
        data = json.loads(exported[0].read_text())
        assert data["text"] == "A gritty vintage synth bass with heavy distortion"
        assert data["seconds_total"] == 3
        assert data["dataset_version"] == "sft-v1-test"
        assert data["curriculum_tier"] == "first_grade"

        assert not list(out_root.rglob("violin_002.json"))

    def test_split_assignment_is_deterministic(self, sft):
        ratios = {"train": 0.8, "valid": 0.1, "test": 0.1}
        first = sft._split_for_id("sound_123", ratios, seed="fixed-seed")
        second = sft._split_for_id("sound_123", ratios, seed="fixed-seed")
        assert first == second
        assert first in ("train", "valid", "test")

    def test_split_ratios_must_sum_to_one(self, sft):
        with pytest.raises(ValueError):
            sft._parse_split_ratios("train:0.8,valid:0.1")
