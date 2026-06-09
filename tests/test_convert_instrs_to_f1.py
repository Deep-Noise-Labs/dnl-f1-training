"""Tests for scripts/convert_instrs_to_f1.py (AISynth → Foundation-1 sidecars)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import get_repo_root

REPO_ROOT = get_repo_root()


def _load_converter():
    path = REPO_ROOT / "scripts" / "convert_instrs_to_f1.py"
    spec = importlib.util.spec_from_file_location("convert_instrs_to_f1", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def conv():
    return _load_converter()


@pytest.fixture
def sample_input_tree(conv, tmp_path: Path) -> Path:
    """Minimal instrs-style tree: one WAV+JSON per split."""
    meta = {
        "title": "Test_Preset_C3.wav",
        "name": "sample001",
        "instrument": "Synth Bass",
        "keywords": "distorted, layered",
        "key": "C3",
        "duration": 3.0,
        "description": "A dry fat bass.",
    }
    for split in conv.SPLITS:
        split_dir = tmp_path / split
        split_dir.mkdir()
        (split_dir / "sample001.wav").write_bytes(b"RIFF")  # placeholder
        (split_dir / "sample001.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
    return tmp_path


class TestBuildF1Text:
    def test_full_schema(self, conv):
        text = conv.build_f1_text(
            {
                "instrument": "Bells",
                "keywords": "bright",
                "description": "Chime-like tone.",
                "key": "C3",
            }
        )
        assert text == "Bells, bright, Chime-like tone., C3"

    def test_fallback_to_title_when_all_empty(self, conv):
        text = conv.build_f1_text({"title": "Fallback.wav", "name": "id1"})
        assert text == "Fallback.wav"

    def test_unknown_instrument_still_builds(self, conv):
        text = conv.build_f1_text(
            {"instrument": "", "keywords": "x", "description": "y", "key": "D3"}
        )
        assert text == "x, y, D3"

    def test_numeric_description_coerced(self, conv):
        text = conv.build_f1_text(
            {
                "instrument": "Pad",
                "keywords": "soft",
                "description": 42.0,
                "key": "C3",
            }
        )
        assert text == "Pad, soft, 42, C3"

    def test_list_keywords_joined(self, conv):
        text = conv.build_f1_text(
            {
                "instrument": "Lead",
                "keywords": ["bright", "sharp"],
                "description": "Analog lead.",
                "key": "G3",
            }
        )
        assert text == "Lead, bright, sharp, Analog lead., G3"


class TestConvertSidecar:
    def test_required_f1_fields(self, conv):
        out = conv.convert_sidecar({"instrument": "Pad", "duration": 3.0})
        assert out["seconds_start"] == 0
        assert out["seconds_total"] == 3
        assert "text" in out
        assert out["source_format"] == "instrs_C3_midasheng"
        assert out["instrument"] == "Pad"

    def test_duration_rounding(self, conv):
        out = conv.convert_sidecar({"instrument": "X", "duration": 2.6})
        assert out["seconds_total"] == 3


class TestProcessSplit:
    def test_writes_json_and_symlink(self, conv, sample_input_tree, tmp_path):
        out_root = tmp_path / "out"
        ok, skipped, errs = conv.process_split(
            sample_input_tree,
            out_root,
            "train",
            symlink_wav=True,
            dry_run=False,
            limit=None,
        )
        assert ok == 1
        assert skipped == 0
        assert errs == []
        out_json = out_root / "train" / "sample001.json"
        assert out_json.exists()
        data = json.loads(out_json.read_text())
        assert data["text"].startswith("Synth Bass")
        assert data["seconds_total"] == 3
        out_wav = out_root / "train" / "sample001.wav"
        assert out_wav.is_symlink()
        assert out_wav.resolve() == (sample_input_tree / "train" / "sample001.wav").resolve()

    def test_missing_json_sidecar(self, conv, sample_input_tree, tmp_path):
        (sample_input_tree / "train" / "orphan.wav").write_bytes(b"x")
        out_root = tmp_path / "out"
        ok, skipped, errs = conv.process_split(
            sample_input_tree,
            out_root,
            "train",
            symlink_wav=False,
            dry_run=False,
            limit=None,
        )
        assert ok == 1
        assert skipped == 1
        assert any("missing JSON" in e for e in errs)

    def test_dry_run_no_output_files(self, conv, sample_input_tree, tmp_path):
        out_root = tmp_path / "out"
        ok, _, errs = conv.process_split(
            sample_input_tree,
            out_root,
            "valid",
            symlink_wav=False,
            dry_run=True,
            limit=None,
        )
        assert ok == 1
        assert errs == []
        assert not out_root.exists()


class TestCli:
    def test_cli_dry_run_exit_zero(self, sample_input_tree, tmp_path):
        out_root = tmp_path / "cli_out"
        script = REPO_ROOT / "scripts" / "convert_instrs_to_f1.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--input",
                str(sample_input_tree),
                "--output",
                str(out_root),
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "train:" in proc.stdout
