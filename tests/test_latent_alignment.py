"""Regression tests for the latent time-dimension alignment fix.

Root cause (fixed in stable_audio_tools/data/dataset.py):
  The ADP U-Net uses factors=[1,2,2,4] → total temporal stride=16.  The
  latent time dimension fed to the model must therefore be divisible by 16.
  Pre-encoding with sample_size=132300 produced 129-step latents
  (132300/1024=129.19, floored to 129).  129 % 16 == 1, causing:

    RuntimeError: Sizes of tensors must match except in dimension 1.
    Expected size 36 but got size 33 for tensor number 1 in the list.

  at the first skip-connection cat in adp.py during the upsampling path.

Fix:
  1. PreEncodedLatentsDataset.__getitem__ crops the time dimension to the
     nearest multiple of _LATENT_TIME_MULTIPLE (16) before returning.
  2. model_config_3s.json sample_size changed to 131072 (= 128 × 1024).
  3. vertex_job_pre_encode.yaml SAMPLE_SIZE updated to 131072.

Test structure:
  - TestModelConfigSampleSize / TestVertexYamlSampleSize — pure JSON/YAML
    checks, no torch required; always run in CI.
  - TestPreEncodedLatentsDatasetCrop — exercises the actual dataset class;
    requires torch + numpy, skipped automatically when not installed.
"""
import importlib
import json
import math
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Detect optional heavy dependencies once at collection time.
_torch_available = importlib.util.find_spec("torch") is not None
_numpy_available = importlib.util.find_spec("numpy") is not None
_requires_torch = pytest.mark.skipif(
    not (_torch_available and _numpy_available),
    reason="torch and numpy are required for dataset crop tests",
)


# ---------------------------------------------------------------------------
# Model config alignment tests  (no torch required — always run in CI)
# ---------------------------------------------------------------------------
class TestModelConfigSampleSize:
    """Verify that model_config_3s.json uses a sample_size compatible with the U-Net."""

    def _load_config(self):
        cfg_path = REPO_ROOT / "models" / "foundation1_3s" / "model_config_3s.json"
        return json.loads(cfg_path.read_text())

    def test_sample_size_divisible_by_vae_times_adp_stride(self):
        """sample_size must be divisible by (VAE downsampling_ratio × ADP total stride)."""
        cfg = self._load_config()
        sample_size = cfg["sample_size"]
        vae_ratio = cfg["model"]["pretransform"]["config"]["downsampling_ratio"]
        factors = cfg["model"]["diffusion"]["config"]["factors"]
        adp_stride = math.prod(factors)
        required_multiple = vae_ratio * adp_stride  # 1024 × 16 = 16384
        assert sample_size % required_multiple == 0, (
            f"sample_size={sample_size} is not divisible by {required_multiple} "
            f"(VAE ratio {vae_ratio} × ADP stride {adp_stride}). "
            f"Remainder: {sample_size % required_multiple}. "
            f"Use {(sample_size // required_multiple) * required_multiple} instead."
        )

    def test_latent_time_divisible_by_adp_stride(self):
        """Latent time dimension (sample_size / VAE ratio) must be divisible by ADP stride."""
        cfg = self._load_config()
        sample_size = cfg["sample_size"]
        vae_ratio = cfg["model"]["pretransform"]["config"]["downsampling_ratio"]
        factors = cfg["model"]["diffusion"]["config"]["factors"]
        adp_stride = math.prod(factors)
        latent_time = sample_size / vae_ratio
        assert latent_time == int(latent_time), (
            f"sample_size={sample_size} / VAE ratio={vae_ratio} = {latent_time} "
            f"is not an integer — sample_size must be an exact multiple of {vae_ratio}."
        )
        latent_time = int(latent_time)
        assert latent_time % adp_stride == 0, (
            f"Latent time {latent_time} is not divisible by ADP stride {adp_stride}. "
            f"Remainder: {latent_time % adp_stride}. "
            f"This causes a skip-connection RuntimeError during training."
        )

    def test_sample_size_is_131072(self):
        """sample_size must be 131072 (= 128 × 1024) after the alignment fix."""
        cfg = self._load_config()
        assert cfg["sample_size"] == 131072, (
            f"Expected sample_size=131072, got {cfg['sample_size']}. "
            "If you changed sample_size, ensure it remains a multiple of 16384."
        )


# ---------------------------------------------------------------------------
# Vertex YAML alignment tests  (no torch required — always run in CI)
# ---------------------------------------------------------------------------
class TestVertexYamlSampleSize:
    """Verify that Vertex YAML SAMPLE_SIZE env vars match the model config."""

    def _get_env_value(self, yaml_path: pathlib.Path, env_name: str):
        import yaml as pyyaml
        data = pyyaml.safe_load(yaml_path.read_text())
        for spec in data.get("workerPoolSpecs", []):
            for entry in spec.get("containerSpec", {}).get("env", []):
                if entry.get("name") == env_name:
                    return entry["value"]
        return None

    def test_pre_encode_yaml_sample_size(self):
        """vertex_job_pre_encode.yaml SAMPLE_SIZE must be 131072."""
        yaml_path = REPO_ROOT / "scripts" / "vertex_job_pre_encode.yaml"
        value = self._get_env_value(yaml_path, "SAMPLE_SIZE")
        assert value is not None, "SAMPLE_SIZE not found in vertex_job_pre_encode.yaml"
        assert int(value) == 131072, (
            f"Expected SAMPLE_SIZE=131072 in vertex_job_pre_encode.yaml, got {value}. "
            "Must match model_config_3s.json sample_size."
        )

    def test_pre_encode_sample_size_divisible_by_16384(self):
        """SAMPLE_SIZE in pre-encode YAML must be divisible by 16384 (16 × 1024)."""
        yaml_path = REPO_ROOT / "scripts" / "vertex_job_pre_encode.yaml"
        value = self._get_env_value(yaml_path, "SAMPLE_SIZE")
        if value is None:
            pytest.skip("SAMPLE_SIZE not found in vertex_job_pre_encode.yaml")
        sample_size = int(value)
        assert sample_size % 16384 == 0, (
            f"SAMPLE_SIZE={sample_size} is not divisible by 16384. "
            f"Remainder: {sample_size % 16384}."
        )


# ---------------------------------------------------------------------------
# EXPECTED_LATENT_SHAPE constant tests  (no torch required — always run in CI)
# ---------------------------------------------------------------------------
class TestExpectedLatentShapeConstant:
    """Verify that PreEncodedLatentsDataset.EXPECTED_LATENT_SHAPE is consistent
    with model_config_3s.json and the ADP U-Net stride requirement.

    This class guards against the exact bug that caused the second training
    failure: EXPECTED_LATENT_SHAPE was (64, 129) while the dataset had been
    re-encoded with sample_size=131072 → 128 tokens, causing a ValueError
    in _validate_latent_shapes before any training step ran.
    """

    def _load_config(self):
        cfg_path = REPO_ROOT / "models" / "foundation1_3s" / "model_config_3s.json"
        return json.loads(cfg_path.read_text())

    def _get_expected_shape_from_source(self):
        """Parse EXPECTED_LATENT_SHAPE directly from dataset.py source without importing torch."""
        import re
        src = (REPO_ROOT / "stable_audio_tools" / "data" / "dataset.py").read_text()
        m = re.search(r"EXPECTED_LATENT_SHAPE\s*=\s*\((\d+),\s*(\d+)\)", src)
        assert m, "EXPECTED_LATENT_SHAPE not found in dataset.py"
        return int(m.group(1)), int(m.group(2))

    def test_expected_shape_channels_is_64(self):
        """Channel dimension of EXPECTED_LATENT_SHAPE must be 64 (io_channels)."""
        channels, _ = self._get_expected_shape_from_source()
        assert channels == 64, (
            f"EXPECTED_LATENT_SHAPE channels={channels}, expected 64 (io_channels from model config)."
        )

    def test_expected_shape_tokens_matches_model_config(self):
        """Token dimension of EXPECTED_LATENT_SHAPE must equal sample_size / VAE ratio."""
        cfg = self._load_config()
        sample_size = cfg["sample_size"]
        vae_ratio = cfg["model"]["pretransform"]["config"]["downsampling_ratio"]
        expected_tokens = sample_size // vae_ratio
        _, actual_tokens = self._get_expected_shape_from_source()
        assert actual_tokens == expected_tokens, (
            f"EXPECTED_LATENT_SHAPE tokens={actual_tokens} does not match "
            f"model_config sample_size={sample_size} / VAE ratio={vae_ratio} = {expected_tokens}. "
            f"Update EXPECTED_LATENT_SHAPE in dataset.py to (64, {expected_tokens})."
        )

    def test_expected_shape_tokens_is_128(self):
        """Token dimension must be 128 after the sample_size=131072 fix."""
        _, tokens = self._get_expected_shape_from_source()
        assert tokens == 128, (
            f"EXPECTED_LATENT_SHAPE tokens={tokens}, expected 128. "
            "If sample_size changed, update EXPECTED_LATENT_SHAPE accordingly."
        )

    def test_expected_shape_tokens_divisible_by_adp_stride(self):
        """Token dimension must be divisible by ADP U-Net temporal stride (16)."""
        cfg = self._load_config()
        factors = cfg["model"]["diffusion"]["config"]["factors"]
        adp_stride = math.prod(factors)
        _, tokens = self._get_expected_shape_from_source()
        assert tokens % adp_stride == 0, (
            f"EXPECTED_LATENT_SHAPE tokens={tokens} is not divisible by ADP stride={adp_stride}. "
            f"This will cause a skip-connection RuntimeError during training."
        )


# ---------------------------------------------------------------------------
# Dataset crop unit tests  (requires torch + numpy)
# ---------------------------------------------------------------------------
def _make_latent_pair(tmp_dir: pathlib.Path, time_steps: int, channels: int = 64):
    """Write a (channels, time_steps) .npy + matching .json pair."""
    import numpy as np  # guarded by _requires_torch skipif
    arr = np.random.randn(channels, time_steps).astype(np.float32)
    npy_path = tmp_dir / "sample.npy"
    json_path = tmp_dir / "sample.json"
    np.save(npy_path, arr)
    metadata = {
        "prompt": "test",
        "seconds_start": 0,
        "seconds_total": 3,
        "padding_mask": [1] * time_steps,
    }
    json_path.write_text(json.dumps(metadata))
    return npy_path, json_path


@pytest.fixture(scope="module")
def PreEncodedLatentsDataset():
    """Import the dataset class from the repo (not an installed package)."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from stable_audio_tools.data.dataset import PreEncodedLatentsDataset as DS
    return DS


@_requires_torch
class TestPreEncodedLatentsDatasetCrop:
    """Verify that __getitem__ crops latent time to a multiple of 16."""

    def test_129_steps_cropped_to_128(self, tmp_path, PreEncodedLatentsDataset):
        """Legacy 129-step latents must be cropped to 128 (divisible by 16)."""
        import torch
        _make_latent_pair(tmp_path, time_steps=129)
        ds = PreEncodedLatentsDataset(root_path=str(tmp_path))
        latent, info = ds[0]
        assert latent.shape[1] == 128, (
            f"Expected latent time=128 after crop, got {latent.shape[1]}"
        )
        assert isinstance(info["padding_mask"], torch.Tensor)
        assert info["padding_mask"].shape[0] == 128, (
            f"padding_mask must be cropped to 128, got {info['padding_mask'].shape[0]}"
        )

    def test_128_steps_unchanged(self, tmp_path, PreEncodedLatentsDataset):
        """128-step latents (already valid) must not be modified."""
        _make_latent_pair(tmp_path, time_steps=128)
        ds = PreEncodedLatentsDataset(root_path=str(tmp_path))
        latent, info = ds[0]
        assert latent.shape[1] == 128
        assert info["padding_mask"].shape[0] == 128

    def test_144_steps_unchanged(self, tmp_path, PreEncodedLatentsDataset):
        """144-step latents (144 % 16 == 0) must not be modified."""
        _make_latent_pair(tmp_path, time_steps=144)
        ds = PreEncodedLatentsDataset(root_path=str(tmp_path))
        latent, info = ds[0]
        assert latent.shape[1] == 144

    def test_crop_is_multiple_of_16_for_range(self, tmp_path, PreEncodedLatentsDataset):
        """For any input time T in 129..143, output time must be divisible by 16."""
        for T in range(129, 144):
            sub = tmp_path / f"T{T}"
            sub.mkdir()
            _make_latent_pair(sub, time_steps=T)
            ds = PreEncodedLatentsDataset(root_path=str(sub))
            latent, _ = ds[0]
            assert latent.shape[1] % 16 == 0, (
                f"T={T}: output time {latent.shape[1]} is not divisible by 16"
            )
            expected = (T // 16) * 16
            assert latent.shape[1] == expected, (
                f"T={T}: expected {expected}, got {latent.shape[1]}"
            )

    def test_channels_preserved(self, tmp_path, PreEncodedLatentsDataset):
        """Cropping must not alter the channel dimension."""
        _make_latent_pair(tmp_path, time_steps=129, channels=64)
        ds = PreEncodedLatentsDataset(root_path=str(tmp_path))
        latent, _ = ds[0]
        assert latent.shape[0] == 64

    def test_latent_time_multiple_class_attribute(self, PreEncodedLatentsDataset):
        """_LATENT_TIME_MULTIPLE must be 16 (product of ADP factors [1,2,2,4])."""
        assert PreEncodedLatentsDataset._LATENT_TIME_MULTIPLE == 16, (
            "ADP U-Net factors=[1,2,2,4] require stride=16; "
            f"_LATENT_TIME_MULTIPLE={PreEncodedLatentsDataset._LATENT_TIME_MULTIPLE}"
        )
