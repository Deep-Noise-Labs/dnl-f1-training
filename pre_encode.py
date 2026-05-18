import argparse
import gc
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.utils import copy_state_dict
from stable_audio_tools.training.clearml_tracking import (
    init_clearml_pre_encode_task,
    ClearMLPreEncodeCallback,
)


def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, model_half=False):
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))

    model.eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)

    print("Done loading model")

    return model, model_config


class StrictDimensionValidator:
    """Validates audio dimensions before encoding to catch issues early."""
    
    def __init__(
        self,
        expected_sample_size: int = 132300,  # 44100 Hz × 3s
        expected_channels: int = 2,
        expected_sample_rate: int = 44100,
        strict_mode: bool = True,
        validation_report_path: str = None
    ):
        self.expected_sample_size = expected_sample_size
        self.expected_channels = expected_channels
        self.expected_sample_rate = expected_sample_rate
        self.strict_mode = strict_mode
        self.validation_report_path = validation_report_path
        self.bad_files = []
        self.total_checked = 0
    
    def validate_batch(self, audio: torch.Tensor, metadata: list) -> tuple:
        """Validate a batch of audio samples.
        
        Returns:
            Tuple of (valid_mask, list of valid samples, list of invalid metadata)
        """
        self.total_checked += audio.shape[0]
        valid_indices = []
        invalid_entries = []
        
        for i, (sample, md) in enumerate(zip(audio, metadata)):
            issues = []
            
            # Check for NaN/Inf
            if torch.isnan(sample).any():
                issues.append("Contains NaN values")
            if torch.isinf(sample).any():
                issues.append("Contains Inf values")
            
            # Check sample length (allow 5% tolerance for rounding)
            sample_length = sample.shape[-1] if sample.ndim > 1 else len(sample)
            length_tolerance = int(self.expected_sample_size * 0.05)
            if abs(sample_length - self.expected_sample_size) > length_tolerance:
                issues.append(f"Sample length {sample_length} != expected {self.expected_sample_size}")
            
            # Check channels
            channels = sample.shape[0] if sample.ndim > 1 else 1
            if channels != self.expected_channels:
                issues.append(f"Channel count {channels} != expected {self.expected_channels}")
            
            # Check for all zeros (potential silent/corrupt audio)
            if torch.all(sample == 0):
                issues.append("Audio is all zeros")
            
            # Check for very low amplitude (might be silent)
            abs_max = torch.abs(sample).max().item()
            if abs_max < 1e-6:
                issues.append(f"Very low amplitude: max={abs_max:.2e}")
            
            if issues:
                if self.strict_mode:
                    invalid_entries.append({
                        "index": i,
                        "issues": issues,
                        "file_path": md.get("path", "unknown"),
                        "sample_shape": list(sample.shape),
                    })
                    if self.validation_report_path:
                        self.bad_files.append({
                            "path": md.get("path", "unknown"),
                            "issues": issues,
                            "shape": list(sample.shape),
                        })
                else:
                    # In non-strict mode, still log but include in batch
                    print(f"  Warning: Sample {i} has issues: {issues}")
                    valid_indices.append(i)
            else:
                valid_indices.append(i)
        
        if invalid_entries:
            print(f"  REJECTED {len(invalid_entries)} samples due to validation issues:")
            for entry in invalid_entries:
                print(f"    - Sample {entry['index']}: {entry['issues']} (path: {entry['file_path']})")
        
        # Create valid mask
        valid_mask = torch.zeros(audio.shape[0], dtype=torch.bool)
        for idx in valid_indices:
            valid_mask[idx] = True
        
        return valid_mask, valid_indices, invalid_entries
    
    def save_report(self):
        """Save validation report to file."""
        if self.validation_report_path and self.bad_files:
            report = {
                "total_checked": self.total_checked,
                "bad_files": self.bad_files,
                "expected_sample_size": self.expected_sample_size,
                "expected_channels": self.expected_channels,
                "expected_sample_rate": self.expected_sample_rate,
            }
            with open(self.validation_report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Saved validation report to {self.validation_report_path}")


class AsyncBatchWriter:
    """Non-blocking parallel file I/O writer for latents."""
    
    def __init__(self, num_threads: int = 4, flush_interval: int = 50):
        self.num_threads = num_threads
        self.flush_interval = flush_interval
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.pending_futures = []
        self.write_count = 0
    
    def write_batch(self, items: list):
        """Submit a batch of writes to the thread pool.
        
        Args:
            items: List of tuples (latent_path, latent_data, metadata_path, metadata)
        """
        for latent_path, latent_data, metadata_path, metadata in items:
            future = self.executor.submit(self._write_single, latent_path, latent_data, metadata_path, metadata)
            self.pending_futures.append(future)
        
        self.write_count += len(items)
        
        # Periodically wait for some futures to prevent memory buildup
        if self.write_count % self.flush_interval == 0:
            self._drain_futures()
    
    def _write_single(self, latent_path, latent_data, metadata_path, metadata):
        """Write a single latent and its metadata to disk."""
        try:
            # Save latent as numpy file
            with open(latent_path, "wb") as f:
                np.save(f, latent_data)
            
            # Save metadata to json file
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
        except Exception as e:
            print(f"Error writing {latent_path}: {e}")
    
    def _drain_futures(self):
        """Wait for completed futures to free memory."""
        completed = [f for f in self.pending_futures if f.done()]
        for future in completed:
            try:
                future.result()  # Raises any exceptions
            except Exception as e:
                print(f"Write error: {e}")
        self.pending_futures = [f for f in self.pending_futures if not f.done()]
    
    def wait(self):
        """Wait for all pending writes to complete."""
        self._drain_futures()
        for future in self.pending_futures:
            try:
                future.result()
            except Exception as e:
                print(f"Write error: {e}")
        self.pending_futures = []
        print(f"AsyncBatchWriter: Completed {self.write_count} writes")
    
    def __del__(self):
        self.wait()
        self.executor.shutdown(wait=False)


class PreEncodedLatentsInferenceWrapper(pl.LightningModule):
    def __init__(
        self, 
        model,
        output_path,
        is_discrete=False,
        model_half=False,
        model_config=None,
        dataset_config=None,
        sample_size=1920000,
        args_dict=None,
        strict_validate=True,
        num_io_threads=4,
        enable_profiling=False
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.output_path = Path(output_path)
        self.strict_validate = strict_validate
        self.enable_profiling = enable_profiling
        
        # Initialize validator if strict mode is enabled
        self.validator = None
        if strict_validate:
            validation_report_path = str(self.output_path / "bad_audio_files.json")
            self.validator = StrictDimensionValidator(
                expected_sample_size=args_dict.get('expected_sample_size', 132300),
                expected_channels=args_dict.get('expected_channels', 2),
                expected_sample_rate=args_dict.get('expected_sample_rate', 44100),
                strict_mode=True,
                validation_report_path=validation_report_path
            )
        
        # Initialize async batch writer
        self.writer = AsyncBatchWriter(num_threads=num_io_threads, flush_interval=50)
        self.batch_count = 0

    def prepare_data(self):
        # runs on rank 0
        self.output_path.mkdir(parents=True, exist_ok=True)
        details_path = self.output_path / "details.json"
        if not details_path.exists():  # Only save if it doesn't exist
            details = {
                "model_config": self.hparams.model_config,
                "dataset_config": self.hparams.dataset_config,
                "sample_size": self.hparams.sample_size,
                "args": self.hparams.args_dict
            }
            details_path.write_text(json.dumps(details))

    def setup(self, stage=None):
        # runs on each device
        process_dir = self.output_path / str(self.global_rank)
        process_dir.mkdir(parents=True, exist_ok=True)

    def validation_step(self, batch, batch_idx):
        if self.enable_profiling:
            batch_start = time.time()
        
        audio, metadata = batch

        if audio.ndim == 4 and audio.shape[0] == 1:
            audio = audio[0]

        # Memory management: only run gc every 10 batches instead of every batch
        self.batch_count += 1
        if self.batch_count % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Run strict dimension validation if enabled
        if self.validator is not None:
            valid_mask, valid_indices, invalid_entries = self.validator.validate_batch(audio, metadata)
            if len(valid_indices) == 0:
                print(f"  Batch {batch_idx}: All samples rejected by validator, skipping")
                return
            audio = audio[valid_mask]
            metadata = [metadata[i] for i in valid_indices]

        if self.hparams.model_half:
            audio = audio.to(torch.float16)

        with torch.no_grad():
            if not self.hparams.is_discrete:
                latents = self.model.encode(audio)
            else:
                _, info = self.model.encode(audio, return_info=True)
                latents = info[self.model.bottleneck.tokens_id]

        latents = latents.cpu().numpy()

        # Prepare batch of writes for async writer
        write_items = []
        for i, latent in enumerate(latents):
            latent_id = f"{self.global_rank:03d}{batch_idx:06d}{i:04d}"

            latent_path = self.output_path / str(self.global_rank) / f"{latent_id}.npy"

            md = metadata[i].copy() if isinstance(metadata[i], dict) else dict(metadata[i])
            padding_mask = F.interpolate(
                md["padding_mask"].unsqueeze(0).unsqueeze(1).float(),
                size=latent.shape[1],
                mode="nearest"
            ).squeeze().int()
            md["padding_mask"] = padding_mask.cpu().numpy().tolist()

            # Convert tensors in md to serializable types
            for k, v in list(md.items()):
                if isinstance(v, torch.Tensor):
                    md[k] = v.cpu().numpy().tolist()

            metadata_path = self.output_path / str(self.global_rank) / f"{latent_id}.json"
            write_items.append((str(latent_path), latent, str(metadata_path), md))
        
        # Submit batch write to async writer
        self.writer.write_batch(write_items)
        
        if self.enable_profiling:
            batch_time = time.time() - batch_start
            print(f"  Batch {batch_idx}: {len(write_items)} samples encoded in {batch_time:.2f}s")

    def configure_optimizers(self):
        return None
    
    def on_validation_end(self):
        # Flush all pending async writes
        print("Flushing pending writes...")
        self.writer.wait()
        
        # Save validation report if validator was used
        if self.validator is not None:
            self.validator.save_report()
            print(f"Validation summary: checked {self.validator.total_checked} samples, "
                  f"rejected {len(self.validator.bad_files)}")


def main(args):
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    model, model_config = load_model(
        model_config=model_config,
        model_ckpt_path=args.ckpt_path,
        model_half=args.model_half
    )

    data_loader = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=args.sample_size,
        audio_channels=model_config.get("audio_channels", 2)
    )

    pl_module = PreEncodedLatentsInferenceWrapper(
        model=model,
        output_path=args.output_path,
        is_discrete=args.is_discrete,
        model_half=args.model_half,
        model_config=args.model_config,
        dataset_config=args.dataset_config,
        sample_size=args.sample_size,
        args_dict=vars(args),
        strict_validate=not args.no_strict_validate,
        num_io_threads=args.num_io_threads,
        enable_profiling=args.enable_profiling
    )

    # ClearML: track the pre-encoding run (no-op if ClearML is not installed)
    clearml_task = init_clearml_pre_encode_task(args, model_config, dataset_config)
    clearml_pre_encode_callback = ClearMLPreEncodeCallback(
        task=clearml_task,
        total_batches=args.limit_batches,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        num_nodes = args.num_nodes,
        strategy=args.strategy,
        precision="bf16-mixed" if args.model_half else "32",
        max_steps=args.limit_batches if args.limit_batches else -1,
        logger=False,  # Disable logging since we're just doing inference
        enable_checkpointing=False,
        callbacks=[clearml_pre_encode_callback],
    )
    trainer.validate(pl_module, data_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode audio dataset to VAE latents using PyTorch Lightning')
    parser.add_argument('--model-config', type=str, help='Path to model config', required=False)
    parser.add_argument('--ckpt-path', type=str, help='Path to unwrapped autoencoder model checkpoint', required=False)
    parser.add_argument('--model-half', action='store_true', help='Whether to use half precision (bf16-mixed)')
    parser.add_argument('--dataset-config', type=str, help='Path to dataset config file', required=True)
    parser.add_argument('--output-path', type=str, help='Path to output folder', required=True)
    parser.add_argument('--batch-size', type=int, help='Batch size for encoding', default=32)
    parser.add_argument('--sample-size', type=int, help='Number of audio samples to pad/crop to (default: 132300 = 44100 Hz × 3s)', default=132300)
    parser.add_argument('--is-discrete', action='store_true', help='Whether the model is discrete')
    parser.add_argument('--num-nodes', type=int, help='Number of GPU nodes', default=1)
    parser.add_argument('--num-workers', type=int, help='Number of dataloader workers', default=8)
    parser.add_argument('--num-io-threads', type=int, help='Number of threads for async file I/O', default=4)
    parser.add_argument('--strategy', type=str, help='PyTorch Lightning strategy', default='auto')
    parser.add_argument('--limit-batches', type=int, help='Limit number of batches (optional)', default=None)
    parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset')
    
    # Validation arguments
    parser.add_argument('--no-strict-validate', action='store_true', help='Disable strict audio dimension validation')
    parser.add_argument('--expected-sample-size', type=int, default=132300, help='Expected audio sample size (default: 132300 = 44100 Hz × 3s)')
    parser.add_argument('--expected-channels', type=int, default=2, help='Expected audio channels (default: 2 for stereo)')
    parser.add_argument('--expected-sample-rate', type=int, default=44100, help='Expected audio sample rate (default: 44100)')
    
    # Profiling
    parser.add_argument('--enable-profiling', action='store_true', help='Enable batch timing profiling')
    
    args = parser.parse_args()
    main(args)