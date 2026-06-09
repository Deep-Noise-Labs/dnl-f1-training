"""Load conditioning fields from WAV sibling JSON sidecars (no torch dependency)."""

from __future__ import annotations

import json
import os

_SIDECAR_CONDITIONING_KEYS = ("text", "prompt", "seconds_start", "seconds_total")


def merge_audio_json_sidecar(info: dict, audio_path: str) -> dict:
    """Merge CLAP/duration conditioning from a WAV sibling ``.json`` sidecar into *info*."""
    json_path = audio_path.rsplit(".", 1)[0] + ".json"
    if not os.path.isfile(json_path):
        return info
    with open(json_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)
    for key in _SIDECAR_CONDITIONING_KEYS:
        if key in sidecar:
            info.setdefault(key, sidecar[key])
    if "prompt" not in info and "text" in info:
        info["prompt"] = info["text"]
    elif "prompt" not in info and "text" in sidecar:
        info["prompt"] = sidecar["text"]
    return info
