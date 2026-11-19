"""Compute device selection with MPS fallback enabled prior to importing torch."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info() -> dict[str, object]:
    device = get_device()
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(idx)
    elif device.type == "mps":
        device_name = "Apple Silicon MPS"
    else:
        device_name = "CPU"

    return {
        "device": device.type,
        "device_name": device_name,
        "torch_version": torch.__version__,
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
    }
