from typing import Literal
import torch
from loguru import logger

def get_device() -> Literal["mps", "cuda", "cpu"]:
    """Select device for training, preferring MPS on Mac."""
    if torch.backends.mps.is_available():
        logger.info("Using MPS device for training.")
        return "mps"
    elif torch.cuda.is_available():
        logger.info("Using CUDA device for training.")
        return "cuda"
    else:
        logger.info("Using CPU device for training.")
        return "cpu"

def get_dtype(device: str) -> torch.dtype:
    """Select dtype for training."""
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif device == "cuda":
        return torch.float16
    elif device == "mps":
        return torch.bfloat16  # MPS only supports float32 reliably. bfloat16 or float16 is experimental.
    else:
        return torch.float32
