import os
import time
import math
import pickle
from typing import Optional

import numpy as np
import torch
from torch import nn
from loguru import logger
import click
from pydantic import BaseModel, Field, ValidationError

from nanogpt.model import GPTConfig, GPT

class TrainConfig(BaseModel):
    """Configuration for GPT training."""
    out_dir: str = Field(default='out', description="Output directory for checkpoints")
    eval_interval: int = Field(default=2000, description="Interval for evaluation")
    log_interval: int = Field(default=1, description="Interval for logging")
    eval_iters: int = Field(default=200, description="Number of iterations for evaluation")
    eval_only: bool = Field(default=False, description="Run evaluation only")
    always_save_checkpoint: bool = Field(default=True, description="Always save checkpoint after eval")
    init_from: str = Field(default='scratch', description="Model initialization mode")
    dataset: str = Field(default='', description="Dataset name")
    gradient_accumulation_steps: int = Field(default=40, description="Gradient accumulation steps")
    batch_size: int = Field(default=12, description="Micro-batch size")
    block_size: int = Field(default=1024, description="Block size")
    n_layer: int = Field(default=12, description="Number of transformer layers")
    n_head: int = Field(default=12, description="Number of attention heads")
    n_embd: int = Field(default=768, description="Embedding dimension")
    dropout: float = Field(default=0.0, description="Dropout rate")
    bias: bool = Field(default=False, description="Use bias in LayerNorm/Linear")
    learning_rate: float = Field(default=6e-4, description="Max learning rate")
    max_iters: int = Field(default=600000, description="Total training iterations")
    weight_decay: float = Field(default=1e-1, description="Weight decay for optimizer")
    beta1: float = Field(default=0.9, description="AdamW beta1")
    beta2: float = Field(default=0.95, description="AdamW beta2")
    grad_clip: float = Field(default=1.0, description="Gradient clipping value")
    decay_lr: bool = Field(default=True, description="Use learning rate decay")
    warmup_iters: int = Field(default=2000, description="Warmup iterations")
    lr_decay_iters: int = Field(default=600000, description="LR decay iterations")
    min_lr: float = Field(default=6e-5, description="Minimum learning rate")
    compile: bool = Field(default=True, description="Use torch.compile for model")
    seed: int = Field(default=1337, description="Random seed")

def get_device() -> str:
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
        return torch.float32  # MPS only supports float32 reliably
    else:
        return torch.float32

def get_batch(data_dir: str, split: str, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a batch from memmapped dataset."""
    file_name = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(os.path.join(data_dir, file_name), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def estimate_loss(model: nn.Module, ctx, data_dir: str, eval_iters: int, batch_size: int, block_size: int, device: str) -> dict[str, float]:
    """Estimate train/val loss."""
    out = {}
    model.eval()
    with torch.no_grad():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(data_dir, split, batch_size, block_size, device)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it: int, cfg: TrainConfig) -> float:
    """Cosine learning rate decay with warmup."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

@click.command()
@click.option('--config', type=click.Path(exists=True), default=None, help='Path to config file (json/yaml)')
@click.option('--batch-size', type=int, default=None, help='Override batch size')
@click.option('--eval-only', is_flag=True, default=False, help='Run evaluation only')
def main(config, batch_size, eval_only):
    """
    Train GPT model on single device.
    """
    # Load config
    try:
        if config:
            import json
            with open(config, 'r') as f:
                cfg_dict = json.load(f)
            cfg = TrainConfig(**cfg_dict)
        else:
            cfg = TrainConfig()
    except ValidationError as e:
        logger.error(f"Config validation error: {e}")
        return

    if batch_size is not None:
        cfg.batch_size = batch_size
    if eval_only:
        cfg.eval_only = True

    logger.info(f"Training config: {cfg}")

    # Device and dtype
    device = get_device()
    dtype = get_dtype(device)
    torch.manual_seed(cfg.seed)
    data_dir = os.path.expanduser(os.path.join('~/Documents/GitHub/nanogpt/data', cfg.dataset))
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Attempt to derive vocab_size from dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size: Optional[int] = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta.get('vocab_size')
        logger.info(f"Found vocab_size = {meta_vocab_size} in {meta_path}")

    # Model config
    model_args = dict(
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        block_size=cfg.block_size,
        bias=cfg.bias,
        vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
        dropout=cfg.dropout,
        n_embeddings=cfg.n_embd,
        n_layers=cfg.n_layer,
    )
    logger.info(f"Model args: {model_args}")

    # Model init
    if cfg.init_from == 'scratch':
        logger.info("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif cfg.init_from == 'resume':
        logger.info(f"Resuming training from {cfg.out_dir}")
        ckpt_path = os.path.join(cfg.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        for k in model_args.keys():
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.load_state_dict(checkpoint['model'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    else:
        logger.error(f"Unknown init_from: {cfg.init_from}")
        return

    model.to(device)
    ctx = torch.autocast(device_type=device, dtype=dtype) if device != 'cpu' else torch.cpu.amp.autocast(dtype=dtype)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    iter_num = 0
    best_val_loss = float('inf')

    # Compile model if requested
    if cfg.compile and hasattr(torch, "compile"):
        logger.info("Compiling the model with torch.compile...")
        model = torch.compile(model)

    # Training loop
    X, Y = get_batch(data_dir, 'train', cfg.batch_size, cfg.block_size, device)
    t0 = time.time()
    running_loss = None

    while True:
        lr = get_lr(iter_num, cfg) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluation and checkpointing
        if iter_num % cfg.eval_interval == 0:
            losses = estimate_loss(model, ctx, data_dir, cfg.eval_iters, cfg.batch_size, cfg.block_size, device)
            logger.info(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': cfg.dict(),
                    }
                    ckpt_path = os.path.join(cfg.out_dir, 'ckpt.pt')
                    torch.save(checkpoint, ckpt_path)
                    logger.info(f"Saved checkpoint to {ckpt_path}")
        if iter_num == 0 and cfg.eval_only:
            break

        # Forward/backward/update
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            logits, loss = model(X, Y)
        loss.backward()
        if cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Prefetch next batch
        X, Y = get_batch(data_dir, 'train', cfg.batch_size, cfg.block_size, device)

        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.log_interval == 0:
            lossf = loss.item()
            running_loss = lossf if running_loss is None else 0.9 * running_loss + 0.1 * lossf
            logger.info(f"Iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

        iter_num += 1
        if iter_num > cfg.max_iters:
            logger.info("Reached max_iters, stopping training.")
            break

if __name__ == "__main__":
    main()
