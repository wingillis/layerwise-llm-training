"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os
from pathlib import Path
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext
import gc

import wandb
import torch

from typing import Any
from tqdm import tqdm
from nanochat.approximated_gpt import WeightApproxGPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.utils import lr_multiplier_factory, get_muon_momentum, find_optimal_batch_size 
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model
from dotenv import load_dotenv

# -----------------------------------------------------------------------------

load_dotenv(override=True)
print_banner()

# -----------------------------------------------------------------------------
# User settings
from pydantic_settings import BaseSettings, SettingsConfigDict
import tomllib

class TrainSettings(BaseSettings):
    # wandb run name default ("dummy" is special - we won't log to wandb)
    run: str = "dummy"
    # Runtime
    # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
    device_type: str = ""
    # Model architecture
    # the depth of the Transformer model to train, rest of the kwargs are derived
    depth: int = 20
    # max context length
    max_seq_len: int = 2048
    # Training horizon. Only one of these 3 will be used, in this order of precedence.
    # explicit number of steps of the optimization (-1 = disable)
    num_iterations: int = -1
    # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
    target_flops: float = -1.0
    # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
    target_param_data_ratio: int = 20
    # Optimization
    # per-device batch size (set to not OOM)
    device_batch_size: int = 32
    # total desired batch size, in #tokens
    total_batch_size: int = 524288
    # learning rate for the embedding parameters (Adam)
    embedding_lr: float = 0.2
    # learning rate for the unembedding parameters (Adam)
    unembedding_lr: float = 0.004
    # weight decay for the embedding/unembedding parameters (Adam)
    weight_decay: float = 0.0
    # learning rate for the matrix parameters (Muon)
    matrix_lr: float = 0.02
    # gradient clipping value (0.0 = disabled)
    grad_clip: float = 1.0
    # ratio of iterations for LR warmup
    warmup_ratio: float = 0.0
    # ratio of iterations for LR warmdown
    warmdown_ratio: float = 0.2
    # final LR is this fraction of the initial LR
    final_lr_frac: float = 0.0
    # resume training from this step of the optimization (-1 = disable)
    resume_from_step: int = -1
    # Evaluation
    # every how many steps to evaluate the model for val bpb
    eval_every: int = 250
    # number of tokens to evaluate val loss on
    eval_tokens: int = 20*524288
    # every how many steps to evaluate the core metric (-1 = disable)
    core_metric_every: int = 2000
    # examples per task in estimating the core metric
    core_metric_max_per_task: int = 500
    # every how many steps to sample from the model
    sample_every: int = 2000
    # every how many steps to save model checkpoints (-1 = disable, and save only at the end of the run)
    save_every: int = -1
    # Output
    # optionally override the model tag for the output checkpoint directory name
    model_tag: str = ""

    reverse_train_order: bool = False

    model_config = SettingsConfigDict(env_prefix="TRAIN_")

    @classmethod
    def load_or_create(cls, config_path: str = "train_config.toml"):
        if Path(config_path).exists():
            with open(config_path, "rb") as f:
                config_data = tomllib.load(f)
            return cls(**config_data)
        else:
            # Generate default config file
            with open(config_path, "w") as f:
                f.write("# Training Configuration\n")
                settings = cls()

                for name, field in cls.model_fields.items():
                    value = getattr(settings, name)
                    if isinstance(value, str):
                        f.write(f'{name} = "{value}"\n')
                    else:
                        f.write(f'{name} = {value}\n')
            print(f"Created default configuration file at {config_path}")
            return settings

    def model_post_init(self, __context: Any) -> None:
        if self.device_type == "":
            self.device_type = autodetect_device_type()

import sys

updated_params = {}
if len(sys.argv) > 1:
    for item in sys.argv[1:]:
        k, v = item.split("=")
        k = k.replace("--", "").replace("-", "_")
        updated_params[k] = v
        print(f"Setting {k} to {v}")

settings = TrainSettings.load_or_create().model_copy(update=updated_params)

# Map settings back to local variables to minimize code changes below
device_type = settings.device_type
max_seq_len = settings.max_seq_len
num_iterations = settings.num_iterations
device_batch_size = settings.device_batch_size
total_batch_size = settings.total_batch_size
grad_clip = settings.grad_clip
resume_from_step = settings.resume_from_step
eval_every = settings.eval_every
eval_tokens = settings.eval_tokens
core_metric_every = settings.core_metric_every
core_metric_max_per_task = settings.core_metric_max_per_task
sample_every = settings.sample_every
save_every = settings.save_every
model_tag = settings.model_tag

user_config = settings.model_dump()
# -----------------------------------------------------------------------------

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = settings.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=settings.run, config=user_config)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = settings.depth
model_dim = settings.depth * 64 # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(1, (model_dim + 127) // 128) # head dim 128 (the division here is ceil div)
num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Initialize the Model

# Create a new model with random weights
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)

freeze_every = settings.num_iterations // settings.depth
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = WeightApproxGPT(model_config, freeze_every=freeze_every, reverse_train_order=settings.reverse_train_order)
model.to_empty(device=device)
model.init_weights()

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = Path(get_base_dir() or ".")
output_dirname = model_tag if model_tag else f"d{settings.depth}" # e.g. d12
checkpoint_dir = base_dir / "base_checkpoints" / output_dirname
resuming = resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
# model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or settings.target_param_data_ratio > 0 or settings.target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif settings.target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(settings.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif settings.target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = settings.target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(unembedding_lr=settings.unembedding_lr, embedding_lr=settings.embedding_lr, matrix_lr=settings.matrix_lr, weight_decay=settings.weight_decay)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data # free up the memory

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
tokens_dir = base_dir / "tokenized_data"
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state(device_batch_size, max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers
get_lr_multiplier = lr_multiplier_factory(warmup_ratio=settings.warmup_ratio, warmdown_ratio=settings.warmdown_ratio, num_iterations=num_iterations, final_lr_frac=settings.final_lr_frac)
# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

step = 0
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
total_training_time = 0 # total wall-clock time of training

if resuming:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# -----------------------------------------------------------------------------
# Training loop
pbar = tqdm(range(num_iterations), desc="training")
for mstep in pbar:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes, step)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            [opt.state_dict() for opt in optimizers], # optimizer states
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y, step=step)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # gradient clipping
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item() # GPU tensor -> CPU float (note: cpu-gpu sync point)
    # step the optimizers
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 2:
        total_training_time += dt # only count the time after the first few steps
    pf = { "loss": debiased_smooth_loss, "tok_per_sec": round(tok_per_sec), }
    if grad_clip_enabled:
        pf["grad_norm"] = round(grad_norm, 2)
    pbar.set_postfix(pf)

    if step % 10 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/gate_level": step // freeze_every,
        }
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        wandb_run.log(log_data)

    # state update
    step += 1

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": settings.warmup_ratio,
        "warmdown_ratio": settings.warmdown_ratio,
        "final_lr_frac": settings.final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
