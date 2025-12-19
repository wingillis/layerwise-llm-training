"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_efficient_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os
import time
from pathlib import Path
from dataclasses import asdict

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import tyro
import wandb
import torch
from tqdm import tqdm
from dotenv import load_dotenv

from nanochat.approximated_gpt import WeightApproxGPT, WeightApproxGPTConfig
from nanochat.dataloader import (
    tokenizing_distributed_data_loader,
    tokenizing_distributed_data_loader_with_state,
)
from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    DummyWandb,
    print_banner,
    get_base_dir,
)
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.utils import (
    lr_multiplier_factory,
    get_muon_momentum,
)
from nanochat.engine import Engine
from nanochat.settings import TrainSettings
from nanochat.gradient_monitor import GradientMonitor
from nanochat.weight_monitor import WeightMonitor
from scripts.base_eval import evaluate_model


get_max_memory = torch.cuda.max_memory_allocated
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def parse_settings(settings: TrainSettings):
    """Parse CLI arguments and load/create settings."""
    diff = set(settings.model_dump().items()) - set(
        TrainSettings().model_dump().items()
    )
    print0(f"Different settings from default: {diff}")
    updated_params = TrainSettings().load_or_create().model_copy(update=dict(diff))

    return updated_params


def setup_compute():
    """Initialize DDP/compute context.

    Returns:
        tuple: (ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, synchronize)
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init("cuda")
    synchronize = torch.cuda.synchronize

    return (
        ddp,
        ddp_rank,
        ddp_local_rank,
        ddp_world_size,
        device,
        synchronize,
    )


def setup_wandb(settings, master_process, user_config):
    """Initialize wandb or dummy wandb for logging."""
    use_dummy_wandb = settings.run == "dummy" or not master_process
    return (
        DummyWandb()
        if use_dummy_wandb
        else wandb.init(project="nanochat", name=settings.run, config=user_config)
    )


def setup_tokenizer(device):
    """Initialize tokenizer and get vocab info.

    Returns:
        tuple: (tokenizer, token_bytes, vocab_size)
    """
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")
    return (tokenizer, token_bytes, vocab_size)


def compute_model_config(
    settings,
    vocab_size,
    ddp_world_size,
    device_batch_size,
    total_batch_size,
):
    """Calculate model configuration and batch parameters.

    Returns:
        tuple: (model_config_kwargs, num_layers, model_dim, grad_accum_steps)
    """
    model_dim = settings.depth * 64  # aspect ratio 64
    num_heads = max(1, (model_dim + 127) // 128)  # head dim 128 (ceil div)
    num_kv_heads = num_heads  # default 1:1 GQA ratio

    print0(f"total num_layers: {settings.depth}")
    print0(f"model_dim: {model_dim}")
    print0(f"num_heads: {num_heads}")
    print0(f"num_kv_heads: {num_kv_heads}")

    # Calculate gradient accumulation
    tokens_per_fwdbwd = device_batch_size * settings.max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    assert total_batch_size % world_tokens_per_fwdbwd == 0
    grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

    print0(
        f"Tokens / micro-batch / rank: {device_batch_size} x {settings.max_seq_len} = {tokens_per_fwdbwd:,}"
    )
    print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
    print0(
        f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}"
    )

    model_config = WeightApproxGPTConfig(
        sequence_len=settings.max_seq_len,
        vocab_size=vocab_size,
        n_layer=settings.depth,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim,
        approx_type=settings.approx_type,
        approx_mlp_proj=settings.approx_mlp_proj,
        mlp_proj_rank=settings.mlp_proj_rank,
        approx_lm_head=settings.approx_lm_head,
        lm_head_rank=settings.lm_head_rank,
        build_by_layer=settings.build_by_layer,
        freeze_previous_weights=settings.freeze_previous_weights,
       )

    return model_config, grad_accum_steps


def create_model(
    model_config,
    device,
    checkpoint_dir,
    resume_from_step,
    ddp_rank,
    build_by_layer: bool,
):
    """Create model on meta device, move to device, init weights, and handle checkpoint loading.

    Returns:
        tuple: (model, orig_model, model_config, optimizer_data, meta_data, num_params)
    """
    with torch.device("meta"):
        model = WeightApproxGPT(
            model_config,
            freeze_every=None,
        )
    model.to_empty(device=device)
    model.init_weights()

    optimizer_data = None
    meta_data = None
    resuming = resume_from_step != -1

    if resuming:
        print0(f"Resuming optimization from step {resume_from_step}")
        model_data, optimizer_data, meta_data = load_checkpoint(
            checkpoint_dir, resume_from_step, device, load_optimizer=True, rank=ddp_rank
        )
        model.load_state_dict(model_data, strict=True, assign=True)
        del model_data

    orig_model = model
    num_params = sum(p.numel() for p in model.parameters())
    print0(f"Number of parameters: {num_params:,}")
    model = torch.compile(model, dynamic=build_by_layer)

    return (
        model,
        orig_model,
        optimizer_data,
        meta_data,
        num_params,
    )


def compute_training_iterations(settings, num_params, total_batch_size, num_iterations):
    """Calculate number of training iterations based on settings.

    Returns:
        tuple: (num_iterations, total_tokens)
    """
    assert num_iterations > 0 or settings.target_param_data_ratio > 0

    if num_iterations > 0:
        print0(f"Using user-provided number of iterations: {num_iterations:,}")
    elif settings.target_param_data_ratio > 0:
        target_tokens = settings.target_param_data_ratio * num_params
        num_iterations = target_tokens // total_batch_size
        print0(
            f"Calculated number of iterations from target data:param ratio: {num_iterations:,}"
        )
    else:
        raise ValueError("No training horizon specified")

    total_tokens = total_batch_size * num_iterations
    print0(f"Total number of training tokens: {total_tokens:,}")
    print0(
        f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}"
    )

    return (num_iterations, total_tokens)


def setup_optimizers(model, settings):
    """Create optimizers and load state if resuming.

    Returns:
        tuple: (adamw_optimizer, muon_optimizer)
    """
    optimizers = model.setup_optimizers(
        unembedding_lr=settings.unembedding_lr,
        embedding_lr=settings.embedding_lr,
        matrix_lr=settings.matrix_lr,
        weight_decay=settings.weight_decay,
    )

    return optimizers


def setup_dataloaders(device, meta_data, device_batch_size, max_seq_len):
    """Create train loader and val loader builder.

    Returns:
        tuple: (train_loader, build_val_loader, x, y, dataloader_state_dict)
    """
    dataloader_resume_state_dict = (
        None if meta_data is None else meta_data["dataloader_state_dict"]
    )

    train_loader = tokenizing_distributed_data_loader_with_state(
        device_batch_size,
        max_seq_len,
        split="train",
        device=device,
        resume_state_dict=dataloader_resume_state_dict,
    )

    def build_val_loader():
        return tokenizing_distributed_data_loader(
            device_batch_size, max_seq_len, split="val", device=device
        )

    x, y, dataloader_state_dict = next(train_loader)  # kick off first batch load

    return (train_loader, build_val_loader, x, y, dataloader_state_dict)


def setup_lr_scheduler(settings, num_iterations):
    """Create LR multiplier function."""
    return lr_multiplier_factory(
        warmup_ratio=settings.warmup_ratio,
        warmdown_ratio=settings.warmdown_ratio,
        num_iterations=num_iterations,
        final_lr_frac=settings.final_lr_frac,
    )


def init_loop_state(meta_data, resume_from_step):
    """Initialize or restore loop state variables.

    Returns:
        dict: Loop state with step, min_val_bpb, smooth_train_loss, total_training_time
    """
    state = {
        "step": 0,
        "min_val_bpb": float("inf"),
        "smooth_train_loss": 0,
        "total_training_time": 0,
    }

    resuming = resume_from_step != -1
    if resuming and meta_data is not None:
        state["step"] = meta_data["step"]
        loop_state = meta_data["loop_state"]
        state["min_val_bpb"] = loop_state["min_val_bpb"]
        state["smooth_train_loss"] = loop_state["smooth_train_loss"]
        state["total_training_time"] = loop_state["total_training_time"]

    return state


def evaluate_validation_bpb(
    model,
    build_val_loader,
    eval_batch_size,
    settings,
    token_bytes,
    step,
    ddp_world_size,
):
    """Evaluate validation bpb.

    Returns:
        tuple: (val_bpb, should_log)
    """
    with torch.no_grad():
        model.eval()
        val_loader = build_val_loader()
        eval_steps = settings.eval_tokens // (
            eval_batch_size * settings.max_seq_len * ddp_world_size
        )
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes, step)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
    model.train()
    return val_bpb


def train_loop(
    model,
    orig_model,
    optimizers,
    train_loader,
    build_val_loader,
    x,
    y,
    dataloader_state_dict,
    tokenizer,
    token_bytes,
    settings,
    user_config,
    checkpoint_dir,
    num_iterations,
    total_batch_size,
    grad_accum_steps,
    freeze_every,
    device,
    synchronize,
    wandb_run,
    master_process,
    ddp_rank,
    ddp_world_size,
    get_lr_multiplier,
    loop_state,
):
    """Main training loop."""

    device_batch_size = settings.device_batch_size
    # Use smaller batch size for evaluation since loss_reduction='none' requires more memory
    eval_batch_size = max(1, int(device_batch_size * 0.7))

    def build_val_loader():
        return tokenizing_distributed_data_loader(
            eval_batch_size, settings.max_seq_len, split="val", device=device
        )

    step = loop_state["step"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

    muon_optimizer = optimizers[1]
    val_bpb = float("inf")
    results = {}

    # Initialize gradient and weight monitors
    gradient_monitor = GradientMonitor(orig_model)
    weight_monitor = WeightMonitor(orig_model, log_frequency=settings.log_weights_every)

    pbar = tqdm(range(num_iterations), desc="training")
    for mstep in pbar:  # pyrefly: ignore
        last_step = step == num_iterations

        # Evaluate validation bpb
        if last_step or step % settings.eval_every == 0 and step > 0:
            val_bpb = evaluate_validation_bpb(
                model,
                build_val_loader,
                eval_batch_size,
                settings,
                token_bytes,
                step,
                ddp_world_size,
            )
            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb
            wandb_run.log(
                {
                    "step": step,
                    "total_training_time": total_training_time,
                    "val/bpb": val_bpb,
                }
            )

        # Evaluate CORE metric
        results = {}
        if settings.core_metric_every > 0 and (
            last_step or (step > 0 and step % settings.core_metric_every == 0)
        ):
            model.eval()
            with autocast_ctx:
                results = evaluate_model(
                    orig_model,
                    tokenizer,
                    device,
                    max_per_task=settings.core_metric_max_per_task,
                )
            print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
            wandb_run.log(
                {
                    "step": step,
                    "core_metric": results["core_metric"],
                    "centered_results": results["centered_results"],
                }
            )
            model.train()

        # Sample from model
        if master_process and (
            last_step or (step > 0 and step % settings.sample_every == 0)
        ):
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
            engine = Engine(orig_model, tokenizer)
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                with autocast_ctx:
                    sample, _ = engine.generate_batch(
                        tokens, num_samples=1, max_tokens=16, temperature=0
                    )
                print0(tokenizer.decode(sample[0]))
            model.train()

        # Save checkpoint
        if last_step or (
            step > 0
            and step != settings.resume_from_step
            and settings.save_every > 0
            and step % settings.save_every == 0
        ):
            save_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict(),
                [opt.state_dict() for opt in optimizers],
                {
                    "step": step,
                    "val_bpb": val_bpb,
                    "model_config": asdict(model.config),
                    "user_config": user_config,
                    "device_batch_size": device_batch_size,
                    "max_seq_len": settings.max_seq_len,
                    "dataloader_state_dict": dataloader_state_dict,
                    "loop_state": {
                        "min_val_bpb": min_val_bpb,
                        "smooth_train_loss": smooth_train_loss,
                        "total_training_time": total_training_time,
                    },
                },
                rank=ddp_rank,
            )

        if last_step:
            break

        # Training step
        synchronize()
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                if settings.build_by_layer:
                    loss = model(x, y, step=step)
                else:
                    loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, dataloader_state_dict = next(train_loader)

        # Collect per-layer gradients after backward pass
        layer_grad_data = {}
        if step % settings.log_gradients_every == 0:
            layer_grad_data = gradient_monitor.collect_grad_norms(step)

        # Collect per-layer weight statistics
        layer_weight_data = weight_monitor.collect_weight_stats(step)

        # Gradient clipping
        grad_clip_enabled = settings.grad_clip > 0.0
        if grad_clip_enabled:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                orig_model.parameters(), settings.grad_clip
            )
            grad_norm = grad_norm_tensor.item()

        # Step optimizers
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

        # Logging
        ema_beta = 0.9
        smooth_train_loss = (
            ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
        )
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        tok_per_sec = int(total_batch_size / dt)

        if step > 2:
            total_training_time += dt

        pf = {"loss": debiased_smooth_loss, "tok_per_sec": round(tok_per_sec)}
        if grad_clip_enabled:
            pf["grad_norm"] = round(grad_norm, 2)

        pbar.set_postfix(pf)  # pyrefly: ignore

        if step % 10 == 0:
            log_data = {
                "step": step,
                "total_training_time": total_training_time,
                "train/loss": debiased_smooth_loss,
                "train/lrm": lrm,
                "train/dt": dt,
                "train/tok_per_sec": tok_per_sec,
                "train/gate_level": step // freeze_every,
            }
            if grad_clip_enabled:
                log_data["train/grad_norm"] = grad_norm
            # Add per-layer gradient data if collected
            if layer_grad_data:
                log_data.update(layer_grad_data)
            # Add per-layer weight data if collected
            if layer_weight_data:
                log_data.update(layer_weight_data)
            wandb_run.log(log_data)

        step += 1

    pbar.close()  # pyrefly: ignore

    return {
        "min_val_bpb": min_val_bpb,
        "val_bpb": val_bpb,
        "results": results,
        "total_training_time": total_training_time,
    }


def log_final_results(
    total_training_time,
    min_val_bpb,
    val_bpb,
    results,
    user_config,
    num_params,
    num_iterations,
    total_tokens,
    total_batch_size,
    ddp_world_size,
    settings,
    mfu,
    wandb_run,
):
    """Print final stats, log to report, and cleanup."""
    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {total_training_time / 60:.2f}m")
    print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

    from nanochat.report import get_report

    get_report().log(
        section="Base model training",
        data=[
            user_config,
            {
                "Number of parameters": num_params,
                "Calculated number of iterations": num_iterations,
                "Number of training tokens": total_tokens,
                "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
                "DDP world size": ddp_world_size,
                "warmup_ratio": settings.warmup_ratio,
                "warmdown_ratio": settings.warmdown_ratio,
                "final_lr_frac": settings.final_lr_frac,
            },
            {
                "Minimum validation bpb": min_val_bpb,
                "Final validation bpb": val_bpb,
                "CORE metric estimate": results.get("core_metric", None),
                "MFU %": f"{mfu:.2f}%",
                "Total training time": f"{total_training_time / 60:.2f}m",
                "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
            },
        ],
    )

    wandb_run.finish()


def main(settings: TrainSettings):
    load_dotenv(override=True)
    # print_banner()

    # Parse settings
    settings = parse_settings(settings)
    user_config = settings
    print0(settings)

    # Extract commonly used settings
    num_iterations = settings.num_iterations
    device_batch_size = settings.device_batch_size
    total_batch_size = settings.total_batch_size
    resume_from_step = settings.resume_from_step
    model_tag = settings.model_tag

    # Setup compute
    (
        ddp,
        ddp_rank,
        ddp_local_rank,
        ddp_world_size,
        device,
        synchronize,
    ) = setup_compute()
    master_process = ddp_rank == 0

    # Setup wandb
    wandb_run = setup_wandb(settings, master_process, user_config)

    # Setup tokenizer
    tokenizer, token_bytes, vocab_size = setup_tokenizer(device)

    # Compute model config
    model_config, grad_accum_steps = compute_model_config(
        settings,
        vocab_size,
        ddp_world_size,
        device_batch_size,
        total_batch_size,
    )

    # Setup checkpoint directory
    base_dir = Path(get_base_dir() or ".")
    output_dirname = model_tag if model_tag else f"d{settings.depth}"
    checkpoint_dir = base_dir / "base_checkpoints" / output_dirname

    # Create model
    (
        model,
        orig_model,
        optimizer_data,
        meta_data,
        num_params,
    ) = create_model(
        model_config,
        device,
        checkpoint_dir,
        resume_from_step,
        ddp_rank,
        build_by_layer=settings.build_by_layer,
    )
    n_params = sum(p.numel() for p in model.transformer.h.parameters())
    print0(f"N params in transformer: {n_params:,}")
    print0(f"N params in lm head: {sum(p.numel() for p in model.lm_head.parameters()):,}")
    print0(f"N params in embedding: {sum(p.numel() for p in model.transformer.wte.parameters()):,}")
    print0(f"N params: {num_params:,}")

    # Compute training iterations
    num_iterations, total_tokens = compute_training_iterations(
        settings, num_params, total_batch_size, num_iterations
    )
    print0(f"Updating num_iterations to {num_iterations} in settings")
    settings = settings.model_copy(update={"num_iterations": num_iterations})
    # explicitly set here after we compute num_iterations
    model.freeze_every = num_iterations // settings.depth

    # Setup optimizers
    optimizers = setup_optimizers(model, settings)

    # Setup dataloaders
    (train_loader, build_val_loader, x, y, dataloader_state_dict) = setup_dataloaders(
        device, meta_data, device_batch_size, settings.max_seq_len
    )

    # Setup LR scheduler
    get_lr_multiplier = setup_lr_scheduler(settings, num_iterations)

    # Initialize loop state
    loop_state = init_loop_state(meta_data, resume_from_step)

    # Calculate freeze_every for train loop
    freeze_every = num_iterations // settings.depth
    print0(f"Adding layer every {freeze_every} iterations")

    # Run training loop
    train_results = train_loop(
        model,
        orig_model,
        optimizers,
        train_loader,
        build_val_loader,
        x,
        y,
        dataloader_state_dict,
        tokenizer,
        token_bytes,
        settings,
        user_config,
        checkpoint_dir,
        num_iterations,
        total_batch_size,
        grad_accum_steps,
        freeze_every,
        device,
        synchronize,
        wandb_run,
        master_process,
        ddp_rank,
        ddp_world_size,
        get_lr_multiplier,
        loop_state,
    )

    # Log final results
    log_final_results(
        train_results["total_training_time"],
        train_results["min_val_bpb"],
        train_results["val_bpb"],
        train_results["results"],
        user_config,
        num_params,
        num_iterations,
        total_tokens,
        total_batch_size,
        ddp_world_size,
        settings,
        train_results["mfu"],
        wandb_run,
    )

    # Cleanup
    compute_cleanup()


if __name__ == "__main__":
    config = tyro.cli(TrainSettings)
    main(config)
