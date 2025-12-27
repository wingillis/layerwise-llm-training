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
from dataclasses import asdict, dataclass

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


# ============================================================================
# State Data Classes
# ============================================================================


@dataclass
class ComputeState:
    """DDP/compute context state."""
    ddp: bool
    ddp_rank: int
    ddp_local_rank: int
    ddp_world_size: int
    device: torch.device
    synchronize: callable
    master_process: bool
    wandb_run: object


@dataclass
class TokenizerState:
    """Tokenizer-related state."""
    tokenizer: object
    token_bytes: object
    vocab_size: int


@dataclass
class ModelContext:
    """Model and optimizer state."""
    model: WeightApproxGPT
    orig_model: WeightApproxGPT
    config: WeightApproxGPTConfig
    num_params: int
    optimizers: tuple
    meta_data: dict | None
    freeze_every: int


@dataclass
class DataContext:
    """Data loading state."""
    train_loader: object
    build_val_loader: callable
    x: torch.Tensor
    y: torch.Tensor
    dataloader_state_dict: dict


@dataclass
class TrainingLoopState:
    """Mutable training loop state."""
    step: int
    min_val_bpb: float
    smooth_train_loss: float
    total_training_time: float
    val_bpb: float = float("inf")
    results: dict | None = None

    def to_dict(self) -> dict:
        """Convert to dict for checkpoint serialization."""
        return {
            "step": self.step,
            "min_val_bpb": self.min_val_bpb,
            "smooth_train_loss": self.smooth_train_loss,
            "total_training_time": self.total_training_time,
        }

    @classmethod
    def from_checkpoint(cls, meta_data: dict, resume_from_step: int) -> "TrainingLoopState":
        """Create from checkpoint metadata."""
        if resume_from_step != -1 and meta_data is not None:
            loop_state = meta_data["loop_state"]
            return cls(
                step=meta_data["step"],
                min_val_bpb=loop_state["min_val_bpb"],
                smooth_train_loss=loop_state["smooth_train_loss"],
                total_training_time=loop_state["total_training_time"],
            )
        return cls(
            step=0,
            min_val_bpb=float("inf"),
            smooth_train_loss=0.0,
            total_training_time=0.0,
        )


# ============================================================================
# Setup Functions
# ============================================================================


def parse_settings(settings: TrainSettings):
    """Parse CLI arguments and load/create settings."""
    diff = set(settings.model_dump().items()) - set(
        TrainSettings().model_dump().items()
    )
    print0(f"Different settings from default: {diff}")
    updated_params = TrainSettings().load_or_create().model_copy(update=dict(diff))

    return updated_params


def setup_environment(settings, user_config):
    """Initialize DDP/compute context and wandb.

    Returns:
        ComputeState: DDP/compute context including wandb run
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init("cuda")
    synchronize = torch.cuda.synchronize
    master_process = ddp_rank == 0

    use_dummy_wandb = settings.run == "dummy" or not master_process
    wandb_run = (
        DummyWandb()
        if use_dummy_wandb
        else wandb.init(project="nanochat", name=settings.run, config=user_config)
    )

    return ComputeState(
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_local_rank=ddp_local_rank,
        ddp_world_size=ddp_world_size,
        device=device,
        synchronize=synchronize,
        master_process=master_process,
        wandb_run=wandb_run,
    )


def setup_tokenizer(device):
    """Initialize tokenizer and get vocab info.

    Returns:
        TokenizerState: Tokenizer, token bytes, and vocab size
    """
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")
    return TokenizerState(
        tokenizer=tokenizer,
        token_bytes=token_bytes,
        vocab_size=vocab_size,
    )


def setup_model(
    settings,
    tokenizer_state,
    compute_state,
    device_batch_size,
    total_batch_size,
    checkpoint_dir,
):
    """Configure, create, and setup model with optimizers.

    Returns:
        tuple: (ModelContext, grad_accum_steps)
    """
    # Calculate model configuration
    model_dim = settings.depth * 64  # aspect ratio 64
    num_heads = max(1, (model_dim + 127) // 128)  # head dim 128 (ceil div)
    num_kv_heads = num_heads  # default 1:1 GQA ratio

    print0(f"total num_layers: {settings.depth}")
    print0(f"model_dim: {model_dim}")
    print0(f"num_heads: {num_heads}")
    print0(f"num_kv_heads: {num_kv_heads}")

    # Calculate gradient accumulation
    tokens_per_fwdbwd = device_batch_size * settings.max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * compute_state.ddp_world_size
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
        vocab_size=tokenizer_state.vocab_size,
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

    # Create model
    with torch.device("meta"):
        model = WeightApproxGPT(
            model_config,
            freeze_every=None,
        )
    model.to_empty(device=compute_state.device)
    model.init_weights()

    optimizer_data = None
    meta_data = None
    resuming = settings.resume_from_step != -1

    if resuming:
        print0(f"Resuming optimization from step {settings.resume_from_step}")
        model_data, optimizer_data, meta_data = load_checkpoint(
            checkpoint_dir, settings.resume_from_step, compute_state.device, load_optimizer=True, rank=compute_state.ddp_rank
        )
        model.load_state_dict(model_data, strict=True, assign=True)
        del model_data

    orig_model = model
    num_params = sum(p.numel() for p in model.parameters())
    print0(f"Number of parameters: {num_params:,}")
    model = torch.compile(model, dynamic=settings.build_by_layer, mode="default")

    # Setup optimizers
    optimizers = model.setup_optimizers(
        unembedding_lr=settings.unembedding_lr,
        embedding_lr=settings.embedding_lr,
        matrix_lr=settings.matrix_lr,
        weight_decay=settings.weight_decay,
    )

    # freeze_every will be set later in main() after num_iterations is computed
    model_context = ModelContext(
        model=model,
        orig_model=orig_model,
        config=model_config,
        num_params=num_params,
        optimizers=optimizers,
        meta_data=meta_data,
        freeze_every=0,  # Will be set later
    )

    return model_context, grad_accum_steps


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


def setup_dataloaders(device, meta_data, device_batch_size, max_seq_len):
    """Create train loader and val loader builder.

    Returns:
        DataContext: Data loaders and first batch
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

    return DataContext(
        train_loader=train_loader,
        build_val_loader=build_val_loader,
        x=x,
        y=y,
        dataloader_state_dict=dataloader_state_dict,
    )


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
        TrainingLoopState: Loop state with step, min_val_bpb, smooth_train_loss, total_training_time
    """
    return TrainingLoopState.from_checkpoint(meta_data, resume_from_step)


def train_loop(
    model_context: ModelContext,
    data_context: DataContext,
    tokenizer_state: TokenizerState,
    compute_state: ComputeState,
    loop_state: TrainingLoopState,
    settings,
    user_config,
    checkpoint_dir,
    num_iterations,
    total_batch_size,
    grad_accum_steps,
    get_lr_multiplier,
):
    """Main training loop."""

    device_batch_size = settings.device_batch_size
    # Use smaller batch size for evaluation since loss_reduction='none' requires more memory
    eval_batch_size = max(1, int(device_batch_size * 0.7))

    def build_val_loader():
        return tokenizing_distributed_data_loader(
            eval_batch_size, settings.max_seq_len, split="val", device=compute_state.device
        )

    step = loop_state.step
    min_val_bpb = loop_state.min_val_bpb
    smooth_train_loss = loop_state.smooth_train_loss
    total_training_time = loop_state.total_training_time

    muon_optimizer = model_context.optimizers[1]
    val_bpb = float("inf")
    results = {}

    # Initialize gradient and weight monitors
    gradient_monitor = GradientMonitor(model_context.orig_model)
    weight_monitor = WeightMonitor(model_context.orig_model, log_frequency=settings.log_weights_every)

    pbar = tqdm(range(num_iterations), desc="training")
    for mstep in pbar:  # pyrefly: ignore
        last_step = step == num_iterations

        # Evaluate validation bpb (inlined from evaluate_validation_bpb)
        if last_step or step % settings.eval_every == 0 and step > 0:
            with torch.no_grad():
                model_context.model.eval()
                val_loader = build_val_loader()
                eval_steps = settings.eval_tokens // (
                    eval_batch_size * settings.max_seq_len * compute_state.ddp_world_size
                )
                with autocast_ctx:
                    val_bpb = evaluate_bpb(model_context.model, val_loader, eval_steps, tokenizer_state.token_bytes, step)
                print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
            model_context.model.train()

            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb
            compute_state.wandb_run.log(
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
            model_context.model.eval()
            with autocast_ctx:
                results = evaluate_model(
                    model_context.orig_model,
                    tokenizer_state.tokenizer,
                    compute_state.device,
                    max_per_task=settings.core_metric_max_per_task,
                )
            print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
            compute_state.wandb_run.log(
                {
                    "step": step,
                    "core_metric": results["core_metric"],
                    "centered_results": results["centered_results"],
                }
            )
            model_context.model.train()

        # Sample from model
        if compute_state.master_process and (
            last_step or (step > 0 and step % settings.sample_every == 0)
        ):
            model_context.model.eval()
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            engine = Engine(model_context.orig_model, tokenizer_state.tokenizer)
            for prompt in prompts:
                tokens = tokenizer_state.tokenizer(prompt, prepend="<|bos|>")
                with autocast_ctx:
                    sample, _ = engine.generate_batch(
                        tokens, num_samples=1, max_tokens=16, temperature=0
                    )
                print0(tokenizer_state.tokenizer.decode(sample[0]))
            model_context.model.train()

        # Save checkpoint
        if last_step or (
            step > 0
            and step != settings.resume_from_step
            and settings.save_every > 0
            and step % settings.save_every == 0
        ):
            # Create updated loop state for checkpoint
            checkpoint_loop_state = TrainingLoopState(
                step=step,
                min_val_bpb=min_val_bpb,
                smooth_train_loss=smooth_train_loss,
                total_training_time=total_training_time,
            )
            save_checkpoint(
                checkpoint_dir,
                step,
                model_context.orig_model.state_dict(),
                [opt.state_dict() for opt in model_context.optimizers],
                {
                    "step": step,
                    "val_bpb": val_bpb,
                    "model_config": asdict(model_context.config),
                    "user_config": user_config,
                    "device_batch_size": device_batch_size,
                    "max_seq_len": settings.max_seq_len,
                    "dataloader_state_dict": data_context.dataloader_state_dict,
                    "loop_state": checkpoint_loop_state.to_dict(),
                },
                rank=compute_state.ddp_rank,
            )

        if last_step:
            break

        # Training step
        compute_state.synchronize()
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                if settings.build_by_layer:
                    loss = model_context.model(data_context.x, data_context.y, step=step)
                else:
                    loss = model_context.model(data_context.x, data_context.y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            data_context.x, data_context.y, data_context.dataloader_state_dict = next(data_context.train_loader)

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
                model_context.orig_model.parameters(), settings.grad_clip
            )
            grad_norm = grad_norm_tensor.item()

        # Step optimizers
        lrm = get_lr_multiplier(step)
        for opt in model_context.optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        muon_momentum = get_muon_momentum(step)
        for group in muon_optimizer.param_groups:
            group["momentum"] = muon_momentum
        for opt in model_context.optimizers:
            opt.step()
        model_context.model.zero_grad(set_to_none=True)
        compute_state.synchronize()
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
                "train/gate_level": step // model_context.freeze_every,
            }
            if grad_clip_enabled:
                log_data["train/grad_norm"] = grad_norm
            # Add per-layer gradient data if collected
            if layer_grad_data:
                log_data.update(layer_grad_data)
            # Add per-layer weight data if collected
            if layer_weight_data:
                log_data.update(layer_weight_data)
            compute_state.wandb_run.log(log_data)

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

    # Setup environment (compute + wandb)
    compute_state = setup_environment(settings, user_config)

    # Setup tokenizer
    tokenizer_state = setup_tokenizer(compute_state.device)

    # Setup checkpoint directory
    base_dir = Path(get_base_dir() or ".")
    output_dirname = model_tag if model_tag else f"d{settings.depth}"
    checkpoint_dir = base_dir / "base_checkpoints" / output_dirname

    # Setup model (config + model + optimizers)
    model_context, grad_accum_steps = setup_model(
        settings,
        tokenizer_state,
        compute_state,
        device_batch_size,
        total_batch_size,
        checkpoint_dir,
    )

    # Print parameter breakdown
    n_params = sum(p.numel() for p in model_context.model.transformer.h.parameters())
    print0(f"N params in transformer: {n_params:,}")
    print0(f"N params in lm head: {sum(p.numel() for p in model_context.model.lm_head.parameters()):,}")
    print0(f"N params in embedding: {sum(p.numel() for p in model_context.model.transformer.wte.parameters()):,}")
    print0(f"N params: {model_context.num_params:,}")

    # Compute training iterations
    num_iterations, total_tokens = compute_training_iterations(
        settings, model_context.num_params, total_batch_size, num_iterations
    )
    print0(f"Updating num_iterations to {num_iterations} in settings")
    settings = settings.model_copy(update={"num_iterations": num_iterations})

    # Calculate and set freeze_every
    freeze_every = num_iterations // settings.depth
    model_context.freeze_every = freeze_every
    model_context.model.freeze_every = freeze_every
    print0(f"Adding layer every {freeze_every} iterations")

    # Setup dataloaders
    data_context = setup_dataloaders(
        compute_state.device, model_context.meta_data, device_batch_size, settings.max_seq_len
    )

    # Setup LR scheduler
    get_lr_multiplier = setup_lr_scheduler(settings, num_iterations)

    # Initialize loop state
    loop_state = init_loop_state(model_context.meta_data, resume_from_step)

    # Run training loop
    train_results = train_loop(
        model_context,
        data_context,
        tokenizer_state,
        compute_state,
        loop_state,
        settings,
        user_config,
        checkpoint_dir,
        num_iterations,
        total_batch_size,
        grad_accum_steps,
        get_lr_multiplier,
    )

    # Log final results
    log_final_results(
        train_results["total_training_time"],
        train_results["min_val_bpb"],
        train_results["val_bpb"],
        train_results["results"],
        user_config,
        model_context.num_params,
        num_iterations,
        total_tokens,
        total_batch_size,
        compute_state.ddp_world_size,
        settings,
        train_results.get("mfu", 0.0),
        compute_state.wandb_run,
    )

    # Cleanup
    compute_cleanup()


if __name__ == "__main__":
    config = tyro.cli(TrainSettings)
    main(config)
