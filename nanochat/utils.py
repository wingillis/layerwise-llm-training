import gc
import hashlib
import json
from contextlib import nullcontext
from dataclasses import asdict as dataclass_asdict
from datetime import datetime
from filelock import FileLock
from functools import wraps
from pathlib import Path

import torch
from einops import einsum, rearrange
from nanochat.approximated_gpt import (
    WeightApproxGPT,
    WeightApproxGPTConfig,
    ApproxLinearSVD,
    ApproxLinearABBA,
    ApproxWeightMLP,
    ApproxWeightBlock,
    LinformerCausalSelfAttention,
)
from nanochat.common import print0, get_base_dir
from nanochat.gpt import GPT, GPTConfig


def disk_cache(key_fn, cache_name, cache_dir=None):
    """
    Decorator that caches function results to disk.

    Args:
        key_fn: A function that takes (*args, **kwargs) and returns a
                JSON-serializable dict representing the unique cache key.
        cache_name: Name for this cache (used in directory/file naming)
        cache_dir: Optional custom cache directory. Defaults to get_base_dir()

    The decorated function will have an additional `force_recalculate` kwarg
    that bypasses the cache when True.

    Returns:
        Decorated function that checks cache before executing.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Pop force_recalculate from kwargs (not passed to original function)
            force_recalculate = kwargs.pop("force_recalculate", False)

            # Determine cache directory
            base = Path(cache_dir) if cache_dir else Path(get_base_dir())
            cache_path = base / cache_name
            cache_path.mkdir(parents=True, exist_ok=True)

            # Generate cache key
            key_dict = key_fn(*args, **kwargs)
            key_json = json.dumps(key_dict, sort_keys=True)
            key_hash = hashlib.sha256(key_json.encode()).hexdigest()[:16]
            cache_file = cache_path / f"{key_hash}.json"
            lock_file = cache_path / f"{key_hash}.lock"

            # Check cache (with file locking for thread safety)
            if not force_recalculate:
                with FileLock(str(lock_file)):
                    if cache_file.exists():
                        try:
                            with open(cache_file, "r", encoding="utf-8") as f:
                                cached = json.load(f)
                            print0(
                                f"[disk_cache] Cache hit for {func.__name__} (key={key_hash})"
                            )
                            return cached["result"]
                        except (json.JSONDecodeError, KeyError):
                            # Corrupted cache file, will recalculate
                            pass

            # Cache miss or force_recalculate - run the function
            result = func(*args, **kwargs)

            # Save to cache (with file locking)
            with FileLock(str(lock_file)):
                cache_entry = {
                    "key": key_dict,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_entry, f, indent=2)

            return result

        return wrapper

    return decorator


def _test_batch_size_fit(
    batch_size: int,
    model: WeightApproxGPT,
    device: str,
    max_seq_len: int,
    freeze_every: int,
    embedding_lr: float,
    unembedding_lr: float,
    matrix_lr: float,
    weight_decay: float,
) -> bool:
    """
    Test if a specific batch size fits in memory using the provided model.

    This function tests batch size on an actual model without modifying its state.
    It saves and restores:
    - Model training/eval mode
    - Existing optimizer states (AdamW and Muon)
    - Gate level for incremental training models
    - All gradients (cleared before and after testing)

    The test performs a complete training step:
    1. Forward pass with mixed precision
    2. Backward pass (peak memory usage)
    3. Optimizer step (allocates optimizer state)

    Args:
        batch_size: Batch size to test
        model: WeightApproxGPT model to test with
        device: Target device for testing
        max_seq_len: Sequence length for test data
        freeze_every: Layer freezing parameter
        embedding_lr: Learning rate for embeddings (AdamW)
        unembedding_lr: Learning rate for lm_head (AdamW)
        matrix_lr: Learning rate for linear layers (Muon)
        weight_decay: Weight decay for AdamW

    Returns:
        True if batch size fits in memory, False if OOM occurs

    Raises:
        RuntimeError: For unexpected errors (not OOM)
    """
    try:
        # Clean slate
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Save model state
        was_training = model.training
        model.train()

        # Save existing optimizers if they exist
        existing_adamw = getattr(model, "adamw_optimizer", None)
        existing_muon = getattr(model, "muon_optimizer", None)

        # Save gate level for incremental training
        original_gate_level = getattr(model, "prev_gate_level", 0)

        # Clear any existing gradients
        model.zero_grad(set_to_none=True)

        # Setup fresh optimizers for testing
        test_optimizers = model.setup_optimizers(
            unembedding_lr=unembedding_lr,
            embedding_lr=embedding_lr,
            matrix_lr=matrix_lr,
            weight_decay=weight_decay,
        )

        # Create dummy data using model's vocab size from config
        vocab_size = model.config.vocab_size
        x = torch.randint(
            0,
            vocab_size,
            (batch_size, max_seq_len),
            dtype=torch.int32,
            device=device,
        )
        y = torch.randint(
            0,
            vocab_size,
            (batch_size, max_seq_len),
            dtype=torch.int64,
            device=device,
        )

        # Forward pass with autocast
        autocast_ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16)

        with autocast_ctx:
            loss = model(x, y, step=0)

        # Backward pass (peak memory usage)
        loss.backward()

        # Optimizer step (ensure state is allocated)
        for opt in test_optimizers:
            opt.step()

        # Cleanup gradients after test
        model.zero_grad(set_to_none=True)
        del x, y, loss

        # Restore original optimizers
        model.adamw_optimizer = existing_adamw
        model.muon_optimizer = existing_muon

        # Restore original gate level
        model.prev_gate_level = original_gate_level

        # Restore original training state
        if not was_training:
            model.eval()

        torch.cuda.empty_cache()
        gc.collect()

        return True

    except torch.cuda.OutOfMemoryError:
        # Expected failure - cleanup and return False
        # Restore original state even on failure
        print0("Hit torch OOM error")
        try:
            model.zero_grad(set_to_none=True)
            model.adamw_optimizer = existing_adamw
            model.muon_optimizer = existing_muon
            model.prev_gate_level = original_gate_level
            if not was_training:
                model.eval()
        except Exception:
            pass  # Ignore errors during cleanup

        return False

    except RuntimeError as e:
        print0(f"RuntimeError: {e}")
        # Check if it's an OOM error on other devices
        if "out of memory" in str(e).lower():
            # Cleanup and return False
            try:
                model.zero_grad(set_to_none=True)
                model.adamw_optimizer = existing_adamw
                model.muon_optimizer = existing_muon
                model.prev_gate_level = original_gate_level
                if not was_training:
                    model.eval()
            except Exception as e:
                print0(f"Failed to restore state after oom: {e}")
                pass
            return False

        # Restore state on unexpected error
        try:
            model.zero_grad(set_to_none=True)
            model.adamw_optimizer = existing_adamw
            model.muon_optimizer = existing_muon
            model.prev_gate_level = original_gate_level
            if not was_training:
                model.eval()
        except Exception as e:
            print0(f"Failed to restore state: {e}")
            pass
        raise e

    finally:
        torch.cuda.empty_cache()
        gc.collect()


def _config_to_dict(config):
    """Convert a config (dataclass or Pydantic model) to a dict."""
    if isinstance(config, dict):
        # Already a dict
        return config
    elif hasattr(config, "model_dump"):
        # Pydantic v2
        return config.model_dump()
    else:
        # Assume dataclass
        return dataclass_asdict(config)


def _batch_size_key_fn(
    model,
    max_seq_len,
    freeze_every=None,
    embedding_lr=0.2,
    unembedding_lr=0.004,
    matrix_lr=0.02,
    weight_decay=0.0,
    safety_factor=0.7,
    **kwargs,
):
    """Generate a cache key for find_optimal_batch_size from its arguments."""
    # Extract config and device info from model
    config = model.config
    device = model.get_device()
    device_type = "cuda" if device.type == "cuda" else "cpu"

    # For caching, we need to include enough information to ensure
    # the same batch size will work for the same setup
    return {
        "config_class": type(config).__name__,
        "config": _config_to_dict(config),
        "device_type": device_type,
        "max_seq_len": max_seq_len,
        "freeze_every": freeze_every,
        "embedding_lr": embedding_lr,
        "unembedding_lr": unembedding_lr,
        "matrix_lr": matrix_lr,
        "weight_decay": weight_decay,
        "safety_factor": safety_factor,
    }


@disk_cache(key_fn=_batch_size_key_fn, cache_name="batch_size_cache")
def find_optimal_batch_size(
    model: WeightApproxGPT,
    max_seq_len: int,
    max_batch_size: int = 128,
    min_batch_size: int = 1,
    safety_factor: float = 0.9,
    verbose: bool = True,
    freeze_every: int | None = None,
    embedding_lr: float = 0.2,
    unembedding_lr: float = 0.004,
    matrix_lr: float = 0.02,
    weight_decay: float = 0.0,
) -> int:
    """
    Find the optimal batch size for training a WeightApproxGPT model without OOM errors.

    Uses binary search to find the largest batch size that fits in memory,
    testing both forward and backward passes with full optimizer state allocation.

    Args:
        model: WeightApproxGPT model to test batch size on
        max_seq_len: Sequence length to test with
        max_batch_size: Upper bound for binary search (default: 128)
        min_batch_size: Lower bound for binary search (default: 1)
        safety_factor: Multiply result by this factor for safety margin (default: 0.9)
        verbose: Print progress during search (default: True)
        freeze_every: freeze_every parameter for LayeredGPT (default: large number)
        embedding_lr: Learning rate for embeddings (AdamW)
        unembedding_lr: Learning rate for lm_head (AdamW)
        matrix_lr: Learning rate for linear layers (Muon)
        weight_decay: Weight decay for AdamW

    Returns:
        Optimal batch size (int) with safety factor applied

    Example:
        >>> model = WeightApproxGPT(config)
        >>> optimal_bs = find_optimal_batch_size(model, max_seq_len=2048)
        >>> print(f"Use batch_size={optimal_bs}")

    Notes:
        - Only works reliably on CUDA (CPU/MPS may not have clear OOM boundaries)
        - Takes several minutes for large models (O(log n) tests)
        - Tests on the actual model state without modifying it
        - Returns min_batch_size if even that doesn't fit (check logs!)
    """
    # Validate model type
    if not isinstance(model, WeightApproxGPT):
        raise TypeError("model must be a WeightApproxGPT instance")

    # Extract device and device_type from model
    device = device_type = "cuda"
    config = model.config

    if freeze_every is None:
        freeze_every = 1000000  # Large number to effectively disable

    if verbose:
        print0(f"\n{'=' * 60}")
        print0("Finding optimal batch size for WeightApproxGPT")
        print0(f"{'=' * 60}")
        print0(
            f"Model: n_layer={len(model.transformer.h)}, n_embd={config.n_embd}, "
            f"n_head={config.n_head}, vocab_size={config.vocab_size}"
        )
        print0(f"Sequence length: {max_seq_len}")
        print0(f"Search range: [{min_batch_size}, {max_batch_size}]")
        print0(f"Safety factor: {safety_factor:.1%}")
        print0(f"Device: {device} ({device_type})")
        print0("")

    # Binary search
    low, high = min_batch_size, max_batch_size
    best_working_batch = min_batch_size
    tests_run = 0

    while low <= high:
        mid = (low + high) // 2
        tests_run += 1

        if verbose:
            print0(f"[Test {tests_run}] Trying batch_size={mid}...", end=" ")

        # Test with the provided model
        success = _test_batch_size_fit(
            batch_size=mid,
            model=model,
            device=device,
            max_seq_len=max_seq_len,
            freeze_every=freeze_every,
            embedding_lr=embedding_lr,
            unembedding_lr=unembedding_lr,
            matrix_lr=matrix_lr,
            weight_decay=weight_decay,
        )

        if success:
            best_working_batch = mid
            if verbose:
                print0("SUCCESS")
            low = mid + 1
        else:
            if verbose:
                print0("OOM")
            high = mid - 1

    # Apply safety factor
    recommended = int(best_working_batch * safety_factor)
    recommended = max(1, recommended)  # Ensure at least 1

    if verbose:
        print0("")
        print0(f"{'=' * 60}")
        print0(f"Results after {tests_run} tests:")
        print0(f"  Max working batch size: {best_working_batch}")
        print0(f"  Recommended (with {safety_factor:.0%} safety): {recommended}")

        if best_working_batch == min_batch_size:
            print0(f"")
            print0(f"WARNING: Only minimum batch size ({min_batch_size}) fits!")
            print0(f"         Consider reducing model size or sequence length.")

        print0(f"{'=' * 60}\n")

    torch.cuda.empty_cache()

    return recommended


def lr_multiplier_factory(warmup_ratio, warmdown_ratio, num_iterations, final_lr_frac):
    # Learning rate scheduler
    def get_lr_multiplier(it):
        warmup_iters = round(warmup_ratio * num_iterations)
        warmdown_iters = round(warmdown_ratio * num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        elif it <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * final_lr_frac

    return get_lr_multiplier


# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


def _convert_linear_weight(approx_linear) -> torch.Tensor:
    """
    Convert an approximated linear layer weight to dense form.

    Args:
        approx_linear: The approximated linear layer (ApproxLinearSVD or ApproxLinearABBA)

    Returns:
        Dense weight tensor of shape (out_features, in_features)
    """
    # Use the reconstruct_weight method from the approximation class
    return approx_linear.linear.reconstruct_weight()


def materialize_weights(
    approx_model: WeightApproxGPT,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    copy_weights: bool = True,
) -> GPT:
    """
    Convert a trained WeightApproxGPT model to a standard GPT model with dense weights.

    This function materializes all approximated weights (SVD, ABBA, Linformer) to their
    full dense form, creating a standard GPT model that can be used for inference
    or further training without any approximations.

    Args:
        approx_model: Trained WeightApproxGPT model with approximated weights
        device: Target device for the new model (defaults to approx_model's device)
        dtype: Target dtype for the new model (defaults to approx_model's dtype)
        copy_weights: Whether to copy weights (True) or move them (False)

    Returns:
        GPT model with reconstructed dense weights

    Raises:
        ValueError: If an unsupported approximation type is encountered
    """
    # Set model to eval mode
    approx_model.eval()

    with torch.no_grad():
        # Create standard GPT config from WeightApproxGPTConfig
        gpt_config = GPTConfig(
            sequence_len=approx_model.config.sequence_len,
            vocab_size=approx_model.config.vocab_size,
            n_layer=approx_model.config.n_layer,
            n_head=approx_model.config.n_head,
            n_kv_head=approx_model.config.n_kv_head,
            n_embd=approx_model.config.n_embd,
        )

        # Create new GPT model
        dense_model = GPT(gpt_config)
        dense_model.eval()

        # Set device and dtype
        if device is None:
            device = approx_model.get_device()
        if dtype is None:
            dtype = approx_model.transformer.wte.weight.dtype

        dense_model = dense_model.to(device=device, dtype=dtype)

        # Copy embeddings and lm_head weights directly
        if copy_weights:
            dense_model.transformer.wte.weight.data.copy_(
                approx_model.transformer.wte.weight.data
            )
            dense_model.lm_head.weight.data.copy_(approx_model.lm_head.weight.data)
        else:
            dense_model.transformer.wte.weight.data = (
                approx_model.transformer.wte.weight.data
            )
            dense_model.lm_head.weight.data = approx_model.lm_head.weight.data

        # Copy rotary embeddings and ensure they're in bfloat16
        if copy_weights:
            dense_model.cos.data.copy_(approx_model.cos.data)
            dense_model.sin.data.copy_(approx_model.sin.data)
        else:
            dense_model.cos.data = approx_model.cos.data
            dense_model.sin.data = approx_model.sin.data

        # Ensure rotary embeddings are in bfloat16 as required by GPT
        dense_model.cos = dense_model.cos.bfloat16()
        dense_model.sin = dense_model.sin.bfloat16()

        # Convert each layer
        for i, (approx_block, dense_block) in enumerate(
            zip(approx_model.transformer.h, dense_model.transformer.h)
        ):
            # Convert attention weights
            _convert_attention_weights(
                approx_block.attn, dense_block.attn, copy_weights
            )

            # Convert MLP weights
            _convert_mlp_weights(approx_block.mlp, dense_block.mlp, copy_weights)

    return dense_model


def _convert_attention_weights(approx_attn, dense_attn, copy_weights: bool = True):
    """Convert attention layer weights from approximated to dense form."""
    # Determine if we're dealing with Linformer or standard/approximated attention
    if isinstance(approx_attn, LinformerCausalSelfAttention):
        # Linformer case - copy the base weights (Q, K, V, proj)
        # The E and F projection matrices are ignored for the dense model
        if copy_weights:
            dense_attn.c_q.weight.data.copy_(approx_attn.c_q.weight.data)
            dense_attn.c_k.weight.data.copy_(approx_attn.c_k.weight.data)
            dense_attn.c_v.weight.data.copy_(approx_attn.c_v.weight.data)
            dense_attn.c_proj.weight.data.copy_(approx_attn.c_proj.weight.data)
        else:
            dense_attn.c_q.weight.data = approx_attn.c_q.weight.data
            dense_attn.c_k.weight.data = approx_attn.c_k.weight.data
            dense_attn.c_v.weight.data = approx_attn.c_v.weight.data
            dense_attn.c_proj.weight.data = approx_attn.c_proj.weight.data
    else:
        # Standard attention case - check if weights are approximated
        # For c_q (query projection)
        if hasattr(approx_attn.c_q, "linear"):
            # Approximated linear layer
            dense_attn.c_q.weight.data = _convert_linear_weight(approx_attn.c_q).to(
                device=dense_attn.c_q.weight.device, dtype=dense_attn.c_q.weight.dtype
            )
        else:
            # Standard linear layer
            if copy_weights:
                dense_attn.c_q.weight.data.copy_(approx_attn.c_q.weight.data)
            else:
                dense_attn.c_q.weight.data = approx_attn.c_q.weight.data

        # For c_k (key projection)
        if hasattr(approx_attn.c_k, "linear"):
            # Approximated linear layer
            dense_attn.c_k.weight.data = _convert_linear_weight(approx_attn.c_k).to(
                device=dense_attn.c_k.weight.device, dtype=dense_attn.c_k.weight.dtype
            )
        else:
            # Standard linear layer
            if copy_weights:
                dense_attn.c_k.weight.data.copy_(approx_attn.c_k.weight.data)
            else:
                dense_attn.c_k.weight.data = approx_attn.c_k.weight.data

        # For c_v (value projection)
        if hasattr(approx_attn.c_v, "linear"):
            # Approximated linear layer
            dense_attn.c_v.weight.data = _convert_linear_weight(approx_attn.c_v).to(
                device=dense_attn.c_v.weight.device, dtype=dense_attn.c_v.weight.dtype
            )
        else:
            # Standard linear layer
            if copy_weights:
                dense_attn.c_v.weight.data.copy_(approx_attn.c_v.weight.data)
            else:
                dense_attn.c_v.weight.data = approx_attn.c_v.weight.data

        # For c_proj (output projection)
        if hasattr(approx_attn.c_proj, "linear"):
            # Approximated linear layer
            dense_attn.c_proj.weight.data = _convert_linear_weight(
                approx_attn.c_proj
            ).to(
                device=dense_attn.c_proj.weight.device,
                dtype=dense_attn.c_proj.weight.dtype,
            )
        else:
            # Standard linear layer
            if copy_weights:
                dense_attn.c_proj.weight.data.copy_(approx_attn.c_proj.weight.data)
            else:
                dense_attn.c_proj.weight.data = approx_attn.c_proj.weight.data


def _convert_mlp_weights(approx_mlp, dense_mlp, copy_weights: bool = True):
    """Convert MLP layer weights from approximated to dense form."""
    # Determine if we're dealing with approximated or standard MLP
    if isinstance(approx_mlp, ApproxWeightMLP):
        # ApproxWeightMLP case
        dense_mlp.c_fc.weight.data = _convert_linear_weight(approx_mlp.c_fc).to(
            device=dense_mlp.c_fc.weight.device, dtype=dense_mlp.c_fc.weight.dtype
        )

        dense_mlp.c_proj.weight.data = _convert_linear_weight(approx_mlp.c_proj).to(
            device=dense_mlp.c_proj.weight.device, dtype=dense_mlp.c_proj.weight.dtype
        )
    else:
        # Standard MLP case - just copy weights directly
        if copy_weights:
            dense_mlp.c_fc.weight.data.copy_(approx_mlp.c_fc.weight.data)
            dense_mlp.c_proj.weight.data.copy_(approx_mlp.c_proj.weight.data)
        else:
            dense_mlp.c_fc.weight.data = approx_mlp.c_fc.weight.data
            dense_mlp.c_proj.weight.data = approx_mlp.c_proj.weight.data
