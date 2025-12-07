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
from nanochat.approximated_gpt import WeightApproxGPT
from nanochat.common import print0, get_base_dir


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
            force_recalculate = kwargs.pop('force_recalculate', False)

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
                            with open(cache_file, 'r', encoding='utf-8') as f:
                                cached = json.load(f)
                            print0(f"[disk_cache] Cache hit for {func.__name__} (key={key_hash})")
                            return cached['result']
                        except (json.JSONDecodeError, KeyError):
                            # Corrupted cache file, will recalculate
                            pass

            # Cache miss or force_recalculate - run the function
            result = func(*args, **kwargs)

            # Save to cache (with file locking)
            with FileLock(str(lock_file)):
                cache_entry = {
                    'key': key_dict,
                    'result': result,
                    'timestamp': datetime.now().isoformat(),
                }
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_entry, f, indent=2)

            return result

        return wrapper
    return decorator


def _test_batch_size_fit(
    batch_size: int,
    config,  # GPTConfig
    device: torch.device,
    device_type: str,
    max_seq_len: int,
    freeze_every: int,
    reverse_train_order: bool,
    embedding_lr: float,
    unembedding_lr: float,
    matrix_lr: float,
    weight_decay: float,
) -> bool:
    """
    Test if a specific batch size fits in memory with full training setup.

    Returns True if successful, False if OOM occurs.
    """
    try:
        # Clean slate
        if device_type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Create model (same pattern as main training)
        with torch.device("meta"):
            test_model = WeightApproxGPT(config, freeze_every=freeze_every,
                                   reverse_train_order=reverse_train_order)
        test_model.to_empty(device=device)
        test_model.init_weights()
        test_model.train()

        # Setup optimizers (allocates optimizer state memory)
        test_optimizers = test_model.setup_optimizers(
            unembedding_lr=unembedding_lr,
            embedding_lr=embedding_lr,
            matrix_lr=matrix_lr,
            weight_decay=weight_decay
        )

        # Create dummy data
        x = torch.randint(0, config.vocab_size, (batch_size, max_seq_len),
                         dtype=torch.int32, device=device)
        y = torch.randint(0, config.vocab_size, (batch_size, max_seq_len),
                         dtype=torch.int64, device=device)

        # Forward pass with autocast
        autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) \
                       if device_type == "cuda" else nullcontext()
        with autocast_ctx:
            loss = test_model(x, y, step=0)

        # Backward pass (peak memory usage)
        loss.backward()

        # Optimizer step (ensure state is allocated)
        for opt in test_optimizers:
            opt.step()

        # Successful - cleanup
        test_model.zero_grad(set_to_none=True)
        del test_model, test_optimizers, x, y, loss
        if device_type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return True

    except torch.cuda.OutOfMemoryError:
        # Expected failure - cleanup and return False
        if device_type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return False

    except RuntimeError as e:
        # Check if it's an OOM error on other devices
        if "out of memory" in str(e).lower():
            gc.collect()
            return False
        # Unexpected error - cleanup and re-raise
        if device_type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        raise e


def _config_to_dict(config):
    """Convert a config (dataclass or Pydantic model) to a dict."""
    if hasattr(config, 'model_dump'):
        # Pydantic v2
        return config.model_dump()
    else:
        # Assume dataclass
        return dataclass_asdict(config)


def _batch_size_key_fn(config, device, device_type, max_seq_len,
                       freeze_every=None, reverse_train_order=False,
                       embedding_lr=0.2, unembedding_lr=0.004,
                       matrix_lr=0.02, weight_decay=0.0,
                       safety_factor=0.9, **kwargs):
    """Generate a cache key for find_optimal_batch_size from its arguments."""
    return {
        'config_class': type(config).__name__,
        'config': _config_to_dict(config),
        'device_type': device_type,
        'max_seq_len': max_seq_len,
        'freeze_every': freeze_every,
        'reverse_train_order': reverse_train_order,
        'embedding_lr': embedding_lr,
        'unembedding_lr': unembedding_lr,
        'matrix_lr': matrix_lr,
        'weight_decay': weight_decay,
        'safety_factor': safety_factor,
    }


@disk_cache(key_fn=_batch_size_key_fn, cache_name="batch_size_cache")
def find_optimal_batch_size(
    config,  # GPTConfig
    device: torch.device,
    device_type: str,
    max_seq_len: int,
    max_batch_size: int = 128,
    min_batch_size: int = 1,
    safety_factor: float = 0.9,
    verbose: bool = True,
    freeze_every: int | None = None,
    reverse_train_order: bool = False,
    embedding_lr: float = 0.2,
    unembedding_lr: float = 0.004,
    matrix_lr: float = 0.02,
    weight_decay: float = 0.0,
) -> int:
    """
    Find the optimal batch size for training LayeredGPT without OOM errors.

    Uses binary search to find the largest batch size that fits in memory,
    testing both forward and backward passes with full optimizer state allocation.

    This function mimics the actual training setup to ensure accurate results:
    - Creates model with meta device initialization
    - Allocates optimizer states (AdamW + Muon)
    - Runs forward pass with autocast
    - Runs backward pass (peak memory)
    - Steps optimizers to allocate state

    Args:
        config: GPTConfig for the model architecture
        device: torch.device to test on (e.g., torch.device("cuda"))
        device_type: "cuda", "cpu", or "mps"
        max_seq_len: Sequence length to test with
        max_batch_size: Upper bound for binary search (default: 128)
        min_batch_size: Lower bound for binary search (default: 1)
        safety_factor: Multiply result by this factor for safety margin (default: 0.9)
        verbose: Print progress during search (default: True)
        freeze_every: freeze_every parameter for LayeredGPT (default: large number)
        reverse_train_order: reverse_train_order for LayeredGPT (default: False)
        embedding_lr: Learning rate for embeddings (AdamW)
        unembedding_lr: Learning rate for lm_head (AdamW)
        matrix_lr: Learning rate for linear layers (Muon)
        weight_decay: Weight decay for AdamW

    Returns:
        Optimal batch size (int) with safety factor applied

    Example:
        >>> config = GPTConfig(n_layer=12, n_embd=768, n_head=6, ...)
        >>> device = torch.device("cuda")
        >>> optimal_bs = find_optimal_batch_size(config, device, "cuda", max_seq_len=2048)
        >>> print(f"Use batch_size={optimal_bs}")

    Notes:
        - Only works reliably on CUDA (CPU/MPS may not have clear OOM boundaries)
        - Takes several minutes for large models (O(log n) tests)
        - Cleans up memory between tests to avoid false OOMs
        - Returns min_batch_size if even that doesn't fit (check logs!)
    """
    if device_type not in ["cuda", "cpu", "mps"]:
        raise ValueError(f"device_type must be 'cuda', 'cpu', or 'mps', got '{device_type}'")

    if freeze_every is None:
        freeze_every = 1000000  # Large number to effectively disable

    if verbose:
        print0(f"\n{'='*60}")
        print0(f"Finding optimal batch size for LayeredGPT")
        print0(f"{'='*60}")
        print0(f"Model config: n_layer={config.n_layer}, n_embd={config.n_embd}, "
               f"n_head={config.n_head}, vocab_size={config.vocab_size}")
        print0(f"Sequence length: {max_seq_len}")
        print0(f"Search range: [{min_batch_size}, {max_batch_size}]")
        print0(f"Safety factor: {safety_factor:.1%}")
        print0(f"Device: {device} ({device_type})")
        print0(f"")

    # Binary search
    low, high = min_batch_size, max_batch_size
    best_working_batch = min_batch_size
    tests_run = 0

    while low <= high:
        mid = (low + high) // 2
        tests_run += 1

        if verbose:
            print0(f"[Test {tests_run}] Trying batch_size={mid}...", end=" ")

        success = _test_batch_size_fit(
            batch_size=mid,
            config=config,
            device=device,
            device_type=device_type,
            max_seq_len=max_seq_len,
            freeze_every=freeze_every,
            reverse_train_order=reverse_train_order,
            embedding_lr=embedding_lr,
            unembedding_lr=unembedding_lr,
            matrix_lr=matrix_lr,
            weight_decay=weight_decay,
        )

        if success:
            best_working_batch = mid
            if verbose:
                print0(f"SUCCESS")
            low = mid + 1
        else:
            if verbose:
                print0(f"OOM")
            high = mid - 1

    # Apply safety factor
    recommended = int(best_working_batch * safety_factor)
    recommended = max(1, recommended)  # Ensure at least 1

    if verbose:
        print0(f"")
        print0(f"{'='*60}")
        print0(f"Results after {tests_run} tests:")
        print0(f"  Max working batch size: {best_working_batch}")
        print0(f"  Recommended (with {safety_factor:.0%} safety): {recommended}")

        if best_working_batch == min_batch_size:
            print0(f"")
            print0(f"WARNING: Only minimum batch size ({min_batch_size}) fits!")
            print0(f"         Consider reducing model size or sequence length.")

        print0(f"{'='*60}\n")

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
