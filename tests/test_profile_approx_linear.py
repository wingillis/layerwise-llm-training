#!/usr/bin/env python3
"""
Profiling test to compare ApproxLinearSVD vs nn.Linear performance.

This test profiles both forward and backward passes using torch.profiler
and memory tracking to identify bottlenecks in ApproxLinearSVD.

Run with:
    # All profiling tests (~5-10 min)
    uv run python -m pytest tests/test_profile_approx_linear.py -v -s

    # Skip slow tests (for CI, ~30 sec)
    uv run python -m pytest tests/test_profile_approx_linear.py -v -m "not slow"

    # Single rank test only
    uv run python -m pytest tests/test_profile_approx_linear.py::test_profile_single_rank -v -s
"""

import pytest
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass, field, asdict
import gc
import json
from pathlib import Path

from nanochat.approximated_gpt import ApproxLinearSVD


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class ProfileConfig:
    """Configuration for profiling tests."""
    # Model dimensions matching production settings
    small_dims: tuple[int, int] = (640, 2560)    # depth=10 MLP (640 → 4*640)
    medium_dims: tuple[int, int] = (1280, 5120)  # depth=20 MLP (1280 → 4*1280)

    # Test parameters
    batch_size: int = 8
    seq_len: int = 1024
    ranks: list[int] = field(default_factory=lambda: [8, 16, 32, 64])

    # Profiling control
    warmup_iters: int = 10   # Warmup iterations before profiling
    profile_iters: int = 20  # Iterations to profile

    require_cuda: bool = True


@dataclass
class ProfilerResult:
    """Results from torch.profiler analysis."""
    total_time_ms: float
    cuda_time_ms: float
    cpu_time_ms: float
    num_cuda_kernels: int
    matmul_time_ms: float
    elementwise_time_ms: float


@dataclass
class ComparisonResult:
    """Results comparing ApproxLinearSVD vs nn.Linear."""
    # Configuration
    in_features: int
    out_features: int
    rank: int
    batch_size: int
    seq_len: int

    # nn.Linear baseline
    linear_time_ms: float
    linear_memory_mb: float
    linear_num_kernels: int
    linear_matmul_time_ms: float
    linear_elementwise_time_ms: float

    # ApproxLinearSVD
    approx_time_ms: float
    approx_memory_mb: float
    approx_num_kernels: int
    approx_matmul_time_ms: float
    approx_elementwise_time_ms: float

    # Comparison metrics
    speedup: float  # linear_time / approx_time (>1 means approx is faster)
    memory_savings: float  # (linear - approx) / linear (as fraction)


# ============================================================================
# CUDA Validation Helpers
# ============================================================================

def check_cuda_available() -> torch.device:
    """Validate CUDA availability, skip test if not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, GPU profiling requires CUDA")
    return torch.device("cuda")


def reset_cuda_memory():
    """Reset CUDA memory stats for clean profiling."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


# ============================================================================
# Warmup and Profiling Functions
# ============================================================================

def warmup_layer(layer: nn.Module, input_tensor: torch.Tensor, num_iters: int = 10):
    """
    Warmup layer to compile CUDA kernels.

    Critical: First forward/backward triggers kernel compilation (10-100x slower).
    Warmup ensures we measure steady-state performance, not compilation overhead.
    """
    layer.train()
    for _ in range(num_iters):
        output = layer(input_tensor)
        loss = output.sum()
        loss.backward()
        layer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()  # Ensure all kernels complete


def track_memory_usage(
    layer: nn.Module,
    input_tensor: torch.Tensor,
    num_iters: int = 5
) -> dict[str, float]:
    """
    Track peak memory usage during forward + backward pass.

    Returns memory metrics in MB:
    - peak_allocated_mb: Peak memory allocated
    - peak_reserved_mb: Peak memory reserved by allocator
    - avg_allocated_mb: Average across iterations
    """
    reset_cuda_memory()
    memory_samples = []

    layer.train()
    for _ in range(num_iters):
        torch.cuda.reset_peak_memory_stats()

        output = layer(input_tensor)
        loss = output.sum()
        loss.backward()
        layer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()

        peak_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
        memory_samples.append({'allocated': peak_allocated, 'reserved': peak_reserved})

    return {
        'peak_allocated_mb': max(s['allocated'] for s in memory_samples),
        'peak_reserved_mb': max(s['reserved'] for s in memory_samples),
        'avg_allocated_mb': sum(s['allocated'] for s in memory_samples) / len(memory_samples)
    }


def extract_profiler_metrics(prof: profile) -> ProfilerResult:
    """
    Extract key metrics from profiler events.

    Categorizes CUDA kernels:
    - Matmul: gemm, matmul, mm operations
    - Elementwise: add, mul, addcmul operations
    """
    events = prof.key_averages()

    total_cuda_time = 0
    total_cpu_time = 0
    matmul_time = 0
    elementwise_time = 0
    num_kernels = 0

    for event in events:
        # CPU time
        cpu_time = event.cpu_time_total / 1000  # Convert μs → ms
        total_cpu_time += cpu_time

        # CUDA time
        if event.device_type == torch.profiler.DeviceType.CUDA:
            cuda_time = event.device_time / 1000  # Convert μs → ms
            total_cuda_time += cuda_time
            num_kernels += 1

            name = event.key.lower()
            if any(op in name for op in ['gemm', 'matmul', 'mm', 'bmm']):
                matmul_time += cuda_time
            elif any(op in name for op in ['add', 'mul', 'cmul']):
                elementwise_time += cuda_time

    return ProfilerResult(
        total_time_ms=total_cuda_time,
        cuda_time_ms=total_cuda_time,
        cpu_time_ms=total_cpu_time,
        num_cuda_kernels=num_kernels,
        matmul_time_ms=matmul_time,
        elementwise_time_ms=elementwise_time
    )


def profile_layer(
    layer: nn.Module,
    input_tensor: torch.Tensor,
    num_iters: int = 20
) -> ProfilerResult:
    """
    Profile layer using torch.profiler with scheduling.

    Uses profiler schedule to reduce overhead:
    - wait=1: Skip first iteration
    - warmup=1: Warmup profiler
    - active=3: Profile these iterations
    - repeat=2: Repeat cycle twice
    """
    layer.train()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # Disable for performance
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2)
    ) as prof:
        for _ in range(num_iters):
            output = layer(input_tensor)
            loss = output.sum()
            loss.backward()
            layer.zero_grad(set_to_none=True)
            prof.step()

    torch.cuda.synchronize()
    return extract_profiler_metrics(prof)


# ============================================================================
# Comparison Logic
# ============================================================================

def compare_single_config(
    in_features: int, out_features: int, rank: int,
    batch_size: int, seq_len: int,
    device: torch.device, config: ProfileConfig
) -> ComparisonResult:
    """
    Compare nn.Linear vs ApproxLinearSVD for single configuration.

    Process:
    1. Create input tensor with requires_grad=True
    2. Profile nn.Linear baseline (warmup → profile → memory)
    3. Clean up and reset CUDA memory
    4. Profile ApproxLinearSVD (warmup → profile → memory)
    5. Compute comparison metrics (speedup, memory savings)
    """
    torch.manual_seed(42)
    reset_cuda_memory()

    # Create input
    input_tensor = torch.randn(
        batch_size, seq_len, in_features,
        device=device, dtype=torch.float32, requires_grad=True
    )

    # 1. Profile nn.Linear
    linear = nn.Linear(in_features, out_features, bias=False).to(device)
    linear = torch.compile(linear, dynamic=False)
    warmup_layer(linear, input_tensor, config.warmup_iters)
    linear_profile = profile_layer(linear, input_tensor, config.profile_iters)
    linear_memory = track_memory_usage(linear, input_tensor)

    del linear
    reset_cuda_memory()

    # 2. Profile ApproxLinearSVD
    approx = ApproxLinearSVD(in_features, out_features, rank=rank, bias=False).to(device)
    approx = torch.compile(approx, dynamic=False, mode='max-autotune')
    warmup_layer(approx, input_tensor, config.warmup_iters)
    approx_profile = profile_layer(approx, input_tensor, config.profile_iters)
    approx_memory = track_memory_usage(approx, input_tensor)

    # 3. Compute metrics
    speedup = linear_profile.total_time_ms / approx_profile.total_time_ms
    memory_savings = (
        (linear_memory['peak_allocated_mb'] - approx_memory['peak_allocated_mb'])
        / linear_memory['peak_allocated_mb']
    )

    return ComparisonResult(
        in_features=in_features, out_features=out_features, rank=rank,
        batch_size=batch_size, seq_len=seq_len,
        linear_time_ms=linear_profile.total_time_ms,
        linear_memory_mb=linear_memory['peak_allocated_mb'],
        linear_num_kernels=linear_profile.num_cuda_kernels,
        linear_matmul_time_ms=linear_profile.matmul_time_ms,
        linear_elementwise_time_ms=linear_profile.elementwise_time_ms,
        approx_time_ms=approx_profile.total_time_ms,
        approx_memory_mb=approx_memory['peak_allocated_mb'],
        approx_num_kernels=approx_profile.num_cuda_kernels,
        approx_matmul_time_ms=approx_profile.matmul_time_ms,
        approx_elementwise_time_ms=approx_profile.elementwise_time_ms,
        speedup=speedup,
        memory_savings=memory_savings
    )


def compare_across_ranks(
    in_features: int, out_features: int, ranks: list[int],
    batch_size: int, seq_len: int,
    device: torch.device, config: ProfileConfig
) -> list[ComparisonResult]:
    """Compare across multiple ranks for single dimension."""
    results = []
    for rank in ranks:
        print(f"  Testing rank={rank}...", end=" ", flush=True)
        result = compare_single_config(
            in_features, out_features, rank, batch_size, seq_len, device, config
        )
        results.append(result)
        print(f"Done (speedup={result.speedup:.2f}x)")
    return results


# ============================================================================
# Output Formatting
# ============================================================================

def display_results_table(
    results: dict[str, list[ComparisonResult]],
    config: ProfileConfig
):
    """
    Display formatted comparison table.

    Shows speedup (>1 means approx is faster) and memory savings percentage.
    """
    for dim_name, dim_results in results.items():
        if not dim_results:
            continue

        first = dim_results[0]
        print(f"\n{dim_name.upper()} DIMENSIONS: "
              f"{first.in_features} → {first.out_features} "
              f"(batch={first.batch_size}, seq={first.seq_len})")
        print("="*80)

        # Header
        print(f"{'Rank':>4} | {'nn.Linear':^19} | {'ApproxSVD':^19} | "
              f"{'Speedup':^7} | {'Memory':^6} | {'Kernels':^9}")
        print(f"{'':>4} | {'Time(ms)':>8} {'Mem(MB)':>6} | "
              f"{'Time(ms)':>8} {'Mem(MB)':>6} | "
              f"{'x':^7} | {'Saved':^6} | {'L / A':^9}")
        print("-"*80)

        # Data rows
        for r in dim_results:
            speedup_str = f"{r.speedup:>5.2f}x"
            memory_str = f"{r.memory_savings*100:>5.1f}%"

            print(f"{r.rank:>4} | "
                  f"{r.linear_time_ms:>8.2f} {r.linear_memory_mb:>6.0f} | "
                  f"{r.approx_time_ms:>8.2f} {r.approx_memory_mb:>6.0f} | "
                  f"{speedup_str:^7} | {memory_str:^6} | "
                  f"{r.linear_num_kernels:>2} / {r.approx_num_kernels:<2}")

        print("-"*80)


def save_results_json(results: dict, filename: str):
    """Save detailed results to JSON for further analysis."""
    output = {
        dim_name: [asdict(r) for r in dim_results]
        for dim_name, dim_results in results.items()
    }

    filepath = Path(filename)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to: {filepath.absolute()}")


# ============================================================================
# Test Functions (Pytest Entry Points)
# ============================================================================

@pytest.mark.slow
def test_profile_approx_linear_comprehensive():
    """
    Comprehensive profiling comparison test.

    Tests both small and medium dimensions across all ranks.
    Marked as @pytest.mark.slow - skip with: pytest -m "not slow"
    """
    device = check_cuda_available()
    config = ProfileConfig()

    print("\n" + "="*80)
    print("PROFILING: ApproxLinearSVD vs nn.Linear")
    print("="*80)

    all_results = {}

    # Test small dimensions (640 → 2560)
    print(f"\n[1/2] Small dimensions: {config.small_dims[0]} → {config.small_dims[1]}")
    small_results = compare_across_ranks(
        *config.small_dims, config.ranks, config.batch_size,
        config.seq_len, device, config
    )
    all_results['small'] = small_results

    # Test medium dimensions (1280 → 5120)
    print(f"\n[2/2] Medium dimensions: {config.medium_dims[0]} → {config.medium_dims[1]}")
    medium_results = compare_across_ranks(
        *config.medium_dims, config.ranks, config.batch_size,
        config.seq_len, device, config
    )
    all_results['medium'] = medium_results

    # Display and save results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    display_results_table(all_results, config)
    save_results_json(all_results, "profiling_results.json")


@pytest.mark.slow
def test_profile_small_model_only():
    """Quick profiling test for small model only."""
    device = check_cuda_available()
    config = ProfileConfig()

    print("\n" + "="*80)
    print("PROFILING: Small Model Only")
    print("="*80)

    print(f"\nSmall dimensions: {config.small_dims[0]} → {config.small_dims[1]}")
    results = compare_across_ranks(
        *config.small_dims, config.ranks, config.batch_size,
        config.seq_len, device, config
    )

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    display_results_table({'small': results}, config)


def test_profile_single_rank():
    """
    Fast test for single rank (not marked slow).
    Use for CI or quick validation.
    """
    device = check_cuda_available()
    config = ProfileConfig()

    print("\n" + "="*80)
    print("PROFILING: Single Rank Test (rank=16)")
    print("="*80)

    result = compare_single_config(
        *config.small_dims, rank=16, batch_size=config.batch_size,
        seq_len=config.seq_len, device=device, config=config
    )

    # Sanity checks
    assert result.approx_time_ms > 0, "ApproxLinearSVD time should be positive"
    assert result.linear_time_ms > 0, "nn.Linear time should be positive"
    assert result.speedup > 0, "Speedup should be positive"

    print(f"\nRank 16 Performance:")
    print(f"  nn.Linear time:     {result.linear_time_ms:.2f} ms")
    print(f"  ApproxLinearSVD:    {result.approx_time_ms:.2f} ms")
    print(f"  Speedup:            {result.speedup:.2f}x")
    print(f"  Memory savings:     {result.memory_savings*100:.1f}%")
    print(f"  Kernels (L/A):      {result.linear_num_kernels} / {result.approx_num_kernels}")


if __name__ == "__main__":
    # For manual testing without pytest
    torch.manual_seed(42)
    test_profile_single_rank()
