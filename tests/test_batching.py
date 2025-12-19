#!/usr/bin/env python3
"""
Test script for verifying batching and head dimension handling in WeightApproxGPT.

This test isolates the bug where batch dimension is confused with n_head dimension.
Bug symptoms: With depth=10 (n_head=5), batch_size=8, the output shape shows (5, T, C)
instead of (8, T, C).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from nanochat.approximated_gpt import (
    WeightApproxGPT,
    WeightApproxGPTConfig,
    ApproxLinearSVD,
    ApproxLinearABBA,
    ApproxLinear,
    ApproxWeightMLP,
    ApproxWeightBlock,
    CausalSelfAttention,
    norm,
)


# Test configuration matching the failing case
@dataclass
class TestConfig:
    depth: int = 10
    batch_size: int = 8
    seq_len: int = 256  # Smaller for faster tests
    vocab_size: int = 1000
    mlp_proj_rank: int = 16

    @property
    def n_embd(self):
        return self.depth * 64  # 640 for depth=10

    @property
    def n_head(self):
        return max(1, (self.n_embd + 127) // 128)  # 5 for n_embd=640


def test_approx_linear_svd_shapes():
    """Test ApproxLinearSVD with various batch sizes - isolate einsum."""
    print("\n=== Testing ApproxLinearSVD shapes ===")
    cfg = TestConfig()

    in_features = cfg.n_embd  # 640
    out_features = 4 * cfg.n_embd  # 2560
    rank = cfg.mlp_proj_rank

    layer = ApproxLinearSVD(in_features, out_features, rank=rank, bias=False)

    # Test various batch sizes
    for batch_size in [1, 4, 8, 16]:
        x = torch.randn(batch_size, cfg.seq_len, in_features)
        y = layer(x)

        expected_shape = (batch_size, cfg.seq_len, out_features)
        actual_shape = tuple(y.shape)

        print(f"  batch={batch_size}: input={tuple(x.shape)} -> output={actual_shape}")
        assert actual_shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
        )

    print("  PASSED: ApproxLinearSVD preserves batch dimension")


def test_approx_linear_abba_shapes():
    """Test ApproxLinearABBA with various batch sizes."""
    print("\n=== Testing ApproxLinearABBA shapes ===")
    cfg = TestConfig()

    in_features = cfg.n_embd
    out_features = 4 * cfg.n_embd
    rank = cfg.mlp_proj_rank

    layer = ApproxLinearABBA(in_features, out_features, rank=rank, bias=False)

    for batch_size in [1, 4, 8, 16]:
        x = torch.randn(batch_size, cfg.seq_len, in_features)
        y = layer(x)

        expected_shape = (batch_size, cfg.seq_len, out_features)
        actual_shape = tuple(y.shape)

        print(f"  batch={batch_size}: input={tuple(x.shape)} -> output={actual_shape}")
        assert actual_shape == expected_shape, (
            f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
        )

    print("  PASSED: ApproxLinearABBA preserves batch dimension")


def test_approx_linear_wrapper_shapes():
    """Test ApproxLinear wrapper with both SVD and ABBA."""
    print("\n=== Testing ApproxLinear wrapper shapes ===")
    cfg = TestConfig()

    in_features = cfg.n_embd
    out_features = 4 * cfg.n_embd
    rank = cfg.mlp_proj_rank

    for approx_type in ["svd", "abba"]:
        layer = ApproxLinear(
            in_features, out_features, approx_type=approx_type, rank=rank, bias=False
        )

        x = torch.randn(cfg.batch_size, cfg.seq_len, in_features)
        y = layer(x)

        expected_shape = (cfg.batch_size, cfg.seq_len, out_features)
        actual_shape = tuple(y.shape)

        print(f"  {approx_type}: input={tuple(x.shape)} -> output={actual_shape}")
        assert actual_shape == expected_shape, (
            f"Shape mismatch for {approx_type}: expected {expected_shape}, got {actual_shape}"
        )

    print("  PASSED: ApproxLinear wrapper preserves batch dimension")


def test_mlp_shapes():
    """Test ApproxWeightMLP preserves batch dimension."""
    print("\n=== Testing ApproxWeightMLP shapes ===")
    cfg = TestConfig()

    for approx_type in ["svd", "abba"]:
        config = WeightApproxGPTConfig(
            sequence_len=cfg.seq_len,
            vocab_size=cfg.vocab_size,
            n_layer=1,
            n_head=cfg.n_head,
            n_kv_head=cfg.n_head,
            n_embd=cfg.n_embd,
            approx_type=approx_type,
            approx_mlp_proj=True,
            mlp_proj_rank=cfg.mlp_proj_rank,
            build_by_layer=False,
        )

        mlp = ApproxWeightMLP(config)

        x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.n_embd)
        y = mlp(x)

        expected_shape = (cfg.batch_size, cfg.seq_len, cfg.n_embd)
        actual_shape = tuple(y.shape)

        print(f"  {approx_type}: input={tuple(x.shape)} -> output={actual_shape}")
        assert actual_shape == expected_shape, (
            f"MLP shape mismatch for {approx_type}: expected {expected_shape}, got {actual_shape}"
        )

    print("  PASSED: ApproxWeightMLP preserves batch dimension")


def test_attention_shapes():
    """Test CausalSelfAttention transpose/reshape operations."""
    print("\n=== Testing CausalSelfAttention shapes ===")
    cfg = TestConfig()

    config = WeightApproxGPTConfig(
        sequence_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        n_layer=1,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_head,
        n_embd=cfg.n_embd,
        build_by_layer=False,
    )

    attn = CausalSelfAttention(config, layer_idx=0)

    # Create dummy rotary embeddings
    head_dim = cfg.n_embd // cfg.n_head
    cos = torch.randn(1, cfg.seq_len, 1, head_dim // 2)
    sin = torch.randn(1, cfg.seq_len, 1, head_dim // 2)
    cos_sin = (cos, sin)

    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.n_embd)
    y = attn(x, cos_sin)

    expected_shape = (cfg.batch_size, cfg.seq_len, cfg.n_embd)
    actual_shape = tuple(y.shape)

    print(f"  input={tuple(x.shape)} -> output={actual_shape}")
    print(f"  n_head={cfg.n_head}, batch_size={cfg.batch_size}")
    assert actual_shape == expected_shape, (
        f"Attention shape mismatch: expected {expected_shape}, got {actual_shape}"
    )

    print("  PASSED: CausalSelfAttention preserves batch dimension")


def test_block_shapes():
    """Test ApproxWeightBlock end-to-end."""
    print("\n=== Testing ApproxWeightBlock shapes ===")
    cfg = TestConfig()

    for approx_type in ["svd", "abba"]:
        config = WeightApproxGPTConfig(
            sequence_len=cfg.seq_len,
            vocab_size=cfg.vocab_size,
            n_layer=1,
            n_head=cfg.n_head,
            n_kv_head=cfg.n_head,
            n_embd=cfg.n_embd,
            approx_type=approx_type,
            approx_mlp_proj=True,
            mlp_proj_rank=cfg.mlp_proj_rank,
            build_by_layer=False,
                    )

        block = ApproxWeightBlock(config, layer_idx=0)

        # Create dummy rotary embeddings
        head_dim = cfg.n_embd // cfg.n_head
        cos = torch.randn(1, cfg.seq_len, 1, head_dim // 2)
        sin = torch.randn(1, cfg.seq_len, 1, head_dim // 2)
        cos_sin = (cos, sin)

        x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.n_embd)
        y = block(x, cos_sin)

        expected_shape = (cfg.batch_size, cfg.seq_len, cfg.n_embd)
        actual_shape = tuple(y.shape)

        print(f"  {approx_type}: input={tuple(x.shape)} -> output={actual_shape}")
        assert actual_shape == expected_shape, (
            f"Block shape mismatch for {approx_type}: expected {expected_shape}, got {actual_shape}"
        )

    print("  PASSED: ApproxWeightBlock preserves batch dimension")


def test_block_shapes_compiled():
    """Test ApproxWeightBlock with torch.compile."""
    print("\n=== Testing ApproxWeightBlock with torch.compile ===")
    cfg = TestConfig()

    for approx_type in ["svd", "abba"]:
        config = WeightApproxGPTConfig(
            sequence_len=cfg.seq_len,
            vocab_size=cfg.vocab_size,
            n_layer=1,
            n_head=cfg.n_head,
            n_kv_head=cfg.n_head,
            n_embd=cfg.n_embd,
            approx_type=approx_type,
            approx_mlp_proj=True,
            mlp_proj_rank=cfg.mlp_proj_rank,
            build_by_layer=False,
                    )

        block = ApproxWeightBlock(config, layer_idx=0)
        compiled_block = torch.compile(block, dynamic=False)

        # Create dummy rotary embeddings
        head_dim = cfg.n_embd // cfg.n_head
        cos = torch.randn(1, cfg.seq_len, 1, head_dim // 2)
        sin = torch.randn(1, cfg.seq_len, 1, head_dim // 2)
        cos_sin = (cos, sin)

        x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.n_embd)
        y = compiled_block(x, cos_sin)

        expected_shape = (cfg.batch_size, cfg.seq_len, cfg.n_embd)
        actual_shape = tuple(y.shape)

        print(f"  {approx_type} (compiled): input={tuple(x.shape)} -> output={actual_shape}")
        assert actual_shape == expected_shape, (
            f"Compiled block shape mismatch for {approx_type}: expected {expected_shape}, got {actual_shape}"
        )

    print("  PASSED: Compiled ApproxWeightBlock preserves batch dimension")


def test_full_model_no_compile():
    """Test complete model WITHOUT torch.compile."""
    print("\n=== Testing full model WITHOUT torch.compile ===")
    cfg = TestConfig()

    # Create model config matching failing case
    config = WeightApproxGPTConfig(
        sequence_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.depth,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_head,
        n_embd=cfg.n_embd,
        approx_type="svd",
        approx_mlp_proj=True,
        mlp_proj_rank=cfg.mlp_proj_rank,
        build_by_layer=False,  # Build all layers upfront, no compile
            )

    # Create model without compilation by modifying the class behavior
    # We'll build blocks manually without compile
    model = nn.Module()
    model.config = config

    wte = nn.Embedding(config.vocab_size, config.n_embd)
    h = nn.ModuleList([
        ApproxWeightBlock(config, layer_idx=i)
        for i in range(config.n_layer)
    ])
    lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    # Test forward pass manually
    x = torch.randint(0, config.vocab_size, (cfg.batch_size, cfg.seq_len))

    # Embedding
    emb = wte(x)
    emb = norm(emb)
    print(f"  After embedding: {tuple(emb.shape)}")

    # Create rotary embeddings
    head_dim = config.n_embd // config.n_head
    seq_len = config.sequence_len * 10
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    cos_sin = (cos[:, :cfg.seq_len], sin[:, :cfg.seq_len])

    # Process through blocks
    out = emb
    for i, block in enumerate(h):
        out = block(out, cos_sin)
        print(f"  After block {i}: {tuple(out.shape)}")

        # Check shape after each block
        expected_shape = (cfg.batch_size, cfg.seq_len, cfg.n_embd)
        actual_shape = tuple(out.shape)
        assert actual_shape == expected_shape, (
            f"Shape mismatch after block {i}: expected {expected_shape}, got {actual_shape}"
        )

    out = norm(out)

    # LM head
    logits = lm_head(out)
    print(f"  Final logits: {tuple(logits.shape)}")

    expected_logits_shape = (cfg.batch_size, cfg.seq_len, config.vocab_size)
    actual_logits_shape = tuple(logits.shape)
    assert actual_logits_shape == expected_logits_shape, (
        f"Logits shape mismatch: expected {expected_logits_shape}, got {actual_logits_shape}"
    )

    print("  PASSED: Full model without compile preserves batch dimension")


def test_full_model_with_compile():
    """Test complete model WITH torch.compile (build_by_layer=False)."""
    print("\n=== Testing full model WITH torch.compile ===")
    cfg = TestConfig()

    config = WeightApproxGPTConfig(
        sequence_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.depth,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_head,
        n_embd=cfg.n_embd,
        approx_type="svd",
        approx_mlp_proj=True,
        mlp_proj_rank=cfg.mlp_proj_rank,
        build_by_layer=False,
            )

    freeze_every = 1000  # Large value to avoid adding layers
    model = WeightApproxGPT(config, freeze_every=freeze_every)
    model.eval()

    x = torch.randint(0, config.vocab_size, (cfg.batch_size, cfg.seq_len))

    with torch.no_grad():
        logits = model(x)

    expected_shape = (cfg.batch_size, cfg.seq_len, config.vocab_size)
    actual_shape = tuple(logits.shape)

    print(f"  Input: {tuple(x.shape)}")
    print(f"  Output logits: {actual_shape}")
    print(f"  Expected: {expected_shape}")

    assert actual_shape == expected_shape, (
        f"Model output shape mismatch: expected {expected_shape}, got {actual_shape}"
    )

    print("  PASSED: Full model with compile preserves batch dimension")


def test_build_by_layer_shapes():
    """Test model with build_by_layer=True (exact failing configuration)."""
    print("\n=== Testing model with build_by_layer=True ===")
    cfg = TestConfig()

    config = WeightApproxGPTConfig(
        sequence_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        n_layer=1,  # Start with 1 layer
        n_head=cfg.n_head,
        n_kv_head=cfg.n_head,
        n_embd=cfg.n_embd,
        approx_type="svd",
        approx_mlp_proj=True,
        mlp_proj_rank=cfg.mlp_proj_rank,
        build_by_layer=True,  # This is the failing config
        freeze_previous_weights=False,
            )

    freeze_every = 100  # Add layer every 100 steps
    model = WeightApproxGPT(config, freeze_every=freeze_every)
    model.eval()

    print(f"  Initial layers: {len(model.transformer.h)}")
    print(f"  n_head: {cfg.n_head}, batch_size: {cfg.batch_size}")

    x = torch.randint(0, config.vocab_size, (cfg.batch_size, cfg.seq_len))

    # Test with step=0 (single layer)
    with torch.no_grad():
        logits = model(x, step=0)

    expected_shape = (cfg.batch_size, cfg.seq_len, config.vocab_size)
    actual_shape = tuple(logits.shape)

    print(f"  Step 0 - Input: {tuple(x.shape)}, Output: {actual_shape}")
    assert actual_shape == expected_shape, (
        f"Step 0 shape mismatch: expected {expected_shape}, got {actual_shape}"
    )

    # Switch to train mode and add more layers
    model.train()

    # Test with step=100 (should add layer)
    with torch.no_grad():
        logits = model(x, step=freeze_every)

    actual_shape = tuple(logits.shape)
    print(f"  Step {freeze_every} - Layers: {len(model.transformer.h)}, Output: {actual_shape}")
    assert actual_shape == expected_shape, (
        f"Step {freeze_every} shape mismatch: expected {expected_shape}, got {actual_shape}"
    )

    # Test with step=200 (should add another layer)
    with torch.no_grad():
        logits = model(x, step=2 * freeze_every)

    actual_shape = tuple(logits.shape)
    print(f"  Step {2*freeze_every} - Layers: {len(model.transformer.h)}, Output: {actual_shape}")
    assert actual_shape == expected_shape, (
        f"Step {2*freeze_every} shape mismatch: expected {expected_shape}, got {actual_shape}"
    )

    print("  PASSED: build_by_layer model preserves batch dimension")


def test_multiple_batch_sizes():
    """Test with multiple batch sizes to ensure no hardcoded dimensions."""
    print("\n=== Testing multiple batch sizes ===")
    cfg = TestConfig()

    config = WeightApproxGPTConfig(
        sequence_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.depth,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_head,
        n_embd=cfg.n_embd,
        approx_type="svd",
        approx_mlp_proj=True,
        mlp_proj_rank=cfg.mlp_proj_rank,
        build_by_layer=False,
            )

    freeze_every = 1000
    model = WeightApproxGPT(config, freeze_every=freeze_every)
    model.eval()

    # Test batch sizes including n_head=5 and batch_size=8
    for batch_size in [1, 4, 5, 8, 16]:
        x = torch.randint(0, config.vocab_size, (batch_size, cfg.seq_len))

        with torch.no_grad():
            logits = model(x)

        expected_shape = (batch_size, cfg.seq_len, config.vocab_size)
        actual_shape = tuple(logits.shape)

        print(f"  batch_size={batch_size}: output={actual_shape}")
        assert actual_shape == expected_shape, (
            f"Batch size {batch_size} shape mismatch: expected {expected_shape}, got {actual_shape}"
        )

    print("  PASSED: All batch sizes work correctly")


def test_training_loop_shapes():
    """Test shapes during actual training with gradients."""
    print("\n=== Testing training loop shapes ===")
    cfg = TestConfig()

    config = WeightApproxGPTConfig(
        sequence_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        n_layer=1,  # Start with 1 layer
        n_head=cfg.n_head,
        n_kv_head=cfg.n_head,
        n_embd=cfg.n_embd,
        approx_type="svd",
        approx_mlp_proj=True,
        mlp_proj_rank=cfg.mlp_proj_rank,
        build_by_layer=True,
        freeze_previous_weights=False,
            )

    freeze_every = 5  # Add layer every 5 steps for faster testing
    model = WeightApproxGPT(config, freeze_every=freeze_every)
    model.train()

    print(f"  n_head: {cfg.n_head}, batch_size: {cfg.batch_size}")

    # Simulate training loop
    for step in range(20):
        x = torch.randint(0, config.vocab_size, (cfg.batch_size, cfg.seq_len))
        y = torch.randint(0, config.vocab_size, (cfg.batch_size, cfg.seq_len))

        # Forward with loss (training mode)
        loss = model(x, y, step=step)

        # Check loss is scalar
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"

        # Backward pass
        loss.backward()

        # Zero gradients (simulate optimizer step)
        model.zero_grad(set_to_none=True)

        if step % 5 == 0:
            print(f"  Step {step}: layers={len(model.transformer.h)}, loss={loss.item():.4f}")

    print("  PASSED: Training loop maintains correct shapes")


def test_training_with_loss_reduction_none():
    """Test shapes when using loss_reduction='none' (used in evaluate_bpb)."""
    print("\n=== Testing loss_reduction='none' shapes ===")
    cfg = TestConfig()

    config = WeightApproxGPTConfig(
        sequence_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.depth,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_head,
        n_embd=cfg.n_embd,
        approx_type="svd",
        approx_mlp_proj=True,
        mlp_proj_rank=cfg.mlp_proj_rank,
        build_by_layer=False,
            )

    freeze_every = 1000
    model = WeightApproxGPT(config, freeze_every=freeze_every)
    model.eval()

    x = torch.randint(0, config.vocab_size, (cfg.batch_size, cfg.seq_len))
    y = torch.randint(0, config.vocab_size, (cfg.batch_size, cfg.seq_len))

    with torch.no_grad():
        loss = model(x, y, loss_reduction='none', step=0)

    # loss_reduction='none' should return (B*T,) shaped loss
    expected_numel = cfg.batch_size * cfg.seq_len
    actual_numel = loss.numel()

    print(f"  Input: ({cfg.batch_size}, {cfg.seq_len})")
    print(f"  Loss shape: {tuple(loss.shape)}, numel: {actual_numel}")
    print(f"  Expected numel: {expected_numel}")

    assert actual_numel == expected_numel, (
        f"Loss numel mismatch: expected {expected_numel}, got {actual_numel}"
    )

    print("  PASSED: loss_reduction='none' returns correct shape")


def test_exact_user_config():
    """Test with exact user configuration from train_config.toml."""
    print("\n=== Testing exact user configuration ===")

    # From train_config.toml
    depth = 10
    max_seq_len = 2048
    device_batch_size = 4

    # Calculated values
    n_embd = depth * 64  # 640
    n_head = max(1, (n_embd + 127) // 128)  # 5
    vocab_size = 65536  # User's actual vocab size

    print(f"  depth={depth}, n_embd={n_embd}, n_head={n_head}")
    print(f"  device_batch_size={device_batch_size}, max_seq_len={max_seq_len}")

    config = WeightApproxGPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=1,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
        approx_type="svd",
        approx_mlp_proj=True,
        mlp_proj_rank=16,
        build_by_layer=True,
        freeze_previous_weights=True,  # User has this True
            )

    freeze_every = 507  # 5075 / 10 = 507.5
    model = WeightApproxGPT(config, freeze_every=freeze_every)
    model.train()

    # First forward pass
    x = torch.randint(0, vocab_size, (device_batch_size, max_seq_len))
    y = torch.randint(0, vocab_size, (device_batch_size, max_seq_len))

    loss = model(x, y, step=0)
    print(f"  Step 0: loss shape={loss.shape}")

    # After adding a layer
    loss = model(x, y, step=freeze_every)
    print(f"  Step {freeze_every}: layers={len(model.transformer.h)}, loss shape={loss.shape}")

    # Inference mode
    model.eval()
    with torch.no_grad():
        logits = model(x)

    expected_shape = (device_batch_size, max_seq_len, vocab_size)
    actual_shape = tuple(logits.shape)

    print(f"  Inference logits: {actual_shape}")
    print(f"  Expected: {expected_shape}")

    assert actual_shape == expected_shape, (
        f"Logits shape mismatch: expected {expected_shape}, got {actual_shape}"
    )

    print("  PASSED: Exact user config works correctly")


if __name__ == "__main__":
    torch.manual_seed(42)

    # Run isolated component tests first
    test_approx_linear_svd_shapes()
    test_approx_linear_abba_shapes()
    test_approx_linear_wrapper_shapes()
    test_mlp_shapes()
    test_attention_shapes()
    test_block_shapes()
    test_block_shapes_compiled()

    # Then full model tests
    test_full_model_no_compile()
    test_full_model_with_compile()
    test_build_by_layer_shapes()
    test_multiple_batch_sizes()

    # Training-specific tests
    test_training_loop_shapes()
    test_training_with_loss_reduction_none()
    test_exact_user_config()

    print("\n" + "=" * 50)
    print("All batching tests passed!")
    print("=" * 50)
