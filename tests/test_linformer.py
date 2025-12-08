#!/usr/bin/env python3
"""
Test script for Linformer implementation
"""

import torch
from nanochat.approximated_gpt import WeightApproxGPTConfig, WeightApproxGPT


def test_linformer_basic():
    """Test basic Linformer functionality"""
    print("Testing Linformer implementation...")

    # Create config with Linformer enabled
    config = WeightApproxGPTConfig(
        sequence_len=256,
        n_embd=128,
        n_head=4,
        n_kv_head=4,
        n_layer=1,
        vocab_size=1000,
        use_linformer=True,
        linformer_proj_dim=64,
        linformer_sharing="layerwise",
        build_by_layer=False,  # Simpler for testing
        approx_type="abba",
        mlp_proj_rank=16,
    )

    # Create model
    model = WeightApproxGPT(config, freeze_every=1000)
    model.eval()

    # Create dummy input
    batch_size = 2
    seq_len = 256  # Match config.sequence_len
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass (no targets = inference mode, returns logits)
    with torch.no_grad():
        output = model(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: ({batch_size}, {seq_len}, {config.vocab_size})")

    # Verify output shape - forward returns logits of shape (B, T, vocab_size)
    assert output.shape == (batch_size, seq_len, config.vocab_size), (
        f"Output shape mismatch: {output.shape} != ({batch_size}, {seq_len}, {config.vocab_size})"
    )

    print("  Basic Linformer test passed!")


def test_linformer_vs_standard():
    """Compare Linformer vs standard attention outputs"""
    print("\nTesting Linformer vs standard attention...")

    # Create two identical configs, one with Linformer, one without
    config_std = WeightApproxGPTConfig(
        sequence_len=256,
        n_embd=256,
        n_head=8,
        n_kv_head=8,
        n_layer=1,
        vocab_size=1000,
        use_linformer=False,
        build_by_layer=False,
        approx_type="svd",
        mlp_proj_rank=16,
    )

    config_lin = WeightApproxGPTConfig(
        sequence_len=256,
        n_embd=256,
        n_head=8,
        n_kv_head=8,
        n_layer=1,
        vocab_size=1000,
        use_linformer=True,
        linformer_proj_dim=64,
        linformer_sharing="layerwise",
        build_by_layer=False,
        approx_type="svd",
        mlp_proj_rank=16,
    )

    # Create models
    model_std = WeightApproxGPT(config_std, freeze_every=1000)
    model_lin = WeightApproxGPT(config_lin, freeze_every=1000)
    model_std.eval()
    model_lin.eval()

    # Copy weights (except Linformer projections)
    model_lin.transformer.wte.weight.data = (
        model_std.transformer.wte.weight.data.clone()
    )
    model_lin.lm_head.weight.data = model_std.lm_head.weight.data.clone()

    # Copy attention and MLP weights (but not Linformer projections)
    for (name_std, param_std), (name_lin, param_lin) in zip(
        model_std.named_parameters(), model_lin.named_parameters()
    ):
        if "linformer" not in name_lin and "shared_linformer" not in name_lin:
            if param_std.shape == param_lin.shape:
                param_lin.data = param_std.data.clone()

    # Create dummy input
    batch_size = 1
    seq_len = 256
    x = torch.randint(0, config_std.vocab_size, (batch_size, seq_len))

    # Forward passes
    with torch.no_grad():
        output_std = model_std(x)
        output_lin = model_lin(x)

    print(f"  Standard attention output shape: {output_std.shape}")
    print(f"  Linformer output shape: {output_lin.shape}")

    # Check that outputs are different (due to approximation)
    diff = torch.abs(output_std - output_lin).mean()
    print(f"  Mean absolute difference: {diff:.6f}")

    # The outputs should be different but have reasonable scale
    assert diff > 1e-6, "Linformer and standard outputs too similar"
    assert diff < 100, "Outputs too different"

    print("  Linformer vs standard test passed!")


def test_different_sharing_strategies():
    """Test different parameter sharing strategies"""
    print("\nTesting different Linformer sharing strategies...")

    sharing_strategies = ["none", "headwise", "keyvalue", "layerwise"]

    for strategy in sharing_strategies:
        print(f"  Testing {strategy} sharing...")

        config = WeightApproxGPTConfig(
            sequence_len=128,
            n_embd=256,
            n_head=8,
            n_kv_head=8,
            n_layer=2,
            vocab_size=1000,
            use_linformer=True,
            linformer_proj_dim=32,
            linformer_sharing=strategy,
            build_by_layer=False,
            approx_type="svd",
            mlp_proj_rank=16,
        )

        # Create model
        model = WeightApproxGPT(config, freeze_every=1000)
        model.eval()

        # Count Linformer parameters
        linformer_params = sum(
            p.numel() for n, p in model.named_parameters() if "linformer" in n.lower()
        )

        print(f"    Linformer parameters: {linformer_params:,}")

        # Forward pass
        batch_size = 1
        seq_len = 128
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, seq_len, config.vocab_size)
        print(f"    {strategy} sharing works!")

    print("  All sharing strategies test passed!")


def test_parameter_counts():
    """Test that parameter sharing reduces parameter count"""
    print("\nTesting parameter counts for different sharing strategies...")

    base_config = dict(
        sequence_len=256,
        n_embd=256,
        n_head=8,
        n_kv_head=8,
        n_layer=2,
        vocab_size=1000,
        use_linformer=True,
        linformer_proj_dim=64,
        build_by_layer=False,
        approx_type="svd",
        mlp_proj_rank=16,
    )

    param_counts = {}
    for strategy in ["none", "headwise", "keyvalue", "layerwise"]:
        config = WeightApproxGPTConfig(**base_config, linformer_sharing=strategy)
        model = WeightApproxGPT(config, freeze_every=1000)
        linformer_params = sum(
            p.numel() for n, p in model.named_parameters() if "linformer" in n.lower()
        )
        param_counts[strategy] = linformer_params
        print(f"  {strategy}: {linformer_params:,} Linformer parameters")

    # Verify that sharing reduces parameters
    # layerwise should have fewest, none should have most
    assert param_counts["layerwise"] <= param_counts["headwise"], (
        "layerwise should have <= params than headwise"
    )
    assert param_counts["headwise"] <= param_counts["none"], (
        "headwise should have <= params than none"
    )

    print("  Parameter count test passed!")


def test_with_layer_building():
    """Test Linformer works with build_by_layer=True"""
    print("\nTesting Linformer with layer building...")

    config = WeightApproxGPTConfig(
        sequence_len=128,
        n_embd=128,
        n_head=4,
        n_kv_head=4,
        n_layer=1,  # Start with 1 layer
        vocab_size=1000,
        use_linformer=True,
        linformer_proj_dim=32,
        linformer_sharing="layerwise",
        build_by_layer=True,
        approx_type="svd",
        mlp_proj_rank=16,
    )

    model = WeightApproxGPT(config, freeze_every=1000)
    model.eval()

    # Test with initial layer
    batch_size = 1
    seq_len = 128
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output1 = model(x)
    assert output1.shape == (batch_size, seq_len, config.vocab_size)
    print(f"  Initial layer count: {len(model.transformer.h)}")

    # Add a block manually
    model.add_block(layer_idx=1)
    print(f"  After adding block: {len(model.transformer.h)} layers")

    with torch.no_grad():
        output2 = model(x)
    assert output2.shape == (batch_size, seq_len, config.vocab_size)

    # Verify shared projection is used
    if config.linformer_sharing == "layerwise":
        assert model.shared_linformer_proj is not None
        # Both blocks should reference the same projection
        block0_proj = model.transformer.h[0].attn.linformer_proj
        block1_proj = model.transformer.h[1].attn.linformer_proj
        assert block0_proj is block1_proj, (
            "Layerwise sharing should use same projection"
        )

    print("  Layer building test passed!")


def test_variable_sequence_length():
    """Test Linformer with sequences shorter than config.sequence_len"""
    print("\nTesting variable sequence lengths...")

    config = WeightApproxGPTConfig(
        sequence_len=256,  # Max sequence length
        n_embd=128,
        n_head=4,
        n_kv_head=4,
        n_layer=1,
        vocab_size=1000,
        use_linformer=True,
        linformer_proj_dim=64,
        linformer_sharing="layerwise",
        build_by_layer=False,
        approx_type="svd",
        mlp_proj_rank=16,
    )

    model = WeightApproxGPT(config, freeze_every=1000)
    model.eval()

    # Test with shorter sequence
    batch_size = 2
    short_seq_len = 64  # Shorter than config.sequence_len

    x = torch.randint(0, config.vocab_size, (batch_size, short_seq_len))

    with torch.no_grad():
        output = model(x)

    assert output.shape == (batch_size, short_seq_len, config.vocab_size)
    print(f"  Short sequence ({short_seq_len}) works!")

    # Test with full sequence
    x_full = torch.randint(0, config.vocab_size, (batch_size, config.sequence_len))
    with torch.no_grad():
        output_full = model(x_full)

    assert output_full.shape == (batch_size, config.sequence_len, config.vocab_size)
    print(f"  Full sequence ({config.sequence_len}) works!")

    print("  Variable sequence length test passed!")


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Run tests
    test_linformer_basic()
    test_linformer_vs_standard()
    test_different_sharing_strategies()
    test_parameter_counts()
    test_with_layer_building()
    test_variable_sequence_length()

    print("\nAll Linformer tests passed!")
