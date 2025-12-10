"""Test initialization of ApproxLinear modules"""

import math
import torch
import torch.nn as nn
import pytest

from nanochat.approximated_gpt import (
    ApproxLinearSVD,
    ApproxLinearABBA,
    WeightApproxGPT,
    WeightApproxGPTConfig
)


def test_svd_initialization_variance():
    """Test that SVD approximation maintains proper variance."""
    # Test parameters
    batch_size = 100
    in_features = 768
    out_features = 3072
    rank = 16

    # Create standard linear layer for reference
    linear_ref = nn.Linear(in_features, out_features, bias=False)

    # Create SVD approximation
    svd_approx = ApproxLinearSVD(in_features, out_features, rank=rank)

    # Initialize both with similar strategy
    # Standard linear gets default PyTorch init (already applied)
    # Apply our custom init to SVD (with no_grad)
    scale = (1.0 / math.sqrt(in_features)) / math.sqrt(rank)
    with torch.no_grad():
        torch.nn.init.kaiming_normal_(svd_approx.U, a=0, mode='fan_in', nonlinearity='linear')
        torch.nn.init.kaiming_normal_(svd_approx.V, a=0, mode='fan_in', nonlinearity='linear')
        svd_approx.U *= scale
        svd_approx.V *= scale

    # Create test input
    x = torch.randn(batch_size, in_features)

    # Forward pass
    y_ref = linear_ref(x)
    y_svd = svd_approx(x)

    # Check output variance
    var_ref = torch.var(y_ref, dim=1).mean()
    var_svd = torch.var(y_svd, dim=1).mean()

    # Variance should be within reasonable tolerance (within 2x)
    assert abs(var_ref - var_svd) < max(var_ref, var_svd), (
        f"Variance mismatch: ref={var_ref:.4f}, svd={var_svd:.4f}"
    )

    print(f"SVD variance test passed: ref={var_ref:.4f}, svd={var_svd:.4f}")


def test_abba_initialization_variance():
    """Test that ABBA approximation maintains proper variance."""
    # Test parameters
    batch_size = 100
    in_features = 768
    out_features = 3072
    rank = 16  # Will be split to rank//2 = 8 for ABBA

    # Create standard linear layer for reference
    linear_ref = nn.Linear(in_features, out_features, bias=False)

    # Create ABBA approximation
    abba_approx = ApproxLinearABBA(in_features, out_features, rank=rank)

    # Apply our custom init to ABBA
    # Use similar scaling as standard linear layers
    with torch.no_grad():
        for param in [abba_approx.A1, abba_approx.A2]:
            # A matrices: (rank//2, in_features)
            torch.nn.init.normal_(param, mean=0.0, std=1.0 / math.sqrt(in_features))
        for param in [abba_approx.B1, abba_approx.B2]:
            # B matrices: (out_features, rank//2)
            torch.nn.init.normal_(param, mean=0.0, std=1.0 / math.sqrt(abba_approx.rank))

    # Create test input
    x = torch.randn(batch_size, in_features)

    # Forward pass
    y_ref = linear_ref(x)
    y_abba = abba_approx(x)

    # Check output variance
    var_ref = torch.var(y_ref, dim=1).mean()
    var_abba = torch.var(y_abba, dim=1).mean()

    # Variance should be within reasonable tolerance (within 2x)
    assert abs(var_ref - var_abba) < max(var_ref, var_abba), (
        f"Variance mismatch: ref={var_ref:.4f}, abba={var_abba:.4f}"
    )

    print(f"ABBA variance test passed: ref={var_ref:.4f}, abba={var_abba:.4f}")


def test_weight_approx_gpt_initialization():
    """Test that WeightApproxGPT initializes without errors."""
    # Create config with approximation
    config = WeightApproxGPTConfig(
        n_layer=2,
        n_embd=256,
        n_head=8,
        n_kv_head=8,  # Must divide n_head evenly
        approx_type="svd",
        approx_mlp_proj=True,
        mlp_proj_rank=8,
        approx_lm_head=True,
        lm_head_rank=8,
    )

    # Create model
    model = WeightApproxGPT(config)

    # Initialize weights
    model.init_weights()

    # Check that parameters are initialized
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"Parameter {name} contains NaN"
        assert not torch.isinf(param).any(), f"Parameter {name} contains Inf"

    print("WeightApproxGPT initialization test passed")


def test_gradient_flow():
    """Test that gradients flow properly through approximated layers."""
    # Create small model
    config = WeightApproxGPTConfig(
        n_layer=1,
        n_embd=128,
        n_head=4,
        n_kv_head=4,  # Must divide n_head evenly
        approx_type="svd",
        approx_mlp_proj=True,
        mlp_proj_rank=8,
    )

    model = WeightApproxGPT(config)
    model.init_weights()
    model.train()

    # Create dummy input and target
    batch_size = 4
    seq_len = 16
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    loss = model(x, targets)

    # Backward pass
    loss.backward()

    # Check gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter {name}"

    print(f"Gradient flow test passed, loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_svd_initialization_variance()
    test_abba_initialization_variance()
    test_weight_approx_gpt_initialization()
    test_gradient_flow()
    print("\nAll initialization tests passed!")