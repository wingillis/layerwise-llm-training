"""
Simple test to verify materialize_weights preserves model behavior.
Can be used as a quick regression test.
"""

import torch
from nanochat.approximated_gpt import WeightApproxGPT, WeightApproxGPTConfig
from nanochat.utils import materialize_weights


def test_materialization_preserves_behavior():
    """Test that materialized model produces identical outputs."""
    # Small test model
    config = WeightApproxGPTConfig(
        sequence_len=64,
        vocab_size=100,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        approx_type="abba",
        approx_mlp_proj=True,
        mlp_proj_rank=8,
        build_by_layer=False,
    )

    # Create and initialize model
    model = WeightApproxGPT(config, freeze_every=1000)
    model.init_weights()
    model.eval()

    # Materialize weights
    dense_model = materialize_weights(model)
    dense_model.eval()

    print("Original model type:", type(model))
    print("Dense model type:", type(dense_model))

    # Test input
    x = torch.randint(0, config.vocab_size, (1, 32))

    # Compare outputs
    with torch.no_grad():
        original_out = model(x)
        materialized_out = dense_model(x)

    # Check they're identical
    assert torch.allclose(original_out, materialized_out, atol=1e-6), (
        "Materialized model doesn't match original!"
    )

    print("✓ Materialization preserves model behavior")
    return True


if __name__ == "__main__":
    torch.manual_seed(42)
    test_materialization_preserves_behavior()
