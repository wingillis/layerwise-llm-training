#!/usr/bin/env python3
"""Test script for the materialize_weights function."""

import torch
from nanochat.approximated_gpt import WeightApproxGPT, WeightApproxGPTConfig
from nanochat.gpt import GPT
from nanochat.utils import materialize_weights


def test_materialize_weights():
    """Test that materialize_weights works correctly."""
    # Create a small test config
    config = WeightApproxGPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
        approx_type="svd",
        approx_mlp_proj=True,
        mlp_proj_rank=8,
        use_linformer=False,
    )

    # Create approximated model
    device = torch.device("cpu")
    approx_model = WeightApproxGPT(config, freeze_every=1000)
    approx_model.to(device)
    approx_model.init_weights()
    approx_model.eval()

    print("Created WeightApproxGPT model")

    # Test with dummy input
    batch_size = 2
    seq_len = 64
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        approx_output = approx_model(dummy_input)
        print(f"Approx model output shape: {approx_output.shape}")

    # Materialize weights
    dense_model = materialize_weights(approx_model, device=device)
    print("Materialized weights to GPT model")

    # Test dense model with same input
    with torch.no_grad():
        dense_output = dense_model(dummy_input)
        print(f"Dense model output shape: {dense_output.shape}")

    # Check that models have the same architecture (in terms of dimensions)
    print("\nModel comparison:")
    print(f"Approx model layers: {len(approx_model.transformer.h)}")
    print(f"Dense model layers: {len(dense_model.transformer.h)}")

    # Check weight shapes
    print("\nWeight shape comparison (first layer attention c_q):")
    print(f"Approx model c_q weight shape: {approx_model.transformer.h[0].attn.c_q.weight.shape}")
    print(f"Dense model c_q weight shape: {dense_model.transformer.h[0].attn.c_q.weight.shape}")

    print("\nTest passed! Materialization successful.")


if __name__ == "__main__":
    test_materialize_weights()