import torch
from nanochat.approximated_gpt import (
    WeightApproxGPT,
    ApproxWeightMLP,
)
from nanochat.gpt import GPT, GPTConfig


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
        dense_attn.c_proj.weight.data = _convert_linear_weight(approx_attn.c_proj).to(
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
