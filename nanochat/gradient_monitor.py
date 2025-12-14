"""
Gradient monitoring utilities for per-layer gradient observability.
"""

class GradientMonitor:
    """Collects and formats per-layer gradient norms for logging."""

    def __init__(self, model):
        self.model = model

    def collect_grad_norms(self, step: int) -> dict[str, float]:
        """Collect gradient norms for all active layers.

        Args:
            step: Current training step (used for potential future extensions)

        Returns:
            Dictionary mapping gradient names to their L2 norms
        """
        grad_data = {}

        # Get current gate level (number of active layers)
        gate_level = self.model.prev_gate_level if hasattr(self.model, 'prev_gate_level') else len(self.model.transformer.h) - 1

        # Embedding layer gradients
        if self.model.transformer.wte.weight.grad is not None:
            grad_data["train/gradients/embedding/norm"] = self.model.transformer.wte.weight.grad.norm().item()

        # Per-layer gradients
        for i in range(gate_level + 1):
            block = self.model.transformer.h[i]
            layer_prefix = f"train/gradients/layer_{i}"

            # Attention gradients
            self._collect_attention_grads(block, layer_prefix, grad_data)

            # MLP gradients
            self._collect_mlp_grads(block, layer_prefix, grad_data)

        # LM Head gradients
        if hasattr(self.model, 'lm_head'):
            if hasattr(self.model.lm_head, 'linear'):
                # Approximated lm_head
                if hasattr(self.model.lm_head.linear, 'U') and self.model.lm_head.linear.U.grad is not None:
                    grad_data["train/gradients/lm_head/U/norm"] = self.model.lm_head.linear.U.grad.norm().item()
                if hasattr(self.model.lm_head.linear, 'V') and self.model.lm_head.linear.V.grad is not None:
                    grad_data["train/gradients/lm_head/V/norm"] = self.model.lm_head.linear.V.grad.norm().item()
                # ABBA parameters
                if hasattr(self.model.lm_head.linear, 'A1') and self.model.lm_head.linear.A1.grad is not None:
                    grad_data["train/gradients/lm_head/A1/norm"] = self.model.lm_head.linear.A1.grad.norm().item()
                if hasattr(self.model.lm_head.linear, 'A2') and self.model.lm_head.linear.A2.grad is not None:
                    grad_data["train/gradients/lm_head/A2/norm"] = self.model.lm_head.linear.A2.grad.norm().item()
                if hasattr(self.model.lm_head.linear, 'B1') and self.model.lm_head.linear.B1.grad is not None:
                    grad_data["train/gradients/lm_head/B1/norm"] = self.model.lm_head.linear.B1.grad.norm().item()
                if hasattr(self.model.lm_head.linear, 'B2') and self.model.lm_head.linear.B2.grad is not None:
                    grad_data["train/gradients/lm_head/B2/norm"] = self.model.lm_head.linear.B2.grad.norm().item()
            else:
                # Standard lm_head
                if self.model.lm_head.weight.grad is not None:
                    grad_data["train/gradients/lm_head/norm"] = self.model.lm_head.weight.grad.norm().item()

        return grad_data

    def _collect_attention_grads(self, block, prefix: str, grad_data: dict[str, float]) -> None:
        """Collect gradients for attention components."""
        attn = block.attn
        attn_prefix = f"{prefix}/attn"

        # Query projection
        self._collect_linear_grads(attn.c_q, f"{attn_prefix}/q", grad_data)

        # Key projection
        self._collect_linear_grads(attn.c_k, f"{attn_prefix}/k", grad_data)

        # Value projection
        self._collect_linear_grads(attn.c_v, f"{attn_prefix}/v", grad_data)

        # Output projection
        self._collect_linear_grads(attn.c_proj, f"{attn_prefix}/proj", grad_data)

    def _collect_mlp_grads(self, block, prefix: str, grad_data: dict[str, float]) -> None:
        """Collect gradients for MLP components."""
        mlp = block.mlp
        mlp_prefix = f"{prefix}/mlp"

        # Feed-forward expansion
        self._collect_linear_grads(mlp.c_fc, f"{mlp_prefix}/fc", grad_data)

        # Feed-forward contraction
        self._collect_linear_grads(mlp.c_proj, f"{mlp_prefix}/proj", grad_data)

    def _collect_linear_grads(self, linear_layer, prefix: str, grad_data: dict[str, float]) -> None:
        """Collect gradients from a linear layer (standard or approximated)."""
        if hasattr(linear_layer, 'linear'):
            # Approximated layer (SVD or ABBA)
            linear = linear_layer.linear

            # For SVD: has U and V
            if hasattr(linear, 'U') and linear.U.grad is not None:
                grad_data[f"{prefix}/U/norm"] = linear.U.grad.norm().item()
            if hasattr(linear, 'V') and linear.V.grad is not None:
                grad_data[f"{prefix}/V/norm"] = linear.V.grad.norm().item()

            # For ABBA: has A1, A2, B1, B2
            if hasattr(linear, 'A1') and linear.A1.grad is not None:
                grad_data[f"{prefix}/A1/norm"] = linear.A1.grad.norm().item()
            if hasattr(linear, 'A2') and linear.A2.grad is not None:
                grad_data[f"{prefix}/A2/norm"] = linear.A2.grad.norm().item()
            if hasattr(linear, 'B1') and linear.B1.grad is not None:
                grad_data[f"{prefix}/B1/norm"] = linear.B1.grad.norm().item()
            if hasattr(linear, 'B2') and linear.B2.grad is not None:
                grad_data[f"{prefix}/B2/norm"] = linear.B2.grad.norm().item()
        else:
            # Standard linear layer
            if linear_layer.weight.grad is not None:
                grad_data[f"{prefix}/norm"] = linear_layer.weight.grad.norm().item()