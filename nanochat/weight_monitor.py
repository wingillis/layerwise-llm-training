"""
Weight monitoring utilities for per-layer weight observability.
"""

import torch
from typing import Dict, Any


class WeightMonitor:
    """Collects and formats per-layer weight statistics for logging."""

    def __init__(self, model, log_frequency: int = 10):
        self.model = model
        self.log_frequency = log_frequency

    def _should_log(self, step: int) -> bool:
        """Check if we should log at this step."""
        return step % self.log_frequency == 0

    def _compute_weight_stats(self, weight: torch.Tensor, prefix: str) -> Dict[str, float]:
        """Compute mean, std, and norm for a weight tensor."""
        stats = {}
        with torch.no_grad():
            stats[f"{prefix}/mean"] = weight.mean().item()
            stats[f"{prefix}/std"] = weight.std().item()
            stats[f"{prefix}/norm"] = weight.norm().item()
        return stats

    def _collect_standard_linear_weights(self, linear_layer, prefix: str, weight_data: Dict[str, float]) -> None:
        """Collect weights from a standard linear layer."""
        if hasattr(linear_layer, 'weight'):
            weight_data.update(self._compute_weight_stats(linear_layer.weight, prefix))

    def _collect_svd_weights(self, svd_layer, prefix: str, weight_data: Dict[str, float]) -> None:
        """Collect weights from an SVD approximated layer."""
        # Log individual parameter stats
        weight_data.update(self._compute_weight_stats(svd_layer.U, f"{prefix}/U"))
        weight_data.update(self._compute_weight_stats(svd_layer.V, f"{prefix}/V"))

        # Reconstruct and log full weight matrix stats
        reconstructed = svd_layer.reconstruct_weight()
        weight_data.update(self._compute_weight_stats(reconstructed, f"{prefix}/reconstructed"))

    def _collect_abba_weights(self, abba_layer, prefix: str, weight_data: Dict[str, float]) -> None:
        """Collect weights from an ABBA approximated layer."""
        # Log individual parameter stats
        weight_data.update(self._compute_weight_stats(abba_layer.A1, f"{prefix}/A1"))
        weight_data.update(self._compute_weight_stats(abba_layer.A2, f"{prefix}/A2"))
        weight_data.update(self._compute_weight_stats(abba_layer.B1, f"{prefix}/B1"))
        weight_data.update(self._compute_weight_stats(abba_layer.B2, f"{prefix}/B2"))

        # Reconstruct and log full weight matrix stats
        reconstructed = abba_layer.reconstruct_weight()
        weight_data.update(self._compute_weight_stats(reconstructed, f"{prefix}/reconstructed"))

    def _collect_linear_weights(self, linear_layer, prefix: str, weight_data: Dict[str, float]) -> None:
        """Collect weights from a linear layer (standard or approximated)."""
        if hasattr(linear_layer, 'linear'):
            # Approximated layer (SVD or ABBA)
            linear = linear_layer.linear

            # Check if it's SVD (has U and V)
            if hasattr(linear, 'U') and hasattr(linear, 'V'):
                self._collect_svd_weights(linear, prefix, weight_data)
            # Check if it's ABBA (has A1, A2, B1, B2)
            elif hasattr(linear, 'A1') and hasattr(linear, 'A2'):
                self._collect_abba_weights(linear, prefix, weight_data)
        else:
            # Standard linear layer
            self._collect_standard_linear_weights(linear_layer, prefix, weight_data)

    def _collect_attention_weights(self, block, prefix: str, weight_data: Dict[str, float]) -> None:
        """Collect weights for attention components."""
        attn = block.attn
        attn_prefix = f"{prefix}/attn"

        # Query projection
        self._collect_linear_weights(attn.c_q, f"{attn_prefix}/q", weight_data)

        # Key projection
        self._collect_linear_weights(attn.c_k, f"{attn_prefix}/k", weight_data)

        # Value projection
        self._collect_linear_weights(attn.c_v, f"{attn_prefix}/v", weight_data)

        # Output projection
        self._collect_linear_weights(attn.c_proj, f"{attn_prefix}/proj", weight_data)

    def _collect_mlp_weights(self, block, prefix: str, weight_data: Dict[str, float]) -> None:
        """Collect weights for MLP components."""
        mlp = block.mlp
        mlp_prefix = f"{prefix}/mlp"

        # Feed-forward expansion
        self._collect_linear_weights(mlp.c_fc, f"{mlp_prefix}/fc", weight_data)

        # Feed-forward contraction
        self._collect_linear_weights(mlp.c_proj, f"{mlp_prefix}/proj", weight_data)

    def collect_weight_stats(self, step: int) -> Dict[str, float]:
        """Collect weight statistics for all active layers.

        Args:
            step: Current training step

        Returns:
            Dictionary mapping weight names to their statistics
        """
        if not self._should_log(step):
            return {}

        weight_data = {}

        # Get current gate level (number of active layers)
        gate_level = self.model.prev_gate_level if hasattr(self.model, 'prev_gate_level') else len(self.model.transformer.h) - 1

        # Embedding layer weights
        if hasattr(self.model.transformer, 'wte') and self.model.transformer.wte.weight is not None:
            weight_data.update(self._compute_weight_stats(self.model.transformer.wte.weight, "train/weights/embedding"))

        # Per-layer weights
        for i in range(gate_level + 1):
            block = self.model.transformer.h[i]
            layer_prefix = f"train/weights/layer_{i}"

            # Attention weights
            self._collect_attention_weights(block, layer_prefix, weight_data)

            # MLP weights
            self._collect_mlp_weights(block, layer_prefix, weight_data)

        # LM Head weights
        if hasattr(self.model, 'lm_head'):
            if hasattr(self.model.lm_head, 'linear'):
                # Approximated lm_head
                linear = self.model.lm_head.linear

                # Check if it's SVD (has U and V)
                if hasattr(linear, 'U') and hasattr(linear, 'V'):
                    self._collect_svd_weights(linear, "train/weights/lm_head", weight_data)
                # Check if it's ABBA (has A1, A2, B1, B2)
                elif hasattr(linear, 'A1') and hasattr(linear, 'A2'):
                    self._collect_abba_weights(linear, "train/weights/lm_head", weight_data)
            else:
                # Standard lm_head
                if hasattr(self.model.lm_head, 'weight') and self.model.lm_head.weight is not None:
                    weight_data.update(self._compute_weight_stats(self.model.lm_head.weight, "train/weights/lm_head"))

        return weight_data