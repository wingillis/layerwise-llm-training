"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

from functools import partial
from dataclasses import dataclass
from typing import Literal
from einops import einsum, rearrange
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.gpt import GPT, MLP


@dataclass
class WeightApproxGPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # Low-rank approximation settings
    approx_type: Literal["svd", "abba"] = (
        "svd"  # Type of linear approximation: "svd" or "abba"
    )
    approx_mlp_proj: bool = True  # Whether to use low-rank MLP weights approximation
    mlp_proj_rank: int = 16  # Rank for low-rank MLP weights approximation
    approx_lm_head: bool = False  # Whether to use low-rank approximation for lm_head
    lm_head_rank: int = 16  # Rank for low-rank lm_head approximation
    # Layer-wise training settings
    build_by_layer: bool = True  # Whether to build model layer-by-layer incrementally
    freeze_previous_weights: bool = (
        False  # Whether to freeze previous blocks during training
    )
    # Whether to use low-rank approximation for attention projections
    approx_attn: bool = False
    # Rank for low-rank attention approximation
    attn_rank: int = 16
    # Whether to use gradient checkpointing
    gradient_checkpointing: bool = False
    # Whether to tie embedding and lm_head weights (parameter sharing)
    tie_embeddings: bool = False


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last time into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    out = out.to(x.dtype)  # ensure input/output dtypes match
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        # Use ApproxLinear for attention projections if configured
        if config.approx_attn:
            self.c_q: nn.Module = ApproxLinear(
                self.n_embd,
                self.n_head * self.head_dim,
                config.approx_type,
                config.attn_rank,
                bias=False,
            )
            self.c_k: nn.Module = ApproxLinear(
                self.n_embd,
                self.n_kv_head * self.head_dim,
                config.approx_type,
                config.attn_rank,
                bias=False,
            )
            self.c_v: nn.Module = ApproxLinear(
                self.n_embd,
                self.n_kv_head * self.head_dim,
                config.approx_type,
                config.attn_rank,
                bias=False,
            )
            self.c_proj: nn.Module = ApproxLinear(
                self.n_embd,
                self.n_embd,
                config.approx_type,
                config.attn_rank,
                bias=False,
            )
        else:
            self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
            self.c_k = nn.Linear(
                self.n_embd, self.n_kv_head * self.head_dim, bias=False
            )
            self.c_v = nn.Linear(
                self.n_embd, self.n_kv_head * self.head_dim, bias=False
            )
            self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = (
            apply_rotary_emb(q, cos, sin),
            apply_rotary_emb(k, cos, sin),
        )  # QK rotary embedding
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Attention with causal masking
        enable_gqa = (
            self.n_head != self.n_kv_head
        )  # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=enable_gqa
        )

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class ApproxLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        approx_type: Literal["svd", "abba"] = "svd",
        rank: int = 16,
        bias: bool = False,
    ):
        super().__init__()
        min_features = min(in_features, out_features)
        if rank > min_features:
            print0(
                f"Warning! rank (r={rank}) is larger than the smallest feature: {min_features}. Setting to this value."
            )
            rank = min_features

        if approx_type == "svd":
            self.linear: nn.Module = ApproxLinearSVD(
                in_features, out_features, rank, bias
            )
        elif approx_type == "abba":
            self.linear: nn.Module = ApproxLinearABBA(
                in_features, out_features, rank, bias
            )
        else:
            raise ValueError(f"Unknown approximation type: {approx_type}")

    def forward(self, x) -> torch.Tensor:
        return self.linear(x)


class ApproxLinearSVD(nn.Module):
    def __init__(self, in_features, out_features, rank: int = 16, bias: bool = False):
        super().__init__()
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features

        # 1. Low-rank components (Standard LoRA-style initialization)
        # We initialize U to zero so the layer starts as just the diagonal part
        self.V = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.U = nn.Parameter(torch.zeros(rank, out_features))

        # 2. Diagonal component
        self.min_dim = min(in_features, out_features)
        self.D = nn.Parameter(
            torch.ones(1, self.min_dim)
        )  # Initialize to 1s (Identity-ish)

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Low-rank path (x @ V @ U)
        # Using standard @ is often faster/more memory-efficient than complex einsum
        # because it leverages highly optimized BLAS/cuBLAS kernels.
        result = (x @ self.V) @ self.U

        # Step 2: In-place diagonal addition
        # This modification happens directly on the 'result' tensor memory.
        if self.in_features >= self.out_features:
            # Compression or Square: Input is sliced to match Output
            result.addcmul_(x[..., : self.out_features], self.D)
        else:
            # Expansion: Only the first 'in_features' of the output get the diagonal
            result[..., : self.in_features].addcmul_(x, self.D)

        if self.bias is not None:
            result += self.bias

        return result

    def reconstruct_weight(self) -> torch.Tensor:
        """
        Reconstruct dense weight from SVD approximation.

        Given: V (in_features, rank), U (rank, out_features)
        Reconstruct: W = V @ U

        Returns:
            Reconstructed dense weight tensor of shape (in_features, out_features)
        """
        with torch.no_grad():
            return self.V @ self.U


class ApproxLinearABBA(nn.Module):
    def __init__(self, in_features, out_features, rank: int = 16, bias: bool = False):
        super().__init__()
        self.rank = rank // 2
        self.in_features = in_features
        self.out_features = out_features

        # A1: (rank, in_features), B1: (out_features, rank)
        # A2: (rank, in_features), B2: (out_features, rank)
        # Following paper convention: W1 = B1 @ A1, W2 = B2 @ A2
        self.A1 = nn.Parameter(torch.randn(self.rank, in_features) * 0.01)
        self.B1 = nn.Parameter(torch.randn(out_features, self.rank) * 0.01)
        self.A2 = nn.Parameter(torch.randn(self.rank, in_features) * 0.01)
        self.B2 = nn.Parameter(torch.randn(out_features, self.rank) * 0.01)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x) -> torch.Tensor:
        # from: https://arxiv.org/pdf/2505.14238v3
        # Using Khatri-Rao factorization:
        # (B1 @ A1) ⊙ (B2 @ A2) = (B1 ⊙_r B2) @ (A1^T ⊙_r A2^T)^T
        #
        # Row-wise Khatri-Rao: for matrices U (m x r1), V (m x r2)
        # (U ⊙_r V)[i] = outer product of row i of U with row i of V, flattened
        # Result shape: (m, r1*r2)

        # B_kr = B1 ⊙_r B2: (out_features, rank*rank)
        # Each row i: outer product of B1[i,:] and B2[i,:]
        B_kr = einsum(self.B1, self.B2, "o r1, o r2 -> o r1 r2")
        B_kr = rearrange(B_kr, "o r1 r2 -> o (r1 r2)")

        # A_kr = (A1^T ⊙_r A2^T)^T: (rank*rank, in_features)
        # A1^T is (in_features, rank), A2^T is (in_features, rank)
        # Row-wise KR of A1^T and A2^T: (in_features, rank*rank)
        # Then transpose: (rank*rank, in_features)
        A_kr = einsum(self.A1, self.A2, "r1 i, r2 i -> r1 r2 i")
        A_kr = rearrange(A_kr, "r1 r2 i -> (r1 r2) i")

        # Now compute x @ W^T where W = B_kr @ A_kr
        # x: (batch, [seq_len,] in_features)
        # A_kr: (rank^2, in_features)
        # B_kr: (out_features, rank^2)
        #
        # x @ W^T = x @ A_kr^T @ B_kr^T

        # Step 1: x @ A_kr^T -> (batch, [seq_len,] rank^2)
        xA = einsum(x, A_kr, "... i, r i -> ... r")

        # Step 2: result @ B_kr^T -> (batch, [seq_len,] out_features)
        result = einsum(xA, B_kr, "... r, o r -> ... o")

        if self.bias is not None:
            result = result + self.bias

        return result

    def reconstruct_weight(self) -> torch.Tensor:
        """
        Reconstruct dense weight from ABBA approximation.

        Given: A1, A2 (rank, in_features), B1, B2 (out_features, rank)
        Using Khatri-Rao factorization as described in the paper:
        W = (B1 ⊙_r B2) @ (A1^T ⊙_r A2^T)^T

        Returns:
            Reconstructed dense weight tensor of shape (out_features, in_features)
        """
        with torch.no_grad():
            # B_kr = B1 ⊙_r B2
            B_kr = einsum(self.B1, self.B2, "o r1, o r2 -> o r1 r2")
            B_kr = rearrange(B_kr, "o r1 r2 -> o (r1 r2)")

            # A_kr = (A1^T ⊙_r A2^T)^T
            A_kr = einsum(self.A1, self.A2, "r1 i, r2 i -> r1 r2 i")
            A_kr = rearrange(A_kr, "r1 r2 i -> (r1 r2) i")

            # Final weight: W = B_kr @ A_kr
            return B_kr @ A_kr


class ApproxWeightMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = ApproxLinear(
            config.n_embd,
            4 * config.n_embd,
            config.approx_type,
            config.mlp_proj_rank,
            bias=False,
        )
        self.c_proj = ApproxLinear(
            4 * config.n_embd,
            config.n_embd,
            config.approx_type,
            config.mlp_proj_rank,
            bias=False,
        )

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class ApproxWeightBlock(nn.Module):
    def __init__(
        self,
        config: WeightApproxGPTConfig,
        layer_idx,
    ):
        super().__init__()
        self.attn: nn.Module = CausalSelfAttention(config, layer_idx)
        if config.approx_mlp_proj:
            self.mlp: nn.Module = ApproxWeightMLP(config)
        else:
            self.mlp: nn.Module = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        attn_output = self.attn(norm(x), cos_sin, kv_cache)
        x = x + attn_output
        mlp_output = self.mlp(norm(x))
        x = x + mlp_output
        return x


class WeightApproxGPT(GPT):
    def __init__(self, config: WeightApproxGPTConfig, freeze_every: int | None = None):
        nn.Module.__init__(self)
        self.config = config

        blocks = [ApproxWeightBlock(config, layer_idx=i) for i in range(config.n_layer)]

        if config.build_by_layer:
            print0("Building by layer")
            for b in blocks[1:]:
                # freeze weights
                for param in b.parameters():
                    param.requires_grad = False

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(blocks),
            }
        )
        # Handle weight tying between wte and lm_head
        if config.tie_embeddings:
            if config.approx_lm_head:
                raise ValueError(
                    "tie_embeddings and approx_lm_head are mutually exclusive"
                )
            # self.lm_head = None  # Will use wte.weight in forward pass
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.lm_head.weight = self.transformer.wte.weight
        elif config.approx_lm_head:
            self.lm_head = ApproxLinear(
                config.n_embd,
                config.vocab_size,
                config.approx_type,
                config.lm_head_rank,
                bias=False,
            )
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = (
            config.sequence_len * 10
        )  # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer(
            "cos", cos, persistent=False
        )  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)
        # freeze weights of the current training layer every freeze_every layers
        self.freeze_every = freeze_every
        self.prev_gate_level = 0
        # Store optimizer references (set by setup_optimizers)
        self.adamw_optimizer = None
        self.muon_optimizer = None

    def add_block(self, layer_idx):
        # pyrefly: ignore
        for p in self.transformer.h[layer_idx].parameters():
            p.requires_grad = True

    @torch.no_grad()
    def _init_approx_linear_weights(self):
        """Initialize ApproxLinear parameters with dimension-aware scaling.

        Uses Kaiming/He-style initialization adapted for low-rank structures
        to maintain proper variance through the approximation layers.

        When model is created on meta device and moved with to_empty(),
        nn.Parameter values are lost. _init_weights only handles nn.Linear
        and nn.Embedding, so we need to explicitly initialize ApproxLinear params.
        """
        for module in self.modules():
            if isinstance(module, ApproxLinearSVD):
                # SVD: W = U @ V.T, need to maintain variance of effective weight
                fan_in = module.in_features
                rank = module.rank

                # Scale for maintaining variance in W = U @ V.T
                # std(W) ≈ std(U) * std(V) * sqrt(rank)
                scale = 1.0 / (math.sqrt(fan_in) / math.sqrt(rank))

                # Kaiming initialization followed by scaling (with no_grad)
                torch.nn.init.kaiming_normal_(
                    module.U, a=0, mode="fan_in", nonlinearity="linear"
                )
                torch.nn.init.kaiming_normal_(
                    module.V, a=0, mode="fan_in", nonlinearity="linear"
                )

                # Apply scale to maintain proper variance
                module.U *= scale
                module.V *= scale

            elif isinstance(module, ApproxLinearABBA):
                # ABBA: W = B_kr @ A_kr where B_kr and A_kr involve Khatri-Rao products
                # The Khatri-Rao product effectively squares the rank dimension
                # Note: ABBA splits the rank by 2 in its __init__
                fan_in = module.in_features
                actual_rank = module.rank  # This is already rank//2 due to ABBA init

                # Scale accounting for the two-stage computation
                # ABBA computes x @ A_kr @ B_kr where A_kr and B_kr are rank^2 in size
                # We need to scale by sqrt(fan_in) and also account for the rank expansion
                scale = 1.0 / (math.sqrt(fan_in) / math.sqrt(actual_rank))

                # Initialize all four matrices (with no_grad)
                for param in [module.A1, module.A2, module.B1, module.B2]:
                    torch.nn.init.kaiming_normal_(
                        param, a=0, mode="fan_in", nonlinearity="linear"
                    )
                    param *= scale

    def init_weights(self):
        self.apply(self._init_weights)

        # Initialize ApproxLinear parameters (not handled by _init_weights)
        self._init_approx_linear_weights()

        # zero out classifier weights (only for non-ApproxLinear lm_head and non-tied weights)
        # Skip when lm_head is None (weights are tied with wte)
        if self.lm_head is not None and not hasattr(self.lm_head, "linear"):
            # Standard linear layer - zero it out
            torch.nn.init.zeros_(self.lm_head.weight)  # pyrefly: ignore
        # Note: ApproxLinear lm_head keeps its random initialization
        # to provide gradient signal. Zeroing would cause zero output and vanishing gradients.
        # Note: When lm_head is None (tied weights), wte initialization applies to both.

        # zero out c_proj weights in all blocks following the first (AFTER _init_approx_linear_weights)
        for block_num, block in enumerate(
            self.transformer.h[1:], start=1
        ):  # pyrefly: ignore
            print0(f"In block {block_num}")
            # Handle approximated c_proj weights
            if hasattr(block.mlp.c_proj, "linear"):
                if hasattr(block.mlp.c_proj.linear, "U"):
                    # SVD approximation - zero out U
                    torch.nn.init.zeros_(block.mlp.c_proj.linear.U)
                elif hasattr(block.mlp.c_proj.linear, "B1"):
                    # ABBA approximation - zero out B1 and B2
                    torch.nn.init.zeros_(block.mlp.c_proj.linear.B1)
                    # torch.nn.init.zeros_(block.mlp.c_proj.linear.B2)
            else:
                # Standard linear layer
                torch.nn.init.zeros_(block.mlp.c_proj.weight)

            # Handle attention c_proj
            if hasattr(block.attn.c_proj, "linear"):
                if hasattr(block.attn.c_proj.linear, "U"):
                    # SVD approximation - zero out U
                    torch.nn.init.zeros_(block.attn.c_proj.linear.U)
                    # torch.nn.init.zeros_(block.attn.c_proj.linear.V)
                elif hasattr(block.attn.c_proj.linear, "B1"):
                    # ABBA approximation - zero out B1 and B2
                    torch.nn.init.zeros_(block.attn.c_proj.linear.B1)
                    torch.nn.init.zeros_(block.attn.c_proj.linear.B2)
                    # torch.nn.init.zeros_(block.attn.c_proj.linear.A1)
                    # torch.nn.init.zeros_(block.attn.c_proj.linear.A2)
            else:
                # Standard linear layer
                print0("Zeroing out attn.c_proj.weight")
                torch.nn.init.zeros_(block.attn.c_proj.weight)
                # if has bias, zero it out as well
                if block.attn.c_proj.bias is not None:
                    print0("Zeroing out attn.c_proj.bias")
                    torch.nn.init.zeros_(block.attn.c_proj.bias)

            # Handle attention projections (c_q, c_k, c_v)
            for proj_name, proj_layer in [
                ("c_q", block.attn.c_q),
                ("c_k", block.attn.c_k),
                ("c_v", block.attn.c_v),
            ]:
                if hasattr(proj_layer, "linear"):
                    if hasattr(proj_layer.linear, "U"):
                        # SVD approximation - zero out U
                        torch.nn.init.zeros_(proj_layer.linear.U)
                    elif hasattr(proj_layer.linear, "B1"):
                        # ABBA approximation - zero out B1 and B2
                        torch.nn.init.zeros_(proj_layer.linear.B1)
                        torch.nn.init.zeros_(proj_layer.linear.B2)
                else:
                    # Standard linear layer
                    torch.nn.init.zeros_(proj_layer.weight)
                    # if has bias, zero it out as well
                    if proj_layer.bias is not None:
                        torch.nn.init.zeros_(proj_layer.bias)

            with torch.no_grad():
                random_input = torch.randn(1, 1, self.config.n_embd, device="cuda")
                output = block.attn.c_proj(random_input)
                assert torch.allclose(output, torch.zeros_like(output)), (
                    f"Block {block_num} does not produce 0 for attn.c_proj"
                )

        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":  # pyrefly: ignore
            self.transformer.wte.to(dtype=torch.bfloat16)

    def setup_optimizers(
        self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0
    ):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())

        if self.lm_head is None:
            # Weight tying: lm_head shares weights with wte
            lm_head_params = []
            # Adjust assertion for tied weights (lm_head doesn't add params)
            assert len(list(self.parameters())) == len(matrix_params) + len(
                embedding_params
            )
        else:
            lm_head_params = list(self.lm_head.parameters())
            assert len(list(self.parameters())) == len(matrix_params) + len(
                embedding_params
            ) + len(lm_head_params)

        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print0(
                f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
            )

        if self.lm_head is None:
            # Weight tying: use only embedding_lr for tied weights
            adam_groups = [
                dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            ]
        else:
            adam_groups = [
                dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
                dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            ]

        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        # Store optimizer references
        self.adamw_optimizer = adamw_optimizer
        self.muon_optimizer = muon_optimizer
        return optimizers

    def check_gate_level(self, gate_level):
        # New implementation: only support forward direction by adding blocks
        if gate_level != self.prev_gate_level:
            print0(f"Gate level changed from {self.prev_gate_level} to {gate_level}")
        if (
            gate_level != self.prev_gate_level
            and self.training
            and gate_level > self.prev_gate_level  # Only forward direction
        ):
            # Add blocks for each new gate level needed
            self.add_block(gate_level)
            print0(f"Added new block at gate level {gate_level}")

            self.prev_gate_level = gate_level
            print0(f"New gate level: {gate_level}")

            if self.config.freeze_previous_weights:
                # Freeze all PREVIOUS blocks (not the newly added one)
                for i in range(gate_level):
                    for param in self.transformer.h[i].parameters():  # pyrefly: ignore
                        param.requires_grad = False

                # Also freeze embeddings on first block addition
                if gate_level == 1:  # Just added second block
                    for param in self.transformer.wte.parameters():  # pyrefly: ignore
                        param.requires_grad = False

        return gate_level

    def forward(
        self, idx, targets=None, loss_reduction="mean", step=None, kv_cache=None
    ):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), (
            f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        )
        assert idx.device == self.cos.device, (
            f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        )
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (
            self.cos[:, T0 : T0 + T],
            self.sin[:, T0 : T0 + T],
        )  # truncate to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)  # pyrefly: ignore
        x = norm(x)

        if self.config.build_by_layer:
            # Use only blocks up to gate_level
            gate_level = step // self.freeze_every if step is not None else 0
            self.check_gate_level(gate_level)
            # pyrefly: ignore
            blocks_to_use = self.transformer.h[: gate_level + 1]
        else:
            # Use all blocks regardless of gate_level
            blocks_to_use = self.transformer.h

        for block in blocks_to_use:  # pyrefly: ignore
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, cos_sin, kv_cache, use_reentrant=False
                )
            else:
                x = block(x, cos_sin, kv_cache)  # pyrefly: ignore

        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            if self.lm_head is None:
                # Weight tying: use wte.weight transposed
                logits = F.linear(x, self.transformer.wte.weight)
            else:
                logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # logits softcap
            # Keep logits in bfloat16 to save memory - cross_entropy works fine with it
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        else:
            # inference mode: compute and return the logits
            if self.lm_head is None:
                # Weight tying: use wte.weight transposed
                logits = F.linear(x, self.transformer.wte.weight)
            else:
                logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # logits softcap
            return logits

    def estimate_flops(self):
        raise NotImplementedError(
            "estimate_flops is not implemented for ApproximatedGPT"
        )

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
