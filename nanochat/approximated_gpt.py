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

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.gpt import GPT


@dataclass
class GPTConfig:
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
    mlp_proj_rank: int = 16  # Rank for low-rank MLP weights approximation
    # Layer-wise training settings
    build_by_layer: bool = True  # Whether to build model layer-by-layer incrementally
    copy_block_weights: bool = (
        True  # Whether new blocks copy weights from previous layer
    )
    freeze_previous_weights: bool = (
        False  # Whether to freeze previous blocks during training
    )


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
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
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

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)  # number of queries in this forward pass
        # number of keys/values in total (in the cache + current forward pass)
        Tk = k.size(2)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = (
            self.n_head != self.n_kv_head
        )  # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=enable_gqa
            )
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=False, enable_gqa=enable_gqa
            )
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros(
                (Tq, Tk), dtype=torch.bool, device=q.device
            )  # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0:  # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa
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
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.U = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(in_features, rank) * 0.01)

    def forward(self, x) -> torch.Tensor:
        result = einsum(x, self.V, self.U, "b i, i r, r o -> b o")
        if self.bias is not None:
            result = result + self.bias
        return result


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
        # x: (batch, in_features)
        # A_kr: (rank^2, in_features)
        # B_kr: (out_features, rank^2)
        #
        # x @ W^T = x @ A_kr^T @ B_kr^T

        # Step 1: x @ A_kr^T -> (batch, rank^2)
        xA = einsum(x, A_kr, "b i, r i -> b r")

        # Step 2: result @ B_kr^T -> (batch, out_features)
        result = einsum(xA, B_kr, "b r, o r -> b o")

        if self.bias is not None:
            result = result + self.bias

        return result


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
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = ApproxWeightMLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class WeightApproxGPT(GPT):
    def __init__(self, config, freeze_every: int):
        nn.Module.__init__(self)
        self.config = config
        if config.build_by_layer:
            self.transformer = nn.ModuleDict(
                # pyrefly: ignore
                {
                    "wte": torch.compile(
                        nn.Embedding(config.vocab_size, config.n_embd), dynamic=False
                    ),
                    "h": nn.ModuleList(
                        [  # pyrefly: ignore
                            torch.compile(
                                ApproxWeightBlock(config, layer_idx=0), dynamic=False
                            )
                        ]
                    ),
                }
            )
        else:
            self.transformer = nn.ModuleDict(
                {
                    "wte": nn.Embedding(config.vocab_size, config.n_embd),
                    "h": nn.ModuleList(
                        [
                            ApproxWeightBlock(config, layer_idx=layer_idx)
                            for layer_idx in range(config.n_layer)
                        ]
                    ),
                }
            )
            # pyrefly: ignore
            self.transformer = torch.compile(self.transformer, dynamic=False)
        self.lm_head = torch.compile(
            nn.Linear(config.n_embd, config.vocab_size, bias=False), dynamic=False
        )
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
        block = ApproxWeightBlock(self.config, layer_idx)
        if self.config.copy_block_weights:
            block.load_state_dict(
                self.transformer.h[layer_idx - 1].state_dict()
            )  # pyrefly: ignore
        # pyrefly: ignore
        self.transformer.h.append(torch.compile(block, dynamic=False))
        self.config.n_layer += 1

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

        # TODO: figure out if I need this
        # freeze the weights of all blocks except the first training layer
        # loop = self.transformer.h if self.reverse_train_order else self.transformer.h[::-1]
        # for block in loop[1:]:
        #     for param in block.parameters():
        #         param.requires_grad = False

    def setup_optimizers(
        self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0
    ):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
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

    def _add_block_to_optimizer(self, block):
        """Add new block's parameters to the Muon optimizer.

        Handles both Muon (groups by numel) and DistMuon (groups by shape).
        """
        if self.muon_optimizer is None:
            return  # Optimizers not set up yet

        ddp, rank, local_rank, world_size = get_dist_info()
        is_dist_muon = ddp

        # Get all parameters from the new block
        new_params = list(block.parameters())

        # Group parameters by shape (DistMuon) or numel (Muon)
        def key_fn(p):
            return p.shape if is_dist_muon else p.numel()

        new_params_by_key = {}
        for p in new_params:
            key = key_fn(p)
            if key not in new_params_by_key:
                new_params_by_key[key] = []
            new_params_by_key[key].append(p)

        # Add to existing param_group or create new one
        for key, params in new_params_by_key.items():
            added_to_existing = False

            # Try to find existing group with matching key
            for group in self.muon_optimizer.param_groups:
                if len(group["params"]) == 0:
                    continue

                existing_param = group["params"][0]
                group_key = key_fn(existing_param)

                if group_key == key:
                    # Add to existing group
                    group["params"].extend(params)
                    added_to_existing = True
                    if rank == 0:
                        print0(
                            f"Added {len(params)} params to existing Muon group (key={key})"
                        )
                    break

            # Create new group if no match found
            if not added_to_existing:
                new_group = {"params": params}
                if is_dist_muon and len(params) > 0:
                    new_group["zero_buffer"] = torch.zeros_like(params[0])

                self.muon_optimizer.add_param_group(new_group)
                if rank == 0:
                    print0(
                        f"Created new Muon param group with {len(params)} params (key={key})"
                    )

    def check_gate_level(self, gate_level):
        # New implementation: only support forward direction by adding blocks
        if (
            gate_level != self.prev_gate_level
            and self.training
            and gate_level > self.prev_gate_level  # Only forward direction
        ):
            # Add blocks for each new gate level needed
            layer_idx = len(self.transformer.h)  # pyrefly: ignore
            self.add_block(layer_idx)
            print0(f"Added new block at layer {layer_idx}")

            self.prev_gate_level = gate_level
            print0(f"New gate level: {gate_level}")

            if self.config.freeze_previous_weights:
                # Freeze all PREVIOUS blocks (not the newly added one)
                for i in range(layer_idx):
                    for param in self.transformer.h[i].parameters():  # pyrefly: ignore
                        param.requires_grad = False

                # Also freeze embeddings on first block addition
                if layer_idx == 1:  # Just added second block
                    for param in self.transformer.wte.parameters():  # pyrefly: ignore
                        param.requires_grad = False

            # Register the NEW block's parameters with optimizer
            # (always do this since the new block is trainable)
            new_block = self.transformer.h[layer_idx]  # pyrefly: ignore
            self._add_block_to_optimizer(new_block)

        return gate_level

    def forward(
        self, idx, targets=None, kv_cache=None, loss_reduction="mean", step=None
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
        )  # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        gate_level = step // self.freeze_every if step is not None else 0
        # Remove reverse training order logic since we only support forward direction
        # if self.reverse_train_order and gate_level is not None:
        #     gate_level = max(len(self.transformer.h) - gate_level - 1, 0)

        self.check_gate_level(gate_level)

        for i, block in enumerate(self.transformer.h):  # pyrefly: ignore
            # Only support forward direction - process blocks up to gate_level
            if i > gate_level:
                break
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # logits softcap
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # logits softcap
            return logits

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
