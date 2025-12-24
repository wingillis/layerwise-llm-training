import torch
import torch.nn as nn

class ApproxLinearSVD(nn.Module):
    def __init__(self, in_features, out_features, rank: int = 16, bias: bool = False):
        super().__init__()
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.V = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.U = nn.Parameter(torch.randn(rank, out_features) * 0.01) 
        self.min_dim = min(in_features, out_features)
        self.D = nn.Parameter(torch.randn(1, self.min_dim))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = (x @ self.V) @ self.U
        if self.in_features >= self.out_features:
            result.addcmul_(x[..., :self.out_features], self.D)
        else:
            result[..., :self.in_features].addcmul_(x, self.D)
        if self.bias is not None:
            result += self.bias
        return result

def verify_equivalence(in_features, out_features, rank=16):
    print(f"Verifying shape: {in_features} -> {out_features}")
    model = ApproxLinearSVD(in_features, out_features, rank, bias=True)
    x = torch.randn(10, in_features)
    
    # Current implementation
    y_actual = model(x)
    
    # Decomposition 1: Full weight matrix
    # W = V @ U + D_diag
    # We need to construct D_diag carefully matching the slicing logic
    W_lr = model.V @ model.U
    
    W_diag = torch.zeros(in_features, out_features)
    min_dim = model.min_dim
    for i in range(min_dim):
        W_diag[i, i] = model.D[0, i]
    
    W_total = W_lr + W_diag
    y_full = x @ W_total + model.bias
    
    # Decomposition 2: torch.addmm style (conceptually)
    # y = bias + (x@V)@U + (x_sliced * D)
    x_sliced = x[..., :out_features] if in_features >= out_features else x
    diag_term = x_sliced * model.D
    
    # Note: addmm(input, mat1, mat2) is input + mat1 @ mat2
    # So we can do: torch.addmm(bias + diag_term, x @ V, U)
    # But diag_term needs to have the same shape as result.
    diag_term_full = torch.zeros_like(y_actual)
    if in_features >= out_features:
        diag_term_full = x[..., :out_features] * model.D
    else:
        diag_term_full[..., :in_features] = x * model.D
        
    y_addmm = torch.addmm(model.bias + diag_term_full, x @ model.V, model.U)

    # Case 3: D is always in_features
    # Proposed: y = bias + (x@V)@U + (x * D_in) -> but this only works if in == out
    # To make it work for in != out while avoiding slicing x for the *multiplication*:
    # We still need to slice the result of (x * D_in) before adding to y if in > out,
    # or pad it if in < out.
    D_fixed = nn.Parameter(torch.randn(1, in_features))
    
    # Implementing the "fixed D" logic
    # x * D_fixed is (batch, in_features)
    x_scaled = x * D_fixed
    
    # We STILL need to match dimensions to add to y (batch, out_features)
    if in_features >= out_features:
        # Compression: we ignore the extra dimensions of x_scaled
        diag_term_fixed = x_scaled[..., :out_features]
    else:
        # Expansion: we pad with zeros
        diag_term_fixed = torch.zeros_like(y_actual)
        diag_term_fixed[..., :in_features] = x_scaled
    
    y_fixed = (x @ model.V) @ model.U + diag_term_fixed + model.bias

    diff_full = (y_actual - y_full).abs().max().item()
    diff_addmm = (y_actual - y_addmm).abs().max().item()
    diff_fixed = (y_actual - y_fixed).abs().max().item()
    
    print(f"  Max Diff (Full Matrix): {diff_full:.2e}")
    print(f"  Max Diff (addmm Style): {diff_addmm:.2e}")
    print(f"  Fixed-size D test: {diff_fixed:.2e}")
    
    # Using allclose with standard tolerances for float32
    assert torch.allclose(y_actual, y_full, atol=1e-5), f"Full matrix mismatch: {diff_full}"
    assert torch.allclose(y_actual, y_addmm, atol=1e-5), f"addmm style mismatch: {diff_addmm}"
    # assert torch.allclose(y_actual, y_fixed, atol=1e-5), f"Fixed-size D mismatch: {diff_fixed}"

if __name__ == "__main__":
    # Test cases: Square, Compression, Expansion
    verify_equivalence(128, 128)
    verify_equivalence(256, 128)
    verify_equivalence(128, 256)
    print("All tests passed!")
