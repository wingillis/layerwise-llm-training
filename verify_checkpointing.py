
import torch
from nanochat.approximated_gpt import WeightApproxGPT, WeightApproxGPTConfig

def test_gradient_checkpointing():
    print("Testing Gradient Checkpointing...")
    
    # 1. Init model with checkpointing
    config = WeightApproxGPTConfig(
        n_layer=2, 
        n_embd=64, 
        n_head=2, 
        gradient_checkpointing=True
    )
    model = WeightApproxGPT(config)
    
    # Needs to be in training mode for checkpointing to activate (based on our condition)
    model.train()
    
    # 2. Forward pass
    x = torch.randint(0, 100, (2, 16))
    targets = x.clone()
    
    print("Running forward pass...")
    loss = model(x, targets=targets)
    print(f"Loss: {loss.item()}")
    
    # 3. Backward pass
    print("Running backward pass...")
    loss.backward()
    
    print("Gradient checkpointing verification successful!")

if __name__ == "__main__":
    test_gradient_checkpointing()
