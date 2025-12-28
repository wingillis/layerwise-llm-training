## Formulation of the objective:

- to maximize the amount of approximations that are applied to the language model by implementing many optimizations others have previously explored.
- After training the approximated network, the full set of parameters are realized and a few more steps of pre-training is applied
- An ablation study will be performed by removing different combinations of the efficiency optimizations

## The axes of variation:

- training the model one layer at a time: Y/N (assume yes to start)
- Use Hadamard ABBA approximation
- Performing a low-rank approximation of the V matrix: Y/N
- Performing a low-rank approximation of the MLP weights in each block: Y/N
- Freeze weights of the previous layer: Y/N
- What is the optimal rank of each approximation? (do this after testing those other axes of variation)
- Train a full rank for a bit at the end: Y/N

Constraints:
- if building one layer at a time, we are only going to build in the forward direction.

In total, there are 16-32 variants to test (depending on if I want to train full rank for a bit at the end), not including finding the optimal approximation ranks.

Approach for modifying the codebase to allow for testing these axes of variation. 

### One layer at a time

Make a new construction function/method that adds a block to the `self.transformer.h` `ModuleList`. Should take in a few parameters to compile it, and to create a low-rank approximation of any parts of the transformer `Block`.

### Linformer-style Low-rank approximation of attention calculation

Create a low-rank approximation of attention with linformer causal attention.

### Low-rank approximation of the MLP projection weights

Re-write the MLP class to have a custom weight parameter set for the up and down projection weight matrices. These should replace the `nn.Linear` classes inside the MLP.

### Freeze previous layer weights

Modify the GPT-level model to create a method that freezes a layer specified in the function parameters.
