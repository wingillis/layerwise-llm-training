
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
