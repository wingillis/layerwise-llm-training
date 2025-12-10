#!/usr/bin/env bash

# set -e

source .env

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

if [ -z "$WANDB_RUN" ]; then
    echo "WANDB_RUN is not set. Setting to dummy."
    WANDB_RUN="dummy"
fi

# uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# uv run python -m nanochat.dataset -n 8
# uv run python -m nanochat.dataset -n 240 &
# DATASET_DOWNLOAD_PID=$!

# uv run python -m scripts.tok_train --max_chars=2000000000
# uv run python -m scripts.tok_eval

# echo "Waiting for dataset download to complete..."
# wait $DATASET_DOWNLOAD_PID

# pretrain the d20 model
uv run python -m scripts.base_efficient_train --depth 20 --approx-type=svd --run=test-svd 
uv run python -m scripts.base_train --run=base --depth=20 --device_batch_size=4 --num_iterations=10000
uv run python -m scripts.base_efficient_train --depth 20 --approx-type=abba --run=test-abba 
