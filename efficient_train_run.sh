#!/usr/bin/env bash

set -e

source .env

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

# set WANDB_RUN to dummy if it's not set
if [ -z "$WANDB_RUN" ]; then
    echo "WANDB_RUN is not set. Setting to dummy."
    WANDB_RUN="dummy"
fi

# download and tokenize dataset if --tokenize is passed
if [ "$1" == "--tokenize" ]; then
    echo "Downloading and tokenizing dataset..."
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

    uv run python -m nanochat.dataset -n 8
    uv run python -m nanochat.dataset -n 240 &
    DATASET_DOWNLOAD_PID=$!

    uv run python -m scripts.tok_train --max_chars=2000000000
    uv run python -m scripts.tok_eval

    echo "Waiting for dataset download to complete..."
    wait $DATASET_DOWNLOAD_PID
fi


# pretrain the LLM

# uv run python -m scripts.base_train --run=base --depth=20 --device_batch_size=4 --num_iterations=10000
uv run python -m scripts.base_efficient_train --depth 20 --approx-type=svd --run=test-svd 
# NOTE: there's something wrong with abba initialization.
# uv run python -m scripts.base_efficient_train --depth 20 --approx-type=abba --run=test-abba 

# TODO: implement torchrun optimizations for distributed training (if necessary)
