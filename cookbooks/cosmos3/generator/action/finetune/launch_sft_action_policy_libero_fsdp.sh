#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Complete recipe: LIBERO-10 action-policy SFT on Cosmos3-Nano (native FSDP,
# single-node 8-GPU). Single-node counterpart to launch_sft_action_policy_libero.sh
# (HSDP 2x8): same gbs/lr/max_iter, but FSDP (shard 8 x replicate 1) + grad_accum 2
# instead of HSDP (shard 8 x replicate 2) + grad_accum 1. Native FSDP is the better
# fit on a single node - no cross-node all-gather and lower per-rank weight memory.
# Run from this folder with the cosmos-framework venv active (see README):
#   bash launch_sft_action_policy_libero_fsdp.sh
# It prepares the small dependencies, checks for the staged libero_10 dataset, and trains.
# Paths are fixed under this (git-ignored) folder, matching the reasoner finetune
# wrappers, while the TOML and tail-overrides match the cosmos-framework example.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export ASCEND_RT_VISIBLE_DEVICES="2,3"
TOML_FILE="toml/sft_config/action_policy_libero_fsdp.toml"
: "${LIBERO_ROOT:=$PWD/data/LIBERO_LeRobot_v3/libero_10}"
: "${BASE_CHECKPOINT_PATH:=$PWD/checkpoints/Cosmos3-Nano}"
: "${WAN_VAE_PATH:=$PWD/checkpoints/wan22_vae/Wan2.2_VAE.pth}"

# 1. Stage the libero_10 suite (the Table-20 reproduction trains on libero_10 ALONE).
if [[ ! -f "$LIBERO_ROOT/meta/info.json" ]]; then
    echo "Downloading nvidia/LIBERO_LeRobot_v3 (libero_10) ..."
    uvx hf@latest download --repo-type dataset nvidia/LIBERO_LeRobot_v3 \
        --include 'libero_10/**' --local-dir "$(dirname "$LIBERO_ROOT")"
fi
if [[ ! -f "$LIBERO_ROOT/meta/info.json" ]]; then
    cat >&2 <<EOF
ERROR: missing libero_10 dataset at:
  $LIBERO_ROOT

Expected a LeRobotDataset dir containing meta/info.json. Stage it with:
  uvx hf@latest download --repo-type dataset nvidia/LIBERO_LeRobot_v3 \\
      --include 'libero_10/**' --local-dir data/LIBERO_LeRobot_v3
or export LIBERO_ROOT=/path/to/libero_10.
EOF
    exit 1
fi

# 2. Download the Wan2.2 VAE (skipped if present).
if [[ ! -f "$WAN_VAE_PATH" ]]; then
    uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "$(dirname "$WAN_VAE_PATH")"
fi

# 3. Convert the base checkpoint to DCP (skipped if present).
if [[ ! -d "$BASE_CHECKPOINT_PATH" ]]; then
    python -m cosmos_framework.scripts.convert_model_to_dcp -o "$BASE_CHECKPOINT_PATH" --checkpoint-path Cosmos3-Nano
fi

# 4. Train (native FSDP single-node 8-GPU per the TOML; the TOML pins shard_degree=8,
#    so the default NPROC_PER_NODE=8 keeps the gbs=2048 budget). To use a different GPU
#    count, override model.config.parallelism.data_parallel_shard_degree via
#    EXTRA_TAIL_OVERRIDES and adjust trainer.grad_accum_iter to keep gbs constant
#    (e.g. 4 GPUs -> shard 4, grad_accum 4). The TOML reads these paths from the env.
export LIBERO_ROOT
export BASE_CHECKPOINT_PATH
export WAN_VAE_PATH

TAIL_OVERRIDES=()
if [[ -n "${EXTRA_TAIL_OVERRIDES:-}" ]]; then
    # EXTRA_TAIL_OVERRIDES is intentionally word-split to match the framework launcher UX.
    # shellcheck disable=SC2206
    TAIL_OVERRIDES=(${EXTRA_TAIL_OVERRIDES})
fi

export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE:-2}")
TORCHRUN_ARGS+=(--master_port="${MASTER_PORT:-50012}")
[[ -n "${NNODES:-}" ]] && TORCHRUN_ARGS+=(--nnodes="$NNODES")
[[ -n "${NODE_RANK:-}" ]] && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
[[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")

# Attach the VSCode debugger (debugpy server on 0.0.0.0:3002; RANK 0 blocks until
# the client attaches). Set ATTACH_VSCODE_DEBUGGER=1 (or true/yes/y) to enable.
ATTACH_VSCODE_DEBUGGER="${ATTACH_VSCODE_DEBUGGER:-0}"
TRAIN_ARGS=()
if [[ "${ATTACH_VSCODE_DEBUGGER,,}" =~ ^(1|true|yes|y)$ ]]; then
    TRAIN_ARGS+=(--attach_vscode_debugger)
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs/train}"
if (( ${#TAIL_OVERRIDES[@]} )); then
    IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}" torchrun "${TORCHRUN_ARGS[@]}" \
        -m cosmos_framework.scripts.train --sft-toml="$TOML_FILE" \
        "${TRAIN_ARGS[@]}" \
        -- "${TAIL_OVERRIDES[@]}"
else
    IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}" torchrun "${TORCHRUN_ARGS[@]}" \
        -m cosmos_framework.scripts.train --sft-toml="$TOML_FILE" \
        "${TRAIN_ARGS[@]}"
fi
