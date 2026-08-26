#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Cosmos 3 Generator pre-training-style optimizer trial for the causal Edge
# workflow. Run from any directory with the cosmos-framework environment active:
#   bash launch_causal_pretrain_2k_edge.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

: "${DATASET_DIR:=$PWD/data/BridgeData2-Subset-Synthetic-Captions}"
: "${CHECKPOINT_DIR:=$PWD/checkpoints/Cosmos3-Edge}"
: "${COSMOS3_EDGE_PROCESSOR_PATH:=$PWD/checkpoints/Cosmos3-Edge}"
: "${VAE_PATH:=$PWD/checkpoints/wan22_vae/Wan2.2_VAE.pth}"

if [[ ! -f "$DATASET_DIR/sft_dataset_bridge/train/video_dataset_file.jsonl" ]]; then
    uvx hf@latest download --repo-type dataset nvidia/BridgeData2-Subset-Synthetic-Captions \
        --revision 40d018ac1c1a2a4b9734f17fdb21f3d933c49a01 --local-dir "$DATASET_DIR"
fi

if [[ ! -f "$VAE_PATH" ]]; then
    uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "$(dirname "$VAE_PATH")"
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    python -m cosmos_framework.scripts.convert_model_to_dcp -o "$CHECKPOINT_DIR" --checkpoint-path Cosmos3-Edge
fi

export DATASET_PATH="$DATASET_DIR/sft_dataset_bridge"
export BASE_CHECKPOINT_PATH="$CHECKPOINT_DIR"
export WAN_VAE_PATH="$VAE_PATH"
TOML_PATH="$PWD/toml/sft_config/vision_causal_pretrain_2k_edge.toml"
: "${OUTPUT_ROOT:=$PWD/outputs/causal_pretrain_2k_edge}"

TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE:-8}")
TORCHRUN_ARGS+=(--master_port="${MASTER_PORT:-50012}")
[[ -n "${NNODES:-}" ]] && TORCHRUN_ARGS+=(--nnodes="$NNODES")
[[ -n "${NODE_RANK:-}" ]] && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
[[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")

cd "$(python -c 'import pathlib, cosmos_framework; print(pathlib.Path(cosmos_framework.__file__).resolve().parents[1])')"
IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" torchrun "${TORCHRUN_ARGS[@]}" \
    -m cosmos_framework.scripts.train --sft-toml="$TOML_PATH" -- \
    model=mot_causal_fsdp \
    model.config.vlm_config.tokenizer.repository=null \
    model.config.vlm_config.tokenizer.revision=null \
    +model.config.vlm_config.tokenizer.tokenizer_type="$COSMOS3_EDGE_PROCESSOR_PATH" \
    '~dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:0.7,1:0.2,2:0.1}' \
    '+dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:1.0}'
