#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 当前机器的运行环境。
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HF_HUB_OFFLINE=1
export COSMOS_DEVICE=npu

# 当前机器的数据集、权重和输出路径。
export DATASET_DIR="/path/to/BridgeData2-Subset-Synthetic-Captions"
export CHECKPOINT_DIR="/path/to/Cosmos3-Edge-DCP"
export COSMOS3_EDGE_PROCESSOR_PATH="/path/to/Cosmos3-Edge"
export VAE_PATH="/path/to/Wan2.2_VAE.pth"
export OUTPUT_ROOT="/path/to/outputs/train"

# torchrun 多机多卡配置。
# 单机时保持 NNODES=1、NODE_RANK=0、MASTER_ADDR=127.0.0.1。
# 多机时，每台机器使用相同的 NNODES、MASTER_ADDR 和 MASTER_PORT，
# 并分别设置不同的 NODE_RANK（从 0 开始）。
export NPROC_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=50012

bash "$SCRIPT_DIR/launch_sft_vision_edge.sh"
