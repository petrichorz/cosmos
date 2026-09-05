#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

if [[ "$#" -ne 6 ]]; then
    echo "Usage: $0 <name-1> <path-1> <name-2> <path-2> <name-3> <path-3>" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_OUTPUT_ROOT="${PROFILE_OUTPUT_ROOT:-/public/Embodied-AI/outputs/vision-edge-profile}"
BASE_MASTER_PORT="${MASTER_PORT:-50019}"

dataset_names=("$1" "$3" "$5")
dataset_paths=("$2" "$4" "$6")

for index in 0 1 2; do
    dataset_name="${dataset_names[$index]}"
    dataset_path="${dataset_paths[$index]}"
    if [[ ! -f "$dataset_path/meta/info.json" ]] && ! find "$dataset_path" -path '*/meta/info.json' -print -quit | grep -q .; then
        echo "No LeRobot v3 meta/info.json found below: $dataset_path" >&2
        exit 1
    fi

    echo "[$((index + 1))/3] profiling dataset '$dataset_name' at '$dataset_path'"
    BENCHMARK_DATASET_NAME="$dataset_name" \
    DATASET_DIR="$dataset_path" \
    OUTPUT_ROOT="$PROFILE_OUTPUT_ROOT/$dataset_name" \
    MASTER_PORT="$((BASE_MASTER_PORT + index))" \
        bash "$SCRIPT_DIR/launch_sft_vision_edge_local.sh"
done

