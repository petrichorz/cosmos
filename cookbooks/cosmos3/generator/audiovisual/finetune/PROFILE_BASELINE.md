# Cosmos3 Edge 一轮训练基准

该基准保留完整 dataloader、forward、backward、梯度裁剪、优化器和 EMA 训练逻辑，按全局送入训练的有效样本数停止。默认数据读取策略为 PyAV、解码阶段 resize 和 15 FPS 上限。

## 单个数据集

```bash
DATASET_DIR=/path/to/lerobot-v3 \
BENCHMARK_DATASET_NAME=my-dataset \
OUTPUT_ROOT=/path/to/output \
bash cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_edge_local.sh
```

脚本默认使用 8 张 NPU。可通过 `ASCEND_RT_VISIBLE_DEVICES` 和 `NPROC_PER_NODE` 覆盖。`max_iter` 仍是异常兜底；正常情况下达到一个逻辑 epoch 后由所有 rank 同步结束。基准模式默认不保存最终 checkpoint。

## 依次运行三个数据集

```bash
PROFILE_OUTPUT_ROOT=/path/to/profile-output \
bash cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/run_sft_vision_edge_profile_3datasets.sh \
    libero /path/to/libero \
    dataset-b /path/to/dataset-b \
    dataset-c /path/to/dataset-c
```

三个任务串行执行，分别使用独立输出目录和连续的 rendezvous 端口。每个路径必须是单个 LeRobot v3 根目录，或包含多个 LeRobot v3 子目录的父目录。

## 结果

结果位于：

```text
<OUTPUT_ROOT>/cosmos3/causal_pretrain/vision_causal_pretrain_2k_edge/benchmark/
```

- `rank_NNN.jsonl`：每个 rank、每个 iteration 的原始指标。
- `rank_NNN_summary.json`：每个 rank 的稳态百分位汇总。
- `iterations.csv`：按 iteration 聚合所有 rank，适合表格比较。
- `summary.json`：全局汇总。

主要指标包括 dataloader 暴露等待时间、host-to-device、forward、backward、optimizer、worker 视频解码与 resize、packing token 利用率、samples/frames/tokens 吞吐、主进程和 worker RSS/FD，以及 NPU allocated/reserved/peak memory。

`dataloader_train` 是训练进程实际等待 `next(dataloader)` 的时间；由于 worker prefetch，它不等于视频解码耗时。视频读取工作量应同时查看 `worker_step/video_decode_resize_seconds`。

`phase/*` 是不插入 NPU synchronize 的 host wall time，用于低开销观察迭代趋势；精确的 device kernel 和通信耗时应在下一阶段使用 Ascend Profiler 分析。

`sample_overshoot` 表示最后一个多卡 packed step 超过精确 epoch 边界的样本数。停止只能发生在 optimizer step 边界，以保证所有 FSDP rank 执行相同数量的 collective。
