# Ascend NPU 适配:设备无关后端层(解耦 pynvml)

- 日期:2026-07-09
- 范围:`packages/cosmos3/cosmos_framework` 下直接使用 pynvml 的四个模块
- 目标:让 `distributed.init()` 及训练/推理路径在华为 Ascend NPU(pynvml/NVML 不可用)上可运行,同时保持 CUDA 路径行为不变

## 背景与问题

`distributed.py` / `device.py` / `device_monitor.py` / `inference/args.py` 直接调用 pynvml(NVIDIA NVML 的 Python 绑定)。Ascend 无 pynvml/NVML,这些路径无法运行。

`distributed.init()` 有四处与 Ascend 不兼容:

1. 顶层 `import pynvml`(`distributed.py:15`)——pynvml 缺失时整个模块加载失败。
2. `pynvml.nvmlInit()` + `Device(local_rank).get_cpu_affinity()`(`distributed.py:39,42-43`)——NVML 仅 NVIDIA 有。
3. `dist.init_process_group(backend="nccl", ...)`(`distributed.py:55`)——Ascend 用 `hccl`。
4. `libcudart.so` / `cudaDeviceSetLimit`(`distributed.py:62-67`)——CUDA runtime,已被 `INTERNAL` 守卫(外部为 False)。

现有 Ascend 适配(已合入):
- `scripts/train.py:31-35`:`try: import torch_npu; from torch_npu.contrib import transfer_to_npu`。`transfer_to_npu` 把 `torch.cuda.*` 重定向到 `torch.npu.*`,因此 `torch.cuda.set_device` / `torch.cuda.current_device` 在 NPU 上可用。
- `pyproject.toml` `ascend` extra:`torch==2.10.0+cpu` + `torch_npu==2.10.0`(需主机 CANN)。

## 决策(已与用户确认)

1. **范围**:做全框架 pynvml 设备无关层,覆盖 `distributed.py` / `device.py` / `device_monitor.py` / `inference/args.py` 四处。
2. **后端检测**:`try import torch_npu` 且 `torch.npu.is_available()` -> NPU;`elif torch.cuda.is_available()` -> CUDA;else CPU。自动探测,不给 `flags.py` 的 `Device` StrEnum 加 `npu`。
3. **NPU 内存监控**:用 `torch.npu` API 实现(`torch.npu.mem_get_info` 取 free/total,`used = total - free`)。
4. **CPU-设备亲和性(`get_cpu_affinity`)**:**完全去掉,不做任何 CPU-设备绑定 / NUMA pinning**。原 pynvml 亲和性是纯性能优化(把进程钉到设备所在 NUMA 节点的 CPU 核,减少跨 NUMA 访存),非正确性需求;去掉后由 OS 默认调度放置,训练/推理功能不受影响。
5. **`Device` 类(`device.py`)**:**删除**——它仅为亲和性存在,全局无其它调用者。
6. **`device.py` 死代码**:`get_gpu_architecture` / `print_gpu_mem` / `gpu0_has_80gb_or_less` / `force_gc` **删除**(实现时再 grep 确认无外部调用者);`with_torch_device` 保留(pynvml 无关)。
7. **pynvml**:不再在任何模块顶层 import,仅 `CudaBackend` 方法内懒加载。Ascend 环境选 `NpuBackend`,根本不 import pynvml。

## 架构

新增 `cosmos_framework/utils/device_backend.py` 作为唯一设备抽象层。模块加载时探测一次,选定 `BACKEND` 单例,并暴露 `IS_NPU` / `IS_CUDA` / `DIST_BACKEND` 常量。

数据流:
- `distributed.init()` 只用 `IS_CUDA` / `DIST_BACKEND` 两个常量(不需要 `BACKEND` 对象,也不直接用 `IS_NPU`)。
- `device_monitor.py` / `inference/args.py` 用 `BACKEND` 对象 + `MemoryInfo`。
- `device.py` 不再被 `distributed.py` 导入其 `Device` 类(类已删)。

## 接口

```
@dataclass
class MemoryInfo:        # 设备无关返回类型,单位 bytes
    total: int
    used: int
    free: int

class DeviceBackend:     # ABC
    init() -> None
    get_handle(idx: int) -> Any
    get_memory_info(handle) -> MemoryInfo | None
    shutdown() -> None
```

三个实现:
- `CudaBackend`:pynvml 懒加载(方法内 `import pynvml`)。`init`=`nvmlInit`;`get_handle`=`nvmlDeviceGetHandleByIndex`;`get_memory_info` 先 `nvmlDeviceGetMemoryInfo_v2`,捕 `NVMLError_NotSupported` 回退 `nvmlDeviceGetMemoryInfo`(搬运 `args.py` 现有 `_get_nvml_device_memory_info` 逻辑);`shutdown`=`nvmlShutdown`。任何异常返回/记 `None`。
- `NpuBackend`:`init`/`shutdown` no-op;`get_handle(idx)` 返回 `idx`(torch.npu 按 idx 查);`get_memory_info` 用 `torch.npu.mem_get_info(handle)`(假设返回 `(free, total)`,实现时核对),`used = total - free`,异常返回 `None`。
- `CpuBackend`:全 no-op,`get_handle` 返回 `None`,`get_memory_info` 返回 `None`。

常量:`IS_NPU`、`IS_CUDA = (not IS_NPU) and torch.cuda.is_available()`、`DIST_BACKEND = "hccl" if IS_NPU else ("nccl" if IS_CUDA else "gloo")`。

## 文件级改动

### 1. 新建 `cosmos_framework/utils/device_backend.py`
如上"架构""接口"两节。模块末尾:`BACKEND = NpuBackend() if IS_NPU else CudaBackend() if IS_CUDA else CpuBackend()`。

### 2. `cosmos_framework/utils/distributed.py`(只改 `init` + import)
- 删 `import pynvml`(L15)、删 `from cosmos_framework.utils.device import Device`(L21)。
- 加 `from cosmos_framework.utils.device_backend import IS_CUDA, DIST_BACKEND`。
- `init()`:
  - 删整段亲和性(`pynvml.nvmlInit()` + `Device(local_rank)` + `os.sched_setaffinity` try/except,L39-45)。
  - `torch.cuda.set_device(local_rank)` 保留(依赖 `transfer_to_npu` 重定向到 `torch.npu.set_device`)。
  - `dist.init_process_group(backend=DIST_BACKEND, ...)`(原 `"nccl"` 改 `DIST_BACKEND`)。
  - `libcudart` 块守卫改 `if INTERNAL and IS_CUDA:`(原 `if INTERNAL:`)。
  - 早返回 `torch.cuda.current_device()` 保留(同依赖 `transfer_to_npu`)。
  - 两个 `TORCH_NCCL_*` env 保留(HCCL 无害)。

### 3. `cosmos_framework/utils/device.py`(大幅删减)
- 删 `import pynvml`、删 `Device` 类、删 `get_gpu_architecture` / `print_gpu_mem` / `force_gc` / `gpu0_has_80gb_or_less`。
- 保留 `with_torch_device`。

### 4. `cosmos_framework/callbacks/device_monitor.py`
- 删 `import pynvml`,加 `from cosmos_framework.utils.device_backend import BACKEND, MemoryInfo`。
- `on_train_start`:**新增 `BACKEND.init()`**(原靠 `distributed.init()` 顺带 `nvmlInit`,现已移除,必须自己 init),再 `self.handle = BACKEND.get_handle(local_rank)`。
- `every_n_impl`:`memory_info = BACKEND.get_memory_info(self.handle)`;`if memory_info is None:` 记 0 或跳过对应列;否则 `used = memory_info.used / 1024**3` 等。`self.handle` 为 `None`(CPU)时自然走 None 分支。

### 5. `cosmos_framework/inference/args.py`(L1292-1315 推理内存检查)
- 删顶层 `import pynvml`(L12)。
- 该块改为:`BACKEND.init()` -> `handle = BACKEND.get_handle(0)` -> `info = BACKEND.get_memory_info(handle)` -> `BACKEND.shutdown()`。
- `_get_nvml_device_memory_info`(v2 回退)逻辑搬进 `CudaBackend.get_memory_info`,本地函数删除。
- `info` 由裸 NVML meminfo 变 `MemoryInfo`,字段名同为 `.total/.used/.free`,消费端基本不动;`info is None` 时走原"not supported"分支。

## 错误处理

- **pynvml 缺失**:仅 `CudaBackend` 方法内懒加载触发;Ascend 选 `NpuBackend`,不 import pynvml,模块顶层加载不受影响。
- **NPU 内存 API 失败 / 不存在**:`NpuBackend.get_memory_info` 捕异常返回 `None`,调用方(device_monitor/args.py)按 `None` 走"记 0 或 not supported"分支,不中断训练/推理。
- **`distributed.init()`**:亲和性删除后无任何 NVML 调用;`libcudart` 块由 `INTERNAL and IS_CUDA` 双重守卫;`nvmlInit` 不再由 `init()` 调用(改由 `device_monitor`/`args.py` 经 `BACKEND.init()` 按需调用)。
- **`device_monitor` 必须自己 `BACKEND.init()`**:原代码隐式依赖 `distributed.init()` 的 `nvmlInit`,该依赖移除后需显式 init。

## 测试

- `_detect_npu` / `BACKEND` 选择:单测 mock `torch_npu` / `torch.cuda` 可用性,验证三种分支。
- 各后端 `get_memory_info`:mock pynvml / `torch.npu`,验证返回 `MemoryInfo(total,used,free)` 形状与 `None` 降级。
- `inference/args_test.py` 用了 pynvml:同步改为 mock `BACKEND` 或跳过 NPU/CUDA 相关断言。
- NPU 路径无 Ascend CI,靠 mock `torch.npu`;`transfer_to_npu` 重定向行为不在单测覆盖范围(依赖真实 torch_npu)。

## 实现时核对项

1. `torch.npu.mem_get_info(idx)` 的返回顺序/参数(本设计假设 `(free, total)`)。
2. `device_monitor` 当前是否自己 `nvmlInit`(若是,第 4 节"新增 init"调整为保留即可)。
3. `device.py` 待删函数 + `with_torch_device` 的真实外部调用者(grep 确认死代码)。
4. `transfer_to_npu` 是否重定向 `torch.cuda.set_device` / `current_device`(若不全,给 `DeviceBackend` 补 `set_device(idx)` / `current_device()` 方法,`distributed.init()` 改用之)。
