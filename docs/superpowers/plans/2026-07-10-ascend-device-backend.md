# Ascend NPU 设备无关后端层 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 解耦 `distributed/device/device_monitor/args` 对 pynvml 的硬依赖,新增 `device_backend.py` 抽象层(CUDA/NPU/CPU 三后端),使 Ascend NPU 训练/推理可运行。

**Architecture:** 新增 `cosmos_framework/utils/device_backend.py`,模块加载时按 `torch_npu`/`torch.cuda` 可用性自动探测后端,暴露 `BACKEND` 单例与 `IS_NPU`/`IS_CUDA`/`DIST_BACKEND` 常量。pynvml 仅在 `CudaBackend` 方法内懒加载,不再任何模块顶层 import。`distributed.init()` 用常量选 `hccl`/`nccl`;`device_monitor`/`args.py` 用 `BACKEND` 查内存。`Device` 类与 `device.py` 死代码删除;CPU-设备亲和性完全去掉(不做 NUMA pinning)。

**Tech Stack:** Python 3, PyTorch, torch_npu(Ascend), pynvml(CUDA only,懒加载), pytest(monkeypatch mock)。

## Global Constraints

- pynvml **不得**在任何模块顶层 `import`;仅 `CudaBackend` 方法内懒加载。
- 后端检测:`try import torch_npu` 且 `torch.npu.is_available()` -> NPU;`elif torch.cuda.is_available()` -> CUDA;else CPU。不给 `flags.py` 的 `Device` StrEnum 加 `npu`。
- `DIST_BACKEND = "hccl" if IS_NPU else ("nccl" if IS_CUDA else "gloo")`。
- 不做任何 CPU-设备亲和 / NUMA pinning(`get_cpu_affinity` 与 `Device` 类删除)。
- 测试:pytest,`*_test.py` 与源码同目录;pynvml / `torch.npu` 用 `monkeypatch` 注入 fake,无真实 NPU CI。
- `MemoryInfo(total, used, free)` 单位 bytes,作为设备无关内存返回类型。
- 保留 `device.py` 的 `with_torch_device`(pynvml 无关);其余 pynvml 相关代码删除。

---

## File Structure

- **Create** `cosmos_framework/utils/device_backend.py` — 唯一设备抽象层(探测 + `MemoryInfo` + `DeviceBackend` ABC + `Cuda/Npu/Cpu` 后端 + `BACKEND` 单例 + `IS_NPU`/`IS_CUDA`/`DIST_BACKEND`)。
- **Create** `cosmos_framework/utils/device_backend_test.py` — 后端选择与各后端 `get_memory_info` 单测。
- **Modify** `cosmos_framework/utils/distributed.py` — `init()` 改用 `IS_CUDA`/`DIST_BACKEND`,删 `import pynvml`、`from ...device import Device`、亲和性段、`libcudart` 加 CUDA 守卫。
- **Modify** `cosmos_framework/utils/device.py` — 删 `Device` 类 + `get_gpu_architecture`/`print_gpu_mem`/`force_gc`/`gpu0_has_80gb_or_less` + pynvml import;保留 `with_torch_device`。
- **Modify** `cosmos_framework/callbacks/device_monitor.py` — 内存查询改 `BACKEND`,`on_train_start` 加 `BACKEND.init()`,`torch.cuda.temperature/power_draw/utilization/clock_rate` 加 try/except(NPU 安全)。
- **Modify** `cosmos_framework/inference/args.py` — `_get_device_memory_bytes` 改 `BACKEND`,删 `_get_nvml_device_memory_info`,删顶层 `import pynvml`。
- **Modify** `cosmos_framework/inference/args_test.py` — 删已移除函数的 import 与两个 v2 回退测试(迁至 `device_backend_test.py`)。

---

## Task 1: 创建 `device_backend.py` 抽象层

**Files:**
- Create: `cosmos_framework/utils/device_backend.py`
- Create: `cosmos_framework/utils/device_backend_test.py`

**Interfaces:**
- Consumes: `torch`(顶层),`torch_npu`(仅 `_npu_available` 内 try-import),`pynvml`(仅 `CudaBackend` 方法内懒加载)。
- Produces: `MemoryInfo(total, used, free)`,`DeviceBackend`(ABC:`init/get_handle/get_memory_info/shutdown`),`CudaBackend`/`NpuBackend`/`CpuBackend`,`IS_NPU`/`IS_CUDA`/`DIST_BACKEND`,`BACKEND` 单例,`select_backend_kind(npu_available=None, cuda_available=None) -> str`。

- [ ] **Step 1: 写失败测试 — 后端选择 + CpuBackend**

Create `cosmos_framework/utils/device_backend_test.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import sys
import types

import pytest

from cosmos_framework.utils import device_backend
from cosmos_framework.utils.device_backend import (
    CpuBackend,
    MemoryInfo,
    select_backend_kind,
)


def test_select_backend_kind_npu_wins():
    assert select_backend_kind(npu_available=True, cuda_available=True) == "npu"


def test_select_backend_kind_cuda_when_no_npu():
    assert select_backend_kind(npu_available=False, cuda_available=True) == "cuda"


def test_select_backend_kind_cpu_when_neither():
    assert select_backend_kind(npu_available=False, cuda_available=False) == "cpu"


def test_cpu_backend_get_memory_info_is_none():
    backend = CpuBackend()
    backend.init()
    assert backend.get_handle(0) is None
    assert backend.get_memory_info(None) is None
    backend.shutdown()


def test_memory_info_fields():
    info = MemoryInfo(total=100, used=30, free=70)
    assert info.total == 100 and info.used == 30 and info.free == 70
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `cd packages/cosmos3 && python -m pytest cosmos_framework/utils/device_backend_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cosmos_framework.utils.device_backend'`

- [ ] **Step 3: 实现最小骨架 — 探测 + MemoryInfo + ABC + CpuBackend + 常量**

Create `cosmos_framework/utils/device_backend.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Device-agnostic backend layer.

Abstracts pynvml (CUDA) / torch.npu (Ascend NPU) / CPU so the rest of the
framework never imports pynvml at module top level. Selection is automatic:
torch_npu available -> NPU; elif torch.cuda available -> CUDA; else CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401  -- importing registers torch.npu
        return bool(torch.npu.is_available())
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def select_backend_kind(npu_available: Optional[bool] = None, cuda_available: Optional[bool] = None) -> str:
    """Return 'npu' | 'cuda' | 'cpu'. Params injectable for tests."""
    if npu_available is None:
        npu_available = _npu_available()
    if cuda_available is None:
        cuda_available = _cuda_available()
    if npu_available:
        return "npu"
    if cuda_available:
        return "cuda"
    return "cpu"


_KIND = select_backend_kind()
IS_NPU: bool = _KIND == "npu"
IS_CUDA: bool = _KIND == "cuda"
DIST_BACKEND: str = "hccl" if IS_NPU else ("nccl" if IS_CUDA else "gloo")


@dataclass
class MemoryInfo:
    """Device-agnostic memory snapshot (bytes)."""

    total: int
    used: int
    free: int


class DeviceBackend:
    """Abstract device backend."""

    def init(self) -> None:
        raise NotImplementedError

    def get_handle(self, idx: int) -> Any:
        raise NotImplementedError

    def get_memory_info(self, handle: Any) -> Optional[MemoryInfo]:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


class CpuBackend(DeviceBackend):
    """No-op backend for CPU runs (e.g. checkpoint conversion, smoke tests)."""

    def init(self) -> None:
        pass

    def get_handle(self, idx: int) -> Any:
        return None

    def get_memory_info(self, handle: Any) -> Optional[MemoryInfo]:
        return None

    def shutdown(self) -> None:
        pass


def _select_backend() -> DeviceBackend:
    # CudaBackend / NpuBackend wired in Task 1 Steps 7 / 11.
    if IS_NPU:
        return CpuBackend()  # placeholder until NpuBackend lands
    if IS_CUDA:
        return CpuBackend()  # placeholder until CudaBackend lands
    return CpuBackend()


BACKEND: DeviceBackend = _select_backend()
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `cd packages/cosmos3 && python -m pytest cosmos_framework/utils/device_backend_test.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: 写失败测试 — CudaBackend v2 优先 + v1 回退**

Append to `cosmos_framework/utils/device_backend_test.py`:

```python
class _FakeMemInfo:
    def __init__(self, total, used, free):
        self.total, self.used, self.free = total, used, free


def _install_fake_pynvml(monkeypatch, *, v2=None, v1=None):
    fake = types.ModuleType("pynvml")

    class _NVMLError_NotSupported(Exception):
        pass

    fake.NVMLError_NotSupported = _NVMLError_NotSupported
    fake.NVMLError = Exception
    fake.nvmlInit = lambda: None
    fake.nvmlShutdown = lambda: None
    fake.nvmlDeviceGetHandleByIndex = lambda idx: f"handle{idx}"
    if v2 is not None:
        fake.nvmlDeviceGetMemoryInfo_v2 = v2
    fake.nvmlDeviceGetMemoryInfo = v1 if v1 is not None else (lambda h: _FakeMemInfo(0, 0, 0))
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    return fake


def test_cuda_backend_prefers_v2(monkeypatch):
    from cosmos_framework.utils.device_backend import CudaBackend

    expected = _FakeMemInfo(total=1000, used=200, free=800)
    _install_fake_pynvml(
        monkeypatch,
        v2=lambda _h: expected,
        v1=lambda _h: pytest.fail("v1 must not be called when v2 succeeds"),
    )
    b = CudaBackend()
    info = b.get_memory_info("h")
    assert info == MemoryInfo(total=1000, used=200, free=800)


def test_cuda_backend_falls_back_to_v1_when_v2_absent(monkeypatch):
    from cosmos_framework.utils.device_backend import CudaBackend

    expected = _FakeMemInfo(total=1000, used=200, free=800)
    fake = _install_fake_pynvml(monkeypatch, v1=lambda _h: expected)
    # v2 absent -> AttributeError on access
    assert not hasattr(fake, "nvmlDeviceGetMemoryInfo_v2")
    b = CudaBackend()
    info = b.get_memory_info("h")
    assert info == MemoryInfo(total=1000, used=200, free=800)


def test_cuda_backend_falls_back_on_not_supported(monkeypatch):
    from cosmos_framework.utils.device_backend import CudaBackend

    expected = _FakeMemInfo(total=1000, used=200, free=800)
    fake = _install_fake_pynvml(monkeypatch, v1=lambda _h: expected)

    def v2_raises(_h):
        raise fake.NVMLError_NotSupported()

    fake.nvmlDeviceGetMemoryInfo_v2 = v2_raises
    b = CudaBackend()
    info = b.get_memory_info("h")
    assert info == MemoryInfo(total=1000, used=200, free=800)


def test_cuda_backend_returns_none_on_failure(monkeypatch):
    from cosmos_framework.utils.device_backend import CudaBackend

    def v2(_h):
        raise fake.NVMLError_NotSupported()

    def v1(_h):
        raise RuntimeError("boom")

    fake = _install_fake_pynvml(monkeypatch, v2=v2, v1=v1)
    b = CudaBackend()
    assert b.get_memory_info("h") is None
```

- [ ] **Step 6: 运行测试,确认 CudaBackend 测试失败**

Run: `cd packages/cosmos3 && python -m pytest cosmos_framework/utils/device_backend_test.py -v`
Expected: FAIL — `ImportError: cannot import name 'CudaBackend'`

- [ ] **Step 7: 实现 CudaBackend 并接入 `_select_backend`**

Edit `cosmos_framework/utils/device_backend.py` — 在 `CpuBackend` 之后、`_select_backend` 之前插入:

```python
class CudaBackend(DeviceBackend):
    """pynvml-backed CUDA backend. pynvml is imported lazily inside methods."""

    def init(self) -> None:
        import pynvml

        pynvml.nvmlInit()

    def get_handle(self, idx: int) -> Any:
        import pynvml

        return pynvml.nvmlDeviceGetHandleByIndex(idx)

    def get_memory_info(self, handle: Any) -> Optional[MemoryInfo]:
        import pynvml

        try:
            try:
                mi = pynvml.nvmlDeviceGetMemoryInfo_v2(handle)
            except AttributeError:
                mi = pynvml.nvmlDeviceGetMemoryInfo(handle)
            except pynvml.NVMLError_NotSupported:
                mi = pynvml.nvmlDeviceGetMemoryInfo(handle)
        except Exception:
            return None
        return MemoryInfo(total=int(mi.total), used=int(mi.used), free=int(mi.free))

    def shutdown(self) -> None:
        import pynvml

        pynvml.nvmlShutdown()
```

并修改 `_select_backend` 的 CUDA 分支:

```python
def _select_backend() -> DeviceBackend:
    if IS_NPU:
        return CpuBackend()  # placeholder until NpuBackend lands
    if IS_CUDA:
        return CudaBackend()
    return CpuBackend()
```

- [ ] **Step 8: 运行测试,确认通过**

Run: `cd packages/cosmos3 && python -m pytest cosmos_framework/utils/device_backend_test.py -v`
Expected: PASS (9 tests)

- [ ] **Step 9: 写失败测试 — NpuBackend**

Append to `cosmos_framework/utils/device_backend_test.py`:

```python
class _FakeNpu:
    def __init__(self, mem_get_info):
        self._mem_get_info = mem_get_info

    def mem_get_info(self, idx):
        return self._mem_get_info(idx)


def test_npu_backend_memory_info(monkeypatch):
    import torch
    from cosmos_framework.utils.device_backend import NpuBackend

    # torch.npu.mem_get_info mirrors torch.cuda.mem_get_info -> (free, total)
    fake_npu = _FakeNpu(mem_get_info=lambda idx: (8 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    b = NpuBackend()
    assert b.get_handle(0) == 0
    info = b.get_memory_info(0)
    assert info == MemoryInfo(total=16 * 1024**3, used=8 * 1024**3, free=8 * 1024**3)


def test_npu_backend_returns_none_on_failure(monkeypatch):
    import torch
    from cosmos_framework.utils.device_backend import NpuBackend

    def _raise(_idx):
        raise RuntimeError("nope")

    fake_npu = _FakeNpu(mem_get_info=_raise)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    b = NpuBackend()
    assert b.get_memory_info(0) is None
```

- [ ] **Step 10: 运行测试,确认 NpuBackend 测试失败**

Run: `cd packages/cosmos3 && python -m pytest cosmos_framework/utils/device_backend_test.py -v`
Expected: FAIL — `ImportError: cannot import name 'NpuBackend'`

- [ ] **Step 11: 实现 NpuBackend 并接入 `_select_backend`**

Edit `cosmos_framework/utils/device_backend.py` — 在 `CudaBackend` 之后插入:

```python
class NpuBackend(DeviceBackend):
    """torch.npu-backed Ascend backend. No NVML equivalent; uses torch.npu APIs."""

    def init(self) -> None:
        pass

    def get_handle(self, idx: int) -> Any:
        return idx  # torch.npu queries by device index

    def get_memory_info(self, handle: Any) -> Optional[MemoryInfo]:
        try:
            # torch.npu.mem_get_info mirrors torch.cuda.mem_get_info -> (free, total)
            free, total = torch.npu.mem_get_info(handle)
            return MemoryInfo(total=int(total), used=int(total) - int(free), free=int(free))
        except Exception:
            return None

    def shutdown(self) -> None:
        pass
```

并修改 `_select_backend` 的 NPU 分支:

```python
def _select_backend() -> DeviceBackend:
    if IS_NPU:
        return NpuBackend()
    if IS_CUDA:
        return CudaBackend()
    return CpuBackend()
```

- [ ] **Step 12: 运行全部测试,确认通过**

Run: `cd packages/cosmos3 && python -m pytest cosmos_framework/utils/device_backend_test.py -v`
Expected: PASS (11 tests)

- [ ] **Step 13: 提交**

```bash
git add packages/cosmos3/cosmos_framework/utils/device_backend.py packages/cosmos3/cosmos_framework/utils/device_backend_test.py
git commit -m "feat(cosmos3): add device-agnostic backend layer (CUDA/NPU/CPU)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: `distributed.init()` 接入后端常量,删除 pynvml/Device/亲和性

**Files:**
- Modify: `cosmos_framework/utils/distributed.py:15,21,33-68`
- Test: `cosmos_framework/utils/distributed_test.py`(新建)

**Interfaces:**
- Consumes: `device_backend.IS_CUDA`,`device_backend.DIST_BACKEND`。
- Produces: `init()` 在 NPU 用 `hccl`、CUDA 用 `nccl`;`distributed` 模块顶层不再 `import pynvml`、不再 `from ...device import Device`。

- [ ] **Step 1: 写失败测试 — init 使用 DIST_BACKEND 且不导入 pynvml**

Create `cosmos_framework/utils/distributed_test.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import sys


def test_distributed_module_does_not_import_pynvml():
    # Ensure no top-level pynvml import remains.
    for mod_name, mod in list(sys.modules.items()):
        if mod_name == "cosmos_framework.utils.distributed":
            assert "pynvml" not in getattr(mod, "__dict__", {}), "distributed must not bind pynvml"
            break


def test_init_uses_dist_backend(monkeypatch):
    import cosmos_framework.utils.distributed as dist_mod

    captured = {}

    def fake_init_process_group(backend, **kwargs):
        captured["backend"] = backend

    monkeypatch.setattr(dist_mod.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(dist_mod.dist, "init_process_group", fake_init_process_group)
    monkeypatch.setattr(dist_mod.torch.cuda, "set_device", lambda _idx: None)
    monkeypatch.setattr(dist_mod, "INTERNAL", False)
    monkeypatch.setattr(dist_mod, "get_world_size", lambda: 1)

    dist_mod.init()

    assert captured["backend"] == dist_mod.DIST_BACKEND
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `cd packages/cosmos3 && python -m pytest cosmos_framework/utils/distributed_test.py -v`
Expected: FAIL — `init` 仍调用 `pynvml.nvmlInit()`(导入即崩,因测试环境无 pynvml)或 `AttributeError: DIST_BACKEND`

- [ ] **Step 3: 改 `distributed.py` — import 与 init**

Edit `cosmos_framework/utils/distributed.py`:

把 L15 `import pynvml` 删除。
把 L21 `from cosmos_framework.utils.device import Device` 改为:
```python
from cosmos_framework.utils.device_backend import IS_CUDA, DIST_BACKEND
```

把 `init()`(L33-68)整体替换为:
```python
def init() -> int | None:
    """Initialize distributed training."""
    if dist.is_initialized():
        return torch.cuda.current_device()

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    # Set up collective communication backend.
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if dist.is_available():
        # transfer_to_npu (train.py) redirects torch.cuda -> torch.npu on Ascend.
        torch.cuda.set_device(local_rank)
        # Get the timeout value from environment variable
        timeout_seconds = os.getenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", 1800)
        # Convert the timeout to an integer (if it isn't already) and then to a timedelta
        timeout_timedelta = timedelta(seconds=int(timeout_seconds))
        dist.init_process_group(backend=DIST_BACKEND, init_method="env://", timeout=timeout_timedelta)
        log.critical(
            f"Initialized distributed training with local rank {local_rank} with timeout {timeout_seconds}",
            rank0_only=False,
        )
    # Increase the L2 fetch granularity for faster speed (CUDA only).
    if INTERNAL and IS_CUDA:
        _libcudart = ctypes.CDLL("libcudart.so")
        # Set device limit on the current device.
        p_value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(p_value, ctypes.c_int(0x05))
    log.info(f"Training with {get_world_size()} GPUs.")
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `cd packages/cosmos3 && python -m pytest cosmos_framework/utils/distributed_test.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: 提交**

```bash
git add packages/cosmos3/cosmos_framework/utils/distributed.py packages/cosmos3/cosmos_framework/utils/distributed_test.py
git commit -m "feat(cosmos3): wire distributed.init to device_backend, drop pynvml/affinity

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: 清理 `device.py`(删 Device 类与死代码)

**Files:**
- Modify: `cosmos_framework/utils/device.py`(整体重写为仅 `with_torch_device`)
- Test: 无新测试;验证 `import cosmos_framework.utils.device` 与 `distributed` 仍可导入。

**Interfaces:**
- Consumes: 无(此任务为删除)。
- Produces: `device.py` 不再含 pynvml / `Device` 类;仅保留 `with_torch_device`。

- [ ] **Step 1: 重写 `device.py`**

Replace entire contents of `cosmos_framework/utils/device.py` with:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from functools import wraps


def with_torch_device(device):
    """
    Decorator factory that wraps a function to execute within a specific torch device context.

    This decorator ensures that all tensor allocations and operations within the decorated
    function use the specified device by default.

    Args:
        device: The torch device to use (e.g. 'cuda', 'cuda:0', 'cpu', or torch.device object).

    Returns:
        A decorator function that wraps the target function with the specified device context.

    Example:
        @with_torch_device('cuda:0')
        def create_tensors():
            x = torch.randn(10, 10)  # Will be created on cuda:0
            return x
    """
    import torch

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with torch.device(device):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
```

- [ ] **Step 2: 验证导入无破坏**

Run: `cd packages/cosmos3 && python -c "import cosmos_framework.utils.device; import cosmos_framework.utils.distributed; print('ok')"`
Expected: 打印 `ok`(确认 `distributed` 已不再 `from ...device import Device`,且 `device` 模块无 pynvml)

- [ ] **Step 3: 运行既有相关测试**

Run: `cd packages/cosmos3 && python -m pytest cosmos_framework/utils/distributed_test.py cosmos_framework/utils/device_backend_test.py -v`
Expected: PASS

- [ ] **Step 4: 提交**

```bash
git add packages/cosmos3/cosmos_framework/utils/device.py
git commit -m "refactor(cosmos3): drop Device class and pynvml dead code from device.py

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: `device_monitor.py` 接入 BACKEND + NPU 安全

**Files:**
- Modify: `cosmos_framework/callbacks/device_monitor.py:9,94-105,107-150`
- Test: `cosmos_framework/callbacks/device_monitor_test.py`(新建)

**Interfaces:**
- Consumes: `device_backend.BACKEND`,`device_backend.MemoryInfo`。
- Produces: `DeviceMonitor` 在 NPU 上不崩(`torch.cuda.temperature/power_draw/utilization/clock_rate` 加 try/except;内存查询走 `BACKEND`)。

- [ ] **Step 1: 写失败测试 — every_n_impl 用 BACKEND 且 None 安全**

Create `cosmos_framework/callbacks/device_monitor_test.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest

import cosmos_framework.callbacks.device_monitor as dm
from cosmos_framework.utils.device_backend import MemoryInfo


class _FakeBackend:
    def __init__(self, memory_info):
        self._memory_info = memory_info
        self.inited = False

    def init(self):
        self.inited = True

    def get_handle(self, idx):
        return "h"

    def get_memory_info(self, handle):
        return self._memory_info

    def shutdown(self):
        pass


def _make_monitor():
    m = dm.DeviceMonitor()
    m.world_size = 1
    m.rank = 0
    m.handle = "h"
    m.step_size = 1
    m.upload_every_n = 1
    m.log_memory_detail = False  # skip torch.cuda.memory_stats / wandb branch in tests
    return m


def test_every_n_impl_handles_none_memory(monkeypatch):
    monkeypatch.setattr(dm, "BACKEND", _FakeBackend(memory_info=None))
    monkeypatch.setattr(dm.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(dm.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(dm.torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(dm.torch.distributed, "all_gather_object", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(dm.torch.distributed, "barrier", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(dm, "log_prof_data", lambda data_list, it: (None, None))

    m = _make_monitor()
    # Should not raise even though memory_info is None (CPU/no-monitor backend).
    m.every_n_impl(trainer=None, model=None, data_batch={}, output_batch={}, loss=None, iteration=0)


def test_every_n_impl_uses_backend_memory(monkeypatch):
    info = MemoryInfo(total=16 * 1024**3, used=4 * 1024**3, free=12 * 1024**3)
    monkeypatch.setattr(dm, "BACKEND", _FakeBackend(memory_info=info))
    monkeypatch.setattr(dm.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(dm.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(dm.torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(dm.torch.distributed, "all_gather_object", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(dm.torch.distributed, "barrier", lambda *a, **k: None, raising=False)

    captured = {}

    def fake_log_prof_data(data_list, it):
        captured["used"] = data_list[0]["nvml_used_gpu_mem_gb"]
        return None, None

    monkeypatch.setattr(dm, "log_prof_data", fake_log_prof_data)

    m = _make_monitor()
    m.every_n_impl(trainer=None, model=None, data_batch={}, output_batch={}, loss=None, iteration=0)
    assert captured["used"] == pytest.approx(4.0, abs=1e-6)
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `cd packages/cosmos3 && python -m pytest cosmos_framework/callbacks/device_monitor_test.py -v`
Expected: FAIL — `device_monitor` 顶层 `import pynvml` 在无 pynvml 环境崩,或 `BACKEND` 未定义

- [ ] **Step 3: 改 import**

Edit `cosmos_framework/callbacks/device_monitor.py` L9: 删 `import pynvml`,改为在 `from cosmos_framework.utils import distributed, log` 之后加:
```python
from cosmos_framework.utils.device_backend import BACKEND, MemoryInfo  # noqa: F401
```

- [ ] **Step 4: 改 `on_train_start` — 加 BACKEND.init + get_handle**

Edit `cosmos_framework/callbacks/device_monitor.py`,把 `on_train_start`(L94-105)末尾两行:
```python
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
```
替换为:
```python
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        BACKEND.init()
        self.handle = BACKEND.get_handle(local_rank)
```

- [ ] **Step 5: 改 `every_n_impl` — NPU 安全的 cuda 监控 + BACKEND 内存**

Edit `cosmos_framework/callbacks/device_monitor.py`,把 L125-138:
```python
        peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        peak_gpu_mem_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        temp = torch.cuda.temperature()
        try:
            power = torch.cuda.power_draw()
        except Exception as e:
            log.warning(f"Failed to get power draw with error {e}")
            power = 0
        util = torch.cuda.utilization()
        clock = torch.cuda.clock_rate()

        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        nvml_used_gpu_mem_gb = memory_info.used / (1024**3)
        nvml_free_gpu_mem_gb = memory_info.free / (1024**3)
```
替换为:
```python
        peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        peak_gpu_mem_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)

        def _safe_cuda(fn_name: str, default=0):
            # torch.cuda.temperature/power_draw/utilization/clock_rate are absent or
            # unsupported on non-CUDA backends (e.g. Ascend NPU); degrade gracefully.
            try:
                return getattr(torch.cuda, fn_name)()
            except Exception as e:
                log.warning(f"Failed to get {fn_name} with error {e}")
                return default

        temp = _safe_cuda("temperature")
        power = _safe_cuda("power_draw")
        util = _safe_cuda("utilization")
        clock = _safe_cuda("clock_rate")

        memory_info = BACKEND.get_memory_info(self.handle)
        if memory_info is None:
            nvml_used_gpu_mem_gb = 0.0
            nvml_free_gpu_mem_gb = 0.0
        else:
            nvml_used_gpu_mem_gb = memory_info.used / (1024**3)
            nvml_free_gpu_mem_gb = memory_info.free / (1024**3)
```

- [ ] **Step 6: 运行测试,确认通过**

Run: `cd packages/cosmos3 && python -m pytest cosmos_framework/callbacks/device_monitor_test.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: 提交**

```bash
git add packages/cosmos3/cosmos_framework/callbacks/device_monitor.py packages/cosmos3/cosmos_framework/callbacks/device_monitor_test.py
git commit -m "feat(cosmos3): route device_monitor memory through BACKEND, harden cuda calls for NPU

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: `args.py` 接入 BACKEND,删除 `_get_nvml_device_memory_info`

**Files:**
- Modify: `cosmos_framework/inference/args.py:12,1289-1317`
- Modify: `cosmos_framework/inference/args_test.py:20,99-121`

**Interfaces:**
- Consumes: `device_backend.BACKEND`。
- Produces: `args.py` 顶层无 `import pynvml`;`_get_device_memory_bytes` 经 `BACKEND` 取内存;`_get_nvml_device_memory_info` 删除。

- [ ] **Step 1: 写失败测试 — `_get_device_memory_bytes` 走 BACKEND**

Create `cosmos_framework/inference/args_mem_test.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import cosmos_framework.inference.args as args


class _FakeBackend:
    def __init__(self, total):
        self._total = total

    def init(self):
        pass

    def get_handle(self, idx):
        return "h"

    def get_memory_info(self, handle):
        from cosmos_framework.utils.device_backend import MemoryInfo

        return MemoryInfo(total=self._total, used=1, free=self._total - 1)

    def shutdown(self):
        pass


def test_get_device_memory_bytes_uses_backend(monkeypatch):
    # _get_device_memory_bytes is @cache'd; clear it for a deterministic test.
    args._get_device_memory_bytes.cache_clear()
    monkeypatch.setattr(args, "BACKEND", _FakeBackend(total=80 * 1024**3))
    assert args._get_device_memory_bytes() == 80 * 1024**3
    args._get_device_memory_bytes.cache_clear()


def test_get_device_memory_bytes_none_falls_back(monkeypatch):
    args._get_device_memory_bytes.cache_clear()

    class _NoneBackend(_FakeBackend):
        def get_memory_info(self, handle):
            return None

    monkeypatch.setattr(args, "BACKEND", _NoneBackend(total=0))
    # None -> fallback path; value is platform-dependent, just assert it returns an int > 0.
    assert isinstance(args._get_device_memory_bytes(), int)
    args._get_device_memory_bytes.cache_clear()


def test_args_has_no_pynvml_attribute():
    assert "pynvml" not in args.__dict__, "args must not bind pynvml at module level"
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `cd packages/cosmos3 && python -m pytest cosmos_framework/inference/args_mem_test.py -v`
Expected: FAIL — `args` 仍 `import pynvml`(L12),或 `BACKEND` 未定义

- [ ] **Step 3: 改 `args.py` import**

Edit `cosmos_framework/inference/args.py` L12: 删 `import pynvml`,在合适 import 区加:
```python
from cosmos_framework.utils.device_backend import BACKEND
```

- [ ] **Step 4: 重写 `_get_device_memory_bytes`,删除 `_get_nvml_device_memory_info`**

Edit `cosmos_framework/inference/args.py`,把 L1289-1317(`_get_device_memory_bytes` 与 `_get_nvml_device_memory_info`)整体替换为:

```python
@cache
def _get_device_memory_bytes() -> int:
    try:
        BACKEND.init()
        handle = BACKEND.get_handle(0)
        info = BACKEND.get_memory_info(handle)
        if info is not None:
            return info.total
        # Backend returned None (e.g. CPU): fall back to torch device properties.
        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory)
        return 128 * 1024**3  # Default 128GB
    except Exception:
        # Fallback for unified memory architectures where memory info is unsupported.
        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory)
        return 128 * 1024**3  # Default 128GB
    finally:
        try:
            BACKEND.shutdown()
        except Exception:
            pass
```

(确认 `args.py` 顶部已 `import torch`;若否,补加。)

- [ ] **Step 5: 清理 `args_test.py` — 删已移除函数的 import 与两个 v2 测试**

Edit `cosmos_framework/inference/args_test.py`:
- L20 的多行 `from cosmos_framework.inference.args import (...)` 语句中,删去 `_get_nvml_device_memory_info,` 这一项(保留其它名字)。
- 删除 `test_get_nvml_device_memory_info_prefers_v2`(L99-110)与 `test_get_nvml_device_memory_info_falls_back_when_v2_unavailable`(L113-121)整段(其等价测试已在 Task 1 的 `device_backend_test.py` 覆盖 `CudaBackend.get_memory_info`)。

- [ ] **Step 6: 运行测试,确认通过**

Run: `cd packages/cosmos3 && python -m pytest cosmos_framework/inference/args_mem_test.py cosmos_framework/inference/args_test.py -v`
Expected: PASS(新测试通过;`args_test.py` 剩余测试不受影响)

- [ ] **Step 7: 全量回归**

Run: `cd packages/cosmos3 && python -m pytest cosmos_framework/utils/device_backend_test.py cosmos_framework/utils/distributed_test.py cosmos_framework/callbacks/device_monitor_test.py cosmos_framework/inference/args_mem_test.py cosmos_framework/inference/args_test.py -v`
Expected: PASS

- [ ] **Step 8: 提交**

```bash
git add packages/cosmos3/cosmos_framework/inference/args.py packages/cosmos3/cosmos_framework/inference/args_test.py packages/cosmos3/cosmos_framework/inference/args_mem_test.py
git commit -m "feat(cosmos3): route args._get_device_memory_bytes through BACKEND, drop pynvml

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 实现后人工核对(spec 的"实现时核对项")

这些无法在无 NPU 的 CI 覆盖,需在 Ascend 机器上跑一次 convert / 训练确认:

1. `torch.npu.mem_get_info(idx)` 返回顺序确为 `(free, total)`(Task 1 Step 11 假设);若为 `(total, free)` 则调整 `NpuBackend.get_memory_info`。
2. `transfer_to_npu` 是否重定向 `torch.cuda.set_device` / `current_device`(Task 2 依赖);若不全,给 `DeviceBackend` 补 `set_device(idx)` / `current_device()` 方法并在 `distributed.init()` 用之。
3. `device_monitor` 的 `torch.cuda.max_memory_allocated` / `memory_stats` / `reset_peak_memory_stats` 在 NPU 上是否经 `transfer_to_npu` 可用;若否,并入 `_safe_cuda` 守卫。
4. 端到端:在 Ascend 跑一次 `launch_sft_action_policy_libero_fsdp.sh` 的前若干 iter,确认 `distributed.init()` 用 `hccl` 起来、`device_monitor` 不崩。
