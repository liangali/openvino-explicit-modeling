# Qwen3.5 OpenAI-compatible Server — Linux Porting Guide

## 项目概述

在 Windows 上已完成的 Qwen3.5 OpenAI-compatible FastAPI 服务器，需要移植到 Linux。

### 架构

```
Client (OpenAI API) → FastAPI Server → LLMPipeline (text) / modeling_qwen3_5 exe (VL)
                                              ↓
                                     OpenVINO GenAI + Modeling API
                                              ↓
                                     Pre-quantized IR (INT4_ASYM)
```

- **Text chat**: Python `openvino_genai.LLMPipeline` 加载 `openvino_model.xml/bin`
- **VL (vision)**: subprocess 调用 `modeling_qwen3_5` 二进制 + `--mode vl --cache-model`
- **Tool calling**: 解析 Qwen3.5 `<tool_call>` XML 格式
- **Streaming**: SSE via FastAPI StreamingResponse

### 代码仓库

3 个 Git 仓库，都在 `explicit-modeling` 主分支，serving 工作在 `qwen3.5_serving` 分支：

```
openvino-explicit-modeling/    # 顶层：build.bat/sh、serving/ 目录
├── openvino/                  # OpenVINO 核心 (GPU/CPU plugin)  
└── openvino.genai/            # GenAI (Modeling API, LLMPipeline, modeling_qwen3_5 sample)
```

### 关键代码路径

| 文件 | 用途 |
|------|------|
| `serving/server.py` | FastAPI 主入口，/v1/chat/completions 等端点 |
| `serving/engine.py` | Worker pool (N × LLMPipeline B=1) |
| `serving/vl_backend.py` | VL subprocess wrapper（调用 modeling_qwen3_5 exe） |
| `serving/tool_parser.py` | Qwen3.5 `<tool_call>` XML → OpenAI tool_calls JSON |
| `serving/config.py` | CLI 参数 (--model, --device, --quant, --vl-exe 等) |
| `serving/schemas.py` | Pydantic v2 OpenAI request/response models |
| `serving/deploy.py` | 部署打包脚本 |
| `serving/bench.py` | 性能/tool calling 基准测试 |
| `serving/test_server.py` | 27 个 pytest 测试 |

---

## Linux 构建步骤

### 1. 编译 OpenVINO + GenAI

```bash
cd openvino-explicit-modeling
chmod +x build.sh
./build.sh
```

或手动：

```bash
# OpenVINO
cd openvino/build
cmake --build . --config Release -j$(nproc)

# GenAI (含 modeling_qwen3_5 sample + Python bindings)
cd openvino.genai/build
cmake --build . --config Release -j$(nproc)
# 确保 ENABLE_PYTHON=ON 以编译 py_openvino_genai
```

### 2. 生成 IR 缓存

```bash
export OV_GENAI_USE_MODELING_API=1
export OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym
export OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128
export OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=int4_asym

MODEL_DIR=/path/to/Qwen3.5-4B  # 或 35B-A3B

# Text IR
./modeling_qwen3_5 --model $MODEL_DIR --mode text --prompt "hi" --output-tokens 2 --cache-model

# VL IR (需要一张测试图片)
./modeling_qwen3_5 --model $MODEL_DIR --mode vl --prompt "describe" --output-tokens 2 --cache-model --image /path/to/test.jpg

# 生成的文件:
#   $MODEL_DIR/qwen3_5_text_q4a_b4a_g128.xml/bin        (text, ~2.1GB for 4B)
#   $MODEL_DIR/qwen3_5_text_vl_q4a_b4a_g128.xml/bin     (VL text)
#   $MODEL_DIR/qwen3_5_vision.xml/bin                    (vision encoder, FP16)
```

### 3. 部署打包

```bash
# 创建 venv
python3 -m venv serving_env
source serving_env/bin/activate
pip install fastapi uvicorn pydantic numpy httpx

# 安装 openvino_genai Python bindings
# 方法 A: 从 build 目录复制
cp -r openvino.genai/build/openvino_genai serving_env/lib/python3.*/site-packages/

# 方法 B: 使用 deploy.py 自动打包
python deploy.py \
  --model $MODEL_DIR \
  --output /deploy/qwen3_5 \
  --genai-pkg serving_env/lib/python3.*/site-packages/openvino_genai \
  --ov-bin openvino/bin/intel64/Release \
  --tbb-bin openvino/temp/Linux_AMD64/tbb/lib \
  --vl-exe openvino.genai/build/bin/Release/modeling_qwen3_5
```

**注意 deploy.py 中需要 Linux 适配的地方：**
- DLL 列表 → `.so` 文件 (openvino.so, libopenvino_intel_gpu_plugin.so 等)
- `.pyd` → `.so` (py_openvino_genai.cpython-312-x86_64-linux-gnu.so)
- `setup.bat` / `start.bat` → `setup.sh` / `start.sh`
- TBB: `tbb12.dll` → `libtbb.so.12`
- ICU: `icudt70.dll` → `libicudata.so.70`

### 4. 启动服务器

```bash
source serving_env/bin/activate
cd serving
python server.py --model /deploy/qwen3_5/model --device GPU --port 8000 \
  --model-name "Qwen3.5-4B" --quant none \
  --vl-exe /deploy/qwen3_5/runtime/modeling_qwen3_5
```

### 5. 测试

```bash
# 基准测试
python bench.py all --url=http://localhost:8000

# pytest
pytest test_server.py -v --timeout=120
```

---

## Linux 适配要点

### deploy.py 改动清单

1. **运行时文件名映射**（Windows → Linux）：

| Windows | Linux |
|---------|-------|
| `openvino.dll` | `libopenvino.so` |
| `openvino_c.dll` | `libopenvino_c.so` |
| `openvino_ir_frontend.dll` | `libopenvino_ir_frontend.so` |
| `openvino_intel_gpu_plugin.dll` | `libopenvino_intel_gpu_plugin.so` |
| `openvino_intel_cpu_plugin.dll` | `libopenvino_intel_cpu_plugin.so` |
| `openvino_auto_plugin.dll` | `libopenvino_auto_plugin.so` |
| `openvino_auto_batch_plugin.dll` | `libopenvino_auto_batch_plugin.so` |
| `openvino_hetero_plugin.dll` | `libopenvino_hetero_plugin.so` |
| `OpenCL.dll` | `libOpenCL.so` (或系统提供) |
| `tbb12.dll` | `libtbb.so.12` |
| `tbbmalloc.dll` | `libtbbmalloc.so.2` |
| `tbbbind_2_5.dll` | `libtbbbind_2_5.so` (可能不需要) |
| `py_openvino_genai.cp312-win_amd64.pyd` | `py_openvino_genai.cpython-312-x86_64-linux-gnu.so` |
| `openvino_genai.dll` | `libopenvino_genai.so` |
| `openvino_tokenizers.dll` | `libopenvino_tokenizers.so` |
| `icudt70.dll` | `libicudata.so.70` |
| `icuuc70.dll` | `libicuuc.so.70` |
| `modeling_qwen3_5.exe` | `modeling_qwen3_5` |

2. **`__init__.py` DLL 加载**：Linux 不需要 `os.add_dll_directory()`，改用 `LD_LIBRARY_PATH` 或 `RPATH`

3. **setup.bat → setup.sh**：`xcopy` → `cp -r`，venv 路径不同

4. **start.bat → start.sh**：`%~dp0` → `$(dirname "$0")`

### vl_backend.py 改动

- `_find_exe()`: 搜索路径改为 Linux 路径，去掉 `.exe` 后缀
- subprocess PATH: 设 `LD_LIBRARY_PATH` 而非 Windows `PATH`

### __init__.py (deployed)

```python
import os, sys
# Linux: set LD_LIBRARY_PATH before importing the .so
_pkg_dir = os.path.dirname(__file__)
_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = _pkg_dir + (":" + _ld if _ld else "")
# For already-loaded process, use ctypes to pre-load
import ctypes
for lib in ["libopenvino.so", "libopenvino_genai.so", "libopenvino_tokenizers.so"]:
    p = os.path.join(_pkg_dir, lib)
    if os.path.exists(p):
        ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
from .py_openvino_genai import *
__version__ = get_version()
```

---

## 已验证的功能 (Windows)

| 功能 | 状态 | 备注 |
|------|------|------|
| Text chat (非流式) | ✅ | |
| Text chat (流式 SSE) | ✅ | |
| Tool calling (6/6 tests) | ✅ | 包括并行调用、多轮对话 |
| VL (vision-language) | ✅ | 64×64 BMP 测试通过 |
| `<think>` 标签剥离 | ✅ | 流式和非流式 |
| INT4 量化 (int4_asym) | ✅ | 75.35% 权重覆盖 |
| IR 缓存加载 | ✅ | 5s 启动 (vs 11s safetensors) |
| pytest 27/27 | ✅ | |
| bench.py tools 6/6 | ✅ | |

## 已知限制

- **VL 每次请求重加载模型**（subprocess），4B 约 13s 延迟
- **单 worker** 模式，请求串行处理
- 不支持 ContinuousBatching（Qwen3.5 hybrid attention 不兼容 PagedAttention）
- Vision encoder 未量化（FP16，636MB），INT4 会导致 GPU compile hang
- 图片不能太大，受 `--max-pixels` 限制（默认 602112）

## Windows 测试结果 (Qwen3.5-4B, ARL-H GPU)

- **模型加载**: IR 5s / safetensors 11s
- **TTFT** (1K context): ~0.4s
- **Tool calling**: 6/6 PASS, ~5s per call
- **Pytest**: 27/27 passed, 47s total
