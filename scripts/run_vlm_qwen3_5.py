#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
测试 VLMPipeline 加载 Qwen3.5 VL 模型并进行图像+文本推理。

模型要求：config.json 中 model_type 为 qwen3_5 或 qwen3_5_moe，且包含 vision_config。
例如：Qwen3-VL-30B-A3B-Instruct、Qwen3.5-VL 等。

依赖: pip install pillow
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# === 推断 OPENVINO_ROOT，确保 openvino / openvino_genai 可导入 ===
_openvino_root = os.environ.get("OPENVINO_ROOT")
if _openvino_root:
    _openvino_root = Path(_openvino_root).resolve()
else:
    _openvino_root = (Path(__file__).resolve().parent / ".." / "..").resolve()

_ov_python = _openvino_root / "openvino" / "bin" / "intel64" / "Release" / "python"
_ov_genai = _openvino_root / "openvino.genai" / "build"
for _p in (_ov_python, _ov_genai):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# === 必须在 import openvino 之前执行：Windows 下预注册 DLL 搜索路径 ===
# 解决 "cannot import name 'AxisSet' from 'openvino._pyopenvino' (unknown location)"
# 原因：_pyopenvino.pyd 依赖 openvino.dll 等，需在加载前 add_dll_directory
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _dll_paths = [
        _openvino_root / "openvino" / "bin" / "intel64" / "Release",
        _openvino_root / "openvino" / "temp" / "Windows_AMD64" / "tbb" / "bin",
        _openvino_root / "openvino" / "build" / "install" / "runtime" / "bin" / "intel64" / "Release",
        _openvino_root / "openvino.genai" / "build" / "openvino_genai",
        _openvino_root / "openvino.genai" / "build" / "bin" / "Release",
    ]
    for _p in _dll_paths:
        if _p.exists():
            os.add_dll_directory(str(_p))
    # 若 bat 已设置 OPENVINO_LIB_PATHS，也加入（可能有额外路径）
    for _ep in os.environ.get("OPENVINO_LIB_PATHS", "").split(";"):
        _ep = _ep.strip()
        if not _ep:
            continue
        _p = Path(_ep).resolve()
        if _p.exists() and _p.is_dir() and str(_p) != ".":
            os.add_dll_directory(str(_p))

import numpy as np

# 确保 modeling API 已启用
if os.environ.get("OV_GENAI_USE_MODELING_API", "").lower() not in ("1", "true", "yes"):
    os.environ["OV_GENAI_USE_MODELING_API"] = "1"

import openvino_genai
from openvino import Tensor

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Missing dependency: pip install pillow") from None


# 默认路径
DEFAULT_MODEL = os.environ.get(
    "QWEN3_VL_MODEL",
    os.environ.get("YOUTU_LLM_MODEL", r"D:\Data\models\Huggingface\Qwen3.5-35B-A3B"),
)
DEFAULT_IMAGE = (Path(__file__).resolve().parent / ".." / "scripts" / "test.jpg").resolve()


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _streamer(subword: str) -> bool:
    """流式输出回调"""
    print(subword, end="", flush=True)
    return False


def read_image(path: str) -> Tensor:
    """读取图片并转换为 OpenVINO Tensor (RGB, uint8)"""
    pic = Image.open(path).convert("RGB")
    arr = np.array(pic, dtype=np.uint8)
    return Tensor(arr)


def read_images(path: str) -> list[Tensor]:
    """读取单张图片或目录下所有图片"""
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(f)) for f in sorted(entry.iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")]
    return [read_image(path)]


def print_perf_metrics_manual(num_input_tokens: int, num_generated_tokens: int,
                              ttft_ms: float, total_ms: float) -> None:
    """Print perf metrics computed from manual wall-clock timing."""
    # decode steps = tokens after the first one (mirrors modeling_qwen3_5.exe)
    decode_steps = max(0, num_generated_tokens - 1)
    decode_time_ms = total_ms - ttft_ms

    print(f"Prompt token size: {num_input_tokens}")
    print(f"Output token size: {num_generated_tokens}")
    print(f"TTFT: {ttft_ms:.2f} ms")
    print(f"Decode time: {decode_time_ms:.2f} ms")
    if decode_steps > 0:
        tpot_ms = decode_time_ms / decode_steps
        throughput = decode_steps / (decode_time_ms / 1000.0) if decode_time_ms > 0 else 0.0
        print(f"TPOT: {tpot_ms:.2f} ms/token")
        print(f"Throughput: {throughput:.2f} tokens/s")
    else:
        print("TPOT: N/A")
        print("Throughput: N/A")


def main():
    default_group_size = _safe_int(os.environ.get("OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE", "128"), 128)

    parser = argparse.ArgumentParser(
        description="Run VLMPipeline with Qwen3.5 VL model (image + text inference)"
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Path to Qwen3.5 VL model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=str(DEFAULT_IMAGE),
        help=f"Image file or directory (default: scripts/test.jpg)",
    )
    parser.add_argument(
        "--test-image",
        action="store_true",
        help="Use a synthetic test image (336x336) when no image_path provided",
    )
    parser.add_argument(
        "--device",
        default="GPU",
        help="Device: CPU or GPU (default: GPU)",
    )
    parser.add_argument(
        "--prompt",
        default="请描述这张图片的内容。",
        help="Input prompt for the image",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--inflight-quant-mode",
        default=os.environ.get("OV_GENAI_INFLIGHT_QUANT_MODE", ""),
        help="In-flight quantization mode: int4_sym/int4_asym/int8_sym/int8_asym/none",
    )
    parser.add_argument(
        "--inflight-quant-group-size",
        type=int,
        default=default_group_size,
        help="In-flight quantization group size (default: 128)",
    )
    parser.add_argument(
        "--inflight-quant-backup-mode",
        default=os.environ.get("OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE", ""),
        help="Backup quant mode for sensitive layers",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (print result at once)",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable multinomial sampling (default: greedy, aligned with modeling_qwen3_5 sample)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature when --do-sample is enabled (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Sampling top-p when --do-sample is enabled (default: 0.9)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Sampling top-k when --do-sample is enabled (default: 20)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (default: 1.1, aligned with modeling_qwen3_5 sample)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(
            f"Model dir not found: {args.model_dir}\n"
            "Please download a Qwen3.5 VL model (e.g. Qwen3-VL-30B-A3B-Instruct) or set QWEN3_VL_MODEL."
        )

    if args.test_image:
        images = [Tensor(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))]
        print("[INFO] Using synthetic test image 336x336")
    elif not args.image_path or not Path(args.image_path).exists():
        # 默认 scripts/test.jpg 不存在时使用合成图
        images = [Tensor(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))]
        print(f"[INFO] Image not found: {args.image_path}, using synthetic test image")
    else:
        images = read_images(args.image_path)
        if not images:
            raise ValueError(f"No valid images found in: {args.image_path}")

    # 配置 in-flight 量化
    quant_mode = (args.inflight_quant_mode or "").strip().lower()
    if quant_mode in ("", "none", "off", "disable", "disabled", "0"):
        os.environ.pop("OV_GENAI_INFLIGHT_QUANT_MODE", None)
        os.environ.pop("OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE", None)
        os.environ.pop("OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE", None)
        print("In-flight quantization: disabled")
    else:
        os.environ["OV_GENAI_INFLIGHT_QUANT_MODE"] = quant_mode
        os.environ["OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE"] = str(args.inflight_quant_group_size)
        if args.inflight_quant_backup_mode:
            os.environ["OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE"] = args.inflight_quant_backup_mode
        print(
            "In-flight quantization: "
            f"mode={os.environ.get('OV_GENAI_INFLIGHT_QUANT_MODE')} "
            f"group={os.environ.get('OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE')}"
        )

    print(f"Loading model: {args.model_dir}")
    print(f"Device: {args.device}")
    print(f"Image(s): {args.image_path or '(synthetic)'} ({len(images)} image(s))")
    print(f"Prompt: {args.prompt}")
    print()

    enable_cache = {}
    if args.device == "GPU":
        enable_cache["CACHE_DIR"] = "vlm_qwen3_5_cache"

    pipe = openvino_genai.VLMPipeline(args.model_dir, args.device, **enable_cache)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    config.do_sample = args.do_sample
    config.repetition_penalty = args.repetition_penalty
    if args.do_sample:
        config.temperature = args.temperature
        config.top_p = args.top_p
        config.top_k = args.top_k
    else:
        # Greedy path to mirror modeling_qwen3_5 non-thinking mode behavior
        config.temperature = 0.0
        config.top_p = 1.0
        config.top_k = 0

    streamer = None if args.no_stream else _streamer

    # --- 手动计时：记录首 token 时间需要用 streamer 回调 ---
    _timing: dict = {"first_token_ms": None, "generate_start": None}

    _orig_streamer = streamer

    def _timed_streamer(subword: str) -> bool:
        if _timing["first_token_ms"] is None:
            elapsed = (time.perf_counter() - _timing["generate_start"]) * 1000.0
            _timing["first_token_ms"] = elapsed
        if _orig_streamer is not None:
            return _orig_streamer(subword)
        return False

    print("Generated:")
    print("-" * 40)
    _timing["generate_start"] = time.perf_counter()
    result = pipe.generate(
        args.prompt,
        images=images,
        generation_config=config,
        streamer=_timed_streamer,
    )
    _total_ms = (time.perf_counter() - _timing["generate_start"]) * 1000.0
    print()

    metrics = result.perf_metrics
    num_input_tokens = int(metrics.get_num_input_tokens())
    num_generated_tokens = int(metrics.get_num_generated_tokens())
    ttft_ms = _timing["first_token_ms"] if _timing["first_token_ms"] is not None else _total_ms
    print_perf_metrics_manual(num_input_tokens, num_generated_tokens, ttft_ms, _total_ms)
    if args.no_stream:
        text = result.texts[0] if hasattr(result, "texts") and result.texts else str(result)
        print(text)
    print()
    print("-" * 40)
    print("Done.")


if __name__ == "__main__":
    main()
