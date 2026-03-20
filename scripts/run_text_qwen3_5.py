#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import argparse
import os

import openvino_genai

def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _streamer(subword: str) -> bool:
    """流式输出回调：每个 subword 立即打印到终端。返回 False 表示继续，True 表示停止。"""
    print(subword, end="", flush=True)
    return False


def main():
    default_group_size = _safe_int(os.environ.get("OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE", "128"), 128)

    parser = argparse.ArgumentParser(description="Run Youtu-LLM-2B with LLMPipeline")
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=os.environ.get("YOUTU_LLM_MODEL", r"D:\Data\models\Huggingface\Qwen3.5-35B-A3B"),
        help="Path to Youtu-LLM-2B HF model (config.json + *.safetensors)",
    )
    parser.add_argument(
        "--device",
        default="GPU",
        help="Device: CPU or GPU (default: CPU)",
    )
    parser.add_argument(
        "--prompt",
        default="你好，请介绍一下你自己。",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max new tokens to generate (default: 128)",
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
        help="Backup quant mode for sensitive layers, e.g. int4_asym/int8_asym/NONE",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(
            f"Model dir not found: {args.model_dir}\n"
            "Please download Youtu-LLM-2B from HuggingFace or set YOUTU_LLM_MODEL."
        )

    if os.environ.get("OV_GENAI_USE_MODELING_API", "").lower() not in ("1", "true", "yes"):
        print("Warning: OV_GENAI_USE_MODELING_API is not set to 1. Youtu-LLM requires modeling API.")
        os.environ["OV_GENAI_USE_MODELING_API"] = "1"

    # Configure in-flight quantization through environment variables consumed by C++ backend.
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
            f"group={os.environ.get('OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE')} "
            f"backup={os.environ.get('OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE', '<default>')}"
        )

    print(f"Loading model: {args.model_dir}")
    print(f"Device: {args.device}")
    pipe = openvino_genai.LLMPipeline(args.model_dir, args.device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    config.do_sample = True    
    config.temperature = 1.0
    config.top_p = 0.95
    config.top_k = 20
    config.presence_penalty = 1.5
    config.repetition_penalty = 1.0
    config.no_repeat_ngram_size = 3
    
    # prompts = [
    #     "Who is Mark Twain?",
    #     "Who is William Shakespeare?",
    #     "Who is Agatha Christie?",
    #     "Who is Barbara Cartland?",
    #     "Who is Danielle Steel?",
    #     "Who is Harold Robbins?",
    #     "Who is Georges Simenon?",
    #     "Who is Enid Blyton?",
    #     "Who is Sidney Sheldon?",
    #     "Who is Akira Toriyama?",
    #     "Who is Leo Tolstoy?",
    #     "Who is Alexander Pushkin?",
    #     "Who is Stephen King?",
    #     "What is C++?",
    #     "What is Python?",
    #     "What is Java?",
    #     "What is JavaScript?",
    #     "What is Perl?",
    #     "What is OpenCV?",
    #     "Who is the most famous writer?",
    #     "Who is the most famous inventor?",
    #     "Who is the most famous mathematician?",
    #     "Who is the most famous composer?",
    #     "Who is the most famous programmer?",
    #     "Who is the most famous athlete?",
    #     "Who is the most famous ancient Greek scientist?",
    #     "What color will you get when you mix blue and yellow?",
    # ]
    prompts = ["Who is Akira Toriyama?"]
    print("Generated:")
    for p in prompts:
        print(f"\nPrompt: {p}\n", end="")
        result = pipe.generate(p, config, streamer=_streamer)
        print()  # 流式输出后换行

if __name__ == "__main__":
    main()
