#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
DFlash quantization benchmark.

Runs 6 configurations and prints a comparison table:
  Baseline FP16      — pure LLMPipeline, no DFlash
  Baseline INT4      — pure LLMPipeline, target INT4
  DFlash FP16/FP16   — DFlash, target=FP16, draft=FP16
  DFlash INT4/FP16   — DFlash, target=INT4, draft=FP16
  DFlash FP16/INT4   — DFlash, target=FP16, draft=INT4
  DFlash INT4/INT4   — DFlash, target=INT4, draft=INT4

Per-step acceptance and timing metrics are returned via extended_perf_metrics and
printed by this script.

Usage:
  python dflash_benchmark.py <target_dir> <draft_dir> [prompt] [--device GPU] [--max-tokens 512] [--no-think]
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ============================================================================
# Bootstrap (identical to dflash_pipeline_inference.py)
# ============================================================================

_DLL_DIR_HANDLES = []
_BOOTSTRAP_DLL_DIRS: List[Path] = []
_BOOTSTRAP_GENAI_DIR: Optional[Path] = None
_BOOTSTRAP_OPENVINO_PY_DIR: Optional[Path] = None


def _find_build_genai_dir(start_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    search = start_dir
    for _ in range(10):
        candidates.append(search / "build" / "openvino_genai")
        candidates.append(search / "openvino.genai" / "build" / "openvino_genai")
        if search.parent == search:
            break
        search = search.parent
    for c in candidates:
        if c.is_dir() and any(c.glob("py_openvino_genai*.pyd")):
            return c
    return None


def _find_runtime_dll_dirs(genai_dir: Path) -> List[Path]:
    entries: List[Path] = [genai_dir]
    search = genai_dir.parent
    for _ in range(8):
        if search.parent == search:
            break
        search = search.parent
        for build_type in ("Release", "RelWithDebInfo", "Debug"):
            runtime = search / "openvino" / "bin" / "intel64" / build_type
            if runtime.is_dir() and (runtime / "openvino.dll").is_file():
                entries.append(runtime)
                break
        tbb_root = search / "openvino" / "temp"
        if tbb_root.is_dir():
            for tbb_bin in sorted(tbb_root.glob("*/tbb/bin")):
                if (tbb_bin / "tbb12.dll").is_file():
                    entries.append(tbb_bin)
                    break
    unique: List[Path] = []
    seen = set()
    for p in entries:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            unique.append(p)
    return unique


def _find_local_openvino_python_dir(start_dir: Path) -> Optional[Path]:
    search = start_dir
    for _ in range(10):
        for build_type in ("Release", "RelWithDebInfo", "Debug"):
            candidate = search / "openvino" / "bin" / "intel64" / build_type / "python"
            if (candidate / "openvino" / "__init__.py").is_file():
                return candidate
        if search.parent == search:
            break
        search = search.parent
    return None


def _bootstrap_openvino_genai() -> None:
    global _BOOTSTRAP_DLL_DIRS, _BOOTSTRAP_GENAI_DIR, _BOOTSTRAP_OPENVINO_PY_DIR
    script_dir = Path(__file__).resolve().parent
    genai_dir = _find_build_genai_dir(script_dir)
    if genai_dir is None:
        return
    _BOOTSTRAP_GENAI_DIR = genai_dir
    build_dir_str = str(genai_dir.parent)
    if build_dir_str not in sys.path:
        sys.path.insert(0, build_dir_str)
    local_ov_py_dir = _find_local_openvino_python_dir(script_dir)
    if local_ov_py_dir is not None:
        _BOOTSTRAP_OPENVINO_PY_DIR = local_ov_py_dir
        s = str(local_ov_py_dir)
        if s not in sys.path:
            sys.path.insert(0, s)
    dll_dirs = _find_runtime_dll_dirs(genai_dir)
    has_local = any((d / "openvino.dll").is_file() for d in dll_dirs)
    if not has_local:
        spec = importlib.util.find_spec("openvino")
        if spec and spec.origin:
            pkg = Path(spec.origin).resolve().parent
            for c in [pkg / "libs", pkg.parent / "openvino" / "libs"]:
                if c.is_dir() and any(c.glob("*.dll")):
                    dll_dirs.append(c)
    deduped: List[Path] = []
    seen = set()
    for d in dll_dirs:
        k = str(Path(d).resolve())
        if k not in seen:
            seen.add(k)
            deduped.append(Path(d))
    dll_dirs = deduped
    _BOOTSTRAP_DLL_DIRS = dll_dirs
    if not os.environ.get("OPENVINO_LIB_PATHS", "").strip() and dll_dirs:
        os.environ["OPENVINO_LIB_PATHS"] = ";".join(str(d) for d in dll_dirs)
    if not dll_dirs:
        return
    if hasattr(os, "add_dll_directory"):
        for d in dll_dirs:
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(d)))
            except OSError:
                pass
    existing = os.environ.get("PATH", "")
    prepend = ";".join(str(d) for d in dll_dirs)
    os.environ["PATH"] = f"{prepend};{existing}" if existing else prepend


_bootstrap_openvino_genai()

try:
    import openvino_genai as ov_genai
except ImportError as exc:
    print("[bootstrap] Failed to import openvino_genai:", exc, file=sys.stderr)
    raise

# ============================================================================
# Metrics
# ============================================================================

@dataclass
class RunMetrics:
    label: str = ""
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_ms: float = 0.0
    generated_tokens: int = 0

    # DFlash-only (from DFlashPerfMetrics)
    is_dflash: bool = False
    target_quant: str = "FP16"
    draft_quant: str = "FP16"
    draft_steps: int = 0
    accepted_draft_tokens: int = 0
    avg_accepted_per_step: float = 0.0
    draft_acceptance_rate: float = 0.0
    accepted_per_step: list = None
    draft_total_ms: float = 0.0
    avg_draft_step_ms: float = 0.0
    avg_accepted_draft_token_ms: float = 0.0
    draft_decode_count: int = 0
    avg_draft_decode_ms: float = 0.0
    target_verify_count: int = 0
    target_replay_count: int = 0
    target_decode_count: int = 0
    target_verify_total_ms: float = 0.0
    target_replay_total_ms: float = 0.0
    target_decode_total_ms: float = 0.0
    avg_target_verify_ms: float = 0.0
    avg_target_replay_ms: float = 0.0
    avg_target_decode_ms: float = 0.0
    verify_trace_lines: list = None
    snapshot_restore_trace_lines: list = None
    output_text: str = ""

    def __post_init__(self):
        if self.accepted_per_step is None:
            self.accepted_per_step = []
        if self.verify_trace_lines is None:
            self.verify_trace_lines = []
        if self.snapshot_restore_trace_lines is None:
            self.snapshot_restore_trace_lines = []


@dataclass
class StreamMetrics:
    start_time: float = 0.0
    first_chunk_time: float = 0.0
    end_time: float = 0.0
    chunk_count: int = 0
    generated_tokens: int = 0

    def begin(self):
        self.start_time = time.perf_counter()
        self.first_chunk_time = 0.0
        self.end_time = 0.0
        self.chunk_count = 0
        self.generated_tokens = 0

    def on_stream(self, subword):
        now = time.perf_counter()
        if subword:
            if self.first_chunk_time == 0.0:
                self.first_chunk_time = now
            self.chunk_count += 1
        print(subword, end="", flush=True)
        return ov_genai.StreamingStatus.RUNNING

    def finish(self, generated_tokens: int):
        self.end_time = time.perf_counter()
        self.generated_tokens = generated_tokens

    @property
    def e2e_ms(self):
        return max(0.0, (self.end_time - self.start_time) * 1000.0)

    @property
    def ttft_ms(self):
        if self.first_chunk_time == 0.0:
            return self.e2e_ms
        return max(0.0, (self.first_chunk_time - self.start_time) * 1000.0)

    @property
    def tpot_ms(self):
        if self.generated_tokens <= 1 or self.first_chunk_time == 0.0:
            return 0.0
        decode_ms = max(0.0, (self.end_time - self.first_chunk_time) * 1000.0)
        return decode_ms / float(self.generated_tokens - 1)


def build_prompt(prompt: str, no_think: bool) -> str:
    if not no_think:
        return prompt
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<|im_start|>think\n\n<|im_end|>\n"
    )


def run_with_streaming(pipe, full_prompt: str, config, capture_result: bool = False):
    sm = StreamMetrics()
    sm.begin()
    # Use list input so pybind always returns DecodedResults (which carries extended_perf_metrics).
    # Passing a bare str with num_return_sequences=1 returns a plain str and loses the metrics.
    result = pipe.generate([full_prompt], config, sm.on_stream)
    gen_tokens = 0
    if hasattr(result, "tokens") and result.tokens and result.tokens[0]:
        gen_tokens = len(result.tokens[0])
    elif sm.chunk_count > 0:
        gen_tokens = sm.chunk_count
    sm.finish(gen_tokens)

    m = RunMetrics()
    # Capture generated text
    if hasattr(result, "texts") and result.texts:
        m.output_text = result.texts[0]
    m.ttft_ms = sm.ttft_ms
    m.tpot_ms = sm.tpot_ms
    m.e2e_ms  = sm.e2e_ms
    m.generated_tokens = sm.generated_tokens
    
    # Extract DFlash acceptance stats if available
    ext = getattr(result, "extended_perf_metrics", None)
    if ext is not None and hasattr(ext, "draft_steps"):
        m.draft_steps            = ext.draft_steps
        m.accepted_draft_tokens  = ext.accepted_draft_tokens
        m.avg_accepted_per_step  = ext.avg_accepted_per_step
        m.draft_acceptance_rate  = ext.draft_acceptance_rate
        m.accepted_per_step      = list(ext.accepted_per_step)
        m.draft_total_ms         = getattr(ext, "draft_total_ms", 0.0)
        m.avg_draft_step_ms      = getattr(ext, "avg_draft_step_ms", 0.0)
        m.avg_accepted_draft_token_ms = getattr(ext, "avg_accepted_draft_token_ms", 0.0)
        m.draft_decode_count     = getattr(ext, "draft_decode_count", 0)
        m.avg_draft_decode_ms    = getattr(ext, "avg_draft_decode_ms", 0.0)
        m.target_verify_count    = getattr(ext, "target_verify_count", 0)
        m.target_replay_count    = getattr(ext, "target_replay_count", 0)
        m.target_decode_count    = getattr(ext, "target_decode_count", 0)
        m.target_verify_total_ms = getattr(ext, "target_verify_total_ms", 0.0)
        m.target_replay_total_ms = getattr(ext, "target_replay_total_ms", 0.0)
        m.target_decode_total_ms = getattr(ext, "target_decode_total_ms", 0.0)
        m.avg_target_verify_ms   = getattr(ext, "avg_target_verify_ms", 0.0)
        m.avg_target_replay_ms   = getattr(ext, "avg_target_replay_ms", 0.0)
        m.avg_target_decode_ms   = getattr(ext, "avg_target_decode_ms", 0.0)
        m.verify_trace_lines = list(getattr(ext, "verify_trace_lines", []))
        m.snapshot_restore_trace_lines = list(getattr(ext, "snapshot_restore_trace_lines", []))
    return m


# ============================================================================
# Baseline runner (pure LLMPipeline, no DFlash)
# ============================================================================

def run_baseline(
    model_dir: str,
    prompt: str,
    device: str,
    max_tokens: int,
    no_think: bool,
    quant_mode: Optional[str],
) -> RunMetrics:
    """
    Pure LLMPipeline baseline (no speculative decoding).

    quant_mode: None → FP16, "INT4_ASYM" → in-flight INT4 via safetensors loader.
    Works for safetensors model directories; if model_dir already has openvino_model.xml
    it will be bypassed and the model re-loaded from safetensors.
    """
    label = f"Baseline {'INT4' if quant_mode else 'FP16'}"
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    pipe_kwargs: dict = {}
    if quant_mode:
        pipe_kwargs["quantization_mode"] = quant_mode
        pipe_kwargs["quantization_group_size"] = 128
    pipe = ov_genai.LLMPipeline(model_dir, device, **pipe_kwargs)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_tokens

    full_prompt = build_prompt(prompt, no_think)
    print("--- output ---")
    m = run_with_streaming(pipe, full_prompt, config)
    print(f"\n--- end ---")
    print(f"ttft={m.ttft_ms:.1f}ms  tpot={m.tpot_ms:.2f}ms  e2e={m.e2e_ms:.1f}ms  tokens={m.generated_tokens}")

    m.label = label
    m.is_dflash = False
    m.target_quant = "INT4" if quant_mode else "FP16"
    m.draft_quant  = "-"
    return m


# ============================================================================
# DFlash runner
# ============================================================================

def run_dflash(
    model_dir: str,
    draft_dir: str,
    prompt: str,
    device: str,
    max_tokens: int,
    no_think: bool,
    target_quant: Optional[str],   # None = FP16, "INT4_ASYM" = INT4
    draft_quant: Optional[str],
    inference_precision: str = "f16",
) -> RunMetrics:
    tq_label = "INT4" if target_quant else "FP16"
    dq_label = "INT4" if draft_quant  else "FP16"
    prec_label = inference_precision.upper()
    label = f"DFlash  target={tq_label}  draft={dq_label}  prec={prec_label}"
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # Build dflash_model kwargs
    dflash_kwargs: dict = {"inference_precision": inference_precision}
    if target_quant:
        dflash_kwargs["target_quantization_mode"] = target_quant
        dflash_kwargs["target_quantization_group_size"] = 128
    if draft_quant:
        dflash_kwargs["draft_quantization_mode"] = draft_quant
        dflash_kwargs["draft_quantization_group_size"] = 128

    draft = ov_genai.dflash_model(draft_dir, device=device, **dflash_kwargs)
    pipe  = ov_genai.LLMPipeline(model_dir, device=device, dflash_model=draft)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_tokens

    full_prompt = build_prompt(prompt, no_think)
    print("--- output ---")
    # C++ pipeline no longer prints acceptance stats; they come back via extended_perf_metrics
    m = run_with_streaming(pipe, full_prompt, config)
    print(f"\n--- end ---")
    print(f"ttft={m.ttft_ms:.1f}ms  tpot={m.tpot_ms:.2f}ms  e2e={m.e2e_ms:.1f}ms  tokens={m.generated_tokens}")
    if m.is_dflash or m.draft_steps > 0:
        print(f"[DFlash] draft_steps={m.draft_steps}  "
              f"accepted_draft={m.accepted_draft_tokens}  "
              f"avg_per_step={m.avg_accepted_per_step:.2f}  "
              f"acceptance_rate={m.draft_acceptance_rate*100:.1f}%")
        print(f"[DFlash Timing] draft(小模型) 完整decode平均={m.avg_draft_decode_ms:.3f}ms/次 "
              f"(共{m.draft_decode_count}次)")
        _skipped = m.target_verify_count - m.target_replay_count
        _skip_note = f", skipped_replay={_skipped}" if _skipped > 0 else ""
        print(f"[DFlash Timing] target(大模型) 完整decode平均={m.avg_target_decode_ms:.3f}ms/次 "
              f"(verify={m.target_verify_count}次, replay={m.target_replay_count}次{_skip_note})")
        _replay_note = f", replay_avg={m.avg_target_replay_ms:.3f}ms/次" if m.target_replay_count > 0 else ""
        print(f"[DFlash Timing] target细分: verify_avg={m.avg_target_verify_ms:.3f}ms/次{_replay_note}")
        if m.accepted_per_step:
            print(f"[DFlash] per_step={m.accepted_per_step}")

    m.label = label
    m.is_dflash = True
    m.target_quant = tq_label
    m.draft_quant  = dq_label
    return m


# ============================================================================
# Main
# ============================================================================

def build_summary_lines(results: List[RunMetrics], baseline_fp16: Optional[RunMetrics]) -> List[str]:
    # Find Baseline INT4 in results if it exists
    baseline_int4 = None
    for r in results:
        if r.label == "Baseline INT4":
            baseline_int4 = r
            break

    lines: List[str] = []
    lines.append(f"\n{'='*105}")
    lines.append("  BENCHMARK SUMMARY")
    lines.append(f"{'='*105}")
    lines.append("  Note: Speedup computed relative to matching baseline:")
    lines.append("        - target=FP16 configs use Baseline FP16")
    lines.append("        - target=INT4 configs use Baseline INT4")
    lines.append(f"{'='*105}")
    header = (f"{'Config':<30} {'TTFT(ms)':>10} {'TPOT(ms)':>10} {'E2E(ms)':>10} "
              f"{'Tokens':>7} {'Speedup':>9} {'Accept%':>9} {'AvgAcc':>8}")
    lines.append(header)
    lines.append("-" * 105)
    
    for m in results:
        speedup = ""
        
        # Determine which baseline to use for speedup calculation
        baseline_to_use = None
        if m.is_dflash:
            if m.target_quant == "INT4":
                baseline_to_use = baseline_int4
            else:  # target=FP16
                baseline_to_use = baseline_fp16
        else:
            # Skip speedup calculation for baseline entries themselves
            if m.label != "Baseline FP16" and m.label != "Baseline INT4":
                baseline_to_use = baseline_fp16
        
        if baseline_to_use and baseline_to_use.e2e_ms > 0 and m.e2e_ms > 0:
            speedup = f"{baseline_to_use.e2e_ms / m.e2e_ms:.2f}x"
        
        accept_pct = f"{m.draft_acceptance_rate*100:.1f}%" if m.is_dflash else "-"
        avg_acc    = f"{m.avg_accepted_per_step:.2f}" if m.is_dflash else "-"
        lines.append(
            f"{m.label:<30} {m.ttft_ms:>10.1f} {m.tpot_ms:>10.2f} {m.e2e_ms:>10.1f} "
            f"{m.generated_tokens:>7} {speedup:>9} {accept_pct:>9} {avg_acc:>8}"
        )
    lines.append("=" * 105)
    return lines


def print_table(results: List[RunMetrics], baseline_fp16: Optional[RunMetrics]) -> None:
    for line in build_summary_lines(results, baseline_fp16):
        print(line)


def save_run_report(
    args: argparse.Namespace,
    results: List[RunMetrics],
    baseline_fp16: Optional[RunMetrics],
) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    report_dir = repo_root / "dflash优化报告"
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"dflash_benchmark_{timestamp}.txt"

    lines: List[str] = []
    lines.append("DFlash Benchmark Run Report")
    lines.append("=" * 80)
    lines.append(f"timestamp={datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"cwd={Path.cwd()}")
    lines.append(f"command={subprocess.list2cmdline(sys.argv)}")
    lines.append("")
    lines.append("[Run Config]")
    lines.append(f"model_dir={args.model_dir}")
    lines.append(f"draft_model_dir={args.draft_model_dir}")
    lines.append(f"prompt={args.prompt}")
    lines.append(f"device={args.device}")
    lines.append(f"max_tokens={args.max_tokens}")
    lines.append(f"no_think={args.no_think}")
    lines.append(f"precision={args.precision}")
    lines.append("")

    lines.extend(build_summary_lines(results, baseline_fp16))
    lines.append("")
    lines.append("=" * 105)
    lines.append("PER-CONFIG DETAILS")
    lines.append("=" * 105)
    for m in results:
        lines.append(f"[{m.label}]")
        lines.append(f"ttft_ms={m.ttft_ms:.1f}")
        lines.append(f"tpot_ms={m.tpot_ms:.2f}")
        lines.append(f"e2e_ms={m.e2e_ms:.1f}")
        lines.append(f"generated_tokens={m.generated_tokens}")
        lines.append(f"is_dflash={m.is_dflash}")
        lines.append(f"target_quant={m.target_quant}")
        lines.append(f"draft_quant={m.draft_quant}")

        if m.is_dflash:
            lines.append(f"draft_steps={m.draft_steps}")
            lines.append(f"accepted_draft_tokens={m.accepted_draft_tokens}")
            lines.append(f"avg_accepted_per_step={m.avg_accepted_per_step:.4f}")
            lines.append(f"draft_acceptance_rate={m.draft_acceptance_rate:.6f}")
            lines.append(f"accepted_per_step={m.accepted_per_step}")
            lines.append(f"draft_total_ms={m.draft_total_ms:.4f}")
            lines.append(f"avg_draft_step_ms={m.avg_draft_step_ms:.4f}")
            lines.append(f"avg_accepted_draft_token_ms={m.avg_accepted_draft_token_ms:.4f}")
            lines.append(f"draft_decode_count={m.draft_decode_count}")
            lines.append(f"avg_draft_decode_ms={m.avg_draft_decode_ms:.4f}")
            lines.append(f"target_verify_count={m.target_verify_count}")
            lines.append(f"target_replay_count={m.target_replay_count}")
            lines.append(f"target_decode_count={m.target_decode_count}")
            lines.append(f"target_verify_total_ms={m.target_verify_total_ms:.4f}")
            lines.append(f"target_replay_total_ms={m.target_replay_total_ms:.4f}")
            lines.append(f"target_decode_total_ms={m.target_decode_total_ms:.4f}")
            lines.append(f"avg_target_verify_ms={m.avg_target_verify_ms:.4f}")
            lines.append(f"avg_target_replay_ms={m.avg_target_replay_ms:.4f}")
            lines.append(f"avg_target_decode_ms={m.avg_target_decode_ms:.4f}")

            if m.verify_trace_lines:
                lines.append("[Verify Trace]")
                lines.extend(m.verify_trace_lines)

            if m.snapshot_restore_trace_lines:
                lines.append("[Snapshot Restore Trace]")
                lines.extend(m.snapshot_restore_trace_lines)

        if m.output_text:
            lines.append("[Output Text]")
            lines.append(m.output_text)

        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="DFlash quantization benchmark")
    parser.add_argument("model_dir",      nargs="?",
                        default=r"D:\Data\models\Huggingface\Qwen3.5-4B",
                        help="Target (Qwen3.5) model directory")
    parser.add_argument("draft_model_dir", nargs="?",
                        default=r"D:\Data\models\Huggingface\Qwen3.5-4B-DFlash",
                        help="DFlash draft model directory")
    parser.add_argument("prompt",          
                        nargs="?",
                        #default="who are you?",
                        default="Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
                        help="Input prompt")
    parser.add_argument("--device",        default="GPU")
    parser.add_argument("--max-tokens",    type=int, default=512)
    parser.add_argument("--no-think",      action="store_true",
                        help="Disable thinking (Qwen3.5 no-think template)")
    parser.add_argument("--precision",     default="f16", choices=["f16", "f32"],
                        help="Inference precision for DFlash compile (default: f16)")

    # Optional: skip individual configs to save time
    parser.add_argument("--skip-baseline",        action="store_true",
                        help="Skip both Baseline FP16 and Baseline INT4")
    parser.add_argument("--skip-baseline-int4",   action="store_true",
                        help="Skip Baseline INT4 (runs by default)")
    parser.add_argument("--run-baseline-int4",    action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-dflash-fp16-fp16", action="store_true")
    parser.add_argument("--skip-dflash-int4-fp16", action="store_true")
    parser.add_argument("--skip-dflash-fp16-int4", action="store_true")
    parser.add_argument("--skip-dflash-int4-int4", action="store_true")
    
    # Snapshot optimization path (reduces state rollback overhead)
    parser.add_argument("--enable-snapshots", action="store_true",
                        help="Enable snapshot-based state selection (reduces verify latency)")
    parser.add_argument("--enable-serial-verify", action="store_true",
                        help="Enable serial verify mode (token-by-token verify)")
    
    args = parser.parse_args()

    # ── Setup snapshot-related environment variables ──
    if args.enable_snapshots:
        os.environ["OV_GENAI_ENABLE_STATE_SNAPSHOTS"] = "1"
        os.environ["OV_GENAI_FORCE_STATE_SNAPSHOTS"] = "1"
        os.environ["OV_GENAI_USE_FUSED_CONV_OP"] = "1"
        os.environ["OV_GENAI_USE_LINEAR_ATTENTION_OP"] = "1"
        print("[SNAPSHOT] Enabled snapshot-based state selection path")
        print("  Environment variables:")
        print("    OV_GENAI_ENABLE_STATE_SNAPSHOTS=1")
        print("    OV_GENAI_FORCE_STATE_SNAPSHOTS=1")
        print("    OV_GENAI_USE_FUSED_CONV_OP=1")
        print("    OV_GENAI_USE_LINEAR_ATTENTION_OP=1")
    
    if args.enable_serial_verify:
        os.environ["OV_GENAI_DFLASH_SERIAL_VERIFY"] = "1"
        print("[SERIAL_VERIFY] Enabled serial verify mode (per-token verification)")
        print("  Environment variables:")
        print("    OV_GENAI_DFLASH_SERIAL_VERIFY=1")

    results: List[RunMetrics] = []
    baseline_fp16: Optional[RunMetrics] = None

    # ── 1. Baseline FP16 ────────────────────────────────────────────────────
    if not args.skip_baseline:
        m = run_baseline(args.model_dir, args.prompt, args.device,
                         args.max_tokens, args.no_think, quant_mode=None)
        results.append(m)
        baseline_fp16 = m

    # ── 2. Baseline INT4 ────────────────────────────────────────────────────
    # Uses in-flight INT4_ASYM quantization via safetensors loader (same mechanism as DFlash).
    # Works for safetensors model directories; cached openvino_model.xml is bypassed.
    if not args.skip_baseline and not args.skip_baseline_int4:
        m = run_baseline(args.model_dir, args.prompt, args.device,
                         args.max_tokens, args.no_think, quant_mode="INT4_ASYM")
        m.label = "Baseline INT4"
        results.append(m)

    # ── 3. DFlash FP16 / FP16 ───────────────────────────────────────────────
    if not args.skip_dflash_fp16_fp16:
        m = run_dflash(args.model_dir, args.draft_model_dir, args.prompt,
                       args.device, args.max_tokens, args.no_think,
                       target_quant=None, draft_quant=None,
                       inference_precision=args.precision)
        results.append(m)

    # ── 4. DFlash INT4 / FP16 ───────────────────────────────────────────────
    if not args.skip_dflash_int4_fp16:
        m = run_dflash(args.model_dir, args.draft_model_dir, args.prompt,
                       args.device, args.max_tokens, args.no_think,
                       target_quant="INT4_ASYM", draft_quant=None,
                       inference_precision=args.precision)
        results.append(m)

    # ── 5. DFlash FP16 / INT4 ───────────────────────────────────────────────
    if not args.skip_dflash_fp16_int4:
        m = run_dflash(args.model_dir, args.draft_model_dir, args.prompt,
                       args.device, args.max_tokens, args.no_think,
                       target_quant=None, draft_quant="INT4_ASYM",
                       inference_precision=args.precision)
        results.append(m)

    # ── 6. DFlash INT4 / INT4 ───────────────────────────────────────────────
    if not args.skip_dflash_int4_int4:
        m = run_dflash(args.model_dir, args.draft_model_dir, args.prompt,
                       args.device, args.max_tokens, args.no_think,
                       target_quant="INT4_ASYM", draft_quant="INT4_ASYM",
                       inference_precision=args.precision)
        results.append(m)

    print_table(results, baseline_fp16)
    try:
        report_path = save_run_report(args, results, baseline_fp16)
        print(f"[Report] Saved run report: {report_path}")
    except OSError as exc:
        print(f"[Report] Failed to save run report: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
