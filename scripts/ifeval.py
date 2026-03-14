"""IFEval (Instruction-Following Evaluation) benchmark for modeling_qwen3_5.exe.

Runs the 541-prompt IFEval benchmark from Google Research (arXiv:2311.07911)
using modeling_qwen3_5.exe and evaluates with the official checker code.

IFEval tests whether an LLM can follow verifiable format constraints such as
"write at least 300 words", "do not use commas", "respond in all caps", etc.
There are 25 instruction types and 541 prompts containing 834 total constraints.
Evaluation is fully deterministic (no LLM judge needed).

Supports batch testing across multiple model/quant/think combinations (like wwb.py).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent                      # openvino-explicit-modeling
WORKSPACE_DIR = REPO_DIR.parent                    # D:\data\code\Openvino_new_arch_2026
RESULTS_BASE = WORKSPACE_DIR / "results_ifeval"    # output outside repo

OV_BIN = WORKSPACE_DIR / "openvino" / "bin" / "intel64" / "Release"
TBB_BIN = WORKSPACE_DIR / "openvino" / "temp" / "Windows_AMD64" / "tbb" / "bin"
GENAI_DLL = WORKSPACE_DIR / "openvino.genai" / "build" / "openvino_genai"
GENAI_BIN = WORKSPACE_DIR / "openvino.genai" / "build" / "bin"
EXE = GENAI_BIN / "modeling_qwen3_5.exe"

DEFAULT_MODEL_ROOT = Path(r"d:\data\models\Huggingface")

# ---------------------------------------------------------------------------
# Model names (same as wwb.py)
# ---------------------------------------------------------------------------
MODEL_NAMES = [
    "Qwen3.5-0.8B",
    "Qwen3.5-2B",
    "Qwen3.5-4B",
    "Qwen3.5-9B",
    "Qwen3.5-35B-A3B",
]

# ---------------------------------------------------------------------------
# Quantization presets (same as wwb.py)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuantPreset:
    mode: str
    group_size: int
    backup_mode: str

    @property
    def disabled(self) -> bool:
        return self.mode.lower() == "none"

    @property
    def tag(self) -> str:
        if self.disabled:
            return "none"
        return f"{self.mode}_g{self.group_size}_{self.backup_mode}"

    @property
    def short_tag(self) -> str:
        """Short tag for folder naming, e.g. 'int4asym' or 'int4a_b8a' or 'none'."""
        if self.disabled:
            return "none"
        if self.mode == self.backup_mode:
            return self.mode.replace("_", "")
        m = self.mode.replace("int4_asym", "int4a").replace("int8_asym", "int8a")
        b = self.backup_mode.replace("int4_asym", "b4a").replace("int8_asym", "b8a")
        return f"{m}_{b}"

    @property
    def display(self) -> str:
        if self.disabled:
            return "[none, none, none]"
        return f"[{self.mode}, {self.group_size}, {self.backup_mode}]"


QUANT_PRESETS: Dict[int, QuantPreset] = {
    1: QuantPreset("int4_asym", 128, "int4_asym"),
    2: QuantPreset("int4_asym", 128, "int8_asym"),
    3: QuantPreset("none", 0, "none"),
}

# ---------------------------------------------------------------------------
# Index selection parser (same pattern as wwb.py)
# ---------------------------------------------------------------------------
def parse_index_selection(spec: str, min_index: int, max_index: int,
                          arg_name: str, allow_all: bool) -> List[int]:
    """Parse a selector string like '1,3,4' or '1~5' or 'all' into a list of ints."""
    if not spec:
        return list(range(min_index, max_index + 1))

    if allow_all and spec.strip().lower() == "all":
        return list(range(min_index, max_index + 1))

    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"`{arg_name}` is empty.")

    chosen: List[int] = []
    seen = set()
    for token in tokens:
        range_match = re.fullmatch(r"(\d+)\s*[~-]\s*(\d+)", token)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if start <= end:
                expanded = range(start, end + 1)
            else:
                expanded = range(start, end - 1, -1)
            for idx in expanded:
                if idx < min_index or idx > max_index:
                    raise ValueError(
                        f"Index out of range in {arg_name}: {idx}. "
                        f"Valid range is {min_index}~{max_index}.")
                if idx not in seen:
                    chosen.append(idx)
                    seen.add(idx)
            continue

        if not token.isdigit():
            raise ValueError(f"Invalid selector in {arg_name}: `{token}`.")
        idx = int(token)
        if idx < min_index or idx > max_index:
            raise ValueError(
                f"Index out of range in {arg_name}: {idx}. "
                f"Valid range is {min_index}~{max_index}.")
        if idx not in seen:
            chosen.append(idx)
            seen.add(idx)

    return chosen


EXAMPLES = """
Examples:

  # Single model, default settings (quant=1, temperature=1.0)
  python scripts/ifeval.py --models 2

  # Multiple models with quant comparison
  python scripts/ifeval.py --models 2,3 --quant-list 1,2

  # All quant presets on a single model
  python scripts/ifeval.py --models 2 --quant-list all

  # Compare thinking vs non-thinking
  python scripts/ifeval.py --models 2 --think 0,1

  # Full matrix: 2 models x 3 quants x 2 think modes = 12 runs
  python scripts/ifeval.py --models 2,3 --quant-list all --think 0,1

  # Quick smoke test with only the first 10 prompts
  python scripts/ifeval.py --models 2 --limit 10

  # Custom sampling parameters
  python scripts/ifeval.py --models 2 --temperature 0.7 --top-p 0.9 --top-k 50

  # Greedy decoding (temperature=0 uses argmax, ignores top-p/top-k)
  python scripts/ifeval.py --models 2 --temperature 0

  # Custom model root directory
  python scripts/ifeval.py --model-root d:\\models --models 1

  # Resume an interrupted run
  python scripts/ifeval.py --models 2 --resume path\\to\\responses.jsonl

  # Re-evaluate saved responses without running inference
  python scripts/ifeval.py --resume path\\to\\responses.jsonl

  # Specify a custom output directory
  python scripts/ifeval.py --models 2 --output-dir path\\to\\my_experiment

Model index mapping (--models):
  1. Qwen3.5-0.8B
  2. Qwen3.5-2B
  3. Qwen3.5-4B
  4. Qwen3.5-9B
  5. Qwen3.5-35B-A3B

Quantization presets (--quant-list):
  1: [int4_asym, 128, int4_asym]  - INT4 weights + INT4 backup (default, fastest)
  2: [int4_asym, 128, int8_asym]  - INT4 weights + INT8 backup (better accuracy)
  3: [none, none, none]           - No quantization (FP16/BF16, best accuracy)

Sampling parameters (passed to modeling_qwen3_5.exe):
  --temperature    Sampling temperature (default: 1.0, 0 = greedy argmax)
  --top-p          Nucleus sampling threshold (default: 0.95)
  --top-k          Top-K filtering (default: 20)
  --repetition-penalty   Penalty for repeating tokens (default: 1.0)
  --frequency-penalty    Subtract penalty * token_count from logit (default: 0.0)
  --presence-penalty     Subtract penalty if token appeared (default: 1.5)
  --rng-seed       Random seed for sampling (default: 0 = random)
  --think          Think mode selectors (default: 0). Examples: 0 | 1 | 0,1 | all
"""


def build_env(quant: QuantPreset) -> dict:
    """Build the environment dict matching run.bat."""
    env = os.environ.copy()
    extra_path = f"{OV_BIN};{TBB_BIN};{GENAI_DLL};{GENAI_BIN}"
    env["PATH"] = extra_path + ";" + env.get("PATH", "")
    env["OV_GENAI_USE_MODELING_API"] = "1"
    env["OV_GENAI_SAVE_OV_MODEL"] = "1"
    # Clear any inherited quant env vars first
    for key in ["OV_GENAI_INFLIGHT_QUANT_MODE",
                "OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE",
                "OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE"]:
        env.pop(key, None)
    if not quant.disabled:
        env["OV_GENAI_INFLIGHT_QUANT_MODE"] = quant.mode
        env["OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE"] = str(quant.group_size)
        env["OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE"] = quant.backup_mode
    return env


def build_exe_cmd(model_path: str, prompt_file: str, max_tokens: int,
                  sampling: dict, think: int) -> List[str]:
    """Build the modeling_qwen3_5.exe command line with all sampling parameters."""
    cmd = [
        str(EXE),
        "--model", model_path,
        "--cache-model",
        "--mode", "text",
        "--prompt-file", prompt_file,
        "--output-tokens", str(max_tokens),
        "--think", str(think),
    ]
    # Sampling parameters — always pass them so exe uses our values, not its defaults
    cmd.extend(["--temperature", str(sampling["temperature"])])
    cmd.extend(["--top-p", str(sampling["top_p"])])
    cmd.extend(["--top-k", str(sampling["top_k"])])
    cmd.extend(["--repetition-penalty", str(sampling["repetition_penalty"])])
    cmd.extend(["--frequency-penalty", str(sampling["frequency_penalty"])])
    cmd.extend(["--presence-penalty", str(sampling["presence_penalty"])])
    cmd.extend(["--rng-seed", str(sampling["rng_seed"])])
    return cmd


def run_inference(model_path: str, prompt: str, max_tokens: int,
                  sampling: dict, think: int, env: dict) -> str:
    """Run modeling_qwen3_5.exe on a single prompt and return the response text."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        cmd = build_exe_cmd(model_path, prompt_file, max_tokens, sampling, think)
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            env=env, cwd=str(GENAI_BIN),
        )
        output = result.stdout + result.stderr
    finally:
        os.unlink(prompt_file)

    return parse_response(output)


def parse_response(output: str) -> str:
    """Extract the model response from exe output.

    The exe prints metrics then the response, then CLIntercept noise.
    Format:
        Throughput: 110.32 tokens/s
        <response text>
        CLIntercept is shutting down...
    or:
        Throughput: 110.32 tokens/s
        <response text>
        -=-=-=-=...
        CLIntercept (64-bit) is loading...
    """
    lines = output.split("\n")

    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Throughput:"):
            start = i + 1
            break

    if start is None:
        for i, line in enumerate(lines):
            if line.strip().startswith("TPOT:"):
                start = i + 1
                break

    if start is None:
        return ""

    end = len(lines)
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if ("CLIntercept is shutting down" in stripped
                or "CLIntercept" in stripped and "is loading" in stripped
                or stripped.startswith("-=-=-")):
            end = i
            break

    return "\n".join(lines[start:end]).strip()


def strip_think_content(response: str, think: int) -> str:
    """Remove thinking/reasoning content from model response.

    When think=1, a normal response contains <think>...</think> followed by
    the actual answer.  We split on the closing </think> tag and return only
    the content after it.  If </think> is missing, the model failed to
    complete its reasoning (e.g. ran out of tokens) and we return empty string.

    When think=0, the response has no think tags and is returned as-is.
    """
    if think == 0:
        return response

    # think=1: expect <think>...</think> followed by the answer
    if "</think>" in response:
        idx = response.rfind("</think>")
        return response[idx + len("</think>"):].strip()

    # No </think> found — model failed to finish thinking, return empty
    return ""


def load_ifeval_dataset() -> list[dict]:
    """Load IFEval dataset from local JSONL (bundled from Google Research)."""
    data_file = SCRIPT_DIR / "ifeval_lib" / "input_data.jsonl"
    dataset = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def evaluate(dataset: list[dict], prompt_to_response: dict) -> dict:
    """Run IFEval evaluation and return metrics."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from ifeval_lib.evaluation_lib import (
        InputExample, test_instruction_following_strict,
        test_instruction_following_loose,
    )

    inputs = [
        InputExample(
            key=row["key"],
            instruction_id_list=row["instruction_id_list"],
            prompt=row["prompt"],
            kwargs=row["kwargs"],
        )
        for row in dataset
    ]

    strict_outputs = []
    loose_outputs = []
    for inp in inputs:
        if inp.prompt in prompt_to_response:
            strict_outputs.append(
                test_instruction_following_strict(inp, prompt_to_response))
            loose_outputs.append(
                test_instruction_following_loose(inp, prompt_to_response))

    def compute_metrics(outputs):
        prompt_total = prompt_correct = 0
        inst_total = inst_correct = 0
        for ex in outputs:
            prompt_total += 1
            if all(ex.follow_instruction_list):
                prompt_correct += 1
            inst_total += len(ex.instruction_id_list)
            inst_correct += sum(ex.follow_instruction_list)
        return {
            "prompt_accuracy": prompt_correct / max(prompt_total, 1),
            "instruction_accuracy": inst_correct / max(inst_total, 1),
            "prompt_correct": prompt_correct,
            "prompt_total": prompt_total,
            "instruction_correct": inst_correct,
            "instruction_total": inst_total,
        }

    strict = compute_metrics(strict_outputs)
    loose = compute_metrics(loose_outputs)

    return {
        "prompt_strict_accuracy": strict["prompt_accuracy"],
        "prompt_loose_accuracy": loose["prompt_accuracy"],
        "instruction_strict_accuracy": strict["instruction_accuracy"],
        "instruction_loose_accuracy": loose["instruction_accuracy"],
        "details": {"strict": strict, "loose": loose},
    }


def model_short_name(model_path: str) -> str:
    """Extract short model name from path, e.g. 'Qwen3.5-2B' from full path."""
    return Path(model_path).name


def make_run_dir_name(model_name: str, quant: QuantPreset,
                      think: int, sampling: dict, limit: Optional[int]) -> str:
    """Build a descriptive sub-folder name for a single run.

    Format: {model}_{quant}_think{think}_t{temperature}_n{num_prompts}
    Examples:
        Qwen3.5-2B_int4asym_think0_t0.7_n541
        Qwen3.5-9B_int4a_b8a_think1_t1.0_n10

    The parent batch directory already carries the timestamp.
    """
    parts = [
        model_name,
        quant.short_tag,
        f"think{think}",
        f"t{sampling['temperature']}",
        f"n{limit or 541}",
    ]
    return "_".join(parts)


def format_summary(metrics: dict, model_name: str, quant: QuantPreset,
                   think: int, sampling: dict, max_tokens: int,
                   num_prompts: int,
                   inference_time: float | None = None) -> str:
    """Format the results summary as a string (printed to console and saved to file)."""
    s = metrics["details"]["strict"]
    l = metrics["details"]["loose"]
    avg = (metrics["prompt_strict_accuracy"] + metrics["prompt_loose_accuracy"] +
           metrics["instruction_strict_accuracy"] + metrics["instruction_loose_accuracy"]) / 4

    lines = []
    lines.append("=" * 70)
    lines.append("IFEval Results Summary")
    lines.append("=" * 70)
    lines.append(f"  Model:        {model_name}")
    lines.append(f"  Quant:        {quant.display}")
    lines.append(f"  Think:        {think}")
    lines.append(f"  Temperature:  {sampling['temperature']}")
    lines.append(f"  Top-p:        {sampling['top_p']}")
    lines.append(f"  Top-k:        {sampling['top_k']}")
    lines.append(f"  Rep penalty:  {sampling['repetition_penalty']}")
    lines.append(f"  Freq penalty: {sampling['frequency_penalty']}")
    lines.append(f"  Pres penalty: {sampling['presence_penalty']}")
    lines.append(f"  RNG seed:     {sampling['rng_seed']}")
    lines.append(f"  Max tokens:   {max_tokens}")
    lines.append(f"  Num prompts:  {num_prompts}")
    if inference_time is not None:
        lines.append(f"  Inference:    {inference_time:.1f}s total, "
                     f"{inference_time / max(num_prompts, 1):.1f}s/prompt")
    lines.append("")
    lines.append(f"  Prompt-level  strict accuracy:  {metrics['prompt_strict_accuracy']:.4f}  "
                 f"({s['prompt_correct']}/{s['prompt_total']})")
    lines.append(f"  Prompt-level  loose  accuracy:  {metrics['prompt_loose_accuracy']:.4f}  "
                 f"({l['prompt_correct']}/{l['prompt_total']})")
    lines.append(f"  Inst-level    strict accuracy:  {metrics['instruction_strict_accuracy']:.4f}  "
                 f"({s['instruction_correct']}/{s['instruction_total']})")
    lines.append(f"  Inst-level    loose  accuracy:  {metrics['instruction_loose_accuracy']:.4f}  "
                 f"({l['instruction_correct']}/{l['instruction_total']})")
    lines.append("")
    lines.append(f"  Average (all 4 metrics):        {avg:.4f}")
    lines.append("=" * 70)
    return "\n".join(lines)


def run_single_eval(model_path: str, model_name: str, quant: QuantPreset,
                    quant_idx: int, think: int, sampling: dict,
                    max_tokens: int, dataset: list[dict],
                    limit: Optional[int],
                    resume_path: Optional[str],
                    output_dir: Optional[str],
                    batch_dir: Optional[Path] = None) -> dict:
    """Run a single IFEval evaluation (inference + scoring) and return results dict."""
    # Setup output directory for this run
    if output_dir:
        out_dir = Path(output_dir)
    elif batch_dir:
        out_dir = batch_dir / make_run_dir_name(
            model_name, quant, think, sampling, limit)
    else:
        out_dir = RESULTS_BASE / make_run_dir_name(
            model_name, quant, think, sampling, limit)
    out_dir.mkdir(parents=True, exist_ok=True)

    responses_file = out_dir / "responses.jsonl"

    # Load existing responses if resuming
    prompt_to_response: dict[str, str] = {}
    if resume_path:
        rp = Path(resume_path)
        print(f"  Resuming from {rp}")
        with open(rp, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                prompt_to_response[entry["prompt"]] = entry["response"]
        print(f"    Loaded {len(prompt_to_response)} existing responses")

    remaining = [row for row in dataset if row["prompt"] not in prompt_to_response]

    inference_time = None
    if remaining:
        print(f"  Running inference on {len(remaining)} prompts...")
        env = build_env(quant)
        t0 = time.time()

        for i, row in enumerate(remaining):
            prompt = row["prompt"]
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (len(remaining) - i) if i > 0 else 0

            print(f"    [{i+1}/{len(remaining)}] key={row['key']}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", end="", flush=True)

            try:
                response = run_inference(
                    model_path, prompt, max_tokens, sampling, think, env)
                prompt_to_response[prompt] = response
                print(f"  -> {len(response)} chars")
            except Exception as e:
                print(f"  ERROR: {e}")
                prompt_to_response[prompt] = ""

            with open(responses_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"prompt": prompt, "response": prompt_to_response[prompt]},
                                   ensure_ascii=False) + "\n")

        inference_time = time.time() - t0
        print(f"  Inference complete: {inference_time:.1f}s total, "
              f"{inference_time/len(remaining):.1f}s per prompt")

    # Strip thinking content before evaluation (think=1 outputs reasoning
    # text that would pollute IFEval format checks like word count, commas, etc.)
    eval_responses = {}
    stripped_count = 0
    empty_count = 0
    for p, r in prompt_to_response.items():
        cleaned = strip_think_content(r, think)
        if len(cleaned) != len(r):
            stripped_count += 1
        if not cleaned and r:
            empty_count += 1
        eval_responses[p] = cleaned
    if stripped_count:
        print(f"  Stripped thinking content from {stripped_count}/{len(eval_responses)} responses")
    if empty_count:
        print(f"  WARNING: {empty_count} responses had no </think> tag "
              f"(model failed to finish reasoning, scored as empty)")

    # Evaluate
    print("  Evaluating responses...")
    metrics = evaluate(dataset, eval_responses)

    # Format and print per-run summary
    summary = format_summary(
        metrics, model_name, quant, think, sampling,
        max_tokens, len(dataset), inference_time)
    print()
    print(summary)

    # Save results.json
    results_file = out_dir / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "model_path": model_path,
            "quant_preset": quant_idx,
            "quant": quant.display,
            "think": think,
            "sampling": sampling,
            "max_tokens": max_tokens,
            "num_prompts": len(dataset),
            "inference_time_s": inference_time,
            "metrics": metrics,
        }, f, indent=2, ensure_ascii=False)

    # Save summary.txt
    summary_file = out_dir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(f"  Results saved to: {out_dir}")

    avg = (metrics["prompt_strict_accuracy"] + metrics["prompt_loose_accuracy"] +
           metrics["instruction_strict_accuracy"] + metrics["instruction_loose_accuracy"]) / 4

    return {
        "model_name": model_name,
        "quant_display": quant.display,
        "think": think,
        "temperature": sampling["temperature"],
        "num_prompts": len(dataset),
        "prompt_strict": metrics["prompt_strict_accuracy"],
        "prompt_loose": metrics["prompt_loose_accuracy"],
        "inst_strict": metrics["instruction_strict_accuracy"],
        "inst_loose": metrics["instruction_loose_accuracy"],
        "average": avg,
        "inference_time_s": inference_time,
        "out_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# Summary table (markdown)
# ---------------------------------------------------------------------------
def build_summary_markdown(rows: List[dict], sampling: dict) -> str:
    """Build a markdown summary table from all run results."""
    lines = [
        "# IFEval Batch Summary",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Total runs: {len(rows)}",
        f"- Sampling: temperature={sampling['temperature']}, "
        f"top_p={sampling['top_p']}, top_k={sampling['top_k']}, "
        f"rep_penalty={sampling['repetition_penalty']}, "
        f"freq_penalty={sampling['frequency_penalty']}, "
        f"pres_penalty={sampling['presence_penalty']}, "
        f"rng_seed={sampling['rng_seed']}",
        "",
        "| Model | Quant | Think | #Prompts | Prompt Strict | Prompt Loose "
        "| Inst Strict | Inst Loose | Average | Time (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in rows:
        t = f"{r['inference_time_s']:.0f}" if r['inference_time_s'] is not None else "N/A"
        lines.append(
            f"| {r['model_name']} "
            f"| {r['quant_display']} "
            f"| {r['think']} "
            f"| {r['num_prompts']} "
            f"| {r['prompt_strict']:.4f} "
            f"| {r['prompt_loose']:.4f} "
            f"| {r['inst_strict']:.4f} "
            f"| {r['inst_loose']:.4f} "
            f"| {r['average']:.4f} "
            f"| {t} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    model_mapping = "\n".join(
        f"  {idx}. {name}" for idx, name in enumerate(MODEL_NAMES, start=1))
    quant_mapping = "\n".join(
        f"  {idx}. {preset.display}" for idx, preset in QUANT_PRESETS.items())

    parser = argparse.ArgumentParser(
        description="IFEval (Instruction-Following Evaluation) benchmark for modeling_qwen3_5.exe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )

    # --- Model selection ---
    parser.add_argument(
        "--models", "--models-list",
        dest="models",
        default="2",
        help=(
            "Model index selectors (default: 2). Examples: 1,3,4 | 1~5 | 2,4~5\n"
            "Model index mapping:\n"
            f"{model_mapping}"
        ),
    )
    parser.add_argument(
        "--model-root",
        default=str(DEFAULT_MODEL_ROOT),
        help=f"Model root folder (default: {DEFAULT_MODEL_ROOT}).",
    )

    # --- Quant selection ---
    parser.add_argument(
        "--quant-list",
        default="1",
        help=(
            "Quant preset selectors (default: 1). Examples: 1 | 2,3 | all | 1~3\n"
            "Quant preset mapping:\n"
            f"{quant_mapping}"
        ),
    )

    # --- Think selection ---
    parser.add_argument(
        "--think",
        metavar="THINK_LIST",
        default="0",
        help=(
            "Think mode selectors (default: 0). Examples: 0 | 1 | 0,1 | all\n"
            "  0 = thinking disabled (non-thinking mode)\n"
            "  1 = thinking enabled"
        ),
    )

    # --- Sampling parameters ---
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0). 0 = greedy argmax, "
             "ignores top-p/top-k. Qwen recommends 0.7 for non-thinking mode.")
    parser.add_argument(
        "--top-p", type=float, default=0.95,
        help="Nucleus sampling threshold (default: 0.95). "
             "Only tokens with cumulative probability <= top-p are kept. "
             "Set to 1.0 to disable.")
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Top-K filtering (default: 20). Only the top-K highest-probability "
             "tokens are kept before nucleus sampling. Set to 0 to disable.")
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.0,
        help="Multiplicative penalty for repeating tokens (default: 1.0 = no penalty). "
             "Values > 1.0 discourage repetition.")
    parser.add_argument(
        "--frequency-penalty", type=float, default=0.0,
        help="Subtract penalty * token_count from logit (default: 0.0). "
             "Penalizes tokens proportional to how often they appeared.")
    parser.add_argument(
        "--presence-penalty", type=float, default=1.5,
        help="Subtract penalty if token appeared at all (default: 1.5). "
             "Encourages the model to use new tokens.")
    parser.add_argument(
        "--rng-seed", type=int, default=0,
        help="Random seed for sampling (default: 0 = random). "
             "Set to a fixed value for reproducible results.")

    # --- Other ---
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Maximum output tokens per prompt (default: 2048). "
             "Some IFEval instructions require long responses (300+ words).")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Custom output directory (only useful for single-run mode). "
             "If omitted, results are saved to results_ifeval/<auto_name>/ "
             "under the workspace root.")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to an existing responses.jsonl file. "
             "With models: loads saved responses and only runs inference for "
             "missing prompts. Without models: re-evaluates saved responses "
             "without any inference (evaluation-only mode).")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N prompts (useful for quick smoke tests). "
             "Example: --limit 10 runs ~1 minute instead of ~67 minutes.")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Build sampling dict from args
    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "rng_seed": args.rng_seed,
    }

    # Parse list selectors
    try:
        selected_model_indices = parse_index_selection(
            args.models, 1, len(MODEL_NAMES), "--models", allow_all=False)
    except ValueError as e:
        parser.error(str(e))
    try:
        selected_quant_indices = parse_index_selection(
            args.quant_list, 1, max(QUANT_PRESETS.keys()), "--quant-list", allow_all=True)
        invalid = [idx for idx in selected_quant_indices if idx not in QUANT_PRESETS]
        if invalid:
            parser.error(f"Unsupported quant preset index: {invalid}")
    except ValueError as e:
        parser.error(str(e))
    try:
        selected_think_values = parse_index_selection(
            args.think, 0, 1, "--think", allow_all=True)
    except ValueError as e:
        parser.error(str(e))

    # Resolve model paths
    model_root = Path(args.model_root)
    model_paths = {idx: model_root / MODEL_NAMES[idx - 1] for idx in selected_model_indices}

    # Validate model paths exist
    missing = [str(p) for idx, p in model_paths.items() if not p.exists()]
    if missing and not args.resume:
        print("ERROR: The following model paths do not exist:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 2

    # Load dataset
    print("Loading IFEval dataset (541 prompts)...")
    dataset = load_ifeval_dataset()
    if args.limit:
        dataset = dataset[:args.limit]
    print(f"  Using {len(dataset)} prompts")

    # Count total combinations
    total_combos = len(selected_model_indices) * len(selected_quant_indices) * len(selected_think_values)
    print(f"\nPlanned runs: {len(selected_model_indices)} models x "
          f"{len(selected_quant_indices)} quants x "
          f"{len(selected_think_values)} think modes = {total_combos} total")
    print(f"  Models: {[MODEL_NAMES[i-1] for i in selected_model_indices]}")
    print(f"  Quants: {[QUANT_PRESETS[i].display for i in selected_quant_indices]}")
    print(f"  Think:  {selected_think_values}")
    print(f"  Sampling: temperature={sampling['temperature']}, top_p={sampling['top_p']}, "
          f"top_k={sampling['top_k']}")
    print()

    # Handle resume-only mode (no models needed)
    if args.resume and total_combos == 0:
        # Just re-evaluate
        print("Re-evaluation-only mode (--resume without model selection)")
        # Use a default quant/think for display
        quant = QUANT_PRESETS[1]
        result = run_single_eval(
            model_path="(resumed)", model_name="(resumed)",
            quant=quant, quant_idx=1, think=0,
            sampling=sampling, max_tokens=args.max_tokens,
            dataset=dataset, limit=args.limit,
            resume_path=args.resume, output_dir=args.output_dir,
            batch_dir=batch_dir)
        return 0

    # Create a batch run directory (all sub-runs go inside)
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = RESULTS_BASE / batch_timestamp
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Run all combinations
    all_results: List[dict] = []
    combo_num = 0

    for model_idx in selected_model_indices:
        model_name = MODEL_NAMES[model_idx - 1]
        model_path = str(model_paths[model_idx])

        for quant_idx in selected_quant_indices:
            quant = QUANT_PRESETS[quant_idx]

            for think in selected_think_values:
                combo_num += 1
                print(f"\n{'='*70}")
                print(f"Run {combo_num}/{total_combos}: "
                      f"{model_name} | Q{quant_idx} {quant.display} | "
                      f"Think={think} | Temp={sampling['temperature']}")
                print(f"{'='*70}")

                # Only use --output-dir for single runs
                out_dir = args.output_dir if total_combos == 1 else None

                result = run_single_eval(
                    model_path=model_path,
                    model_name=model_name,
                    quant=quant,
                    quant_idx=quant_idx,
                    think=think,
                    sampling=sampling,
                    max_tokens=args.max_tokens,
                    dataset=dataset,
                    limit=args.limit,
                    resume_path=args.resume,
                    output_dir=out_dir,
                    batch_dir=batch_dir,
                )
                all_results.append(result)

    # Build and print summary table (always, even for a single run)
    summary_md = build_summary_markdown(all_results, sampling)

    print(f"\n{'='*70}")
    print("All runs complete. Summary:")
    print(f"{'='*70}")
    print()
    print(summary_md)

    # Save summary markdown into the batch directory
    summary_file = batch_dir / "batch_summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Batch summary saved to: {summary_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
