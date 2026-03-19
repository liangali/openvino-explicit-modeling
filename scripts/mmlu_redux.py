"""MMLU-Redux benchmark for modeling_qwen3_5.exe.

Runs the MMLU-Redux benchmark (~3042 error-annotated OK questions across 30
subjects, or all ~5330 questions across 57 subjects) using modeling_qwen3_5.exe
and evaluates by matching the model's answer letter (A/B/C/D) against ground
truth.

MMLU-Redux is a quality-filtered re-annotation of the original MMLU benchmark
(Hendrycks et al., 2021). The re-annotation identifies and categorizes errors
in the original test set. By default (--filter-ok), only questions labeled "OK"
are evaluated, giving a cleaner signal. Use --no-filter-ok to include all
questions.

Each question is a 4-choice MCQ in English. Evaluation uses 5-shot prompting
with examples from the MMLU dev split (configurable via --n-shot). If the dev
data is not available, falls back to 0-shot with a warning.

Supports batch testing across multiple model/quant/think combinations
(same CLI as ifeval.py).
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import locale
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
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
RESULTS_BASE = WORKSPACE_DIR / "results_mmlu_redux"

OV_BIN = WORKSPACE_DIR / "openvino" / "bin" / "intel64" / "Release"
TBB_BIN = WORKSPACE_DIR / "openvino" / "temp" / "Windows_AMD64" / "tbb" / "bin"
GENAI_DLL = WORKSPACE_DIR / "openvino.genai" / "build" / "openvino_genai"
GENAI_RUNTIME_BIN = WORKSPACE_DIR / "openvino.genai" / "build" / "bin"
GENAI_BIN = WORKSPACE_DIR / "openvino.genai" / "build" / "bin" / "Release"
EXE = GENAI_BIN / "modeling_qwen3_5.exe"

DEFAULT_MODEL_ROOT = Path(r"d:\data\models\Huggingface")

# MMLU-Redux data directories (auto-downloaded)
DATA_DIR = SCRIPT_DIR / "mmlu_redux_data"
MMLU_DEV_DIR = SCRIPT_DIR / "mmlu_dev_data"

# ---------------------------------------------------------------------------
# Model names (same as ifeval.py / wwb.py)
# ---------------------------------------------------------------------------
MODEL_NAMES = [
    "Qwen3.5-0.8B",
    "Qwen3.5-2B",
    "Qwen3.5-4B",
    "Qwen3.5-9B",
    "Qwen3.5-35B-A3B",
]

# ---------------------------------------------------------------------------
# Quantization presets (same as ifeval.py)
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
# MMLU-Redux subjects (57 subjects from aryopg/mmlu-redux)
# ---------------------------------------------------------------------------
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

# Index-to-letter mapping for MMLU-Redux (answers are stored as 0-3)
INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}

# ---------------------------------------------------------------------------
# Index selection parser (same as ifeval.py)
# ---------------------------------------------------------------------------
def parse_index_selection(spec: str, min_index: int, max_index: int,
                          arg_name: str, allow_all: bool) -> List[int]:
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
            expanded = range(start, end + 1) if start <= end else range(start, end - 1, -1)
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


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def _find_data_dir() -> Optional[Path]:
    """Locate the directory containing mmlu_*.csv files.

    The aryopg/mmlu-redux repo layout: mmlu_redux_data/mmlu_redux/mmlu_*.csv
    """
    candidates = [
        DATA_DIR / "mmlu_redux",             # git clone layout
        DATA_DIR / "data",                    # alternative layout
        DATA_DIR,                             # flat layout
    ]
    for c in candidates:
        if c.exists() and list(c.glob("mmlu_*.csv")):
            return c
    # Also try without mmlu_ prefix
    for c in candidates:
        if c.exists() and list(c.glob("*.csv")):
            return c
    return None


def _find_dev_dir() -> Optional[Path]:
    """Locate the MMLU dev data directory."""
    if MMLU_DEV_DIR.exists() and list(MMLU_DEV_DIR.glob("*_dev.csv")):
        return MMLU_DEV_DIR
    sub = MMLU_DEV_DIR / "data" / "dev"
    if sub.exists() and list(sub.glob("*_dev.csv")):
        return sub
    return None


def download_mmlu_redux_data() -> None:
    """Download MMLU-Redux dataset from GitHub."""
    data_dir = _find_data_dir()
    if data_dir is not None:
        n = len(list(data_dir.glob("mmlu_*.csv")) or list(data_dir.glob("*.csv")))
        print(f"MMLU-Redux data already exists at {data_dir} ({n} subjects)")
    else:
        print("Downloading MMLU-Redux dataset from GitHub...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/aryopg/mmlu-redux.git",
                 str(DATA_DIR)],
                check=True)
        except Exception as e:
            print(f"  git clone failed: {e}")
            print("\nPlease download manually:")
            print(f"  git clone https://github.com/aryopg/mmlu-redux.git {DATA_DIR}")
            sys.exit(1)

        data_dir = _find_data_dir()
        if data_dir is None:
            print(f"ERROR: Cloned repo but cannot find data CSVs under {DATA_DIR}")
            sys.exit(1)
        n = len(list(data_dir.glob("mmlu_*.csv")) or list(data_dir.glob("*.csv")))
        print(f"  MMLU-Redux data ready: {n} subjects found")

    # Download MMLU dev data for few-shot examples
    dev_dir = _find_dev_dir()
    if dev_dir is not None:
        n = len(list(dev_dir.glob("*_dev.csv")))
        print(f"MMLU dev data already exists at {dev_dir} ({n} subjects)")
        return

    print("Downloading MMLU dev data (for 5-shot examples)...")
    print("  Downloading MMLU data archive from hendrycks/test...")
    MMLU_DEV_DIR.mkdir(parents=True, exist_ok=True)

    # The MMLU data is distributed as a tar.gz; try downloading it
    tar_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
    import tarfile
    import io
    tar_path = MMLU_DEV_DIR / "_mmlu_data.tar"
    try:
        urllib.request.urlretrieve(tar_url, str(tar_path))
        with tarfile.open(str(tar_path)) as tf:
            # Extract only dev/ files
            for member in tf.getmembers():
                if "/dev/" in member.name and member.name.endswith("_dev.csv"):
                    # Extract to flat structure in MMLU_DEV_DIR
                    fname = Path(member.name).name
                    member_f = tf.extractfile(member)
                    if member_f:
                        (MMLU_DEV_DIR / fname).write_bytes(member_f.read())
        tar_path.unlink()
        n = len(list(MMLU_DEV_DIR.glob("*_dev.csv")))
        print(f"  Extracted {n} dev files")
    except Exception as e:
        print(f"  Download failed: {e}")
        if tar_path.exists():
            tar_path.unlink()
        print("  WARNING: Could not download MMLU dev data.")
        print("  The benchmark will run in 0-shot mode (no few-shot examples).")
        print("  To manually download, get data.tar from:")
        print(f"    {tar_url}")
        print(f"  Extract the data/dev/*.csv files to: {MMLU_DEV_DIR}/")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def _parse_choices(choices_str: str) -> List[str]:
    """Parse the choices column from MMLU-Redux CSV.

    The choices are stored as a Python list repr, e.g.:
    "['True, True', 'False, False', 'True, False', 'False, True']"
    """
    try:
        choices = ast.literal_eval(choices_str)
        if isinstance(choices, list):
            return [str(c).strip() for c in choices]
    except (ValueError, SyntaxError):
        pass
    return []


def discover_subjects() -> List[str]:
    """Discover available subjects from the data directory."""
    data_dir = _find_data_dir()
    if data_dir is None:
        return []
    subjects = []
    for csv_file in sorted(data_dir.glob("mmlu_*.csv")):
        # Strip the mmlu_ prefix: mmlu_abstract_algebra.csv -> abstract_algebra
        subject = csv_file.stem
        if subject.startswith("mmlu_"):
            subject = subject[5:]
        subjects.append(subject)
    if not subjects:
        # Fallback: try without mmlu_ prefix
        for csv_file in sorted(data_dir.glob("*.csv")):
            subjects.append(csv_file.stem)
    return subjects


def _find_subject_csv(subject: str) -> Optional[Path]:
    """Find the CSV file for a subject, trying both naming conventions."""
    data_dir = _find_data_dir()
    if data_dir is None:
        return None
    # Try mmlu_ prefix first (aryopg/mmlu-redux layout)
    path = data_dir / f"mmlu_{subject}.csv"
    if path.exists():
        return path
    # Try without prefix
    path = data_dir / f"{subject}.csv"
    if path.exists():
        return path
    return None


def load_subject_data(subject: str, filter_ok: bool = True) -> List[dict]:
    """Load test questions for a subject from the MMLU-Redux dataset.

    The CSV has columns: question, choices, answer, error_type, source,
    correct_answer, potential_reason.

    Returns list of dicts with keys: question, A, B, C, D, answer (letter).
    """
    csv_path = _find_subject_csv(subject)
    if csv_path is None:
        return []

    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter by error_type if requested
            if filter_ok:
                error_type = row.get("error_type", "").strip()
                if error_type.lower() not in ("ok", ""):
                    continue

            choices = _parse_choices(row.get("choices", "[]"))
            if len(choices) < 4:
                continue

            # Convert numeric answer index to letter
            try:
                answer_idx = int(row.get("answer", -1))
            except (ValueError, TypeError):
                continue
            answer_letter = INDEX_TO_LETTER.get(answer_idx, "")
            if not answer_letter:
                continue

            rows.append({
                "question": row.get("question", "").strip(),
                "A": choices[0],
                "B": choices[1],
                "C": choices[2],
                "D": choices[3],
                "answer": answer_letter,
            })
    return rows


def load_dev_examples(subject: str) -> List[dict]:
    """Load dev examples for few-shot prompting from the MMLU dev set.

    The dev CSVs (from hendrycks/test) have no header and columns:
    question, A, B, C, D, answer (letter).
    """
    dev_dir = _find_dev_dir()
    if dev_dir is None:
        return []
    dev_path = dev_dir / f"{subject}_dev.csv"
    if not dev_path.exists():
        return []

    rows = []
    with open(dev_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_list in reader:
            if len(row_list) < 6:
                continue
            rows.append({
                "question": row_list[0],
                "A": row_list[1],
                "B": row_list[2],
                "C": row_list[3],
                "D": row_list[4],
                "answer": row_list[5].strip(),
            })
    return rows


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def format_subject_name(subject: str) -> str:
    """Convert subject_key to display name: 'abstract_algebra' -> 'abstract algebra'."""
    return subject.replace("_", " ")


def build_mmlu_prompt(question_row: dict, dev_rows: List[dict],
                      subject: str, n_shot: int) -> str:
    """Build an MMLU prompt with n-shot examples.

    Format:
        The following are multiple choice questions (with answers) about {subject}.

        {dev_question}
        A. {A}
        B. {B}
        C. {C}
        D. {D}
        Answer: {answer}

        ...

        {test_question}
        A. {A}
        B. {B}
        C. {C}
        D. {D}
        Answer:
    """
    subject_display = format_subject_name(subject)
    header = f"The following are multiple choice questions (with answers) about {subject_display}.\n"

    parts = [header]

    # Few-shot examples
    examples = dev_rows[:n_shot]
    for ex in examples:
        q = (f"{ex['question']}\n"
             f"A. {ex['A']}\n"
             f"B. {ex['B']}\n"
             f"C. {ex['C']}\n"
             f"D. {ex['D']}\n"
             f"Answer: {ex['answer']}\n")
        parts.append(q)

    # Test question
    q = (f"{question_row['question']}\n"
         f"A. {question_row['A']}\n"
         f"B. {question_row['B']}\n"
         f"C. {question_row['C']}\n"
         f"D. {question_row['D']}\n"
         f"Answer:")
    parts.append(q)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------
def parse_mcq_answer(response: str) -> str:
    """Extract the answer letter (A/B/C/D) from the model response.

    The model typically generates an explanation first, then states the answer
    at the end (e.g., "Answer: B" or "答案：D").  We search for explicit
    answer patterns from the END of the response first, then fall back to
    simpler heuristics.

    Returns the letter (uppercase) or empty string if no valid letter found.
    """
    if not response:
        return ""

    # Strategy 1: Look for explicit answer patterns (highest confidence).
    # Use the LAST match — the model often discusses options A/B/C/D in its
    # explanation before stating the final answer at the end.
    answer_patterns = [
        r'(?:answer|Answer)\s*(?:is|:)\s*\**([A-D])\**',  # Answer: D / answer is B
        r'(?:the answer|The answer)\s+is\s+\**([A-D])\**', # The answer is C
        r'答案[是为]?[：:]\s*\**([A-D])\**',        # 答案：D / 答案是：**D**
        r'正确答案[是为]\s*\**([A-D])\**',           # 正确答案是C
        r'答案[是为]\s*\**([A-D])\**',              # 答案是D / 答案为D
        r'(?:应选|应该选|选)\s*\**([A-D])\**',        # 应选D
        r'故选\s*\**([A-D])\**',                    # 故选D
    ]
    for pattern in answer_patterns:
        matches = list(re.finditer(pattern, response))
        if matches:
            return matches[-1].group(1)

    # Strategy 2: Response starts with just a letter (model directly answered)
    first = response.strip()
    if first and first[0] in "ABCD" and (len(first) == 1 or first[1] in ' \t\n\r.。,，\r'):
        return first[0]

    # Strategy 3: Last standalone A-D letter in the response
    matches = list(re.finditer(r'\b([A-D])\b', response))
    if matches:
        return matches[-1].group(1)

    # Strategy 4: Last A-D character anywhere
    for ch in reversed(response):
        if ch in "ABCD":
            return ch

    return ""


# ---------------------------------------------------------------------------
# Inference (same as ifeval.py)
# ---------------------------------------------------------------------------
def build_env(quant: QuantPreset) -> dict:
    env = os.environ.copy()
    extra_path = f"{OV_BIN};{TBB_BIN};{GENAI_DLL};{GENAI_RUNTIME_BIN};{GENAI_BIN}"
    env["PATH"] = extra_path + ";" + env.get("PATH", "")
    env["OV_GENAI_USE_MODELING_API"] = "1"
    env["OV_GENAI_SAVE_OV_MODEL"] = "1"
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
    cmd = [
        str(EXE),
        "--model", model_path,
        "--cache-model",
        "--mode", "text",
        "--prompt-file", prompt_file,
        "--output-tokens", str(max_tokens),
        "--think", str(think),
    ]
    cmd.extend(["--temperature", str(sampling["temperature"])])
    cmd.extend(["--top-p", str(sampling["top_p"])])
    cmd.extend(["--top-k", str(sampling["top_k"])])
    cmd.extend(["--repetition-penalty", str(sampling["repetition_penalty"])])
    cmd.extend(["--frequency-penalty", str(sampling["frequency_penalty"])])
    cmd.extend(["--presence-penalty", str(sampling["presence_penalty"])])
    cmd.extend(["--rng-seed", str(sampling["rng_seed"])])
    return cmd


def _decode_subprocess_bytes(data: bytes | None) -> str:
    if not data:
        return ""
    encodings = []
    for enc in ("utf-8", locale.getpreferredencoding(False), "mbcs"):
        if enc and enc.lower() not in {e.lower() for e in encodings}:
            encodings.append(enc)
    for enc in encodings:
        try:
            return data.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def _collect_subprocess_output(result: subprocess.CompletedProcess[bytes]) -> str:
    return (
        _decode_subprocess_bytes(result.stdout) +
        _decode_subprocess_bytes(result.stderr)
    )


def parse_response(output: str) -> str:
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
    if think == 0:
        return response
    if "</think>" in response:
        idx = response.rfind("</think>")
        return response[idx + len("</think>"):].strip()
    return ""


def run_inference(model_path: str, prompt: str, max_tokens: int,
                  sampling: dict, think: int, env: dict) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        f.write(prompt)
        prompt_file = f.name
    try:
        cmd = build_exe_cmd(model_path, prompt_file, max_tokens, sampling, think)
        result = subprocess.run(
            cmd, capture_output=True, timeout=300,
            env=env, cwd=str(GENAI_BIN),
        )
        output = _collect_subprocess_output(result)
    finally:
        os.unlink(prompt_file)
    return parse_response(output)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_mmlu(results_by_subject: Dict[str, List[dict]]) -> dict:
    """Compute MMLU-Redux metrics.

    Returns:
        {
            "macro_avg": float,
            "micro_avg": float,
            "per_subject": { subject: { "accuracy", "correct", "total", "empty" } },
            "total_correct": int,
            "total_questions": int,
            "empty_answers": int,
        }
    """
    per_subject = {}
    total_correct = 0
    total_questions = 0
    empty_answers = 0

    for subject, results in sorted(results_by_subject.items()):
        correct = sum(1 for r in results if r["predicted"] == r["answer"])
        total = len(results)
        empty = sum(1 for r in results if r["predicted"] == "")
        acc = correct / max(total, 1)

        per_subject[subject] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "empty": empty,
        }
        total_correct += correct
        total_questions += total
        empty_answers += empty

    all_accs = [v["accuracy"] for v in per_subject.values()]
    macro_avg = sum(all_accs) / max(len(all_accs), 1)
    micro_avg = total_correct / max(total_questions, 1)

    return {
        "macro_avg": macro_avg,
        "micro_avg": micro_avg,
        "per_subject": per_subject,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "empty_answers": empty_answers,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def make_run_dir_name(model_name: str, quant: QuantPreset,
                      think: int, sampling: dict, limit: Optional[int],
                      total_questions: int) -> str:
    parts = [
        model_name,
        quant.short_tag,
        f"think{think}",
        f"t{sampling['temperature']}",
        f"n{limit or total_questions}",
    ]
    return "_".join(parts)


def format_summary(metrics: dict, model_name: str, quant: QuantPreset,
                   think: int, sampling: dict, max_tokens: int,
                   n_shot: int, filter_ok: bool,
                   inference_time: float | None = None) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("MMLU-Redux Results Summary")
    lines.append("=" * 70)
    lines.append(f"  Model:        {model_name}")
    lines.append(f"  Quant:        {quant.display}")
    lines.append(f"  Think:        {think}")
    lines.append(f"  N-shot:       {n_shot}")
    lines.append(f"  Filter OK:    {filter_ok}")
    lines.append(f"  Temperature:  {sampling['temperature']}")
    lines.append(f"  Top-p:        {sampling['top_p']}")
    lines.append(f"  Top-k:        {sampling['top_k']}")
    lines.append(f"  Max tokens:   {max_tokens}")
    lines.append(f"  Subjects:     {len(metrics['per_subject'])}")
    lines.append(f"  Questions:    {metrics['total_questions']}")
    lines.append(f"  Correct:      {metrics['total_correct']}")
    lines.append(f"  Empty:        {metrics['empty_answers']}")
    if inference_time is not None:
        lines.append(f"  Inference:    {inference_time:.1f}s total, "
                     f"{inference_time / max(metrics['total_questions'], 1):.1f}s/question")
    lines.append("")
    lines.append(f"  Macro Average:    {metrics['macro_avg']:.4f}  "
                 f"(avg of {len(metrics['per_subject'])} subject accuracies)")
    lines.append(f"  Micro Average:    {metrics['micro_avg']:.4f}  "
                 f"({metrics['total_correct']}/{metrics['total_questions']})")
    lines.append("")

    # Top/bottom 5 subjects
    sorted_subjects = sorted(metrics["per_subject"].items(),
                            key=lambda x: x[1]["accuracy"], reverse=True)
    lines.append("  Top 5 subjects:")
    for subject, info in sorted_subjects[:5]:
        lines.append(f"    {subject:45s}  {info['accuracy']:.4f}  "
                     f"({info['correct']}/{info['total']})")

    lines.append("  Bottom 5 subjects:")
    for subject, info in sorted_subjects[-5:]:
        lines.append(f"    {subject:45s}  {info['accuracy']:.4f}  "
                     f"({info['correct']}/{info['total']})")

    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------
def run_single_eval(model_path: str, model_name: str, quant: QuantPreset,
                    quant_idx: int, think: int, sampling: dict,
                    max_tokens: int, n_shot: int, filter_ok: bool,
                    subjects: List[str],
                    limit: Optional[int],
                    resume_path: Optional[str],
                    output_dir: Optional[str],
                    batch_dir: Optional[Path] = None) -> dict:
    """Run a single MMLU-Redux evaluation and return results dict."""

    # Collect all questions across subjects
    all_questions = []
    dev_warned = False
    for subject in subjects:
        test_rows = load_subject_data(subject, filter_ok=filter_ok)
        dev_rows = load_dev_examples(subject) if n_shot > 0 else []
        if n_shot > 0 and not dev_rows and not dev_warned:
            print("  WARNING: MMLU dev data not found. Using 0-shot. "
                  "Run with --download to fetch dev examples.")
            dev_warned = True

        for row in test_rows:
            prompt = build_mmlu_prompt(row, dev_rows, subject, n_shot)
            all_questions.append({
                "subject": subject,
                "question": row["question"],
                "answer": row["answer"].strip().upper(),
                "prompt": prompt,
                "id": f"{subject}_{len(all_questions)}",
            })

    if limit and limit < len(all_questions):
        all_questions = all_questions[:limit]

    total_questions = len(all_questions)
    n_subjects = len(set(q["subject"] for q in all_questions))
    print(f"  Total questions: {total_questions} across {n_subjects} subjects")

    # Setup output directory
    if output_dir:
        out_dir = Path(output_dir)
    elif batch_dir:
        out_dir = batch_dir / make_run_dir_name(
            model_name, quant, think, sampling, limit, total_questions)
    else:
        out_dir = RESULTS_BASE / make_run_dir_name(
            model_name, quant, think, sampling, limit, total_questions)
    out_dir.mkdir(parents=True, exist_ok=True)

    answers_file = out_dir / "answers.jsonl"

    # Load existing answers if resuming
    existing_answers: dict[str, dict] = {}
    if resume_path:
        rp = Path(resume_path)
        if rp.exists():
            print(f"  Resuming from {rp}")
            with open(rp, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    existing_answers[entry["id"]] = entry
            print(f"    Loaded {len(existing_answers)} existing answers")
    elif answers_file.exists():
        print(f"  Auto-resuming from {answers_file}")
        with open(answers_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    existing_answers[entry["id"]] = entry
        print(f"    Loaded {len(existing_answers)} existing answers")

    remaining = [q for q in all_questions if q["id"] not in existing_answers]

    inference_time = None
    if remaining:
        print(f"  Running inference on {len(remaining)} questions...")
        env = build_env(quant)
        t0 = time.time()

        for i, q in enumerate(remaining):
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (len(remaining) - i) if i > 0 else 0

            print(f"    [{i+1}/{len(remaining)}] {q['subject']}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", end="", flush=True)

            try:
                response = run_inference(
                    model_path, q["prompt"], max_tokens, sampling, think, env)
                cleaned = strip_think_content(response, think)
                predicted = parse_mcq_answer(cleaned)
                print(f"  -> {predicted} (truth={q['answer']})"
                      f"{'  OK' if predicted == q['answer'] else '  WRONG'}")
            except Exception as e:
                print(f"  ERROR: {e}")
                response = ""
                predicted = ""

            entry = {
                "id": q["id"],
                "subject": q["subject"],
                "question": q["question"],
                "answer": q["answer"],
                "predicted": predicted,
                "response": response,
            }
            existing_answers[q["id"]] = entry

            with open(answers_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        inference_time = time.time() - t0
        print(f"  Inference complete: {inference_time:.1f}s total, "
              f"{inference_time/len(remaining):.1f}s per question")

    # Group results by subject
    results_by_subject: Dict[str, List[dict]] = {}
    for q in all_questions:
        entry = existing_answers.get(q["id"], {
            "predicted": "", "answer": q["answer"]
        })
        subject = q["subject"]
        if subject not in results_by_subject:
            results_by_subject[subject] = []
        results_by_subject[subject].append({
            "predicted": entry.get("predicted", ""),
            "answer": q["answer"],
        })

    # Evaluate
    print("  Evaluating...")
    metrics = evaluate_mmlu(results_by_subject)

    summary = format_summary(
        metrics, model_name, quant, think, sampling,
        max_tokens, n_shot, filter_ok, inference_time)
    print()
    print(summary)

    # Save results
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "model_path": model_path,
            "quant_preset": quant_idx,
            "quant": quant.display,
            "think": think,
            "n_shot": n_shot,
            "filter_ok": filter_ok,
            "sampling": sampling,
            "max_tokens": max_tokens,
            "metrics": metrics,
            "inference_time_s": inference_time,
        }, f, indent=2, ensure_ascii=False)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(f"  Results saved to: {out_dir}")

    return {
        "model_name": model_name,
        "quant_display": quant.display,
        "think": think,
        "temperature": sampling["temperature"],
        "total_questions": metrics["total_questions"],
        "macro_avg": metrics["macro_avg"],
        "micro_avg": metrics["micro_avg"],
        "empty_answers": metrics["empty_answers"],
        "inference_time_s": inference_time,
        "out_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# Summary table (markdown)
# ---------------------------------------------------------------------------
def build_summary_markdown(rows: List[dict], sampling: dict) -> str:
    lines = [
        "# MMLU-Redux Batch Summary",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Total runs: {len(rows)}",
        f"- Sampling: temperature={sampling['temperature']}, "
        f"top_p={sampling['top_p']}, top_k={sampling['top_k']}",
        "",
        "| Model | Quant | Think | #Q | Macro Avg | Micro Avg "
        "| Empty | Time (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        t = f"{r['inference_time_s']:.0f}" if r['inference_time_s'] is not None else "N/A"
        lines.append(
            f"| {r['model_name']} "
            f"| {r['quant_display']} "
            f"| {r['think']} "
            f"| {r['total_questions']} "
            f"| {r['macro_avg']:.4f} "
            f"| {r['micro_avg']:.4f} "
            f"| {r['empty_answers']} "
            f"| {t} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
EXAMPLES = """
Examples:

  # Download MMLU-Redux dataset and dev examples (first time only)
  python scripts/mmlu_redux.py --download

  # Single model, default settings
  python scripts/mmlu_redux.py --models 3

  # Multiple models
  python scripts/mmlu_redux.py --models 1~4

  # Quick smoke test (first 10 questions only)
  python scripts/mmlu_redux.py --models 3 --limit 10

  # Include all questions (not just error-filtered OK ones)
  python scripts/mmlu_redux.py --models 3 --no-filter-ok

  # 0-shot evaluation
  python scripts/mmlu_redux.py --models 3 --n-shot 0

  # Greedy decoding
  python scripts/mmlu_redux.py --models 3 --temperature 0

  # Compare thinking vs non-thinking
  python scripts/mmlu_redux.py --models 3 --think 0,1

  # Resume an interrupted run
  python scripts/mmlu_redux.py --models 3 --resume path\\to\\answers.jsonl

Model index mapping (--models):
  1. Qwen3.5-0.8B
  2. Qwen3.5-2B
  3. Qwen3.5-4B
  4. Qwen3.5-9B
  5. Qwen3.5-35B-A3B

Quantization presets (--quant-list):
  1: [int4_asym, 128, int4_asym]  - INT4 weights + INT4 backup (default)
  2: [int4_asym, 128, int8_asym]  - INT4 weights + INT8 backup
  3: [none, none, none]           - No quantization (FP16/BF16)
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MMLU-Redux benchmark for modeling_qwen3_5.exe "
                    "(57 subjects, ~3042 OK / ~5330 total English MCQ questions)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )

    # --- Model selection ---
    parser.add_argument(
        "--models", "--models-list", dest="models", default="3",
        help="Model index selectors (default: 3). Examples: 1,3,4 | 1~5 | 2,4~5")
    parser.add_argument(
        "--model-root", default=str(DEFAULT_MODEL_ROOT),
        help=f"Model root folder (default: {DEFAULT_MODEL_ROOT}).")

    # --- Quant selection ---
    parser.add_argument(
        "--quant-list", default="1",
        help="Quant preset selectors (default: 1). Examples: 1 | 2,3 | all | 1~3")

    # --- Think selection ---
    parser.add_argument(
        "--think", metavar="THINK_LIST", default="0",
        help="Think mode selectors (default: 0). Examples: 0 | 1 | 0,1 | all")

    # --- Sampling parameters ---
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0). 0 = greedy.")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus sampling threshold (default: 0.95).")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-K filtering (default: 20).")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty (default: 1.0).")
    parser.add_argument("--frequency-penalty", type=float, default=0.0,
                        help="Frequency penalty (default: 0.0).")
    parser.add_argument("--presence-penalty", type=float, default=0.0,
                        help="Presence penalty (default: 0.0). "
                             "Keep at 0 for MCQ to avoid encouraging verbose output.")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="Random seed (default: 0 = random).")

    # --- MMLU-Redux specific ---
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Maximum output tokens per question (default: 512). "
             "The model generates an explanation before stating the answer letter; "
             "512 tokens is enough for the explanation + final answer.")
    parser.add_argument(
        "--n-shot", type=int, default=5,
        help="Number of few-shot examples from MMLU dev set (default: 5). "
             "Falls back to 0 if dev data is unavailable.")
    parser.add_argument(
        "--filter-ok", dest="filter_ok", action="store_true", default=True,
        help="Only evaluate questions annotated as 'OK' (default: enabled).")
    parser.add_argument(
        "--no-filter-ok", dest="filter_ok", action="store_false",
        help="Evaluate all questions, including those with annotation errors.")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Custom output directory (single-run mode only).")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to an existing answers.jsonl to resume from.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N questions (across all subjects).")
    parser.add_argument(
        "--download", action="store_true",
        help="Download MMLU-Redux dataset and MMLU dev data, then exit.")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Handle --download
    if args.download:
        download_mmlu_redux_data()
        if "--models" not in sys.argv and "--resume" not in sys.argv:
            print("\nDataset downloaded. Run again without --download to start evaluation.")
            return 0

    # Ensure data exists
    if _find_data_dir() is None:
        print("MMLU-Redux data not found. Downloading...")
        download_mmlu_redux_data()

    # Build sampling dict
    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "rng_seed": args.rng_seed,
    }

    # Parse selectors
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

    missing = [str(p) for idx, p in model_paths.items() if not p.exists()]
    if missing and not args.resume:
        print("ERROR: The following model paths do not exist:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 2

    # Discover subjects
    subjects = discover_subjects()
    if not subjects:
        print("ERROR: No MMLU-Redux subjects found. Run with --download first.",
              file=sys.stderr)
        return 2
    print(f"MMLU-Redux dataset: {len(subjects)} subjects "
          f"(filter_ok={args.filter_ok})")

    # Count total combinations
    total_combos = (len(selected_model_indices) * len(selected_quant_indices)
                    * len(selected_think_values))
    print(f"\nPlanned runs: {len(selected_model_indices)} models x "
          f"{len(selected_quant_indices)} quants x "
          f"{len(selected_think_values)} think modes = {total_combos} total")
    print(f"  Models: {[MODEL_NAMES[i-1] for i in selected_model_indices]}")
    print(f"  Quants: {[QUANT_PRESETS[i].display for i in selected_quant_indices]}")
    print(f"  Think:  {selected_think_values}")
    print(f"  N-shot: {args.n_shot}")
    print()

    # Batch directory
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

                out_dir = args.output_dir if total_combos == 1 else None

                result = run_single_eval(
                    model_path=model_path,
                    model_name=model_name,
                    quant=quant,
                    quant_idx=quant_idx,
                    think=think,
                    sampling=sampling,
                    max_tokens=args.max_tokens,
                    n_shot=args.n_shot,
                    filter_ok=args.filter_ok,
                    subjects=subjects,
                    limit=args.limit,
                    resume_path=args.resume,
                    output_dir=out_dir,
                    batch_dir=batch_dir,
                )
                all_results.append(result)

    # Summary
    summary_md = build_summary_markdown(all_results, sampling)
    print(f"\n{'='*70}")
    print("All runs complete. Summary:")
    print(f"{'='*70}")
    print()
    print(summary_md)

    summary_file = batch_dir / "batch_summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Batch summary saved to: {summary_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
