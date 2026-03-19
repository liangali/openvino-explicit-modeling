"""C-Eval benchmark for modeling_qwen3_5.exe.

Runs the C-Eval benchmark (52 subjects, ~1346 val questions) using
modeling_qwen3_5.exe and evaluates by matching the model's answer letter
(A/B/C/D) against the ground truth.

C-Eval is a comprehensive Chinese evaluation suite covering four categories:
STEM (20 subjects), Social Science (10), Humanities (11), and Other (11).
Each question is a 4-choice MCQ in Chinese.  Evaluation uses 5-shot
prompting with examples drawn from the dev split (configurable via --n-shot).

C-Eval Hard is a subset of 8 challenging STEM subjects (advanced math,
discrete math, probability & statistics, college chemistry, college physics,
high-school math, high-school chemistry, high-school physics).

Supports batch testing across multiple model/quant/think combinations
(same CLI as ifeval.py).
"""
from __future__ import annotations

import argparse
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
import zipfile
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
RESULTS_BASE = WORKSPACE_DIR / "results_ceval"     # output outside repo

OV_BIN = WORKSPACE_DIR / "openvino" / "bin" / "intel64" / "Release"
TBB_BIN = WORKSPACE_DIR / "openvino" / "temp" / "Windows_AMD64" / "tbb" / "bin"
GENAI_DLL = WORKSPACE_DIR / "openvino.genai" / "build" / "openvino_genai"
GENAI_RUNTIME_BIN = WORKSPACE_DIR / "openvino.genai" / "build" / "bin"
GENAI_BIN = WORKSPACE_DIR / "openvino.genai" / "build" / "bin" / "Release"
EXE = GENAI_BIN / "modeling_qwen3_5.exe"

DEFAULT_MODEL_ROOT = Path(r"d:\data\models\Huggingface")

# C-Eval data directory (auto-downloaded)
DATA_DIR = SCRIPT_DIR / "ceval_data"

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
# C-Eval subject mapping: subject_key -> (chinese_name, category)
# ---------------------------------------------------------------------------
CEVAL_SUBJECT_MAPPING: Dict[str, tuple] = {
    # STEM (20)
    "advanced_mathematics": ("高等数学", "STEM"),
    "college_chemistry": ("大学化学", "STEM"),
    "college_physics": ("大学物理", "STEM"),
    "college_programming": ("大学编程", "STEM"),
    "computer_architecture": ("计算机组成", "STEM"),
    "computer_network": ("计算机网络", "STEM"),
    "discrete_mathematics": ("离散数学", "STEM"),
    "electrical_engineer": ("注册电气工程师", "STEM"),
    "high_school_biology": ("高中生物", "STEM"),
    "high_school_chemistry": ("高中化学", "STEM"),
    "high_school_mathematics": ("高中数学", "STEM"),
    "high_school_physics": ("高中物理", "STEM"),
    "metrology_engineer": ("注册计量师", "STEM"),
    "middle_school_biology": ("初中生物", "STEM"),
    "middle_school_chemistry": ("初中化学", "STEM"),
    "middle_school_mathematics": ("初中数学", "STEM"),
    "middle_school_physics": ("初中物理", "STEM"),
    "operating_system": ("操作系统", "STEM"),
    "probability_and_statistics": ("概率统计", "STEM"),
    "veterinary_medicine": ("兽医学", "STEM"),
    # Social Science (10)
    "business_administration": ("工商管理", "Social Science"),
    "college_economics": ("大学经济学", "Social Science"),
    "education_science": ("教育学", "Social Science"),
    "environmental_impact_assessment_engineer": ("环境影响评价工程师", "Social Science"),
    "fire_engineer": ("注册消防工程师", "Social Science"),
    "high_school_geography": ("高中地理", "Social Science"),
    "high_school_politics": ("高中政治", "Social Science"),
    "law": ("法学", "Social Science"),
    "logic": ("逻辑学", "Social Science"),
    "mao_zedong_thought": ("毛泽东思想和中国特色社会主义理论体系概论", "Social Science"),
    # Humanities (11)
    "art_studies": ("艺术学", "Humanities"),
    "chinese_language_and_literature": ("中国语言文学", "Humanities"),
    "college_chinese": ("大学语文", "Humanities"),
    "high_school_chinese": ("高中语文", "Humanities"),
    "high_school_history": ("高中历史", "Humanities"),
    "ideological_and_moral_cultivation": ("思想道德修养与法律基础", "Humanities"),
    "legal_professional": ("法律职业资格", "Humanities"),
    "modern_chinese_history": ("近代史纲要", "Humanities"),
    "professional_tour_guide": ("导游资格", "Humanities"),
    "teacher_qualification": ("教师资格", "Humanities"),
    "marxism": ("马克思主义基本原理", "Humanities"),
    # Social Science (additional subjects found in the dataset)
    "middle_school_geography": ("初中地理", "Social Science"),
    "middle_school_history": ("初中历史", "Social Science"),
    "middle_school_politics": ("初中政治", "Social Science"),
    # Other (9)
    "accountant": ("注册会计师", "Other"),
    "basic_medicine": ("基础医学", "Other"),
    "civil_servant": ("公务员", "Other"),
    "clinical_medicine": ("临床医学", "Other"),
    "physician": ("医师资格", "Other"),
    "plant_protection": ("植物保护", "Other"),
    "sports_science": ("体育学", "Other"),
    "tax_accountant": ("税务师", "Other"),
    "urban_and_rural_planner": ("城乡规划师", "Other"),
}

CEVAL_HARD_SUBJECTS = [
    "advanced_mathematics",
    "discrete_mathematics",
    "probability_and_statistics",
    "college_chemistry",
    "college_physics",
    "high_school_mathematics",
    "high_school_chemistry",
    "high_school_physics",
]

CEVAL_CATEGORIES = ["STEM", "Social Science", "Humanities", "Other"]

# ---------------------------------------------------------------------------
# Index selection parser (same as ifeval.py)
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
def _find_data_root() -> Optional[Path]:
    """Locate the directory containing C-Eval CSV files.

    The ModelScope zip extracts to ceval_data/data/ceval/ with all CSVs in a
    flat directory (accountant_val.csv, accountant_dev.csv, ...).  Other layouts
    may have val/ and dev/ sub-directories.  This function returns the first
    directory that contains *_val.csv files.
    """
    candidates = [
        DATA_DIR / "data" / "ceval",        # ModelScope zip layout
        DATA_DIR / "val",                    # val/dev subdirectory layout
        DATA_DIR / "data" / "val",
        DATA_DIR,                            # flat layout
    ]
    for c in candidates:
        if c.exists() and list(c.glob("*_val.csv")):
            return c
    return None


def _find_val_dir() -> Optional[Path]:
    """Locate the directory with *_val.csv files."""
    return _find_data_root()


def _find_dev_dir() -> Optional[Path]:
    """Locate the directory with *_dev.csv files."""
    # In the ModelScope layout, dev and val are in the same flat directory
    root = _find_data_root()
    if root and list(root.glob("*_dev.csv")):
        return root
    # Try dedicated dev/ subdirectories
    candidates = [
        DATA_DIR / "dev",
        DATA_DIR / "data" / "dev",
    ]
    for c in candidates:
        if c.exists() and list(c.glob("*_dev.csv")):
            return c
    return None


def download_ceval_data() -> None:
    """Download C-Eval dataset.

    Tries two methods:
    1. Download zip from ModelScope (works in China without VPN)
    2. git clone from HuggingFace
    """
    if _find_val_dir() is not None:
        val_dir = _find_val_dir()
        n = len(list(val_dir.glob("*_val.csv")))
        print(f"C-Eval data already exists at {val_dir} ({n} subjects)")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Method 1: Download zip from ModelScope
    zip_url = "https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/benchmark/data.zip"
    zip_path = DATA_DIR / "_ceval_download.zip"
    print(f"Downloading C-Eval dataset from ModelScope...")
    print(f"  URL: {zip_url}")

    try:
        urllib.request.urlretrieve(zip_url, str(zip_path))
        print(f"  Extracting to {DATA_DIR}...")
        with zipfile.ZipFile(str(zip_path)) as zf:
            zf.extractall(str(DATA_DIR))
        zip_path.unlink()
    except Exception as e:
        print(f"  ModelScope download failed: {e}")
        print("  Trying HuggingFace git clone...")
        if zip_path.exists():
            zip_path.unlink()
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://huggingface.co/datasets/ceval/ceval-exam",
                 str(DATA_DIR / "data")],
                check=True)
        except Exception as e2:
            print(f"  HuggingFace clone also failed: {e2}")
            print("\nPlease download C-Eval data manually:")
            print(f"  Option A: Download {zip_url}")
            print(f"            Extract to {DATA_DIR}/")
            print(f"  Option B: git clone https://huggingface.co/datasets/ceval/ceval-exam {DATA_DIR}/data")
            sys.exit(1)

    val_dir = _find_val_dir()
    if val_dir is None:
        print(f"ERROR: Downloaded data but cannot find val/ directory under {DATA_DIR}")
        sys.exit(1)

    n = len(list(val_dir.glob("*_val.csv")))
    print(f"  C-Eval data ready: {n} subjects found")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_csv_rows(csv_path: Path) -> List[dict]:
    """Load a C-Eval CSV file and return rows as dicts."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def discover_subjects() -> List[str]:
    """Discover available subjects from the val/ directory."""
    val_dir = _find_val_dir()
    if val_dir is None:
        return []
    subjects = []
    for csv_file in sorted(val_dir.glob("*_val.csv")):
        subject = csv_file.stem.replace("_val", "")
        subjects.append(subject)
    return subjects


def load_subject_data(subject: str, split: str = "val") -> List[dict]:
    """Load questions for a subject from the given split."""
    if split == "val":
        base_dir = _find_val_dir()
    elif split == "dev":
        base_dir = _find_dev_dir()
    else:
        raise ValueError(f"Unknown split: {split}")

    if base_dir is None:
        return []

    csv_path = base_dir / f"{subject}_{split}.csv"
    if not csv_path.exists():
        return []
    return load_csv_rows(csv_path)


def get_subject_info(subject: str) -> tuple:
    """Get (chinese_name, category) for a subject."""
    if subject in CEVAL_SUBJECT_MAPPING:
        return CEVAL_SUBJECT_MAPPING[subject]
    # Fallback: generate a reasonable Chinese name from the English key
    display = subject.replace("_", " ").title()
    return (display, "Other")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def format_mcq_question(question: str, choices: dict) -> str:
    """Format a single MCQ question with choices."""
    lines = [question]
    for letter in "ABCD":
        choice = choices.get(letter, "")
        lines.append(f"A. {choice}" if letter == "A" else
                     f"B. {choice}" if letter == "B" else
                     f"C. {choice}" if letter == "C" else
                     f"D. {choice}")
    return "\n".join(lines)


def build_ceval_prompt(question_row: dict, dev_rows: List[dict],
                       subject_zh: str, n_shot: int) -> str:
    """Build a C-Eval prompt with n-shot examples from dev set.

    Format:
        以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。

        [dev example 1]
        答案：A

        ...

        [val question]
        答案：
    """
    header = f"以下是中国关于{subject_zh}考试的单项选择题，请选出其中的正确答案。\n"

    parts = [header]

    # Few-shot examples from dev set
    examples = dev_rows[:n_shot]
    for ex in examples:
        q_text = format_mcq_question(
            ex["question"],
            {"A": ex["A"], "B": ex["B"], "C": ex["C"], "D": ex["D"]})
        parts.append(f"{q_text}\n答案：{ex['answer']}\n")

    # The actual question
    q_text = format_mcq_question(
        question_row["question"],
        {"A": question_row["A"], "B": question_row["B"],
         "C": question_row["C"], "D": question_row["D"]})
    parts.append(f"{q_text}\n答案：")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------
def parse_mcq_answer(response: str) -> str:
    """Extract the answer letter (A/B/C/D) from the model response.

    The model typically generates an explanation first, then states the answer
    at the end (e.g., "答案：D" or "Answer: B").  We search for explicit
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
        r'答案[是为]?[：:]\s*\**([A-D])\**',        # 答案：D / 答案是：**D** / 答案为：C
        r'正确答案[是为]\s*\**([A-D])\**',           # 正确答案是C
        r'答案[是为]\s*\**([A-D])\**',              # 答案是D / 答案为D
        r'(?:answer|Answer)\s*(?:is|:)\s*\**([A-D])\**',  # Answer: D
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
    """Build the environment dict matching run.bat."""
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
    """Build the modeling_qwen3_5.exe command line."""
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
    """Extract the model response from exe output."""
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
    """Remove <think>...</think> content from model response."""
    if think == 0:
        return response
    if "</think>" in response:
        idx = response.rfind("</think>")
        return response[idx + len("</think>"):].strip()
    return ""


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
def evaluate_ceval(results_by_subject: Dict[str, List[dict]]) -> dict:
    """Compute C-Eval metrics from results grouped by subject.

    Each result dict has keys: 'predicted', 'answer' (ground truth).

    Returns:
        {
            "macro_avg": float,        # average of per-subject accuracies
            "micro_avg": float,        # total_correct / total_questions
            "ceval_hard_avg": float,   # macro avg over hard subjects
            "per_category": { "STEM": float, ... },
            "per_subject": { subject: { "accuracy": float, "correct": int, "total": int } },
            "total_correct": int,
            "total_questions": int,
            "empty_answers": int,
        }
    """
    per_subject = {}
    category_accs = {cat: [] for cat in CEVAL_CATEGORIES}
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

        _, category = get_subject_info(subject)
        if category in category_accs:
            category_accs[category].append(acc)

    # Macro average (average of per-subject accuracies)
    all_accs = [v["accuracy"] for v in per_subject.values()]
    macro_avg = sum(all_accs) / max(len(all_accs), 1)

    # Micro average (total correct / total questions)
    micro_avg = total_correct / max(total_questions, 1)

    # Per-category macro average
    per_category = {}
    for cat in CEVAL_CATEGORIES:
        accs = category_accs[cat]
        per_category[cat] = sum(accs) / max(len(accs), 1) if accs else 0.0

    # C-Eval Hard
    hard_accs = [per_subject[s]["accuracy"] for s in CEVAL_HARD_SUBJECTS
                 if s in per_subject]
    ceval_hard_avg = sum(hard_accs) / max(len(hard_accs), 1) if hard_accs else 0.0

    return {
        "macro_avg": macro_avg,
        "micro_avg": micro_avg,
        "ceval_hard_avg": ceval_hard_avg,
        "per_category": per_category,
        "per_subject": per_subject,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "empty_answers": empty_answers,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def model_short_name(model_path: str) -> str:
    return Path(model_path).name


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
                   n_shot: int,
                   inference_time: float | None = None) -> str:
    """Format C-Eval results summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("C-Eval Results Summary")
    lines.append("=" * 70)
    lines.append(f"  Model:        {model_name}")
    lines.append(f"  Quant:        {quant.display}")
    lines.append(f"  Think:        {think}")
    lines.append(f"  N-shot:       {n_shot}")
    lines.append(f"  Temperature:  {sampling['temperature']}")
    lines.append(f"  Top-p:        {sampling['top_p']}")
    lines.append(f"  Top-k:        {sampling['top_k']}")
    lines.append(f"  Max tokens:   {max_tokens}")
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
    lines.append(f"  C-Eval Hard:      {metrics['ceval_hard_avg']:.4f}  "
                 f"(8 hard STEM subjects)")
    lines.append("")
    lines.append("  Per-category accuracy:")
    for cat in CEVAL_CATEGORIES:
        lines.append(f"    {cat:20s}  {metrics['per_category'].get(cat, 0):.4f}")
    lines.append("")

    # Top/bottom 5 subjects
    sorted_subjects = sorted(metrics["per_subject"].items(),
                            key=lambda x: x[1]["accuracy"], reverse=True)
    lines.append("  Top 5 subjects:")
    for subject, info in sorted_subjects[:5]:
        zh_name, cat = get_subject_info(subject)
        lines.append(f"    {subject:45s}  {info['accuracy']:.4f}  "
                     f"({info['correct']}/{info['total']})  [{cat}]")

    lines.append("  Bottom 5 subjects:")
    for subject, info in sorted_subjects[-5:]:
        zh_name, cat = get_subject_info(subject)
        lines.append(f"    {subject:45s}  {info['accuracy']:.4f}  "
                     f"({info['correct']}/{info['total']})  [{cat}]")

    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------
def run_single_eval(model_path: str, model_name: str, quant: QuantPreset,
                    quant_idx: int, think: int, sampling: dict,
                    max_tokens: int, n_shot: int,
                    subjects: List[str],
                    limit: Optional[int],
                    resume_path: Optional[str],
                    output_dir: Optional[str],
                    batch_dir: Optional[Path] = None) -> dict:
    """Run a single C-Eval evaluation and return results dict."""

    # Collect all questions across subjects
    all_questions = []  # list of (subject, question_row, prompt)
    for subject in subjects:
        val_rows = load_subject_data(subject, "val")
        dev_rows = load_subject_data(subject, "dev")
        subject_zh, _ = get_subject_info(subject)

        for row in val_rows:
            prompt = build_ceval_prompt(row, dev_rows, subject_zh, n_shot)
            all_questions.append({
                "subject": subject,
                "question": row["question"],
                "answer": row["answer"].strip().upper(),
                "prompt": prompt,
                "id": f"{subject}_{row.get('id', len(all_questions))}",
            })

    if limit and limit < len(all_questions):
        all_questions = all_questions[:limit]

    total_questions = len(all_questions)
    print(f"  Total questions: {total_questions} across {len(subjects)} subjects")

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
        # Auto-resume from the output directory
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

    # Group results by subject for evaluation
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
    metrics = evaluate_ceval(results_by_subject)

    summary = format_summary(
        metrics, model_name, quant, think, sampling,
        max_tokens, n_shot, inference_time)
    print()
    print(summary)

    # Save results.json
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "model_path": model_path,
            "quant_preset": quant_idx,
            "quant": quant.display,
            "think": think,
            "n_shot": n_shot,
            "sampling": sampling,
            "max_tokens": max_tokens,
            "metrics": metrics,
            "inference_time_s": inference_time,
        }, f, indent=2, ensure_ascii=False)

    # Save summary.txt
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
        "ceval_hard": metrics["ceval_hard_avg"],
        "stem": metrics["per_category"].get("STEM", 0),
        "social_science": metrics["per_category"].get("Social Science", 0),
        "humanities": metrics["per_category"].get("Humanities", 0),
        "other": metrics["per_category"].get("Other", 0),
        "empty_answers": metrics["empty_answers"],
        "inference_time_s": inference_time,
        "out_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# Summary table (markdown)
# ---------------------------------------------------------------------------
def build_summary_markdown(rows: List[dict], sampling: dict) -> str:
    lines = [
        "# C-Eval Batch Summary",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Total runs: {len(rows)}",
        f"- Sampling: temperature={sampling['temperature']}, "
        f"top_p={sampling['top_p']}, top_k={sampling['top_k']}",
        "",
        "| Model | Quant | Think | #Q | Macro Avg | Micro Avg | C-Eval Hard "
        "| STEM | Social Sci | Humanities | Other | Empty | Time (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            f"| {r['ceval_hard']:.4f} "
            f"| {r['stem']:.4f} "
            f"| {r['social_science']:.4f} "
            f"| {r['humanities']:.4f} "
            f"| {r['other']:.4f} "
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

  # Download C-Eval dataset (first time only)
  python scripts/ceval.py --download

  # Single model, default settings
  python scripts/ceval.py --models 3

  # Multiple models
  python scripts/ceval.py --models 1~4

  # Quick smoke test (first 10 questions only)
  python scripts/ceval.py --models 3 --limit 10

  # Compare thinking vs non-thinking
  python scripts/ceval.py --models 3 --think 0,1

  # Greedy decoding
  python scripts/ceval.py --models 3 --temperature 0

  # 0-shot evaluation (no few-shot examples)
  python scripts/ceval.py --models 3 --n-shot 0

  # Resume an interrupted run
  python scripts/ceval.py --models 3 --resume path\\to\\answers.jsonl

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
        description="C-Eval benchmark for modeling_qwen3_5.exe "
                    "(52 subjects, ~1346 Chinese MCQ questions)",
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

    # --- C-Eval specific ---
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Maximum output tokens per question (default: 512). "
             "The model generates an explanation before stating the answer letter; "
             "512 tokens is enough for the explanation + final answer.")
    parser.add_argument(
        "--n-shot", type=int, default=5,
        help="Number of few-shot examples from dev set (default: 5). "
             "Standard C-Eval uses 5-shot. Set to 0 for zero-shot.")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Custom output directory (single-run mode only).")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to an existing answers.jsonl to resume from.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N questions (across all subjects). "
             "Useful for quick smoke tests.")
    parser.add_argument(
        "--download", action="store_true",
        help="Download C-Eval dataset and exit. "
             "Data is also auto-downloaded on first run.")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Handle --download
    if args.download:
        download_ceval_data()
        if "--models" not in sys.argv and "--resume" not in sys.argv:
            print("\nDataset downloaded. Run again without --download to start evaluation.")
            return 0

    # Ensure data exists (auto-download if needed)
    if _find_val_dir() is None:
        print("C-Eval data not found. Downloading...")
        download_ceval_data()

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
        print("ERROR: No C-Eval subjects found. Run with --download first.", file=sys.stderr)
        return 2
    print(f"C-Eval dataset: {len(subjects)} subjects")

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
