import re
import ast
import os
import json
import glob
import subprocess
import tempfile
from PIL import Image
import sys
import yaml
from datasets import load_dataset

# CLI args: python infer_qwen35.py [MODEL_DIR] [MODE] [SETTING]
#   MODEL_DIR : Qwen3.5 HF model dir (config.json + *.safetensors)
#   MODE      : direct | cot
#   SETTING   : vision | standard (10 options)
if len(sys.argv) == 4:
    MODEL_DIR = sys.argv[1]
    MODE      = sys.argv[2]
    SETTING   = sys.argv[3]
else:
    print("Usage: python infer_qwen35.py [MODEL_DIR] [MODE] [SETTING]")
    print("  e.g.: python infer_qwen35.py C:\\models\\Qwen3.5-VL-7B direct vision")
    MODEL_DIR = r"C:\models\Qwen3.5-VL-7B"   # change to your model path
    MODE      = "direct"
    SETTING   = "vision"

# Config: modeling_qwen3_5.exe path (build output under build\bin\Release\)
EXE_PATH = (
    r"D:\openvino-modeling-api\openvino.genai\build\bin\Release\modeling_qwen3_5.exe"
)

# DLL dirs for exe. Exit 0xC0000135 = STATUS_DLL_NOT_FOUND; add these to PATH before subprocess:
OPENVINO_GENAI_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(EXE_PATH))))
_OPENVINO_ROOT = os.path.join(os.path.dirname(OPENVINO_GENAI_ROOT), "openvino")
EXE_DLL_PATHS = [
    os.path.join(OPENVINO_GENAI_ROOT, "build", "openvino_genai"),
    os.path.join(_OPENVINO_ROOT, "bin", "intel64", "Release"),
    os.path.join(_OPENVINO_ROOT, "runtime", "bin", "intel64", "Release"),
    # oneTBB (tbb12.dll from openvino build)
    os.path.join(_OPENVINO_ROOT, "temp", "Windows_AMD64", "tbb", "bin"),
    os.path.dirname(EXE_PATH),
]


def _build_exe_env():
    """Build subprocess env with DLL paths prepended to PATH."""
    env = os.environ.copy()
    extra = os.pathsep.join(p for p in EXE_DLL_PATHS if os.path.isdir(p))
    if extra:
        env["PATH"] = extra + os.pathsep + env.get("PATH", "")
    return env

# OpenVINO device (GPU / CPU / AUTO)
DEVICE = "GPU"

# Max tokens per question
MAX_NEW_TOKENS = 2048


# Use --cache-model: cache compiled IR in model dir on first run, reuse later (faster startup)
USE_CACHE_MODEL = True

# Subprocess timeout (sec). Single-image infer typically <2 min.
SUBPROCESS_TIMEOUT = 600

# Output filename uses model dir basename
MODEL_NAME = os.path.basename(MODEL_DIR.rstrip(r"\/"))

# Local dataset path (MMMU_Pro_dataset in repo root)
DATASET_PATH = r"D:\MMMU\MMMU_Pro_dataset"

# Load prompts.yaml
with open("prompts.yaml", "r") as _f:
    prompt_config = yaml.safe_load(_f)[MODE]

# Prompt building (same as infer_gpt.py)
def replace_images_tokens(input_string):
    image_order = [int(n) for n in re.findall(r'<image\s+(\d+)>', input_string)]
    input_string = re.sub(r'<image\s+\d+>', '<image>', input_string)
    return input_string, image_order


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    return "\n".join(f"{l}. {o}" for l, o in zip(option_letters, options))


def construct_prompt(doc):
    question = doc["question"]
    parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    return f"{question}\n{parsed_options}\n{prompt_config['standard']}"


def mmmu_doc_to_text(doc):
    return replace_images_tokens(construct_prompt(doc))


def origin_mmmu_doc_to_visual(doc, image_order):
    return [doc[f"image_{idx}"] for idx in image_order]


def vision_mmmu_doc_to_visual(doc):
    return [doc["image"]]

# Exe invocation helpers
def save_pil_to_tempfile(pil_image: Image.Image) -> str:
    """Save PIL image to temp PNG file; caller must delete."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pil_image.save(tmp.name, format="PNG")
    tmp.close()
    return tmp.name


# Regex to find where generated text starts in stdout (after metrics lines)
_METRICS_RE = re.compile(
    r"^(\[|Mode:|Dummy text|Prompt token|Output token|TTFT:|Decode time:|TPOT:|Throughput:)"
)


def parse_exe_output(stdout: str) -> str:
    """Extract generated text from exe stdout, strip metrics lines."""
    lines = stdout.strip().split("\n")
    last_metrics_idx = -1
    for i, line in enumerate(lines):
        if _METRICS_RE.match(line.strip()):
            last_metrics_idx = i
    if 0 <= last_metrics_idx < len(lines) - 1:
        return "\n".join(lines[last_metrics_idx + 1:]).strip()
    # If no metrics found, return full output (may be error)
    return stdout.strip()


def infer_with_exe(prompt: str, image: Image.Image):
    """
    Call modeling_qwen3_5.exe for single-image inference.
    Returns generated text string, or {"error": "..."} on failure.
    """
    tmp_img = save_pil_to_tempfile(image)
    try:
        cmd = [
            EXE_PATH,
            "--model",         MODEL_DIR,
            "--mode",          "vl",
            "--image",         tmp_img,
            "--prompt",        prompt,
            "--device",        DEVICE,
            "--output-tokens", str(MAX_NEW_TOKENS),
        ]
        if USE_CACHE_MODEL:
            cmd.append("--cache-model")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
            env=_build_exe_env(),
        )
        # Print full exe output (metrics + generated text)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            err = result.stderr.strip() or f"exit code {result.returncode}"
            return {"error": err}
        return parse_exe_output(result.stdout)

    except subprocess.TimeoutExpired:
        return {"error": "subprocess timeout"}
    except FileNotFoundError:
        return {"error": f"exe not found: {EXE_PATH}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.unlink(tmp_img)
        except OSError:
            pass

# Per-sample processing
def _clean_image_tokens(prompt: str) -> str:
    """Remove <image> placeholders (exe does not understand them)."""
    return re.sub(r"<image>", "", prompt).strip()


def process_prompt(data):
    """
    Build prompt and pick image per SETTING, call exe, return (response, data).
    """
    if "standard (10 options)" in SETTING:
        prompt, image_order = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data, image_order)

        if not images:
            return {"error": "no images found in data"}, data

        # Exe supports single image only; take first, clear <image> placeholders
        image = images[0]
        prompt = _clean_image_tokens(prompt)

    elif SETTING == "vision":
        # Vision: image contains question, prompt only describes answer format
        prompt = prompt_config["vision"]
        images = vision_mmmu_doc_to_visual(data)
        image = images[0] if images else None
        if image is None:
            return {"error": "no image field in data"}, data

    else:
        return {"error": f"unsupported SETTING: {SETTING}"}, data

    response = infer_with_exe(prompt, image)
    return response, data

# Run and save
def run_and_save():
    def save_results_to_file(results, output_path):
        with open(output_path, "w", encoding="utf-8") as outfile:
            for output, data in results:
                data["response"] = output
                # Drop PIL Image fields so json.dump works
                data = {
                    k: v for k, v in data.items()
                    if not (k.startswith("image_") or k == "image")
                }
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write("\n")

    # Load parquet directly (avoids dataset_infos.json incompatibility with newer datasets)
    parquet_dir = os.path.join(DATASET_PATH, SETTING)
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "test-*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")
    dataset = load_dataset("parquet", data_files={"test": parquet_files}, split="test")
    dataset = dataset[0:10]
    def process_and_save_part(part_data, part_name):
        print(f"Begin processing {part_name}")
        results = []
        output_path = f"./output/{MODEL_NAME}_{part_name}_{MODE}.jsonl"

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    row = json.loads(line)
                    results.append((row["response"], row))
            print(f"Loaded {len(results)} existing results for {part_name}, skipping re-inference.")
        else:
            total = len(part_data)
            for i, data in enumerate(part_data):
                print(f"[{i + 1}/{total}] Processing id={data.get('id', i)}...")
                result, data = process_prompt(data)
                results.append((result, data))
                if (i + 1) % 10 == 0 or i + 1 == total:
                    save_results_to_file(results, output_path)
                    print(f"  -> Checkpoint: saved {i + 1}/{total}")
            print(f"Saved {len(results)} results to {output_path}")

        return output_path

    process_and_save_part(dataset, SETTING)


def main():
    run_and_save()


if __name__ == "__main__":
    main()
