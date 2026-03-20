# Qwen3.5 Benchmark Scripts

One-click scripts to run Qwen3.5 text/VL models with OpenVINO LLMPipeline/VLMPipeline, plus IFEval and MMMU benchmarks.

## Requirements

- Python 3.12 (openvino .pyd is built for cp312)
- OpenVINO and openvino-genai built (see main repo)
- `pip install pillow` (for VLM and MMMU)
- `pip install pyyaml datasets` (for MMMU only)

## Text Model (LLMPipeline)

**Script:** `scripts/run_text_qwen3_5.bat` and `scripts/run_text_qwen3_5.py`

Runs Qwen3.5 text models (e.g. Qwen3.5-35B-A3B, Qwen3.5-MoE) via `openvino_genai.LLMPipeline`.

### Usage

```batch
cd openvino-explicit-modeling\scripts
run_text_qwen3_5.bat
run_text_qwen3_5.bat "D:\Data\models\Huggingface\Qwen3.5-35B-A3B"
run_text_qwen3_5.bat "D:\Data\models\Qwen3.5-MoE-57B-A14B" --device GPU --prompt "Hello"
run_text_qwen3_5.bat --device GPU --max-new-tokens 512
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `model_dir` | from `MODEL` env or `Qwen3.5-35B-A3B` | Path to Qwen3.5 HF model |
| `--device` | CPU | Device: CPU or GPU |
| `--prompt` | "Hello, please introduce yourself." | Input prompt |
| `--max-new-tokens` | 2048 | Max tokens to generate |
| `--inflight-quant-mode` | int4_asym | int4_sym / int4_asym / int8_sym / int8_asym / none |

---

## VLM Model (VLMPipeline)

**Script:** `scripts/run_vlm_qwen3_5.bat` and `scripts/run_vlm_qwen3_5.py`

Runs Qwen3.5 VL models (e.g. Qwen3-VL-30B-A3B-Instruct, Qwen3.5-VL) via `openvino_genai.VLMPipeline`.

### Usage

```batch
cd openvino-explicit-modeling\scripts
run_vlm_qwen3_5.bat
run_vlm_qwen3_5.bat "D:\Data\models\Qwen3-VL-30B-A3B-Instruct" "test.jpg"
run_vlm_qwen3_5.bat "D:\Data\models\Qwen3.5-VL-7B" --test-image --prompt "What is in this image?"
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `model_dir` | from `QWEN3_VL_MODEL` env | Path to Qwen3.5 VL model |
| `image_path` | scripts/test.jpg | Image file or directory |
| `--test-image` | - | Use synthetic 336x336 test image |
| `--device` | GPU | Device: CPU or GPU |
| `--prompt` | "Describe the content of this image." | Input prompt |
| `--max-new-tokens` | 256 | Max tokens |
| `--no-stream` | - | Disable streaming output |

---

## IFEval Benchmark (LLM Instruction-Following)

**Script:** `IFEVAL/run_ifeval.bat` and `IFEVAL/run_ifeval_openvino.py`

Runs IFEval (instruction-following evaluation) using LLMPipeline, then calls `evaluation_main.py`.

- **Input:** `IFEVAL/data/input_data.jsonl`
- **Output:** `IFEVAL/data/{model_name}/ifeval/output.jsonl`

### Usage

```batch
cd openvino-explicit-modeling\IFEVAL
run_ifeval.bat
run_ifeval.bat "D:\Data\models\Qwen3.5-35B-A3B" --device GPU
run_ifeval.bat --limit 2
run_ifeval.bat --skip-eval
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `model_dir` | from `MODEL` env | Qwen3.5 HF model path |
| `--device` | GPU | CPU or GPU |
| `--limit` | 0 (all) | Limit prompts (e.g. 2 for quick test) |
| `--skip-eval` | - | Skip running evaluation after generation |
| `--input-data` | IFEVAL/data/input_data.jsonl | Input jsonl path |
| `--output-base` | IFEVAL/data | Output base dir |

---

## MMMU Benchmark (VLM Multimodal)

**Script:** `MMMU/run_mmmu.bat` and `MMMU/run_mmmu_openvino.py`

Runs MMMU Pro using VLMPipeline, then calls `evaluate.py`.

- **Dataset:** `MMMU/MMMU_Pro_dataset/vision` or `standard (10 options)`
- **Output:** `MMMU/mmmu-pro/output/{model}_{setting}_{mode}.jsonl`

### Usage

```batch
cd openvino-explicit-modeling\MMMU
run_mmmu.bat
run_mmmu.bat "D:\Data\models\Qwen3-35B-A3B" --setting vision --mode direct
run_mmmu.bat --limit 2
run_mmmu.bat --skip-eval
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `model_dir` | from `QWEN3_VL_MODEL` env | Qwen3.5 VL model path |
| `--mode` | direct | direct or cot |
| `--setting` | vision | vision \| standard (10 options) |
| `--device` | GPU | CPU or GPU |
| `--limit` | 0 (all) | Limit samples for testing |
| `--skip-eval` | - | Skip evaluate.py after inference |

### Dataset

If `MMMU_Pro_dataset/vision` has no parquet files, download from HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("MMMU/MMMU_Pro", "vision", split="test")
# Save to MMMU/MMMU_Pro_dataset/vision/ if needed
```

---

## Environment

All scripts set:

- `OV_GENAI_USE_MODELING_API=1`
- `OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym` (for large models)
- `PYTHONPATH` and `PATH` for openvino and openvino.genai

You can override quantization before running, e.g.:

```batch
set OV_GENAI_INFLIGHT_QUANT_MODE=
run_text_qwen3_5.bat
```
