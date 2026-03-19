# OpenVINO Explicit Modeling

Windows-focused PoC workspace for the OpenVINO Explicit Modeling / Modeling API.

This repository provides:

- `repo.bat` to prepare the sibling-repo workspace
- `build.bat` to build both `openvino` and `openvino.genai`
- `run.bat` and `scripts/auto_tests.py` for smoke and functional checks
- benchmark scripts under `scripts/` for LLM accuracy evaluation

## 1. Prerequisites

Install the following on Windows:

```text
git 2.x
cmake 3.x
Visual Studio 2022
uv 0.10+ (required for wheel builds and managed Python provisioning)
python 3.10+
```

## 2. Set Up the Workspace

Clone this repository first, then let `repo.bat` create the sibling checkout
layout for `openvino` and `openvino.genai`.

```bat
cd my_workspace
git clone https://github.com/liangali/openvino-explicit-modeling
cd openvino-explicit-modeling
repo.bat
```

`repo.bat` will:

- clone `openvino`
- clone `openvino.genai`
- check out the `explicit-modeling` branch
- initialize submodules
- apply the oneDNN patch helper in `scripts\apply_onednn_patch.bat`

Expected workspace layout:

```text
my_workspace\
|-- openvino\
|-- openvino.genai\
`-- openvino-explicit-modeling\
```

## 3. Build

Build `openvino` and `openvino.genai` from the workspace:

```bat
cd my_workspace\openvino-explicit-modeling
build.bat
```

Optional: collect built DLL/EXE artifacts into a single package directory:

```bat
cd my_workspace\openvino-explicit-modeling\scripts
python package.py
```

By default, `package.py` writes files under:

```text
my_workspace\package\Release\
```

## 4. Create the Python Environment

The same virtual environment is used for tokenizer conversion, helper scripts,
and benchmark scripts.

```bat
cd my_workspace
python -m venv .venv
.venv\Scripts\activate
pip install openvino-tokenizers
pip install transformers
```

## 5. Download Models

For a minimal smoke-test setup, place Hugging Face models under
`D:\data\models\Huggingface`.

```bat
cd D:\data\models\Huggingface
git clone https://huggingface.co/Qwen/Qwen3-4B
git clone https://huggingface.co/Qwen/Qwen3.5-35B-A3B
```

## 6. Run Samples and Functional Checks

Activate `.venv` before running the commands below.

### Option A: Run executables directly

This is the most explicit way to compare the Modeling API executable with the
existing GenAI samples.

Replace `MY_WORKSPACE` with your actual workspace path.

```bat
.venv\Scripts\activate

set "MY_WORKSPACE=D:\data\code\Openvino_new_arch_2026"
set OV_GENAI_SAVE_OV_MODEL=1
set OV_GENAI_USE_MODELING_API=1
set OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym
set OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128
set OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=int4_asym
set "PATH=%MY_WORKSPACE%\openvino\temp\Windows_AMD64\tbb\bin;%MY_WORKSPACE%\openvino\bin\intel64\Release;%MY_WORKSPACE%\openvino.genai\build\openvino_genai;%PATH%"

cd %MY_WORKSPACE%\openvino.genai\build\bin

modeling_qwen3_5.exe --model D:\data\models\Huggingface\Qwen3.5-35B-A3B --cache-model --mode text --prompt "write opencl gemm kernel and host code" --output-tokens 300
greedy_causal_lm.exe D:\data\models\Huggingface\Qwen3.5-35B-A3B "write opencl gemm kernel and host code" GPU 1 3 300 int4_asym 128 int4_asym
benchmark_genai.exe -m D:\data\models\Huggingface\Qwen3.5-35B-A3B -p "write opencl gemm kernel and host code" -n 1 --mt 300 -d GPU
```

### Option B: Use `run.bat`

`run.bat` validates the expected build output, sets the same environment
variables, and prints ready-to-use sample commands.

```bat
.venv\Scripts\activate
cd my_workspace\openvino-explicit-modeling
run.bat
```

### Option C: Use `auto_tests.py`

`auto_tests.py` runs functional checks and generates a markdown report under:

```text
my_workspace\reports\
```

Examples:

```bat
.venv\Scripts\activate
cd my_workspace

python openvino-explicit-modeling\scripts\auto_tests.py --help
python openvino-explicit-modeling\scripts\auto_tests.py --list
python openvino-explicit-modeling\scripts\auto_tests.py --models-root D:\data\models
python openvino-explicit-modeling\scripts\auto_tests.py --models-root D:\data\models --tests 0,1,2
python openvino-explicit-modeling\scripts\auto_tests.py --models-root D:\data\models --tests 1~5
python openvino-explicit-modeling\scripts\auto_tests.py --models-root D:\data\models --tests 1~5,7,8~10
python openvino-explicit-modeling\scripts\auto_tests.py --models-root D:\data\models --tests all
```

Report contents:

- markdown summary
- TTFT, throughput, duration, and related metrics
- failed cases in `stderr`

Full script documentation is in **[scripts/README.md](scripts/README.md)**.

## 7. Build Python Wheels and Run with Python (Optional)

Build wheels for `openvino`, `openvino_tokenizers`, and `openvino.genai`:

```bat
cd my_workspace\openvino-explicit-modeling
build.bat --wheel --python=3.11.9
```

If `--python` is omitted, `build.bat --wheel` uses the first `python.exe` found
in `PATH`.

Wheel build virtual environments are reused by exact Python version under:

```text
my_workspace\openvino-explicit-modeling\.wheel-build-venv\<python-version>\
```

The wheel files and `wheel.py` are written to versioned output folders such as:

```text
my_workspace\wheel\cp311\
```

Install from the local wheel directory:

```bat
cd my_workspace
.venv\Scripts\activate
pip install --no-index --find-links wheel\cp311 openvino_genai
python wheel\cp311\wheel.py --help
```

Before running `wheel.py`, first generate cached OpenVINO IR files by running
one of the executable samples with `--cache-model`.

Example:

```bat
python wheel\cp311\wheel.py --model D:\data\models\Huggingface\Qwen3.5-35B-A3B\qwen3_5_text_q4a_b4a_g128.xml --prompt "what's ffmpeg?" --device GPU --max-new-tokens 300
```

## 8. LLM Accuracy Benchmarks

Three scripts are provided for model accuracy evaluation:

- `scripts\ifeval.py`
- `scripts\ceval.py`
- `scripts\mmlu_redux.py`

For benchmark setup, dataset download, CLI reference, output structure, and
expected run times, see **[scripts/README.md](scripts/README.md)**.
