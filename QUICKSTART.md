# OpenVINO Modeling API Quick Start

This document explains how to clone, build, and run the OpenVINO Modeling API project.

---

## 1. Clone The Repositories

Clone `openvino`, `openvino.genai`, and `openvino-explicit-modeling` into the same parent directory:

```
openvino-modeling-api/
|-- openvino/
|-- openvino.genai/
`-- openvino-explicit-modeling/
```

---

## 2. Pull Submodules

### openvino

```powershell
cd openvino
git submodule update --init --recursive
git lfs install
git lfs fetch --all
git lfs checkout
```

### openvino.genai

```powershell
cd ..\openvino.genai
git submodule update --init --recursive
git lfs install
git lfs fetch --all
git lfs checkout
```

---

## 3. Build

```powershell
cd ..\openvino-explicit-modeling
build.bat
```

`build.bat` builds both `openvino` and `openvino.genai`. It also creates the `build` directories if needed.

---

## 4. Run Tests

Use `auto_tests.py` for automated testing. Run it from the project root, which contains both `openvino` and `openvino.genai`.

### Basic Usage

```powershell
# Run all tests from the openvino-modeling-api root
cd d:\openvino-modeling-api
python openvino-explicit-modeling\scripts\auto_tests.py

# Specify the project root
python openvino-explicit-modeling\scripts\auto_tests.py --root .

# Specify the model root directory (default: D:\data\models)
python openvino-explicit-modeling\scripts\auto_tests.py --models-root D:\data\models
```

### List Available Tests

```powershell
python openvino-explicit-modeling\scripts\auto_tests.py --list
```

Example output:
```
Models root: D:\data\models
Available tests:
[0] Modeling API Unit Tests -> N/A (ULT) (exe: openvino.genai\build\...\test_modeling_api.exe)
[1] Huggingface Qwen3-0.6B -> Huggingface\Qwen3-0.6B (exe: ...)
...
```

### Select Tests To Run

```powershell
# Run specific test indices: 0, 1, 2
python openvino-explicit-modeling\scripts\auto_tests.py --tests 0 1 2

# Or use comma-separated input
python openvino-explicit-modeling\scripts\auto_tests.py --tests 0,1,2

# Run an index range (1~5 means 1,2,3,4,5)
python openvino-explicit-modeling\scripts\auto_tests.py --tests 1~5

# Combine ranges and single indices
python openvino-explicit-modeling\scripts\auto_tests.py --tests 1~5,7,8~10

# Run all tests
python openvino-explicit-modeling\scripts\auto_tests.py --tests all
```

### Combined Examples

```powershell
# Set the root and model path, then run tests 0 and 1 only
python openvino-explicit-modeling\scripts\auto_tests.py --root . --models-root D:\data\models --tests 0 1

# Run from the openvino-explicit-modeling directory and use the parent as root
cd openvino-explicit-modeling
python scripts\auto_tests.py --root ..
```

---

## 5. Test Output

- A Markdown report is generated in the current directory after the run.
- The report includes TTFT, throughput, duration, and related metrics.
- Failed cases are listed in `stderr`.

---

## Argument Reference

| Argument | Description | Default |
|------|------|--------|
| `--root` | Project root containing `openvino` and `openvino.genai` | Parent of the script directory |
| `--models-root` | Model file root directory | `D:\data\models` |
| `--list` | List available tests and exit | - |
| `--tests` | Test indices to run, supports `0,1,2`, `1~5`, and `all` | Run all |
