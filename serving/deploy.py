"""Deploy Qwen3.5-4B OpenAI-compatible server as a self-contained package.

Creates a deployment directory with:
  - Model IR files (int4_asym quantized)
  - Tokenizer files
  - Server code
  - Setup script

Usage:
  python deploy.py --model C:\data\models\Huggingface\Qwen3.5-4B --output D:\deploy\qwen3_5_4b
  python deploy.py --model C:\data\models\Huggingface\Qwen3.5-4B --output D:\deploy\qwen3_5_4b --include-vl
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

SERVING_DIR = Path(__file__).parent

# Files needed from model directory
MODEL_FILES_TEXT = [
    # IR (text-only, int4_asym gs128) — will be renamed to openvino_model.xml/bin
    ("qwen3_5_text_q4a_b4a_g128.xml", "openvino_model.xml"),
    ("qwen3_5_text_q4a_b4a_g128.bin", "openvino_model.bin"),
    # Tokenizer
    ("openvino_tokenizer.xml", None),
    ("openvino_tokenizer.bin", None),
    ("openvino_detokenizer.xml", None),
    ("openvino_detokenizer.bin", None),
    # Config
    ("config.json", None),
    ("tokenizer_config.json", None),
    ("tokenizer.json", None),
    ("chat_template.jinja", None),
    ("preprocessor_config.json", None),
    ("vocab.json", None),
    ("merges.txt", None),
]

MODEL_FILES_VL = [
    ("qwen3_5_text_vl_q4a_b4a_g128.xml", None),
    ("qwen3_5_text_vl_q4a_b4a_g128.bin", None),
    ("qwen3_5_vision.xml", None),
    ("qwen3_5_vision.bin", None),
    ("video_preprocessor_config.json", None),
]

SERVING_FILES = [
    "config.py",
    "schemas.py",
    "tool_parser.py",
    "engine.py",
    "vl_backend.py",
    "server.py",
    "requirements.txt",
    "README.md",
]


def main():
    parser = argparse.ArgumentParser(description="Deploy Qwen3.5 server package")
    parser.add_argument("--model", required=True, help="Source HF model directory with cached IR")
    parser.add_argument("--output", required=True, help="Output deployment directory")
    parser.add_argument("--include-vl", action="store_true", help="Include VL (vision) IR files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied")
    args = parser.parse_args()

    model_dir = Path(args.model)
    output_dir = Path(args.output)
    model_out = output_dir / "model"
    server_out = output_dir / "serving"

    # Validate source IR exists
    ir_xml = model_dir / "qwen3_5_text_q4a_b4a_g128.xml"
    if not ir_xml.exists():
        print(f"ERROR: IR not found: {ir_xml}")
        print("Run this first to generate IR:")
        print(f"  modeling_qwen3_5.exe --model {model_dir} --mode text --prompt hi --output-tokens 2 --cache-model")
        sys.exit(1)

    # Collect files to copy
    files = []
    for entry in MODEL_FILES_TEXT:
        src_name, dst_name = entry
        dst_name = dst_name or src_name
        src = model_dir / src_name
        if src.exists():
            files.append((src, model_out / dst_name))
        else:
            print(f"  WARN: missing {src_name}")

    if args.include_vl:
        for entry in MODEL_FILES_VL:
            src_name, dst_name = entry
            dst_name = dst_name or src_name
            src = model_dir / src_name
            if src.exists():
                files.append((src, model_out / dst_name))
            else:
                print(f"  WARN: missing VL file {src_name}")

    for f in SERVING_FILES:
        files.append((SERVING_DIR / f, server_out / f))

    # Calculate total size
    total_bytes = sum(s.stat().st_size for s, _ in files if s.exists())
    total_gb = total_bytes / (1024 ** 3)

    print(f"Deployment package: {output_dir}")
    print(f"  Model files: {len([f for _, d in files if 'model' in str(d)])} files")
    print(f"  Server files: {len([f for _, d in files if 'serving' in str(d)])} files")
    print(f"  Total size: {total_gb:.1f} GB")
    print(f"  VL included: {args.include_vl}")

    if args.dry_run:
        print("\nFiles:")
        for src, dst in files:
            size_mb = src.stat().st_size / (1024 ** 2) if src.exists() else 0
            print(f"  {src.name:45s} → {dst.relative_to(output_dir)}  ({size_mb:.1f} MB)")
        return

    # Copy files
    print("\nCopying...")
    model_out.mkdir(parents=True, exist_ok=True)
    server_out.mkdir(parents=True, exist_ok=True)

    for src, dst in files:
        if not src.exists():
            continue
        size_mb = src.stat().st_size / (1024 ** 2)
        print(f"  {src.name:45s} ({size_mb:.1f} MB)")
        shutil.copy2(src, dst)

    # Create launch script
    launch_script = output_dir / "start.bat"
    launch_script.write_text(f"""@echo off
REM Qwen3.5-4B OpenAI API Server — Deployment Launcher
REM
REM Prerequisites:
REM   1. Python 3.12+ with venv
REM   2. Run: python -m venv venv && venv\\Scripts\\activate && pip install -r serving\\requirements.txt
REM   3. Install openvino_genai Python bindings (from explicit-modeling build)
REM
REM The model loads from pre-quantized IR (int4_asym) — no OV_GENAI_USE_MODELING_API needed.

cd /d %~dp0
call venv\\Scripts\\activate.bat 2>nul

cd serving
python server.py --model "%~dp0model" --device GPU --port 8000 --model-name "Qwen3.5-4B" --quant none %*
""", encoding="utf-8")

    # Create setup script
    setup_script = output_dir / "setup.bat"
    setup_script.write_text(f"""@echo off
REM One-time setup for deployment environment
cd /d %~dp0

echo Creating Python venv...
python -m venv venv
call venv\\Scripts\\activate.bat

echo Installing dependencies...
pip install -r serving\\requirements.txt

echo.
echo Setup complete. Now install openvino_genai Python bindings:
echo   1. Copy the built openvino_genai package to venv\\Lib\\site-packages\\
echo   2. Or pip install the built wheel
echo.
echo Then run: start.bat
""", encoding="utf-8")

    print(f"\n✅ Deployment package created: {output_dir}")
    print(f"   model/  — {total_gb:.1f} GB (IR + tokenizer)")
    print(f"   serving/ — server code")
    print(f"   setup.bat — one-time environment setup")
    print(f"   start.bat — launch server")
    print(f"\nTo deploy:")
    print(f"  1. Copy {output_dir} to target machine")
    print(f"  2. Run setup.bat (installs Python deps)")
    print(f"  3. Install openvino_genai bindings")
    print(f"  4. Run start.bat")


if __name__ == "__main__":
    main()
