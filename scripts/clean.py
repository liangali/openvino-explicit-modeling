from __future__ import annotations

import argparse
from pathlib import Path


KEEP_FILENAMES = {
    "openvino_detokenizer.bin",
    "openvino_detokenizer.xml",
    "openvino_tokenizer.bin",
    "openvino_tokenizer.xml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Traverse model folders and delete .bin/.xml files except the "
            "OpenVINO tokenizer/detokenizer artifacts."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"C:\data\models\Huggingface"),
        help="Root directory that contains model folders.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Actually delete files. If omitted, only print statistics and planned deletions.",
    )
    return parser.parse_args()


def collect_target_files(model_dir: Path) -> list[Path]:
    files = []
    for pattern in ("*.bin", "*.xml"):
        files.extend(model_dir.rglob(pattern))
    return sorted(path for path in files if path.is_file())


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if not root.exists():
        print(f"[ERROR] Root directory does not exist: {root}")
        return 1

    if not root.is_dir():
        print(f"[ERROR] Root path is not a directory: {root}")
        return 1

    model_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not model_dirs:
        print(f"[INFO] No model directories found under: {root}")
        return 0

    print("=" * 80)
    print(f"Root directory: {root}")
    print(f"Model directories found: {len(model_dirs)}")
    print(f"Mode: {'DELETE' if args.clean else 'DRY RUN'}")
    print("=" * 80)

    total_found = 0
    total_kept = 0
    total_deleted = 0
    total_failed = 0

    for index, model_dir in enumerate(model_dirs, start=1):
        print()
        print("-" * 80)
        print(f"[{index}/{len(model_dirs)}] Model: {model_dir.name}")
        print(f"Path: {model_dir}")

        target_files = collect_target_files(model_dir)
        if not target_files:
            print("Found 0 matching files (.bin/.xml).")
            continue

        total_found += len(target_files)
        print(f"Found {len(target_files)} matching files (.bin/.xml):")
        for file_path in target_files:
            status = "KEEP" if file_path.name in KEEP_FILENAMES else "DELETE"
            rel_path = file_path.relative_to(model_dir)
            print(f"  [{status}] {rel_path}")

        kept_here = 0
        deleted_here = 0
        failed_here = 0

        for file_path in target_files:
            if file_path.name in KEEP_FILENAMES:
                kept_here += 1
                total_kept += 1
                continue

            rel_path = file_path.relative_to(model_dir)
            if not args.clean:
                print(f"  [DRY-RUN DELETE] {rel_path}")
                deleted_here += 1
                total_deleted += 1
                continue

            try:
                file_path.unlink()
                print(f"  [DELETED] {rel_path}")
                deleted_here += 1
                total_deleted += 1
            except Exception as exc:
                print(f"  [FAILED] {rel_path} -> {exc}")
                failed_here += 1
                total_failed += 1

        print(
            "Summary: "
            f"found={len(target_files)}, kept={kept_here}, "
            f"deleted={deleted_here}, failed={failed_here}"
        )

    print()
    print("=" * 80)
    print("Overall summary")
    print(f"Models scanned : {len(model_dirs)}")
    print(f"Files found    : {total_found}")
    print(f"Files kept     : {total_kept}")
    print(f"Files deleted  : {total_deleted}")
    print(f"Delete failed  : {total_failed}")
    print("=" * 80)
    return 0 if total_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
