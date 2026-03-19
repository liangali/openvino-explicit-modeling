from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if entry and Path(entry).resolve() != SCRIPT_DIR
]

from wheel._commands.pack import pack as pack_wheel


def normalize_dist_name(name: str) -> str:
    return re.sub(r"[-]+", "_", name)


def build_metadata(project: dict, readme_text: str, content_type: str) -> str:
    lines: list[str] = [
        "Metadata-Version: 2.1",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
    ]

    if project.get("description"):
        lines.append(f"Summary: {project['description']}")

    authors = project.get("authors") or []
    if authors:
        author = authors[0]
        name = author.get("name")
        email = author.get("email")
        if name and email:
            lines.append(f"Author-Email: {name} <{email}>")
        elif email:
            lines.append(f"Author-Email: {email}")
        elif name:
            lines.append(f"Author: {name}")

    license_value = project.get("license")
    if isinstance(license_value, dict):
        if license_value.get("text"):
            lines.append(f"License: {license_value['text']}")
        elif license_value.get("file"):
            lines.append(f"License-File: {license_value['file']}")

    for classifier in project.get("classifiers", []):
        lines.append(f"Classifier: {classifier}")

    if project.get("requires-python"):
        lines.append(f"Requires-Python: {project['requires-python']}")

    for dependency in project.get("dependencies", []):
        lines.append(f"Requires-Dist: {dependency}")

    optional_dependencies = project.get("optional-dependencies") or {}
    for extra_name, dependencies in optional_dependencies.items():
        lines.append(f"Provides-Extra: {extra_name}")
        for dependency in dependencies:
            lines.append(f'Requires-Dist: {dependency}; extra == "{extra_name}"')

    lines.append(f"Description-Content-Type: {content_type}")
    lines.append("")
    lines.append(readme_text)
    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def find_first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the expected files exist: {paths}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--wheel-dir", required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    build_dir = Path(args.build_dir).resolve()
    wheel_dir = Path(args.wheel_dir).resolve()
    pyproject_path = source_dir / "pyproject.toml"

    with pyproject_path.open("rb") as handle:
        pyproject = tomllib.load(handle)

    project = pyproject["project"]
    readme = project["readme"]
    readme_path = source_dir / readme["file"]
    readme_text = readme_path.read_text(encoding="utf-8")
    readme_content_type = readme.get("content-type", "text/markdown")

    dist_name = normalize_dist_name(project["name"])
    version = project["version"]
    dist_info_dir_name = f"{dist_name}-{version}.dist-info"

    package_src = source_dir / "python" / "openvino_tokenizers"
    built_version_py = find_first_existing(
        [
            build_dir / "openvino_tokenizers" / "python" / "__version__.py",
            package_src / "__version__.py",
        ]
    )

    tokenizer_dll = find_first_existing(
        [
            build_dir / "openvino_genai" / "openvino_tokenizers.dll",
            build_dir / "bin" / "openvino_tokenizers.dll",
        ]
    )
    icu_uc_dll = find_first_existing(
        [
            build_dir / "_deps" / "icu" / "icu-install" / "Release" / "bin64" / "icuuc70.dll",
            build_dir / "openvino_genai" / "icuuc70.dll",
        ]
    )
    icu_data_dll = find_first_existing(
        [
            build_dir / "_deps" / "icu" / "icu-install" / "Release" / "bin64" / "icudt70.dll",
            build_dir / "openvino_genai" / "icudt70.dll",
        ]
    )

    wheel_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="openvino_tokenizers_wheel_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        staging_root = temp_dir / "staging"
        dist_info_dir = staging_root / dist_info_dir_name
        package_dst = staging_root / "openvino_tokenizers"
        lib_dst = package_dst / "lib"

        shutil.copytree(
            package_src,
            package_dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        copy_file(built_version_py, package_dst / "__version__.py")
        copy_file(tokenizer_dll, lib_dst / "openvino_tokenizers.dll")
        copy_file(icu_uc_dll, lib_dst / "icuuc70.dll")
        copy_file(icu_data_dll, lib_dst / "icudt70.dll")

        dist_info_dir.mkdir(parents=True, exist_ok=True)
        write_text(
            dist_info_dir / "WHEEL",
            "\n".join(
                [
                    "Wheel-Version: 1.0",
                    "Generator: openvino-explicit-modeling",
                    "Root-Is-Purelib: false",
                    "Tag: py3-none-win_amd64",
                    "",
                ]
            ),
        )
        write_text(dist_info_dir / "METADATA", build_metadata(project, readme_text, readme_content_type))

        entry_points = project.get("scripts") or {}
        if entry_points:
            lines = ["[console_scripts]"]
            for name, target in entry_points.items():
                lines.append(f"{name}={target}")
            lines.append("")
            write_text(dist_info_dir / "entry_points.txt", "\n".join(lines))

        copy_file(source_dir / "LICENSE", dist_info_dir / "LICENSE")
        copy_file(source_dir / "SECURITY.md", dist_info_dir / "SECURITY.md")
        copy_file(source_dir / "third-party-programs.txt", dist_info_dir / "third-party-programs.txt")

        pack_wheel(str(staging_root), str(wheel_dir), None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
