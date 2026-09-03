"""Read-only environment diagnostics for the GUI-Agent project."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
from pathlib import Path


PACKAGES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "Pillow")
DEMO_PACKAGES = ("fastapi", "uvicorn", "gradio", "pydantic")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def path_status(name: str, required: bool) -> tuple[bool, str]:
    value = os.environ.get(name, "").strip()
    if not value:
        return (not required, f"{name}: NOT_SET")
    path = Path(value).expanduser()
    return (path.exists(), f"{name}: {'OK' if path.exists() else 'MISSING'} ({path})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Python, packages, CUDA, and configured paths.")
    parser.add_argument("--strict", action="store_true", help="Fail when a required package/path is missing.")
    parser.add_argument("--demo", action="store_true", help="Also require the optional API and Gradio packages.")
    args = parser.parse_args()

    failures: list[str] = []
    print(f"python: {platform.python_version()}")
    for package in PACKAGES:
        version = package_version(package)
        print(f"{package}: {version or 'NOT_INSTALLED'}")
        if version is None:
            failures.append(package)

    if args.demo:
        for package in DEMO_PACKAGES:
            version = package_version(package)
            print(f"{package}: {version or 'NOT_INSTALLED'}")
            if version is None:
                failures.append(package)

    try:
        import torch

        print(f"cuda_available: {torch.cuda.is_available()}")
        print(f"cuda_device_count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"cuda_device_{index}: {torch.cuda.get_device_name(index)}")
        if not torch.cuda.is_available():
            failures.append("cuda")
    except Exception as exc:
        print(f"cuda_check: ERROR ({exc})")
        failures.append("cuda")

    for name, required in (("BASE_MODEL", True), ("CAGUI_ROOT", True), ("SFT_ADAPTER", False), ("ARTIFACT_ROOT", True)):
        ok, message = path_status(name, required)
        print(message)
        if not ok:
            failures.append(name)

    if args.strict and failures:
        raise SystemExit("environment check failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
