#!/usr/bin/env python3
"""CV 스크립트 실행에 필요한 Python 환경을 확인한다."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("YASD_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    cwd = Path.cwd().resolve()
    executable = Path(sys.executable).resolve()
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")

    print(f"python_executable={executable}")
    print(f"cwd={cwd}")
    print(f"CONDA_DEFAULT_ENV={conda_env}")
    print(f"CONDA_PREFIX={conda_prefix}")

    if cwd != PROJECT_ROOT and not str(cwd).startswith(str(PROJECT_ROOT) + os.sep):
        fail(f"working directory must be inside {PROJECT_ROOT}")

    executable_text = str(executable)
    prefix_text = str(Path(conda_prefix).resolve()) if conda_prefix else ""
    in_cv = (
        conda_env == "cv"
        or "/envs/cv/" in executable_text
        or executable_text.endswith("/envs/cv/bin/python")
        or prefix_text.endswith("/envs/cv")
    )
    if not in_cv:
        fail("Python must be executed from the cv conda environment")

    try:
        import pandas as pd
    except Exception as exc:
        fail(f"pandas import failed: {exc!r}")

    try:
        import openpyxl
    except Exception as exc:
        fail(f"openpyxl import failed: {exc!r}")

    print(f"pandas_version={pd.__version__}")
    print(f"openpyxl_version={openpyxl.__version__}")
    print("cv_environment_check=ok")


if __name__ == "__main__":
    main()
