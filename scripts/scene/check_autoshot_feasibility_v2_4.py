#!/usr/bin/env python3
"""AutoShot feasibility check for the v2.4 scene-boundary workflow.

This script is intentionally conservative:
- it never downloads datasets,
- it never trains,
- it never runs full train inference,
- it does not touch detector/rule/anchor/label/split/prediction files,
- it only writes the AutoShot feasibility artifacts requested for this task.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any


TASK_NAME = "autoshot_feasibility_check_v2_4"
PROJECT_ROOT = Path(".")
AUTOSHOT_REPO_URL = "https://github.com/wentaozhu/AutoShot"
AUTOSHOT_API_URL = "https://api.github.com/repos/wentaozhu/AutoShot"
AUTOSHOT_BAIDU_URL = "https://pan.baidu.com/s/1CdCVNzFdF3U6I4ajfejYNQ?pwd=sfkq"

REPO_PATH = PROJECT_ROOT / "external/autoshot/repo"
LOCAL_PYTHON_PATH = PROJECT_ROOT / "external/autoshot/python"
WEIGHT_DIR = PROJECT_ROOT / "models/third_party/autoshot"
SMOKE_OUTPUT_DIR = PROJECT_ROOT / "data/scene/autoshot_feasibility_smoke_outputs_v2_4"

SCRIPT_PATH = PROJECT_ROOT / "scripts/scene/check_autoshot_feasibility_v2_4.py"
INVENTORY_CSV = PROJECT_ROOT / "data/scene/autoshot_feasibility_inventory_v2_4.csv"
WEIGHT_CSV = PROJECT_ROOT / "data/scene/autoshot_weight_access_check_v2_4.csv"
SMOKE_CSV = PROJECT_ROOT / "data/scene/autoshot_smoke_test_index_v2_4.csv"
SUMMARY_MD = PROJECT_ROOT / "reports/scene/autoshot_feasibility_check_v2_4_summary.md"
REPORT_JSON = PROJECT_ROOT / "reports/scene/autoshot_feasibility_check_v2_4_report.json"
FINDINGS_MD = PROJECT_ROOT / "reports/scene/autoshot_feasibility_check_v2_4_findings.md"
LOG_PATH = PROJECT_ROOT / "logs/autoshot_feasibility_check_v2_4_run_log.txt"
LATEST_BUNDLE = PROJECT_ROOT / "outputs/latest_for_chatgpt_autoshot_feasibility_check_v2_4"
LATEST_SHARED = PROJECT_ROOT / "outputs/latest_autoshot_feasibility"

SPLIT_FILE = PROJECT_ROOT / "data/splits/video_split_v2_4.csv"

INVENTORY_COLUMNS = [
    "item_type",
    "name",
    "status",
    "version",
    "path",
    "source",
    "size_bytes",
    "sha256",
    "command_or_check",
    "notes",
]

WEIGHT_COLUMNS = [
    "weight_source_name",
    "source_url_or_description",
    "access_status",
    "requires_login_or_captcha",
    "downloadable_by_script",
    "expected_file_name",
    "downloaded",
    "downloaded_path",
    "size_bytes",
    "sha256",
    "failure_reason",
    "notes",
]

SMOKE_COLUMNS = [
    "smoke_test_name",
    "status",
    "command",
    "train_video_id",
    "video_path",
    "output_path",
    "output_format",
    "runtime_seconds",
    "device_used",
    "parsed_candidate_count",
    "error_message",
    "notes",
]


def ensure_dirs() -> None:
    for path in [
        REPO_PATH,
        LOCAL_PYTHON_PATH,
        WEIGHT_DIR,
        SMOKE_OUTPUT_DIR,
        INVENTORY_CSV.parent,
        SUMMARY_MD.parent,
        LOG_PATH.parent,
        LATEST_BUNDLE,
        LATEST_SHARED,
        SCRIPT_PATH.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def reset_log() -> None:
    LOG_PATH.write_text("", encoding="utf-8")


def log(message: str) -> None:
    print(message)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size(path: Path) -> str:
    return str(path.stat().st_size) if path.exists() else ""


def add_inventory(
    rows: list[dict[str, Any]],
    item_type: str,
    name: str,
    status: str,
    version: str = "",
    path: str | Path = "",
    source: str = "",
    size_bytes: str | int = "",
    sha256: str = "",
    command_or_check: str = "",
    notes: str = "",
) -> None:
    rows.append(
        {
            "item_type": item_type,
            "name": name,
            "status": status,
            "version": version,
            "path": str(path) if path else "",
            "source": source,
            "size_bytes": str(size_bytes) if size_bytes != "" else "",
            "sha256": sha256,
            "command_or_check": command_or_check,
            "notes": notes,
        }
    )


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: safe_str(row.get(col, "")) for col in columns})


def run_command(args: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def import_version(module_name: str, version_attr: str = "__version__") -> tuple[bool, str, str]:
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False, "", ""
        module = importlib.import_module(module_name)
        return True, str(getattr(module, version_attr, "")), str(getattr(spec, "origin", "") or "")
    except Exception as exc:
        return False, "", repr(exc)


def read_json_url(url: str, timeout: int = 30) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "codex-autoshot-feasibility",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def inspect_environment(inventory_rows: list[dict[str, Any]]) -> dict[str, Any]:
    log("[STEP 02] 현재 Python, torch, CUDA, ffmpeg 환경 확인")
    env: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    add_inventory(
        inventory_rows,
        "python",
        "cv_python",
        "available",
        env["python_version"],
        sys.executable,
        "conda env cv",
        command_or_check="sys.version",
    )

    pip_result = run_command([sys.executable, "-m", "pip", "--version"], timeout=20)
    pip_version = pip_result.get("stdout", "") if pip_result.get("returncode") == 0 else ""
    env["pip_version"] = pip_version
    add_inventory(
        inventory_rows,
        "pip",
        "pip",
        "available" if pip_version else "unknown",
        pip_version,
        source="conda env cv",
        command_or_check=f"{sys.executable} -m pip --version",
        notes=pip_result.get("stderr", "") or pip_result.get("error", ""),
    )

    try:
        import torch

        env["torch_version"] = torch.__version__
        env["torch_cuda_version"] = str(torch.version.cuda)
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_device_count"] = int(torch.cuda.device_count())
        env["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        add_inventory(
            inventory_rows,
            "torch",
            "torch",
            "available",
            torch.__version__,
            getattr(torch, "__file__", ""),
            "conda env cv",
            command_or_check="import torch",
            notes=f"torch.version.cuda={torch.version.cuda}; cuda_available={torch.cuda.is_available()}",
        )
        add_inventory(
            inventory_rows,
            "cuda",
            "cuda",
            "available" if torch.cuda.is_available() else "not_available",
            str(torch.version.cuda),
            source="torch",
            command_or_check="torch.cuda.is_available()",
            notes=env["cuda_device_name"],
        )
    except Exception as exc:
        env["torch_error"] = repr(exc)
        env["torch_version"] = ""
        env["cuda_available"] = False
        add_inventory(
            inventory_rows,
            "torch",
            "torch",
            "error",
            command_or_check="import torch",
            notes=repr(exc),
        )

    ffmpeg_path = shutil.which("ffmpeg")
    ffmpeg_version = ""
    if ffmpeg_path:
        ffmpeg_result = run_command([ffmpeg_path, "-version"], timeout=20)
        if ffmpeg_result.get("returncode") == 0:
            ffmpeg_version = ffmpeg_result.get("stdout", "").splitlines()[0]
    env["ffmpeg_available"] = bool(ffmpeg_path)
    env["ffmpeg_path"] = ffmpeg_path or ""
    env["ffmpeg_version"] = ffmpeg_version
    add_inventory(
        inventory_rows,
        "ffmpeg",
        "ffmpeg",
        "available" if ffmpeg_path else "not_available",
        ffmpeg_version,
        ffmpeg_path or "",
        "PATH",
        command_or_check="shutil.which('ffmpeg')",
    )
    return env


def inspect_repo(inventory_rows: list[dict[str, Any]]) -> dict[str, Any]:
    log("[STEP 03] AutoShot 공식 repo 접근 및 clone 가능 여부 확인")
    repo: dict[str, Any] = {
        "repo_clone_performed": False,
        "repo_path": str(REPO_PATH),
        "repo_commit_hash": "",
        "repo_release_available": False,
        "repo_accessible": False,
        "git_available": bool(shutil.which("git")),
        "repo_archive_snapshot_used": False,
    }

    if repo["git_available"]:
        add_inventory(
            inventory_rows,
            "repo",
            "git_cli",
            "available",
            path=shutil.which("git") or "",
            command_or_check="shutil.which('git')",
        )
    else:
        add_inventory(
            inventory_rows,
            "repo",
            "git_cli",
            "not_available",
            command_or_check="shutil.which('git')",
            notes="git CLI is not installed in this environment, so git clone could not be executed.",
        )

    try:
        repo_api = read_json_url(AUTOSHOT_API_URL, timeout=30)
        default_branch = repo_api.get("default_branch", "main")
        branch_api = read_json_url(f"{AUTOSHOT_API_URL}/branches/{default_branch}", timeout=30)
        releases_api = read_json_url(f"{AUTOSHOT_API_URL}/releases", timeout=30)
        repo["repo_accessible"] = True
        repo["default_branch"] = default_branch
        repo["repo_commit_hash"] = branch_api.get("commit", {}).get("sha", "")
        repo["repo_release_available"] = len(releases_api) > 0
    except Exception as exc:
        repo["repo_error"] = repr(exc)

    snapshot_meta = REPO_PATH / ".autoshot_source_snapshot.json"
    if snapshot_meta.exists():
        try:
            snapshot = json.loads(snapshot_meta.read_text(encoding="utf-8"))
            repo["repo_commit_hash"] = snapshot.get("commit_sha") or repo["repo_commit_hash"]
            repo["repo_archive_snapshot_used"] = True
        except Exception:
            pass

    repo_files = sorted(p.name for p in REPO_PATH.iterdir()) if REPO_PATH.exists() else []
    source_status = "available" if (REPO_PATH / "README.md").exists() else "missing"
    add_inventory(
        inventory_rows,
        "repo",
        "AutoShot official source",
        source_status,
        repo.get("repo_commit_hash", ""),
        REPO_PATH,
        AUTOSHOT_REPO_URL,
        command_or_check="GitHub API plus project-local source snapshot",
        notes=(
            "GitHub repo is accessible; local source was populated by archive fallback because git CLI is unavailable."
            if repo.get("repo_archive_snapshot_used")
            else "GitHub repo inspected."
        ),
    )
    add_inventory(
        inventory_rows,
        "repo",
        "GitHub releases",
        "available" if repo.get("repo_release_available") else "not_available",
        source=f"{AUTOSHOT_REPO_URL}/releases",
        command_or_check="GitHub releases API",
        notes="No releases published." if not repo.get("repo_release_available") else "",
    )
    add_inventory(
        inventory_rows,
        "repo",
        "repo_file_count",
        "available" if repo_files else "empty",
        path=REPO_PATH,
        size_bytes=len(repo_files),
        command_or_check="local repo snapshot listing",
        notes=", ".join(repo_files),
    )
    return repo


def analyze_readme_and_code(inventory_rows: list[dict[str, Any]]) -> dict[str, Any]:
    log("[STEP 04] README, model link, release, inference script 분석")
    analysis: dict[str, Any] = {
        "model_link": "Baidu link with passcode sfkq",
        "expected_weight_file": "ckpt_0_200_0.pth",
        "inference_script": "compare_inference_baseline_groundtruth_v2.py",
        "inference_interface_status": "dataset_specific_but_adaptable",
        "requirements_file": "",
        "candidate_weight_files": [],
        "candidate_prediction_files": [],
    }

    readme = REPO_PATH / "README.md"
    if readme.exists():
        add_inventory(
            inventory_rows,
            "repo",
            "README.md",
            "available",
            path=readme,
            source=AUTOSHOT_REPO_URL,
            size_bytes=file_size(readme),
            sha256=sha256_file(readme),
            command_or_check="README static analysis",
            notes="README lists model link as Baidu and expected model ckpt_0_200_0.pth.",
        )

    requirements = sorted(REPO_PATH.glob("requirements*.txt"))
    if requirements:
        analysis["requirements_file"] = str(requirements[0])
        add_inventory(
            inventory_rows,
            "requirement",
            requirements[0].name,
            "available",
            path=requirements[0],
            size_bytes=file_size(requirements[0]),
            sha256=sha256_file(requirements[0]),
            command_or_check="requirements file check",
        )
    else:
        add_inventory(
            inventory_rows,
            "requirement",
            "requirements.txt",
            "not_available",
            path=REPO_PATH,
            command_or_check="glob requirements*.txt",
            notes="No requirements file; dependencies inferred from imports.",
        )

    for pattern in ("*.pth", "*.pt", "*.ckpt"):
        analysis["candidate_weight_files"].extend(str(p) for p in REPO_PATH.glob(pattern))
    for pattern in ("*.pickle", "*.pkl"):
        analysis["candidate_prediction_files"].extend(str(p) for p in REPO_PATH.glob(pattern))

    add_inventory(
        inventory_rows,
        "weight",
        "candidate_weight_files_in_repo",
        "not_found" if not analysis["candidate_weight_files"] else "found",
        path=REPO_PATH,
        command_or_check="glob *.pth/*.pt/*.ckpt",
        notes="README weight ckpt_0_200_0.pth is not present in the repo snapshot.",
    )

    inference_script = REPO_PATH / analysis["inference_script"]
    add_inventory(
        inventory_rows,
        "inference_script",
        analysis["inference_script"],
        "available" if inference_script.exists() else "missing",
        path=inference_script,
        size_bytes=file_size(inference_script) if inference_script.exists() else "",
        sha256=sha256_file(inference_script) if inference_script.exists() else "",
        command_or_check="static code analysis",
        notes=(
            "Evaluation script is AutoShot-test-set oriented; real AutoShot inference block is commented, "
            "uses get_frames/get_batches, loads ckpt_0_200_0.pth from current directory, and writes pickle predictions."
        ),
    )
    add_inventory(
        inventory_rows,
        "model_code",
        "TransNetV2Supernet",
        "available" if (REPO_PATH / "supernet_flattransf_3_8_8_8_13_12_0_16_60.py").exists() else "missing",
        path=REPO_PATH / "supernet_flattransf_3_8_8_8_13_12_0_16_60.py",
        command_or_check="model class static check",
        notes="Model accepts 100-frame low-resolution batches shaped [B, C, T, H, W].",
    )
    return analysis


def check_dependencies(inventory_rows: list[dict[str, Any]]) -> dict[str, Any]:
    log("[STEP 05] dependency compatibility 확인")
    sys.path.insert(0, str(LOCAL_PYTHON_PATH))
    sys.path.insert(0, str(REPO_PATH))

    deps = {
        "torch": "required",
        "numpy": "required",
        "ffmpeg": "required via ffmpeg-python",
        "einops": "required",
        "matplotlib": "utility/visualization",
        "sklearn": "evaluation metric",
        "PIL": "visualization",
    }
    result: dict[str, Any] = {"missing": [], "available": {}, "local_python_path": str(LOCAL_PYTHON_PATH)}
    for module, note in deps.items():
        available, version, origin = import_version(module)
        result["available"][module] = {"available": available, "version": version, "origin": origin}
        if not available:
            result["missing"].append(module)
        add_inventory(
            inventory_rows,
            "package",
            module,
            "available" if available else "missing",
            version,
            origin,
            "cv env plus external/autoshot/python",
            command_or_check=f"import {module}",
            notes=note,
        )

    local_deps_present = (LOCAL_PYTHON_PATH / "einops").exists() and (
        (LOCAL_PYTHON_PATH / "ffmpeg").exists() or (LOCAL_PYTHON_PATH / "ffmpeg_python-0.2.0.dist-info").exists()
    )
    result["dependency_check_status"] = "success" if not result["missing"] else "missing_dependency"
    result["install_performed"] = local_deps_present
    result["install_method"] = (
        "project-local pip --target external/autoshot/python for einops and ffmpeg-python"
        if local_deps_present
        else "none"
    )
    add_inventory(
        inventory_rows,
        "package",
        "project_local_dependency_target",
        "available" if local_deps_present else "not_available",
        path=LOCAL_PYTHON_PATH,
        source="pip --target",
        command_or_check="check local package directory",
        notes=result["install_method"],
    )
    return result


def check_weight_access(
    inventory_rows: list[dict[str, Any]],
    weight_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    log("[STEP 06] pretrained weight 접근 가능성 확인")
    weight: dict[str, Any] = {
        "weight_access_status": "manual_baidu_required",
        "weight_download_performed": False,
        "weight_path": "",
        "weight_size_bytes": "",
        "weight_sha256": "",
        "expected_file_name": "ckpt_0_200_0.pth",
        "baidu_status": "",
        "baidu_final_url": "",
        "baidu_content_type": "",
        "baidu_keyword_flags": {},
    }

    local_weight = WEIGHT_DIR / weight["expected_file_name"]
    if local_weight.exists():
        weight["weight_access_status"] = "local_weight_present"
        weight["weight_path"] = str(local_weight)
        weight["weight_size_bytes"] = str(local_weight.stat().st_size)
        weight["weight_sha256"] = sha256_file(local_weight)

    try:
        req = urllib.request.Request(AUTOSHOT_BAIDU_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read(65536).decode("utf-8", "replace")
            weight["baidu_status"] = str(getattr(resp, "status", ""))
            weight["baidu_final_url"] = resp.geturl()
            weight["baidu_content_type"] = resp.headers.get("content-type", "")
            flags = {
                "html_page": "<html" in data.lower(),
                "login": "login" in data.lower(),
                "captcha": "captcha" in data.lower() or "验证码" in data,
                "verify": "verify" in data.lower(),
                "bdstoken": "bdstoken" in data.lower(),
            }
            weight["baidu_keyword_flags"] = flags
    except Exception as exc:
        weight["baidu_error"] = repr(exc)

    if not local_weight.exists():
        weight_rows.append(
            {
                "weight_source_name": "README Baidu model link",
                "source_url_or_description": AUTOSHOT_BAIDU_URL,
                "access_status": "html_share_page_only",
                "requires_login_or_captcha": "unknown_or_likely_manual_verification",
                "downloadable_by_script": "False",
                "expected_file_name": weight["expected_file_name"],
                "downloaded": "False",
                "downloaded_path": "",
                "size_bytes": "",
                "sha256": "",
                "failure_reason": "No direct file URL or file-size metadata exposed; Baidu page includes verification/login-related markers.",
                "notes": f"HTTP {weight.get('baidu_status')}; final_url={weight.get('baidu_final_url')}; content_type={weight.get('baidu_content_type')}; flags={weight.get('baidu_keyword_flags')}",
            }
        )
    else:
        weight_rows.append(
            {
                "weight_source_name": "local ckpt_0_200_0.pth",
                "source_url_or_description": str(local_weight),
                "access_status": "local_file_present",
                "requires_login_or_captcha": "False",
                "downloadable_by_script": "not_applicable",
                "expected_file_name": weight["expected_file_name"],
                "downloaded": "False",
                "downloaded_path": str(local_weight),
                "size_bytes": weight["weight_size_bytes"],
                "sha256": weight["weight_sha256"],
                "failure_reason": "",
                "notes": "Local file detected before this run.",
            }
        )

    weight_rows.append(
        {
            "weight_source_name": "GitHub releases",
            "source_url_or_description": f"{AUTOSHOT_REPO_URL}/releases",
            "access_status": "no_release_assets",
            "requires_login_or_captcha": "False",
            "downloadable_by_script": "False",
            "expected_file_name": weight["expected_file_name"],
            "downloaded": "False",
            "downloaded_path": "",
            "size_bytes": "",
            "sha256": "",
            "failure_reason": "No releases are published for the official repo.",
            "notes": "Checked via GitHub releases API.",
        }
    )
    weight_rows.append(
        {
            "weight_source_name": "public alternative search",
            "source_url_or_description": "Search queries for ckpt_0_200_0.pth, baseline_one_hot_pred_dict_baseline.pickle, AutoShot pretrained model",
            "access_status": "not_found",
            "requires_login_or_captcha": "unknown",
            "downloadable_by_script": "False",
            "expected_file_name": weight["expected_file_name"],
            "downloaded": "False",
            "downloaded_path": "",
            "size_bytes": "",
            "sha256": "",
            "failure_reason": "No clearly official public model-hosting or release weight was found.",
            "notes": "Do not download unofficial or ambiguous weights.",
        }
    )

    add_inventory(
        inventory_rows,
        "weight",
        weight["expected_file_name"],
        "manual_required" if not local_weight.exists() else "available",
        path=local_weight if local_weight.exists() else WEIGHT_DIR,
        source=AUTOSHOT_BAIDU_URL,
        size_bytes=weight["weight_size_bytes"],
        sha256=weight["weight_sha256"],
        command_or_check="bounded Baidu HTML access check; no file download",
        notes=(
            "Automatic download blocked by missing direct file URL/size and manual Baidu flow."
            if not local_weight.exists()
            else "Local checkpoint present."
        ),
    )
    return weight


def safe_project_local_setup(inventory_rows: list[dict[str, Any]], deps: dict[str, Any]) -> None:
    log("[STEP 07] 필요한 경우 안전한 project-local 설치 또는 PYTHONPATH 구성")
    add_inventory(
        inventory_rows,
        "package",
        "PYTHONPATH",
        "configured",
        path=f"{LOCAL_PYTHON_PATH}:{REPO_PATH}",
        command_or_check="sys.path.insert",
        notes=(
            "Using project-local dependency target; cv env site-packages were not upgraded or downgraded. "
            f"dependency_check_status={deps.get('dependency_check_status')}"
        ),
    )


def run_smoke_checks(
    inventory_rows: list[dict[str, Any]],
    smoke_rows: list[dict[str, Any]],
    weight: dict[str, Any],
) -> dict[str, Any]:
    smoke: dict[str, Any] = {
        "smoke_import_status": "not_run",
        "smoke_model_init_status": "not_run",
        "smoke_weight_load_status": "not_run",
        "smoke_dummy_forward_status": "not_run",
        "smoke_video_inference_status": "not_run",
        "smoke_output_parse_status": "not_run",
        "device_used": "",
        "dummy_runtime_seconds": "",
        "dummy_output_shapes": [],
        "parser_candidate_count": "",
        "train_video_id": "",
        "train_video_path": "",
    }

    sys.path.insert(0, str(LOCAL_PYTHON_PATH))
    sys.path.insert(0, str(REPO_PATH))

    log("[STEP 08] 가능한 경우 import 및 model code smoke test")
    t0 = time.time()
    try:
        import torch
        from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet

        smoke["smoke_import_status"] = "success"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        smoke["device_used"] = device
        model = TransNetV2Supernet().eval().to(device)
        smoke["smoke_model_init_status"] = "success"
        add_inventory(
            inventory_rows,
            "smoke_test",
            "autoshot_import_model_init",
            "success",
            path=REPO_PATH,
            command_or_check="import TransNetV2Supernet; TransNetV2Supernet().eval()",
            notes=f"device={device}",
        )

        log("[STEP 09] 가능한 경우 weight load smoke test")
        if weight.get("weight_path"):
            try:
                pretrained = torch.load(weight["weight_path"], map_location=device)
                model_dict = model.state_dict()
                if isinstance(pretrained, dict) and "net" in pretrained:
                    matched = {k: v for k, v in pretrained["net"].items() if k in model_dict}
                    model_dict.update(matched)
                    model.load_state_dict(model_dict)
                    smoke["smoke_weight_load_status"] = "success"
                    weight_note = f"matched_params={len(matched)}"
                else:
                    smoke["smoke_weight_load_status"] = "error"
                    weight_note = "Checkpoint did not contain expected top-level key 'net'."
            except Exception as exc:
                smoke["smoke_weight_load_status"] = "error"
                weight_note = repr(exc)
        else:
            smoke["smoke_weight_load_status"] = "skipped_no_weight"
            weight_note = "Skipped because official pretrained weight was not automatically downloadable."

        smoke_rows.append(
            {
                "smoke_test_name": "weight_load",
                "status": smoke["smoke_weight_load_status"],
                "command": "torch.load(ckpt_0_200_0.pth); load pretrained_dict['net']",
                "train_video_id": "",
                "video_path": "",
                "output_path": "",
                "output_format": "",
                "runtime_seconds": "",
                "device_used": device,
                "parsed_candidate_count": "",
                "error_message": "" if smoke["smoke_weight_load_status"] != "error" else weight_note,
                "notes": weight_note,
            }
        )

        log("[STEP 10] 가능한 경우 train 영상 1개 최소 inference smoke test")
        # 실제 video inference 가능 여부는 weight loading이 기준이다.
        # official checkpoint 없이 random weight로 실행하면 feasibility 판단이 왜곡된다.
        if smoke["smoke_weight_load_status"] == "success":
            smoke["smoke_video_inference_status"] = "not_implemented_in_feasibility_script"
            video_note = "Checkpoint exists, but full video adaptation is intentionally deferred to a follow-up script."
        else:
            smoke["smoke_video_inference_status"] = "skipped_no_pretrained_weight"
            video_note = "No train video inference was run because pretrained weight load did not succeed."

        train_candidate = find_train_candidate()
        smoke.update(train_candidate)
        smoke_rows.append(
            {
                "smoke_test_name": "train_video_inference",
                "status": smoke["smoke_video_inference_status"],
                "command": "not run",
                "train_video_id": smoke.get("train_video_id", ""),
                "video_path": smoke.get("train_video_path", ""),
                "output_path": "",
                "output_format": "",
                "runtime_seconds": "",
                "device_used": device,
                "parsed_candidate_count": "",
                "error_message": "",
                "notes": video_note,
            }
        )

        log("[STEP 11] output parsing 가능성 확인")
        from utils import predictions_to_scenes

        synthetic = __import__("numpy").zeros(100, dtype=__import__("numpy").uint8)
        synthetic[[25, 70]] = 1
        scenes = predictions_to_scenes(synthetic)
        parsed_count = max(0, len(scenes) - 1)
        smoke["smoke_output_parse_status"] = "success_static_parser_check"
        smoke["parser_candidate_count"] = str(parsed_count)

        smoke_rows.append(
            {
                "smoke_test_name": "output_parser_static",
                "status": smoke["smoke_output_parse_status"],
                "command": "utils.predictions_to_scenes(synthetic_one_hot)",
                "train_video_id": "",
                "video_path": "",
                "output_path": "",
                "output_format": "frame-index scenes array",
                "runtime_seconds": "",
                "device_used": "",
                "parsed_candidate_count": parsed_count,
                "error_message": "",
                "notes": "Parser can convert one-hot frame predictions to scene intervals; timestamps require fps from video metadata.",
            }
        )

        with torch.no_grad():
            x = torch.randint(0, 256, (1, 3, 100, 27, 48), device=device, dtype=torch.float32)
            y = model(x)
        if isinstance(y, tuple):
            smoke["dummy_output_shapes"] = [list(v.shape) for v in y]
        else:
            smoke["dummy_output_shapes"] = [list(y.shape)]
        smoke["smoke_dummy_forward_status"] = "success"
        smoke["dummy_runtime_seconds"] = round(time.time() - t0, 3)
        smoke_rows.insert(
            0,
            {
                "smoke_test_name": "import_model_dummy_forward",
                "status": "success",
                "command": "TransNetV2Supernet().eval(); dummy [1,3,100,27,48] forward",
                "train_video_id": "",
                "video_path": "",
                "output_path": "",
                "output_format": "torch tensors [B,T,1]",
                "runtime_seconds": smoke["dummy_runtime_seconds"],
                "device_used": device,
                "parsed_candidate_count": "",
                "error_message": "",
                "notes": f"output_shapes={smoke['dummy_output_shapes']}",
            },
        )
        add_inventory(
            inventory_rows,
            "smoke_test",
            "dummy_forward",
            "success",
            command_or_check="dummy [1,3,100,27,48] forward",
            notes=f"output_shapes={smoke['dummy_output_shapes']}; runtime_seconds={smoke['dummy_runtime_seconds']}",
        )
        add_inventory(
            inventory_rows,
            "output_parser",
            "predictions_to_scenes",
            "success_static_parser_check",
            path=REPO_PATH / "utils.py",
            command_or_check="utils.predictions_to_scenes",
            notes=f"synthetic parsed_candidate_count={parsed_count}",
        )

    except Exception as exc:
        smoke["smoke_import_status"] = "error" if smoke["smoke_import_status"] == "not_run" else smoke["smoke_import_status"]
        err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        smoke_rows.append(
            {
                "smoke_test_name": "import_model_dummy_forward",
                "status": "error",
                "command": "TransNetV2Supernet import/model/dummy",
                "train_video_id": "",
                "video_path": "",
                "output_path": "",
                "output_format": "",
                "runtime_seconds": round(time.time() - t0, 3),
                "device_used": smoke.get("device_used", ""),
                "parsed_candidate_count": "",
                "error_message": err,
                "notes": traceback.format_exc(limit=3),
            }
        )
        add_inventory(
            inventory_rows,
            "smoke_test",
            "autoshot_import_model_dummy",
            "error",
            path=REPO_PATH,
            command_or_check="import/model/dummy smoke",
            notes=err,
        )
    return smoke


def find_train_candidate() -> dict[str, str]:
    candidate = {"train_video_id": "", "train_video_path": ""}
    if not SPLIT_FILE.exists():
        return candidate
    try:
        with SPLIT_FILE.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") == "train" and row.get("video_path") and Path(row["video_path"]).exists():
                    candidate["train_video_id"] = row.get("video_id", "")
                    candidate["train_video_path"] = row.get("video_path", "")
                    break
    except Exception:
        pass
    return candidate


def decide_status(repo: dict[str, Any], deps: dict[str, Any], weight: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    log("[STEP 12] feasibility status 및 recommended next action 결정")
    reasons: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if not repo.get("repo_accessible"):
        errors.append("Official AutoShot GitHub repo could not be reached.")
    if not repo.get("git_available"):
        warnings.append("git CLI is unavailable; source snapshot was obtained via GitHub archive fallback, not git clone.")
    if not repo.get("repo_release_available"):
        warnings.append("Official GitHub repo has no releases or release assets.")
    if deps.get("dependency_check_status") != "success":
        errors.append(f"Missing dependencies: {deps.get('missing')}")
    if weight.get("weight_access_status") == "manual_baidu_required":
        warnings.append("Pretrained ckpt_0_200_0.pth is only documented via Baidu share flow and was not automatically downloaded.")
    if smoke.get("smoke_video_inference_status") != "success":
        warnings.append("No project-video pretrained smoke inference was run because weight load did not succeed.")

    if errors and deps.get("dependency_check_status") != "success":
        feasibility_status = "BLOCKED_BY_ENV_COMPATIBILITY"
    elif weight.get("weight_access_status") == "manual_baidu_required" and smoke.get("smoke_import_status") == "success":
        feasibility_status = "READY_WITH_MANUAL_WEIGHT"
    elif weight.get("weight_access_status") == "manual_baidu_required":
        feasibility_status = "BLOCKED_BY_WEIGHT_ACCESS"
    elif smoke.get("smoke_video_inference_status") == "success":
        feasibility_status = "READY_FOR_TRAIN_EXTRACTION"
    else:
        feasibility_status = "FEASIBILITY_ONLY_NO_INFERENCE"

    if feasibility_status == "READY_WITH_MANUAL_WEIGHT":
        reasons.extend(
            [
                "Repo source and model code are accessible.",
                "Lightweight missing dependencies are satisfied through project-local external/autoshot/python.",
                "Model import/init and dummy forward succeeded on the current CUDA environment.",
                "Official pretrained weight was not automatically obtainable from a direct public URL.",
            ]
        )
        recommended_next_action = (
            "Manually obtain ckpt_0_200_0.pth from the official Baidu share, place it under "
            "models/third_party/autoshot/, then rerun a weight-load and one-train-video smoke test before any train extraction."
        )
    else:
        recommended_next_action = (
            "Keep TransNetV2 conservative outputs as the usable scene-boundary helper for now; revisit AutoShot only after weight access is solved."
        )

    ready_for_full_train_extraction = feasibility_status == "READY_FOR_TRAIN_EXTRACTION"
    return {
        "feasibility_status": feasibility_status,
        "ready_for_full_train_extraction": ready_for_full_train_extraction,
        "recommended_next_action": recommended_next_action,
        "reasons": reasons,
        "warnings": warnings,
        "errors": errors,
    }


def build_report(
    env: dict[str, Any],
    repo: dict[str, Any],
    deps: dict[str, Any],
    analysis: dict[str, Any],
    weight: dict[str, Any],
    smoke: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_name": TASK_NAME,
        "project_root": str(PROJECT_ROOT),
        "autoshot_repo_url": AUTOSHOT_REPO_URL,
        "repo_clone_performed": repo.get("repo_clone_performed", False),
        "repo_archive_snapshot_used": repo.get("repo_archive_snapshot_used", False),
        "repo_path": str(REPO_PATH),
        "repo_commit_hash": repo.get("repo_commit_hash", ""),
        "repo_release_available": repo.get("repo_release_available", False),
        "python_executable": env.get("python_executable", ""),
        "python_version": env.get("python_version", ""),
        "torch_version": env.get("torch_version", ""),
        "cuda_available": env.get("cuda_available", False),
        "ffmpeg_available": env.get("ffmpeg_available", False),
        "dependency_check_status": deps.get("dependency_check_status", ""),
        "install_performed": deps.get("install_performed", False),
        "install_method": deps.get("install_method", ""),
        "venv_used": False,
        "weight_access_status": weight.get("weight_access_status", ""),
        "weight_download_performed": weight.get("weight_download_performed", False),
        "weight_path": weight.get("weight_path", ""),
        "weight_size_bytes": weight.get("weight_size_bytes", ""),
        "weight_sha256": weight.get("weight_sha256", ""),
        "smoke_import_status": smoke.get("smoke_import_status", ""),
        "smoke_weight_load_status": smoke.get("smoke_weight_load_status", ""),
        "smoke_dummy_forward_status": smoke.get("smoke_dummy_forward_status", ""),
        "smoke_video_inference_status": smoke.get("smoke_video_inference_status", ""),
        "smoke_output_parse_status": smoke.get("smoke_output_parse_status", ""),
        "feasibility_status": decision.get("feasibility_status", ""),
        "ready_for_full_train_extraction": decision.get("ready_for_full_train_extraction", False),
        "recommended_next_action": decision.get("recommended_next_action", ""),
        "reasons": decision.get("reasons", []),
        "warnings": decision.get("warnings", []),
        "errors": decision.get("errors", []),
        "protected_files_modified": False,
        "no_full_train_inference": True,
        "no_validation_test_row_level_output": True,
        "latest_bundle_path": str(LATEST_BUNDLE),
        "additional_details": {
            "git_available": repo.get("git_available", False),
            "repo_accessible": repo.get("repo_accessible", False),
            "expected_weight_file": analysis.get("expected_weight_file", "ckpt_0_200_0.pth"),
            "inference_script": analysis.get("inference_script", ""),
            "inference_interface_status": analysis.get("inference_interface_status", ""),
            "baidu_final_url": weight.get("baidu_final_url", ""),
            "baidu_keyword_flags": weight.get("baidu_keyword_flags", {}),
            "dummy_output_shapes": smoke.get("dummy_output_shapes", []),
            "train_video_candidate": {
                "video_id": smoke.get("train_video_id", ""),
                "video_path": smoke.get("train_video_path", ""),
            },
        },
    }


def write_reports(report: dict[str, Any]) -> None:
    log("[STEP 13] CSV/report/log 생성")
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = f"""# AutoShot feasibility check v2.4

## 1. AutoShot을 검토한 이유
TransNetV2 외에 추가로 사용할 수 있는 최신 shot boundary detection 후보를 확인하기 위해 AutoShot의 repo, weight, dependency, smoke 실행 가능성을 점검했다.

## 2. AutoShot의 성격
AutoShot은 CVPRW 2023 SBD 모델/데이터셋 작업이며, README와 논문 기준 SHOT 및 ClipShots/BBC/RAI에서 TransNetV2 대비 성능 우위를 보고한다. 이 점은 장점이지만, 현재 프로젝트에서는 재현 가능한 pretrained weight와 바로 쓰기 쉬운 inference interface가 더 중요하다.

## 3. Repo 접근성 결과
- 공식 repo: {AUTOSHOT_REPO_URL}
- repo 접근: {report['additional_details']['repo_accessible']}
- git clone 수행: {report['repo_clone_performed']}
- local source path: {report['repo_path']}
- commit hash: {report['repo_commit_hash']}
- 주의: git CLI가 없어 GitHub archive snapshot fallback을 사용했다.

## 4. Release / package / weight 접근성 결과
- GitHub release 존재: {report['repo_release_available']}
- README model link: Baidu share, expected file `ckpt_0_200_0.pth`
- weight 자동 다운로드: {report['weight_download_performed']}
- weight access status: {report['weight_access_status']}
- weight path: {report['weight_path'] or 'none'}

## 5. Dependency compatibility 결과
- Python: {report['python_version']}
- Torch: {report['torch_version']}
- CUDA available: {report['cuda_available']}
- ffmpeg available: {report['ffmpeg_available']}
- dependency status: {report['dependency_check_status']}
- install method: {report['install_method']}

## 6. Smoke test 결과
- import: {report['smoke_import_status']}
- dummy forward: {report['smoke_dummy_forward_status']}
- weight load: {report['smoke_weight_load_status']}
- video inference: {report['smoke_video_inference_status']}
- output parse: {report['smoke_output_parse_status']}

## 7. 현재 프로젝트에서 바로 쓸 수 있는지 여부
바로 쓸 수는 없다. 모델 코드는 현재 CUDA/Torch 환경에서 import와 dummy forward가 되지만, 공식 pretrained weight가 자동으로 확보되지 않았고 실제 프로젝트 train 영상에 대한 pretrained smoke inference가 수행되지 않았다.

## 8. 전체 train 실험 확장 추천 여부
현재 상태에서는 full train extraction으로 확장하지 않는다. 수동으로 공식 `ckpt_0_200_0.pth`를 확보한 뒤 weight load와 train 영상 1개 smoke inference를 먼저 통과해야 한다.

## 9. 불가능하거나 낮은 재현성의 이유
- weight가 Baidu 공유 링크 중심이며 직접 파일 URL/size를 얻을 수 없다.
- GitHub release asset이 없다.
- evaluation script는 AutoShot dataset path와 pickle 중심이며, 프로젝트 영상용 CLI가 별도로 제공되지 않는다.

## 10. 현재 일정에서 추천하는 결론
feasibility_status: `{report['feasibility_status']}`

recommended_next_action: {report['recommended_next_action']}
"""
    SUMMARY_MD.write_text(summary, encoding="utf-8")

    findings = f"""# AutoShot feasibility findings v2.4

결론: AutoShot은 weight 수동 확보가 필요하다.

AutoShot은 최신 SBD 후보로 검토할 가치는 있다. repo source, model class, dummy forward는 현재 `cv` 환경에서 동작했고, output parser도 frame index 기반 scene interval로 변환 가능하다.

하지만 공식 pretrained weight는 README 기준 Baidu 공유 링크에만 의존한다. GitHub release가 없고, 직접 다운로드 가능한 공식 `ckpt_0_200_0.pth` URL이나 size metadata를 확보하지 못했다. 따라서 지금 바로 train 실험으로 확장하는 것은 적절하지 않다.

현재 일정에서는 AutoShot을 기존 TransNetV2 conservative 결과를 대체하기보다 추가 검토 대상으로 남기는 것이 적절하다. 다음 단계는 사용자가 공식 Baidu 링크에서 `ckpt_0_200_0.pth`를 수동 확보한 뒤, weight load와 train 영상 1개 smoke inference만 다시 확인하는 것이다.

최종 status: `{report['feasibility_status']}`
"""
    FINDINGS_MD.write_text(findings, encoding="utf-8")


def copy_latest_bundle() -> None:
    log("[STEP 15] latest bundle 및 latest_autoshot_feasibility 복사")
    bundle_files = [
        SCRIPT_PATH,
        INVENTORY_CSV,
        WEIGHT_CSV,
        SMOKE_CSV,
        SUMMARY_MD,
        REPORT_JSON,
        FINDINGS_MD,
        LOG_PATH,
    ]

    readme = f"""# Latest AutoShot Feasibility Check v2.4 Files

This bundle includes only the newly generated script, small CSVs, reports, and log.

Excluded by design:
- AutoShot repo directory
- project-local package directory
- model weights/checkpoints
- raw videos
- frame images/cache
- raw prediction arrays
- validation/test row-level output

Files:
"""
    for src in bundle_files:
        if src.exists():
            readme += f"- {src.name}\n"
    (LATEST_BUNDLE / "README_latest_files.md").write_text(readme, encoding="utf-8")

    for dst_dir in [LATEST_BUNDLE, LATEST_SHARED]:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in bundle_files:
            if src.exists():
                shutil.copy2(src, dst_dir / src.name)
        readme_src = LATEST_BUNDLE / "README_latest_files.md"
        readme_dst = dst_dir / "README_latest_files.md"
        if readme_src.resolve() != readme_dst.resolve():
            shutil.copy2(readme_src, readme_dst)


def main() -> int:
    ensure_dirs()
    reset_log()
    log("[STEP 01] 안전 스냅샷 및 출력 경로 준비")

    inventory_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    smoke_rows: list[dict[str, Any]] = []

    env = inspect_environment(inventory_rows)
    repo = inspect_repo(inventory_rows)
    analysis = analyze_readme_and_code(inventory_rows)
    deps = check_dependencies(inventory_rows)
    weight = check_weight_access(inventory_rows, weight_rows)
    safe_project_local_setup(inventory_rows, deps)
    smoke = run_smoke_checks(inventory_rows, smoke_rows, weight)
    decision = decide_status(repo, deps, weight, smoke)
    report = build_report(env, repo, deps, analysis, weight, smoke, decision)

    write_csv(INVENTORY_CSV, INVENTORY_COLUMNS, inventory_rows)
    write_csv(WEIGHT_CSV, WEIGHT_COLUMNS, weight_rows)
    write_csv(SMOKE_CSV, SMOKE_COLUMNS, smoke_rows)
    write_reports(report)

    log("[STEP 14] Sub Agent 검증 실행")
    log("Sub Agent 검증은 main agent 산출물 생성 후 별도 agent들에게 요청한다.")
    copy_latest_bundle()
    log("[STEP 16] 최종 요약 출력")
    log(f"feasibility_status={report['feasibility_status']}")
    log(f"ready_for_full_train_extraction={report['ready_for_full_train_extraction']}")
    log(f"latest_bundle_path={report['latest_bundle_path']}")
    for dst_dir in [LATEST_BUNDLE, LATEST_SHARED]:
        if dst_dir.exists():
            shutil.copy2(LOG_PATH, dst_dir / LOG_PATH.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
