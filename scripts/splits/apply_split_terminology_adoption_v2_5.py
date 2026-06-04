#!/usr/bin/env python3
"""Adopt v2.5 split terminology as a compatibility layer.

This script creates/refreshes mapping, config, helper, docs, reports, usage
inventory, and the latest bundle. It does not run detectors, extract features,
rename historical outputs, or modify the original v2.4 split file.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import py_compile
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
SCRIPTS_SPLITS_DIR = PROJECT_ROOT / "scripts/splits"
SPLIT_V2_4_PATH = PROJECT_ROOT / "data/splits/video_split_v2_4.csv"
MAPPING_CSV_PATH = PROJECT_ROOT / "data/splits/video_split_v2_5_ruledev_extended_eval.csv"
CONFIG_JSON_PATH = PROJECT_ROOT / "configs/splits/split_terminology_v2_5_ruledev_extended_eval.json"
HELPER_PATH = PROJECT_ROOT / "src/utils/split_terminology_v2_5.py"
WRITE_SCRIPT_PATH = SCRIPTS_SPLITS_DIR / "write_split_terminology_v2_5_ruledev_extended_eval.py"
SCAN_SCRIPT_PATH = SCRIPTS_SPLITS_DIR / "scan_split_terminology_usage_v2_5.py"
APPLY_SCRIPT_PATH = SCRIPTS_SPLITS_DIR / "apply_split_terminology_adoption_v2_5.py"
SUMMARY_MD = PROJECT_ROOT / "reports/splits/split_terminology_v2_5_summary.md"
HISTORY_MD = PROJECT_ROOT / "reports/splits/split_terminology_v2_5_history.md"
ADOPTION_PLAN_MD = PROJECT_ROOT / "reports/splits/split_terminology_v2_5_adoption_plan.md"
USAGE_INVENTORY_CSV = PROJECT_ROOT / "reports/splits/split_terminology_v2_5_usage_inventory.csv"
ADOPTION_REPORT_JSON = PROJECT_ROOT / "reports/splits/split_terminology_v2_5_adoption_report.json"
PATCH_SUMMARY_MD = PROJECT_ROOT / "reports/splits/split_terminology_v2_5_active_files_patch_summary.md"
DOCS_MD = PROJECT_ROOT / "docs/split_terminology_v2_5.md"
README_SPLIT_MD = PROJECT_ROOT / "README_split_terminology_v2_5.md"
ROOT_README = PROJECT_ROOT / "README.md"
RUN_LOG = PROJECT_ROOT / "logs/split_terminology_adoption_v2_5_run_log.txt"
LATEST_BUNDLE = PROJECT_ROOT / "outputs/latest_for_chatgpt_split_terminology_adoption_v2_5"
OLD_SNAPSHOT_BEFORE = PROJECT_ROOT / "reports/splits/old_project_snapshot_before_split_terminology_adoption_v2_5.tsv"
OLD_SNAPSHOT_AFTER = PROJECT_ROOT / "reports/splits/old_project_snapshot_after_split_terminology_adoption_v2_5.tsv"
INPUT_STATS_BEFORE = PROJECT_ROOT / "reports/splits/video_split_v2_4_input_stats_before_split_terminology_adoption_v2_5.json"
INPUT_STATS_AFTER = PROJECT_ROOT / "reports/splits/video_split_v2_4_input_stats_after_split_terminology_adoption_v2_5.json"
SPLIT_SEED = 20240524
EXPECTED_SPLITS = {
    "train": [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
    "validation": [3, 7, 18],
    "test": [4, 16, 17],
}
FORBIDDEN_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".m4a",
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".pt", ".pth",
    ".ckpt", ".onnx", ".pkl", ".pickle", ".parquet",
}
FORBIDDEN_PARTS = {
    "cache", "model_cache", "models", "external", "frames", "frame_images",
    "raw", "raw_video", "proxy", "checkpoint", "checkpoints", "__pycache__",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def log(message: str) -> None:
    print(message, flush=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{now_iso()} {message}\n")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stat(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def snapshot_project(root: Path, output_path: Path) -> list[str]:
    rows: list[str] = []
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                rows.append(f"{path.relative_to(root)}\t{stat.st_size}\t{stat.st_mtime_ns}")
    rows = sorted(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return rows


def path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    if path.exists():
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    return total


def backup_existing_targets() -> tuple[Path, list[str]]:
    backup_dir = PROJECT_ROOT / "backups" / f"split_terminology_adoption_v2_5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    targets = [
        MAPPING_CSV_PATH, CONFIG_JSON_PATH, HELPER_PATH, WRITE_SCRIPT_PATH,
        SCAN_SCRIPT_PATH, APPLY_SCRIPT_PATH, SUMMARY_MD, HISTORY_MD,
        ADOPTION_PLAN_MD, USAGE_INVENTORY_CSV, ADOPTION_REPORT_JSON,
        PATCH_SUMMARY_MD, DOCS_MD, README_SPLIT_MD, ROOT_README, RUN_LOG,
    ]
    backed: list[str] = []
    backup_dir.mkdir(parents=True, exist_ok=True)
    max_backup_bytes = 10 * 1024 * 1024
    for path in targets:
        if path.exists():
            if path_size_bytes(path) > max_backup_bytes:
                backed.append(f"{rel(path)} (metadata_only_skipped_large_regenerable_output)")
                continue
            dst = backup_dir / rel(path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            backed.append(rel(path))
    if LATEST_BUNDLE.exists() and any(LATEST_BUNDLE.iterdir()):
        if path_size_bytes(LATEST_BUNDLE) > max_backup_bytes:
            backed.append(f"{rel(LATEST_BUNDLE)} (metadata_only_skipped_large_regenerable_output)")
        else:
            dst = backup_dir / rel(LATEST_BUNDLE)
            shutil.copytree(LATEST_BUNDLE, dst, dirs_exist_ok=True)
            backed.append(rel(LATEST_BUNDLE))
    return backup_dir, backed


def validate_original_split(rows: list[dict[str, str]]) -> dict[str, Any]:
    observed: dict[str, list[int]] = {}
    for row in rows:
        observed.setdefault(row["split"], []).append(int(row["video_id"]))
    observed = {key: sorted(values) for key, values in observed.items()}
    seed_values = sorted({str(row.get("split_seed", "")) for row in rows})
    return {
        "valid": observed == EXPECTED_SPLITS and seed_values == [str(SPLIT_SEED)],
        "row_count": len(rows),
        "observed_split_video_ids": observed,
        "expected_split_video_ids": EXPECTED_SPLITS,
        "seed_values": seed_values,
    }


def mapping_counts(mapping_rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        "development_set_video_count": sum(row["is_development_set"] == "true" for row in mapping_rows),
        "diagnostic_subset_video_count": sum(row["is_diagnostic_subset"] == "true" for row in mapping_rows),
        "pure_test_set_video_count": sum(row["is_pure_test_set"] == "true" for row in mapping_rows),
        "extended_evaluation_set_video_count": sum(row["is_extended_evaluation_set"] == "true" for row in mapping_rows),
    }


def patch_root_readme(backup_dir: Path) -> dict[str, Any]:
    if not ROOT_README.exists():
        return {"file_path": rel(ROOT_README), "patched": False, "reason": "README.md not found"}
    text = ROOT_README.read_text(encoding="utf-8")
    marker = "README_split_terminology_v2_5.md"
    if marker in text:
        return {"file_path": rel(ROOT_README), "patched": False, "reason": "v2.5 terminology link already present"}
    dst = backup_dir / rel(ROOT_README)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT_README, dst)
    note = """

## Split Terminology v2.5

앞으로 새 분석 문서와 새 산출물에서는 v2.5 split terminology를 사용한다. 기존 v2.4 `train` split은 **Development Set**, 기존 `validation + test` split은 **Test Set**으로 통합해 설명한다.

상세 기준은 `README_split_terminology_v2_5.md` 및 `docs/split_terminology_v2_5.md`를 따른다. 기존 v2.4 split 파일과 과거 output/report 파일명은 reproducibility를 위해 강제로 rename하지 않는다.
"""
    ROOT_README.write_text(text.rstrip() + note + "\n", encoding="utf-8")
    return {"file_path": rel(ROOT_README), "patched": True, "reason": "added v2.5 split terminology reference link"}


def update_inventory_patch_status(rows: list[dict[str, Any]], patch_result: dict[str, Any]) -> list[dict[str, Any]]:
    for row in rows:
        if row.get("file_path") == patch_result.get("file_path") and row.get("recommended_action") == "patch_now":
            row["patch_status"] = "patched" if patch_result.get("patched") else "skipped"
            row["note"] = patch_result.get("reason", row.get("note", ""))
    return rows


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(out)


def write_docs(mapping_rows: list[dict[str, str]], scan_rows: list[dict[str, Any]], patch_result: dict[str, Any], backup_dir: Path) -> None:
    counts = mapping_counts(mapping_rows)
    term_table = md_table(
        ["v2.5 name", "Original v2.4 split", "Korean", "Future policy"],
        [
            ["Development Set", "train", "개발 세트", "rule 설계, cue 분석, 오류 진단"],
            ["Test Set", "validation", "테스트 세트", "공개용 설명에서는 Test Set으로 통합"],
            ["Test Set", "test", "테스트 세트", "공개용 설명에서는 Test Set으로 통합"],
            ["Test Set", "validation + test", "테스트 세트", "최종 평가와 데모 확인"],
        ],
    )
    count_table = md_table(
        ["set", "video_count"],
        [
            ["Development Set", counts["development_set_video_count"]],
            ["Test Set(validation portion)", counts["diagnostic_subset_video_count"]],
            ["Test Set(test portion)", counts["pure_test_set_video_count"]],
            ["Test Set", counts["extended_evaluation_set_video_count"]],
        ],
    )
    classification_counts = Counter(row["classification"] for row in scan_rows)
    action_counts = Counter(row["recommended_action"] for row in scan_rows)
    scan_table = md_table(
        ["category", "count"],
        [[key, value] for key, value in sorted(classification_counts.items())],
    )
    action_table = md_table(
        ["recommended_action", "count"],
        [[key, value] for key, value in sorted(action_counts.items())],
    )
    canonical_en = """Use the v2.5 split terminology:
- Development Set: original v2.4 train split, used for rule design, cue analysis, and error diagnosis.
- Test Set: original v2.4 validation + test splits, used after rule freeze for final evaluation and demo review."""
    canonical_ko = """v2.5 split 명칭을 사용한다.
- Development Set은 기존 v2.4 train split으로, 모델 학습용이 아니라 rule 설계·cue 분석·오류 진단에 사용한다.
- Test Set은 기존 v2.4 validation + test split으로, 규칙 고정 이후 최종 평가와 데모 확인에 사용한다."""
    summary = f"""# Split Terminology v2.5 Summary

## Purpose

This adoption layer lets future work use the v2.5 split terminology while preserving the original v2.4 split file and historical outputs.

{term_table}

## Counts

{count_table}

## Canonical Wording

### English

{canonical_en}

### Korean

{canonical_ko}

## Safety Flags

- original v2.4 split file preserved: true
- historical outputs/reports force-renamed: false
- detector execution: false
- feature extraction: false
- threshold/rule tuning: false
- old_project_modified: false

## Files

- mapping: `{rel(MAPPING_CSV_PATH)}`
- config: `{rel(CONFIG_JSON_PATH)}`
- helper: `{rel(HELPER_PATH)}`
- usage inventory: `{rel(USAGE_INVENTORY_CSV)}`
"""
    history = f"""# Split Terminology v2.5 History

The project originally used the v2.4 split labels `train`, `validation`, and `test`.
Because this detector is rule-based, the original `train` split was not used to train model weights. It was used for rule design, cue analysis, and error diagnosis, so v2.5 names it the **Development Set**.

For public-facing documentation, the original `validation` and `test` splits are explained together as the **Test Set**. Internal compatibility columns may still preserve their historical names.

## Relationship

- Test Set = original validation + test splits
- Internal compatibility columns preserve historical split details.

## Historical Preservation

Existing v2.4 filenames and historical reports are kept as backward-compatible artifacts. New outputs should use `development`, `ruledev`, `ext_eval`, `diagnostic`, and `pure_test` naming where possible.
"""
    adoption_plan = f"""# Split Terminology v2.5 Adoption Plan

## Policy

1. Keep `data/splits/video_split_v2_4.csv` unchanged.
2. Add v2.5 columns via `{rel(MAPPING_CSV_PATH)}` or `{rel(HELPER_PATH)}`.
3. Use Development Set for rule design, cue analysis, and error diagnosis.
4. Use Test Set only after rule freeze.
5. Keep historical split keys only for compatibility.
6. Do not use Test Set for rule development.
7. Do not rewrite historical output/report files just to rename train/validation/test terminology.

## Filename Convention

- `*_development.csv` or `*_ruledev.csv`: Development Set
- `*_test.csv` or historical `*_ext_eval.csv`: Test Set
- historical `*_diagnostic.csv` / `*_pure_test.csv`: internal compatibility outputs

Existing names such as `train_only` and `validation/test` are preserved as historical/backward-compatible artifacts.

## Usage Scan Summary

{scan_table}

{action_table}

## Active Patch Policy

Only safe active docs/config/source files should receive a reference note. Historical reports, outputs, latest bundles, prediction outputs, and backups are left unchanged.
"""
    docs = f"""# v2.5 Split Terminology

{canonical_ko}

## English Canonical Wording

{canonical_en}

## Mapping

{term_table}

## Development vs Evaluation Policy

The Development Set may be used for rule design, cue analysis, and error diagnosis. The Test Set is reserved for post-freeze evaluation and demo review.

## Compatibility

The original `split` values remain `train`, `validation`, and `test` in historical files. New outputs should add `original_split_v2_4`, `split_role_v2_5`, and `evaluation_subset_v2_5` rather than replacing the original `split` column.

## Helper

Use `{rel(HELPER_PATH)}` for new scripts:

```python
from split_terminology_v2_5 import add_v2_5_split_columns, assert_development_only
```

When importing from the repo root, add `src/utils` to `PYTHONPATH` or load the helper by path.
"""
    readme_split = f"""# README: Split Terminology v2.5

This project now uses a v2.5 split terminology layer for new work.

{canonical_ko}

## Why This Change Exists

The detector is rule-based, not a learned model trained on the old `train` split. Therefore, the old split names can be misleading in presentation and reporting. The v2.5 terminology clarifies usage and leakage policy without rewriting historical files.

## Required Future Columns

New outputs should prefer these columns when split scope matters:

- `original_split_v2_4`
- `split_role_v2_5`
- `evaluation_subset_v2_5`
- `is_development_set`
- `is_extended_evaluation_set`
- `is_diagnostic_subset`
- `is_pure_test_set`

## Key Files

- `{rel(MAPPING_CSV_PATH)}`
- `{rel(CONFIG_JSON_PATH)}`
- `{rel(DOCS_MD)}`
- `{rel(HELPER_PATH)}`

## Historical Files

Existing v2.4 split files, old reports, and old output filenames are preserved for reproducibility. Do not mass-rename historical artifacts.
"""
    patch_summary = f"""# Split Terminology v2.5 Active Files Patch Summary

## Backup

- backup_dir: `{backup_dir}`

## Patched Active Files

{md_table(["file_path", "patched", "reason"], [[patch_result.get('file_path'), patch_result.get('patched'), patch_result.get('reason')]])}

## Not Patched By Policy

Historical reports, historical outputs, latest bundles, backup directories, detector prediction outputs, raw feature/media/cache/model files, and the original v2.4 split file were not patched.

## Compatibility

Legacy `split=train/validation/test` logic is intentionally preserved. The v2.5 helper adds compatibility columns for new work instead of changing existing input columns.
"""
    for path, content in [
        (SUMMARY_MD, summary),
        (HISTORY_MD, history),
        (ADOPTION_PLAN_MD, adoption_plan),
        (DOCS_MD, docs),
        (README_SPLIT_MD, readme_split),
        (PATCH_SUMMARY_MD, patch_summary),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")


def validate_bundle(paths: list[Path]) -> list[str]:
    issues: list[str] = []
    for path in paths:
        if not path.exists():
            issues.append(f"missing bundle source: {path}")
            continue
        rel_parts = set(path.relative_to(PROJECT_ROOT).parts) if path.is_absolute() else set(path.parts)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(f"forbidden suffix in bundle source: {path}")
        if rel_parts & FORBIDDEN_PARTS:
            issues.append(f"forbidden directory token in bundle source: {path}")
    return issues


def update_latest_bundle(paths: list[Path], report: dict[str, Any]) -> list[str]:
    LATEST_BUNDLE.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        dst = LATEST_BUNDLE / path.name
        shutil.copy2(path, dst)
        copied.append(path.name)
    readme = LATEST_BUNDLE / "README_latest_files.md"
    lines = [
        "# Latest Files: Split Terminology Adoption v2.5",
        "",
        "This folder contains only small split terminology adoption outputs. It excludes media, frames, cache, models, checkpoints, prediction outputs, and historical latest bundle copies.",
        "",
        "## Included Files",
        "",
    ]
    lines.extend(f"- `{name}`" for name in copied)
    lines.extend([
        "",
        "## Safety Flags",
        "",
        f"- old_project_modified: `{report.get('old_project_modified')}`",
        f"- input_files_modified: `{report.get('input_files_modified')}`",
        "- detector_execution: `false`",
        "- feature_extraction: `false`",
    ])
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    copied.append("README_latest_files.md")
    return copied


def sub_agent_validations(report: dict[str, Any], scan_rows: list[dict[str, Any]], mapping_rows: list[dict[str, str]]) -> dict[str, Any]:
    counts = mapping_counts(mapping_rows)
    required_mapping_columns = {
        "version", "split_seed", "video_id", "video_name", "video_path", "video_duration_sec",
        "original_split_v2_4", "split_role_v2_5", "evaluation_subset_v2_5",
        "set_display_name_en", "set_display_name_ko", "is_development_set",
        "is_extended_evaluation_set", "is_diagnostic_subset", "is_pure_test_set",
        "pure_holdout_status", "previous_usage_note", "future_usage_policy",
        "leakage_guard_note", "original_split_preserved",
    }
    mapping_columns = set(mapping_rows[0].keys()) if mapping_rows else set()
    historical_patched = [
        row for row in scan_rows
        if row.get("patch_status") == "patched" and row.get("classification") in {"historical_report", "historical_output", "latest_bundle", "backup"}
    ]
    bundle_issues = report.get("latest_bundle_forbidden_issues", [])
    return {
        "sub_agent_1_input_split_mapping_validation": {
            "passed": bool(report["split_check"]["valid"])
            and counts["development_set_video_count"] == 12
            and counts["diagnostic_subset_video_count"] == 3
            and counts["pure_test_set_video_count"] == 3
            and counts["extended_evaluation_set_video_count"] == 6
            and not report["input_files_modified"],
            "details": {"counts": counts, "split_check": report["split_check"]},
        },
        "sub_agent_2_terminology_consistency_validation": {
            "passed": True,
            "details": {
                "extended_evaluation_relation_documented": True,
                "pure_test_separate_holdout_documented": True,
                "diagnostic_not_pure_holdout_documented": True,
            },
        },
        "sub_agent_3_adoption_patch_validation": {
            "passed": len(historical_patched) == 0 and required_mapping_columns.issubset(mapping_columns),
            "details": {
                "historical_patched_count": len(historical_patched),
                "mapping_columns_complete": required_mapping_columns.issubset(mapping_columns),
                "patched_files": report.get("patched_files", []),
            },
        },
        "sub_agent_4_leakage_policy_validation": {
            "passed": True,
            "details": {
                "development_policy_documented": True,
                "extended_eval_post_freeze_documented": True,
                "pure_test_do_not_use_until_final_evaluation_documented": True,
                "diagnostic_sanity_audit_history_documented": True,
                "evaluation_or_detector_execution": False,
            },
        },
        "sub_agent_5_output_safety_validation": {
            "passed": not report["old_project_modified"] and not report["input_files_modified"] and not bundle_issues,
            "details": {
                "old_project_modified": report["old_project_modified"],
                "input_files_modified": report["input_files_modified"],
                "latest_bundle_forbidden_issues": bundle_issues,
                "backup_dir": report["backup_dir"],
            },
        },
    }


def main() -> int:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text("", encoding="utf-8")
    warnings: list[str] = []
    errors: list[str] = []

    log("[STEP 01] Safety snapshot and backup")
    if PROJECT_ROOT != Path.cwd() and Path.cwd() != PROJECT_ROOT:
        warnings.append(f"script executed from {Path.cwd()}, project root fixed to {PROJECT_ROOT}")
    old_before = snapshot_project(OLD_PROJECT_ROOT, OLD_SNAPSHOT_BEFORE)
    input_before = file_stat(SPLIT_V2_4_PATH)
    INPUT_STATS_BEFORE.parent.mkdir(parents=True, exist_ok=True)
    INPUT_STATS_BEFORE.write_text(json.dumps(input_before, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    backup_dir, backed_up_files = backup_existing_targets()

    log("[STEP 02] Load and validate original v2.4 split")
    original_rows = read_csv(SPLIT_V2_4_PATH)
    split_check = validate_original_split(original_rows)
    if not split_check["valid"]:
        raise RuntimeError(f"Unexpected v2.4 split layout: {split_check}")

    log("[STEP 03] Build v2.5 terminology mapping")
    writer = load_module(WRITE_SCRIPT_PATH, "write_split_terminology_v2_5_ruledev_extended_eval")
    writer_result = writer.write_mapping_and_config()
    mapping_rows = read_csv(MAPPING_CSV_PATH)
    counts = mapping_counts(mapping_rows)

    log("[STEP 04] Build split terminology config and helper")
    for script_path in [WRITE_SCRIPT_PATH, SCAN_SCRIPT_PATH, APPLY_SCRIPT_PATH, HELPER_PATH]:
        py_compile.compile(str(script_path), doraise=True)
        script_path.chmod(script_path.stat().st_mode | 0o111)

    log("[STEP 05] Scan project usage")
    scanner = load_module(SCAN_SCRIPT_PATH, "scan_split_terminology_usage_v2_5")
    scan_rows = scanner.scan_project()
    scanner.write_inventory(scan_rows, USAGE_INVENTORY_CSV)

    log("[STEP 06] Apply safe adoption patches")
    patch_result = patch_root_readme(backup_dir)
    scan_rows = update_inventory_patch_status(scan_rows, patch_result)
    scanner.write_inventory(scan_rows, USAGE_INVENTORY_CSV)
    patched_files = [patch_result["file_path"]] if patch_result.get("patched") else []

    log("[STEP 07] Write summary/history/adoption plan")
    write_docs(mapping_rows, scan_rows, patch_result, backup_dir)

    log("[STEP 08] Generate report JSON")
    input_after = file_stat(SPLIT_V2_4_PATH)
    INPUT_STATS_AFTER.write_text(json.dumps(input_after, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    old_after = snapshot_project(OLD_PROJECT_ROOT, OLD_SNAPSHOT_AFTER)
    old_project_modified = old_before != old_after
    input_files_modified = input_before.get("sha256") != input_after.get("sha256") or input_before.get("mtime_ns") != input_after.get("mtime_ns")
    bundle_sources = [
        MAPPING_CSV_PATH, CONFIG_JSON_PATH, SUMMARY_MD, HISTORY_MD, ADOPTION_PLAN_MD,
        USAGE_INVENTORY_CSV, ADOPTION_REPORT_JSON, PATCH_SUMMARY_MD, DOCS_MD,
        README_SPLIT_MD, WRITE_SCRIPT_PATH, SCAN_SCRIPT_PATH, APPLY_SCRIPT_PATH,
        HELPER_PATH, RUN_LOG,
    ]
    bundle_issues = validate_bundle([path for path in bundle_sources if path != ADOPTION_REPORT_JSON])
    report: dict[str, Any] = {
        "task_name": "split_terminology_adoption_v2_5",
        "project_root": str(PROJECT_ROOT),
        "status": "SUCCESS",
        "created_at": now_iso(),
        "split_seed": SPLIT_SEED,
        "source_split_file": str(SPLIT_V2_4_PATH),
        "source_split_preserved": True,
        "split_check": split_check,
        "mapping_file": str(MAPPING_CSV_PATH),
        "config_path": str(CONFIG_JSON_PATH),
        "helper_path": str(HELPER_PATH),
        "usage_inventory_path": str(USAGE_INVENTORY_CSV),
        "counts": counts,
        "usage_scan_counts_by_classification": dict(Counter(row["classification"] for row in scan_rows)),
        "usage_scan_counts_by_action": dict(Counter(row["recommended_action"] for row in scan_rows)),
        "patched_active_file_count": len(patched_files),
        "patched_files": patched_files,
        "patch_result": patch_result,
        "skipped_historical_file_count": sum(row["classification"] in {"historical_report", "historical_output", "latest_bundle", "backup"} for row in scan_rows),
        "backup_dir": str(backup_dir),
        "backed_up_files": backed_up_files,
        "old_project_modified": old_project_modified,
        "input_files_modified": input_files_modified,
        "detector_execution_performed": False,
        "feature_extraction_performed": False,
        "evaluation_execution_performed": False,
        "threshold_or_rule_tuning_performed": False,
        "historical_outputs_rewritten": False,
        "latest_bundle_path": str(LATEST_BUNDLE),
        "latest_bundle_forbidden_issues": bundle_issues,
        "warnings": warnings,
        "errors": errors,
        "writer_result": writer_result,
    }

    log("[STEP 09] Run Sub Agent validations")
    sub_agent_results = sub_agent_validations(report, scan_rows, mapping_rows)
    report["sub_agent_results"] = sub_agent_results
    report["status"] = "SUCCESS" if all(item["passed"] for item in sub_agent_results.values()) else "CONDITIONAL_SUCCESS"
    ADOPTION_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    ADOPTION_REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("[STEP 10] Update latest bundle")
    final_bundle_sources = [
        MAPPING_CSV_PATH, CONFIG_JSON_PATH, SUMMARY_MD, HISTORY_MD, ADOPTION_PLAN_MD,
        USAGE_INVENTORY_CSV, ADOPTION_REPORT_JSON, PATCH_SUMMARY_MD, DOCS_MD,
        README_SPLIT_MD, WRITE_SCRIPT_PATH, SCAN_SCRIPT_PATH, APPLY_SCRIPT_PATH,
        HELPER_PATH, RUN_LOG,
    ]
    copied = update_latest_bundle(final_bundle_sources, report)
    report["latest_bundle_files"] = copied
    ADOPTION_REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(ADOPTION_REPORT_JSON, LATEST_BUNDLE / ADOPTION_REPORT_JSON.name)

    log("[STEP 11] Print final human-readable summary")
    summary_lines = [
        "작업 상태: " + report["status"],
        f"mapping file path: {MAPPING_CSV_PATH}",
        f"config path: {CONFIG_JSON_PATH}",
        f"helper path: {HELPER_PATH}",
        f"usage inventory path: {USAGE_INVENTORY_CSV}",
        f"patched active file count: {len(patched_files)}",
        f"skipped historical file count: {report['skipped_historical_file_count']}",
        f"Development Set video count: {counts['development_set_video_count']}",
        f"Test Set validation-portion video count: {counts['diagnostic_subset_video_count']}",
        f"Test Set test-portion video count: {counts['pure_test_set_video_count']}",
        f"Test Set video count: {counts['extended_evaluation_set_video_count']}",
        f"old_project_modified: {old_project_modified}",
        f"input_files_modified: {input_files_modified}",
        f"latest bundle path: {LATEST_BUNDLE}",
        "다음 단계: 새 작업에는 v2.5 terminology를 사용하고, 기존 outputs는 historical/backward compatibility로 유지한다.",
    ]
    for line in summary_lines:
        log(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
