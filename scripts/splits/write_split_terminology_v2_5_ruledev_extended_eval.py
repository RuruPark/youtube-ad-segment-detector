#!/usr/bin/env python3
"""Create the v2.5 split terminology mapping/config layer.

This is a compatibility/adoption layer. It reads the original v2.4 split file
and writes new v2.5 role columns without modifying the original split file.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(".")
SPLIT_V2_4_PATH = PROJECT_ROOT / "data/splits/video_split_v2_4.csv"
MAPPING_CSV_PATH = PROJECT_ROOT / "data/splits/video_split_v2_5_ruledev_extended_eval.csv"
CONFIG_JSON_PATH = PROJECT_ROOT / "configs/splits/split_terminology_v2_5_ruledev_extended_eval.json"
SPLIT_SEED = 20240524
VERSION = "v2_5_ruledev_extended_eval"
EXPECTED_SPLITS = {
    "train": [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
    "validation": [3, 7, 18],
    "test": [4, 16, 17],
}

MAPPING_COLUMNS = [
    "version",
    "split_seed",
    "video_id",
    "video_name",
    "video_path",
    "video_duration_sec",
    "original_split_v2_4",
    "split_role_v2_5",
    "evaluation_subset_v2_5",
    "set_display_name_en",
    "set_display_name_ko",
    "is_development_set",
    "is_extended_evaluation_set",
    "is_diagnostic_subset",
    "is_pure_test_set",
    "pure_holdout_status",
    "previous_usage_note",
    "future_usage_policy",
    "leakage_guard_note",
    "original_split_preserved",
]

ROLE_BY_SPLIT: dict[str, dict[str, Any]] = {
    "train": {
        "split_role_v2_5": "development",
        "evaluation_subset_v2_5": "none",
        "set_display_name_en": "Development Set",
        "set_display_name_ko": "개발 세트",
        "is_development_set": True,
        "is_extended_evaluation_set": False,
        "is_diagnostic_subset": False,
        "is_pure_test_set": False,
        "pure_holdout_status": "not_holdout",
        "previous_usage_note": "rule_design_cue_analysis_error_diagnosis",
        "future_usage_policy": "rule_development_only",
        "leakage_guard_note": "may_be_used_for_rule_design_and_cue_analysis",
    },
    "validation": {
        "split_role_v2_5": "extended_evaluation",
        "evaluation_subset_v2_5": "diagnostic_subset",
        "set_display_name_en": "Test Set",
        "set_display_name_ko": "테스트 세트",
        "is_development_set": False,
        "is_extended_evaluation_set": True,
        "is_diagnostic_subset": True,
        "is_pure_test_set": False,
        "pure_holdout_status": "not_pure_holdout_used_for_early_sanity_audit",
        "previous_usage_note": "early_detector_sanity_audit_only",
        "future_usage_policy": "use_after_rule_freeze_report_separately_from_pure_test",
        "leakage_guard_note": "not_for_rule_design_after_v2_5_freeze_policy",
    },
    "test": {
        "split_role_v2_5": "extended_evaluation",
        "evaluation_subset_v2_5": "pure_test",
        "set_display_name_en": "Test Set",
        "set_display_name_ko": "테스트 세트",
        "is_development_set": False,
        "is_extended_evaluation_set": True,
        "is_diagnostic_subset": False,
        "is_pure_test_set": True,
        "pure_holdout_status": "pure_holdout",
        "previous_usage_note": "protected_not_used_for_rule_design",
        "future_usage_policy": "final_pure_test_subset_report_separately",
        "leakage_guard_note": "do_not_use_until_final_evaluation",
    },
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


def as_csv_bool(value: bool) -> str:
    return "true" if value else "false"


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


def build_mapping(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for row in rows:
        original_split = row["split"].strip().lower()
        role = ROLE_BY_SPLIT[original_split]
        mapped.append(
            {
                "version": VERSION,
                "split_seed": SPLIT_SEED,
                "video_id": row.get("video_id", ""),
                "video_name": row.get("video_name", ""),
                "video_path": row.get("video_path", ""),
                "video_duration_sec": row.get("video_duration_sec", ""),
                "original_split_v2_4": original_split,
                "original_split_preserved": "true",
                **{key: as_csv_bool(value) if isinstance(value, bool) else value for key, value in role.items()},
            }
        )
    return sorted(mapped, key=lambda item: int(item["video_id"]))


def build_config(mapping_rows: list[dict[str, Any]], split_check: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(row["original_split_v2_4"] for row in mapping_rows)
    diagnostic_ids = [int(row["video_id"]) for row in mapping_rows if row["evaluation_subset_v2_5"] == "diagnostic_subset"]
    pure_test_ids = [int(row["video_id"]) for row in mapping_rows if row["evaluation_subset_v2_5"] == "pure_test"]
    development_ids = [int(row["video_id"]) for row in mapping_rows if row["split_role_v2_5"] == "development"]
    return {
        "task_name": "split_terminology_v2_5_ruledev_extended_eval",
        "version": VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "source_split_file": str(SPLIT_V2_4_PATH),
        "source_split_preserved": True,
        "split_seed": SPLIT_SEED,
        "row_count": len(mapping_rows),
        "original_split_counts": dict(counts),
        "definitions": {
            "Development Set": "original v2.4 train split, used for rule design, cue analysis, and error diagnosis.",
            "Test Set": "original v2.4 validation + test splits, used after rule freeze for final evaluation and demo review.",
        },
        "definitions_ko": {
            "Development Set": "기존 v2.4 train split으로, 규칙 설계와 cue 분석, 오류 진단에 사용한다.",
            "Test Set": "기존 v2.4 validation + test split으로, 규칙 고정 이후 최종 평가와 데모 확인에 사용한다.",
        },
        "relationships": {
            "public_split_policy": "Public-facing documents use Development Set and Test Set.",
            "historical_internal_columns_preserved": True,
        },
        "fixed_video_ids": {
            "development_set": development_ids,
            "diagnostic_subset": diagnostic_ids,
            "pure_test_set": pure_test_ids,
            "extended_evaluation_set": diagnostic_ids + pure_test_ids,
        },
        "usage_policy": {
            "development_set": "rule_development_only",
            "diagnostic_subset": "use_after_rule_freeze_report_separately_from_pure_test",
            "pure_test_set": "final_pure_test_subset_report_separately",
            "extended_evaluation_set": "post_freeze_broader_evaluation_only_with_pure_test_breakout",
        },
        "filename_convention_policy": {
            "development_or_ruledev": "Development Set",
            "ext_eval": "Test Set",
            "diagnostic": "Test Set",
            "pure_test": "Test Set",
            "historical_names_preserved": True,
        },
        "split_check": split_check,
    }


def write_mapping_and_config() -> dict[str, Any]:
    rows = read_csv(SPLIT_V2_4_PATH)
    split_check = validate_original_split(rows)
    if not split_check["valid"]:
        raise RuntimeError(f"Unexpected v2.4 split layout: {split_check}")
    mapping_rows = build_mapping(rows)
    write_csv(MAPPING_CSV_PATH, mapping_rows, MAPPING_COLUMNS)
    config = build_config(mapping_rows, split_check)
    CONFIG_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_JSON_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "mapping_csv": str(MAPPING_CSV_PATH),
        "config_json": str(CONFIG_JSON_PATH),
        "mapping_row_count": len(mapping_rows),
        "split_check": split_check,
    }


def main() -> int:
    result = write_mapping_and_config()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
