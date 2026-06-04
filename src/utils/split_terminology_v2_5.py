"""v2.4 split 값을 공개용 Development Set/Test Set 표기로 보강한다.

원본 split 값인 ``train``, ``validation``, ``test``는 그대로 유지한다.
공개용 문서와 표시명에서는 ``train``을 Development Set으로, ``validation``과
``test``를 Test Set으로 통합해 설명한다. 내부 boolean key와 helper 이름은
기존 산출물 호환을 위해 유지한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEVELOPMENT_SET_NAME = "Development Set"
EXTENDED_EVALUATION_SET_NAME = "Test Set"
DIAGNOSTIC_SUBSET_NAME = "Test Set"
PURE_TEST_SET_NAME = "Test Set"

SPLIT_SEED_V2_4 = 20240524
VERSION = "v2_5_ruledev_extended_eval"

ROLE_BY_ORIGINAL_SPLIT: dict[str, dict[str, Any]] = {
    "train": {
        "split_role_v2_5": "development",
        "evaluation_subset_v2_5": "none",
        "set_display_name_en": DEVELOPMENT_SET_NAME,
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
        "set_display_name_en": DIAGNOSTIC_SUBSET_NAME,
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
        "set_display_name_en": PURE_TEST_SET_NAME,
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


def _extract_original_split(row_or_split: Any) -> str:
    if isinstance(row_or_split, str):
        return row_or_split.strip().lower()
    if isinstance(row_or_split, Mapping):
        for key in ("original_split_v2_4", "split"):
            value = row_or_split.get(key)
            if value is not None:
                return str(value).strip().lower()
    if hasattr(row_or_split, "get"):
        for key in ("original_split_v2_4", "split"):
            value = row_or_split.get(key)
            if value is not None:
                return str(value).strip().lower()
    raise ValueError(f"Cannot infer original v2.4 split from {type(row_or_split)!r}")


def original_split_to_v2_5_role(split: str) -> dict[str, Any]:
    """기존 v2.4 split 값에 대응하는 v2.5 역할 정보를 반환한다."""
    normalized = _extract_original_split(split)
    if normalized not in ROLE_BY_ORIGINAL_SPLIT:
        raise ValueError(f"Unknown original v2.4 split: {split!r}")
    return dict(ROLE_BY_ORIGINAL_SPLIT[normalized])


def is_development_set(row_or_split: Any) -> bool:
    return bool(original_split_to_v2_5_role(_extract_original_split(row_or_split))["is_development_set"])


def is_extended_evaluation_set(row_or_split: Any) -> bool:
    return bool(original_split_to_v2_5_role(_extract_original_split(row_or_split))["is_extended_evaluation_set"])


def is_diagnostic_subset(row_or_split: Any) -> bool:
    return bool(original_split_to_v2_5_role(_extract_original_split(row_or_split))["is_diagnostic_subset"])


def is_pure_test_set(row_or_split: Any) -> bool:
    return bool(original_split_to_v2_5_role(_extract_original_split(row_or_split))["is_pure_test_set"])


def add_v2_5_split_columns(df: Any, original_split_col: str = "split") -> Any:
    """Return a DataFrame-like copy with v2.5 terminology columns added.

    This function is intentionally pandas-light: it works with a pandas
    DataFrame without importing pandas at module import time.
    """
    out = df.copy()
    roles = [original_split_to_v2_5_role(value) for value in out[original_split_col]]
    for column in (
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
    ):
        out[column] = [role[column] for role in roles]
    out["original_split_v2_4"] = list(out[original_split_col])
    out["original_split_preserved"] = True
    return out


def assert_no_pure_test_for_development(df: Any) -> None:
    if "is_pure_test_set" in df:
        pure_count = int(df["is_pure_test_set"].astype(bool).sum())
    else:
        pure_count = sum(is_pure_test_set(value) for value in df["split"])
    if pure_count:
        raise AssertionError(f"Test Set rows are not allowed here: {pure_count}")


def assert_development_only(df: Any) -> None:
    if "is_development_set" in df:
        invalid_count = int((~df["is_development_set"].astype(bool)).sum())
    else:
        invalid_count = sum(not is_development_set(value) for value in df["split"])
    if invalid_count:
        raise AssertionError(f"Expected Development Set only, found non-development rows: {invalid_count}")


def assert_no_extended_eval_rows(df: Any) -> None:
    if "is_extended_evaluation_set" in df:
        extended_count = int(df["is_extended_evaluation_set"].astype(bool).sum())
    else:
        extended_count = sum(is_extended_evaluation_set(value) for value in df["split"])
    if extended_count:
        raise AssertionError(f"Test Set rows are not allowed here: {extended_count}")
