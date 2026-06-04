#!/usr/bin/env python3
"""수동 title mapping을 사용해 v2.2 window label과 segment를 갱신한다."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")

INPUT_CLEAN_LABEL = PROJECT_ROOT / "data/labels/clean_ad_labels_v0.csv"
INPUT_REVIEW_XLSX = PROJECT_ROOT / "data/labels/clean_ad_labels_v0_review.xlsx"
INPUT_MANIFEST_V2 = PROJECT_ROOT / "data/video_metadata/video_manifest_v2.csv"
INPUT_WINDOW_GRID_V2 = PROJECT_ROOT / "data/windows/window_grid_5s_v2.csv"
INPUT_WINDOW_LABELS_V2_1 = PROJECT_ROOT / "data/windows/window_labels_5s_v2_1.csv"
INPUT_AD_SEGMENTS_V2_1 = PROJECT_ROOT / "data/segments/ad_interval_segments_v2_1.csv"
INPUT_CONTEXT_SEGMENTS_V2_1 = PROJECT_ROOT / "data/segments/ad_context_segments_v2_1.csv"
INPUT_COMBINED_SEGMENTS_V2_1 = PROJECT_ROOT / "data/segments/label_interval_context_segments_v2_1.csv"
INPUT_AMBIGUOUS_AUDIT_V2_1 = PROJECT_ROOT / "data/video_metadata/ambiguous_title_mapping_audit_v2_1.csv"
INPUT_REPORT_V2_1 = PROJECT_ROOT / "reports/fix_label_mapping_and_refresh_window_segments_v2_1_report.json"

MANUAL_MAPPING_CANDIDATES = [
    PROJECT_ROOT / "data/video_metadata/manual_video_title_mapping_v2_1.csv",
    PROJECT_ROOT / "data/video_metadata/manual_video_title_mapping_v2_1_template.csv",
    PROJECT_ROOT / "data/video_metadata/manual_video_title_mapping_v2_2.csv",
]

OUTPUT_LABEL = PROJECT_ROOT / "data/labels/clean_ad_labels_v0_scope_refreshed_v2_2.csv"
OUTPUT_MANIFEST = PROJECT_ROOT / "data/video_metadata/video_manifest_v2_2.csv"
OUTPUT_MANUAL_AUDIT = PROJECT_ROOT / "data/video_metadata/manual_mapping_applied_audit_v2_2.csv"
OUTPUT_WINDOW_LABELS = PROJECT_ROOT / "data/windows/window_labels_5s_v2_2.csv"
OUTPUT_AD_SEGMENTS = PROJECT_ROOT / "data/segments/ad_interval_segments_v2_2.csv"
OUTPUT_CONTEXT_SEGMENTS = PROJECT_ROOT / "data/segments/ad_context_segments_v2_2.csv"
OUTPUT_COMBINED_SEGMENTS = PROJECT_ROOT / "data/segments/label_interval_context_segments_v2_2.csv"
OUTPUT_CLIPPED_AUDIT = PROJECT_ROOT / "data/segments/clipped_segment_audit_v2_2.csv"
REPORT_PATH = PROJECT_ROOT / "reports/refresh_window_segments_v2_2_with_manual_mapping_report.json"
SUMMARY_PATH = PROJECT_ROOT / "reports/refresh_window_segments_v2_2_with_manual_mapping_summary.md"
LOG_PATH = PROJECT_ROOT / "logs/refresh_window_segments_v2_2_with_manual_mapping_run_log.txt"
README_PATH = PROJECT_ROOT / "README.md"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt"
LATEST_README = LATEST_DIR / "README_latest_files.md"
SCRIPT_PATH = PROJECT_ROOT / "scripts/labels/refresh_window_segments_v2_2_with_manual_mapping.py"
CHECK_ENV_SCRIPT = PROJECT_ROOT / "scripts/utils/check_cv_environment.py"

CONTEXT_WINDOW_SEC = 10.0
RUN_LOG: list[str] = []


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    RUN_LOG.append(f"[{stamp}] {message}")
    print(message)


def step(number: int, message: str) -> None:
    log(f"[STEP {number}/9] {message}")


def ensure_inside_project(path: Path) -> Path:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and not str(resolved).startswith(str(root) + os.sep):
        raise RuntimeError(f"Refusing to write outside project root: {resolved}")
    return path


def ensure_dirs() -> None:
    for path in [
        PROJECT_ROOT / "data/labels",
        PROJECT_ROOT / "data/video_metadata",
        PROJECT_ROOT / "data/windows",
        PROJECT_ROOT / "data/segments",
        PROJECT_ROOT / "reports",
        PROJECT_ROOT / "logs",
        LATEST_DIR,
    ]:
        ensure_inside_project(path).mkdir(parents=True, exist_ok=True)


def clean_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_title(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.strip().casefold()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^0-9a-z가-힣\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else None


def fmt_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        rounded = round(value, 6)
        if abs(rounded - round(rounded)) < 1e-9:
            return int(round(rounded))
        return rounded
    return value


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def label_valid_mask(df: pd.DataFrame) -> pd.Series:
    if "label_valid" not in df.columns:
        start = df["ad_start_sec"].map(to_float) if "ad_start_sec" in df.columns else pd.Series([None] * len(df))
        end = df["ad_end_sec"].map(to_float) if "ad_end_sec" in df.columns else pd.Series([None] * len(df))
        return start.notna() & end.notna() & (start < end)
    return df["label_valid"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_inside_project(path)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: fmt_value(row.get(column, "")) for column in columns})


def verify_cv_environment() -> tuple[bool, str, list[Any]]:
    warnings: list[Any] = []
    executable = sys.executable
    in_cv = "/envs/cv/" in executable or executable.endswith("/envs/cv/bin/python") or executable.endswith("/envs/cv/bin/python3.10")
    if CHECK_ENV_SCRIPT.exists():
        cmd = ["conda", "run", "-n", "cv", "python", str(CHECK_ENV_SCRIPT)]
        log("Command: " + " ".join(cmd))
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        log("cv check stdout: " + result.stdout.strip().replace("\n", " | "))
        if result.stderr.strip():
            log("cv check stderr: " + result.stderr.strip().replace("\n", " | "))
        if result.returncode != 0:
            warnings.append("cv_environment_check_failed")
            return False, executable, warnings
    if not in_cv:
        warnings.append("current_python_executable_not_in_cv")
    return in_cv, executable, warnings


def find_manual_mapping_file(warnings: list[Any]) -> tuple[Path | None, pd.DataFrame, int]:
    existing = [path for path in MANUAL_MAPPING_CANDIDATES if path.exists()]
    if not existing:
        warnings.append("manual_mapping_file_missing")
        return None, pd.DataFrame(), 0
    existing.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in existing:
        df = pd.read_csv(path)
        required = {"video_filename", "selected_video_id", "selected_video_title", "resolution_note"}
        missing = sorted(required - set(df.columns))
        if missing:
            warnings.append({"manual_mapping_missing_columns": str(path), "missing": missing})
            continue
        filled = df[df["selected_video_id"].map(clean_value).ne("")].copy()
        if not filled.empty:
            return path, filled, len(filled)
    warnings.append("manual_mapping_files_exist_but_selected_video_id_empty")
    return existing[0], pd.DataFrame(columns=["video_filename", "selected_video_id", "selected_video_title", "resolution_note"]), 0


def refresh_labels_all_valid_yes(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    refreshed = clean_df.copy()
    valid_mask = label_valid_mask(refreshed)
    original = refreshed["is_abrupt_transition_ad"].map(clean_value) if "is_abrupt_transition_ad" in refreshed.columns else pd.Series([""] * len(refreshed))
    refreshed["is_abrupt_transition_ad_original"] = original
    refreshed_values: list[str] = []
    sources: list[str] = []
    valid_flags: list[str] = []
    warnings: list[str] = []
    for idx, is_valid in valid_mask.items():
        if is_valid:
            refreshed_values.append("yes")
            sources.append("all_valid_intervals_from_new_ad_labeling_assumed_abrupt_transition_ad")
            valid_flags.append("true")
            warnings.append("")
        else:
            value = clean_value(original.loc[idx])
            refreshed_values.append(value)
            sources.append("preserved_invalid_label_original_value")
            valid_flags.append(bool_text(value in {"yes", "no", "unclear", "not_reviewed"}))
            warnings.append("label_invalid_excluded_from_downstream")
    refreshed["is_abrupt_transition_ad_refreshed"] = refreshed_values
    refreshed["scope_value_source"] = sources
    refreshed["scope_value_valid"] = valid_flags
    refreshed["scope_warning"] = warnings
    refreshed["label_refresh_version"] = "v2_2"
    before = dict(sorted(Counter(original.tolist()).items()))
    after = dict(sorted(Counter(refreshed_values).items()))
    return refreshed, before, after


def candidate_ids_from_audit(audit_df: pd.DataFrame) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    if audit_df.empty:
        return result
    for _, row in audit_df.iterrows():
        filename = clean_value(row.get("video_filename"))
        candidate = clean_value(row.get("candidate_video_id"))
        if filename and candidate and candidate not in result[filename]:
            result[filename].append(candidate)
    return result


def apply_manual_mapping(
    manifest_df: pd.DataFrame,
    refreshed_labels: pd.DataFrame,
    manual_df: pd.DataFrame,
    manual_path: Path | None,
    audit_v2_1_df: pd.DataFrame,
    warnings: list[Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, dict[str, Any]], int, int, int]:
    candidate_ids = candidate_ids_from_audit(audit_v2_1_df)
    manual_by_filename = {
        clean_value(row.get("video_filename")): {
            "selected_video_id": clean_value(row.get("selected_video_id")),
            "selected_video_title": clean_value(row.get("selected_video_title")),
            "resolution_note": clean_value(row.get("resolution_note")),
        }
        for _, row in manual_df.iterrows()
        if clean_value(row.get("video_filename")) and clean_value(row.get("selected_video_id"))
    }

    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    mapping_by_filename: dict[str, dict[str, Any]] = {}
    applied_count = 0
    unresolved_count = 0
    ambiguous_before = int((manifest_df["title_match_status"].astype(str) == "ambiguous").sum()) if "title_match_status" in manifest_df.columns else 0

    for _, row in manifest_df.iterrows():
        row_dict = row.to_dict()
        filename = clean_value(row.get("video_filename"))
        previous_status = clean_value(row.get("title_match_status"))
        selected = manual_by_filename.get(filename)
        mapping_warning = ""
        mapping_status = ""
        mapped_video_id = clean_value(row.get("video_id"))
        mapped_video_title = clean_value(row.get("matched_label_video_title")) or clean_value(row.get("video_title"))

        if previous_status == "ambiguous":
            if selected:
                selected_id = selected["selected_video_id"]
                selected_title = selected["selected_video_title"]
                valid_selected = refreshed_labels[
                    refreshed_labels["video_id"].map(clean_value).eq(selected_id)
                    & refreshed_labels["video_title"].map(clean_value).eq(selected_title)
                    & label_valid_mask(refreshed_labels)
                ]
                if valid_selected.empty:
                    mapping_warning = "manual_selected_video_id_title_not_found_in_valid_labels"
                    mapping_status = "manual_mapping_invalid"
                    unresolved_count += 1
                else:
                    mapped_video_id = selected_id
                    mapped_video_title = selected_title
                    row_dict["video_id"] = selected_id
                    row_dict["video_title"] = selected_title
                    row_dict["matched_label_video_title"] = selected_title
                    row_dict["normalized_matched_label_video_title"] = normalize_title(selected_title)
                    row_dict["title_match_status"] = "manual_resolved"
                    row_dict["title_match_method"] = "manual_mapping_v2_2"
                    mapping_status = "manual_mapping_applied"
                    applied_count += 1
            else:
                mapping_status = "unresolved_ambiguous"
                mapping_warning = "manual_mapping_not_available_for_ambiguous_video"
                mapped_video_id = ""
                mapped_video_title = ""
                unresolved_count += 1
        elif previous_status in {"exact", "normalized", "manual_resolved"}:
            mapping_status = "resolved_existing_title_match"
        else:
            mapping_status = f"unresolved_{previous_status or 'unknown'}"
            mapping_warning = "video_without_label_mapping"
            mapped_video_id = ""
            mapped_video_title = ""

        row_dict["label_mapping_status"] = mapping_status
        row_dict["label_mapping_warning"] = mapping_warning
        row_dict["label_mapping_video_id"] = mapped_video_id
        row_dict["label_mapping_video_title"] = mapped_video_title
        row_dict["manifest_refresh_version"] = "v2_2"
        output_rows.append(row_dict)

        mapping_by_filename[filename] = {
            "mapped_video_id": mapped_video_id,
            "mapped_video_title": mapped_video_title,
            "label_mapping_status": mapping_status,
            "label_mapping_warning": mapping_warning,
            "video_filename": filename,
            "video_path": clean_value(row.get("video_path")),
            "file_stem": clean_value(row.get("file_stem")),
            "title_match_status": clean_value(row_dict.get("title_match_status")),
            "video_duration_sec": to_float(row.get("duration_sec")),
        }

        if previous_status == "ambiguous" or selected:
            audit_rows.append(
                {
                    "video_filename": filename,
                    "previous_match_status": previous_status,
                    "previous_candidate_video_ids": ";".join(candidate_ids.get(filename, [])),
                    "selected_video_id": selected["selected_video_id"] if selected else "",
                    "selected_video_title": selected["selected_video_title"] if selected else "",
                    "resolution_note": selected["resolution_note"] if selected else "",
                    "manual_mapping_source_file": str(manual_path) if manual_path else "",
                    "mapping_applied": bool_text(mapping_status == "manual_mapping_applied"),
                    "mapping_warning": mapping_warning,
                }
            )

    refreshed_manifest = pd.DataFrame(output_rows)
    ambiguous_after = int((refreshed_manifest["title_match_status"].astype(str) == "ambiguous").sum())
    return refreshed_manifest, audit_rows, mapping_by_filename, applied_count, ambiguous_before, max(ambiguous_after, unresolved_count)


def valid_label_rows(df: pd.DataFrame) -> pd.DataFrame:
    labels = df.copy()
    labels["_source_label_row_index"] = labels.index + 2
    labels["_video_id_text"] = labels["video_id"].map(clean_value)
    labels["_video_title_text"] = labels["video_title"].map(clean_value)
    labels["_ad_start_sec_num"] = labels["ad_start_sec"].map(to_float)
    labels["_ad_end_sec_num"] = labels["ad_end_sec"].map(to_float)
    labels["_label_valid_bool"] = label_valid_mask(labels)
    return labels[
        labels["_label_valid_bool"]
        & labels["_video_id_text"].ne("")
        & labels["_video_title_text"].ne("")
        & labels["_ad_start_sec_num"].notna()
        & labels["_ad_end_sec_num"].notna()
        & (labels["_ad_start_sec_num"] < labels["_ad_end_sec_num"])
    ].copy()


def labels_grouped_by_mapping(df: pd.DataFrame, mapping_by_filename: dict[str, dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    resolved_keys = {
        (payload["mapped_video_id"], payload["mapped_video_title"])
        for payload in mapping_by_filename.values()
        if payload.get("mapped_video_id") and payload.get("mapped_video_title")
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for _, row in valid_label_rows(df).iterrows():
        key = (clean_value(row.get("video_id")), clean_value(row.get("video_title")))
        if key not in resolved_keys:
            continue
        grouped[key].append(
            {
                "ad_interval_id": clean_value(row.get("ad_interval_id")),
                "ad_start_sec": float(row["_ad_start_sec_num"]),
                "ad_end_sec": float(row["_ad_end_sec_num"]),
                "scope": clean_value(row.get("is_abrupt_transition_ad_refreshed")),
                "scope_value_source": clean_value(row.get("scope_value_source")),
                "source_label_row_index": int(row["_source_label_row_index"]),
            }
        )
    return grouped


def window_scope(overlaps: list[dict[str, Any]]) -> str:
    if not overlaps:
        return "outside_ad"
    qualifying = sorted({item["scope"] for item in overlaps if item["overlap_ratio"] >= 0.5})
    if len(qualifying) != 1:
        return "mixed_or_boundary"
    value = qualifying[0]
    if value == "yes":
        return "abrupt_ad"
    if value == "no":
        return "out_of_scope_ad"
    if value == "unclear":
        return "uncertain_ad"
    if value == "not_reviewed":
        return "not_reviewed_ad"
    return "mixed_or_boundary"


def create_window_labels(
    window_grid: pd.DataFrame,
    labels: pd.DataFrame,
    manifest: pd.DataFrame,
    mapping_by_filename: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    grouped = labels_grouped_by_mapping(labels, mapping_by_filename)
    rows: list[dict[str, Any]] = []
    for _, window in window_grid.iterrows():
        filename = clean_value(window.get("video_filename"))
        mapping = mapping_by_filename.get(filename, {})
        mapped_id = mapping.get("mapped_video_id", "")
        mapped_title = mapping.get("mapped_video_title", "")
        win_start = float(window.get("window_start_sec"))
        win_end = float(window.get("window_end_sec"))
        win_duration = max(float(window.get("window_duration_sec")), 1e-9)
        overlaps: list[dict[str, Any]] = []
        for label in grouped.get((mapped_id, mapped_title), []):
            overlap_sec = max(0.0, min(win_end, label["ad_end_sec"]) - max(win_start, label["ad_start_sec"]))
            if overlap_sec > 0:
                overlaps.append({**label, "overlap_sec": overlap_sec, "overlap_ratio": overlap_sec / win_duration})
        overlap_total = min(sum(item["overlap_sec"] for item in overlaps), win_duration)
        max_overlap = max(overlaps, key=lambda item: item["overlap_sec"], default=None)
        scopes = sorted({item["scope"] for item in overlaps})
        base = window.to_dict()
        original_window_id = clean_value(base.get("window_id"))
        if mapped_id:
            base["video_id"] = mapped_id
            base["video_title"] = mapped_title
            base["window_id"] = f"{mapped_id}_W{int(base['window_index']):06d}"
        base.update(
            {
                "source_window_id_v2": original_window_id,
                "label_mapping_status": mapping.get("label_mapping_status", "mapping_missing"),
                "label_mapping_warning": mapping.get("label_mapping_warning", ""),
                "label_refresh_version": "v2_2",
                "window_label_refresh_source": "manual_mapping_and_all_valid_abrupt_scope_v2_2",
                "overlap_any_ad": bool_text(bool(overlaps)),
                "overlap_ad_sec": overlap_total,
                "overlap_ad_ratio": overlap_total / win_duration,
                "matched_ad_interval_ids": ";".join(item["ad_interval_id"] for item in overlaps),
                "max_overlap_ad_interval_id": max_overlap["ad_interval_id"] if max_overlap else "",
                "max_overlap_ad_sec": max_overlap["overlap_sec"] if max_overlap else 0,
                "max_overlap_ad_ratio": max_overlap["overlap_ratio"] if max_overlap else 0,
                "matched_abrupt_scope_values": ";".join(scopes),
                "window_label_scope": window_scope(overlaps),
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def create_segment(
    label: pd.Series,
    mapping: dict[str, Any],
    segment_id: str,
    segment_type: str,
    boundary_role: str,
    original_start: float,
    original_end: float,
    context_window_sec: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    start = original_start
    end = original_end
    duration = mapping.get("video_duration_sec")
    segment_clipped = False
    clipping_reason = ""
    clipped_audit: dict[str, Any] | None = None
    if duration is not None:
        clipped_start = max(0.0, start)
        clipped_end = min(float(duration), end)
        segment_clipped = clipped_start != start or clipped_end != end
        start, end = clipped_start, clipped_end
        if segment_clipped:
            clipping_reason = "segment exceeded video duration or started before zero"
            clipped_audit = {
                "source_version": "v2_2",
                "segment_id": segment_id,
                "segment_type": segment_type,
                "video_id": clean_value(label.get("video_id")),
                "video_title": clean_value(label.get("video_title")),
                "video_filename": mapping.get("video_filename", ""),
                "ad_interval_id": clean_value(label.get("ad_interval_id")),
                "original_segment_start_sec": original_start,
                "original_segment_end_sec": original_end,
                "clipped_segment_start_sec": start,
                "clipped_segment_end_sec": end,
                "video_duration_sec": duration,
                "clipping_applied": "true",
                "clipping_reason": clipping_reason,
                "needs_manual_review": "true",
            }
    if start >= end:
        return None, clipped_audit
    row = {
        "segment_id": segment_id,
        "segment_type": segment_type,
        "boundary_role": boundary_role,
        "video_id": clean_value(label.get("video_id")),
        "video_title": clean_value(label.get("video_title")),
        "video_filename": mapping.get("video_filename", ""),
        "video_path": mapping.get("video_path", ""),
        "file_stem": mapping.get("file_stem", ""),
        "ad_interval_id": clean_value(label.get("ad_interval_id")),
        "source_label_row_index": int(label.get("_source_label_row_index")),
        "segment_start_sec": start,
        "segment_end_sec": end,
        "segment_duration_sec": end - start,
        "context_window_sec": context_window_sec,
        "ad_start_sec": float(label.get("_ad_start_sec_num")),
        "ad_end_sec": float(label.get("_ad_end_sec_num")),
        "is_abrupt_transition_ad_original": clean_value(label.get("is_abrupt_transition_ad_original")),
        "is_abrupt_transition_ad_refreshed": clean_value(label.get("is_abrupt_transition_ad_refreshed")),
        "scope_value_source": clean_value(label.get("scope_value_source")),
        "label_valid": "true",
        "video_duration_sec": duration,
        "label_mapping_status": mapping.get("label_mapping_status", ""),
        "segment_refresh_version": "v2_2",
        "segment_clipped": bool_text(segment_clipped),
        "clipping_reason": clipping_reason,
        "segment_valid": "true",
    }
    return row, clipped_audit


def create_segments(labels: pd.DataFrame, mapping_by_filename: dict[str, dict[str, Any]], warnings: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    mapping_by_key = {
        (payload["mapped_video_id"], payload["mapped_video_title"]): payload
        for payload in mapping_by_filename.values()
        if payload.get("mapped_video_id") and payload.get("mapped_video_title")
    }
    ad_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    clipped: list[dict[str, Any]] = []
    empty_count = 0
    for _, label in valid_label_rows(labels).iterrows():
        key = (clean_value(label.get("video_id")), clean_value(label.get("video_title")))
        mapping = mapping_by_key.get(key)
        if not mapping:
            continue
        video_id = clean_value(label.get("video_id"))
        interval_id = clean_value(label.get("ad_interval_id"))
        ad_start = float(label["_ad_start_sec_num"])
        ad_end = float(label["_ad_end_sec_num"])
        specs = [
            (f"{video_id}_{interval_id}_AD", "ad_interval", "", ad_start, ad_end, ""),
            (f"{video_id}_{interval_id}_PRE10", "pre_ad_start_10s", "before_ad_start", max(0.0, ad_start - CONTEXT_WINDOW_SEC), ad_start, CONTEXT_WINDOW_SEC),
            (f"{video_id}_{interval_id}_POST10", "post_ad_end_10s", "after_ad_end", ad_end, ad_end + CONTEXT_WINDOW_SEC, CONTEXT_WINDOW_SEC),
        ]
        for segment_id, segment_type, boundary_role, start, end, context_sec in specs:
            row, audit = create_segment(label, mapping, segment_id, segment_type, boundary_role, start, end, context_sec)
            if audit:
                clipped.append(audit)
            if row is None:
                empty_count += 1
                warnings.append({"empty_segment_skipped": segment_id, "reason": "segment_start_sec_gte_segment_end_sec"})
                continue
            if segment_type == "ad_interval":
                ad_rows.append(row)
            else:
                context_rows.append(row)
    return ad_rows, context_rows, ad_rows + context_rows, clipped, empty_count


def update_readme() -> None:
    section = """## Window and Segment Refresh v2.2

v2_2는 manual mapping과 all-valid-abrupt scope 반영 버전이다. `new_ad_labeling.xlsx`가 전환형 광고만 포함한 라벨 파일이므로 valid interval을 `is_abrupt_transition_ad=yes`로 처리했다.

이후 OpenCV/ResNet scene-change, OCR, audio feature는 v2_2 기준 window/segment를 사용해야 한다. 5초 window는 최종 광고 판정 단위가 아니라 feature 수집 단위이며, 최종 판단은 interval 단위 rule-based detector에서 수행할 예정이다.
"""
    text = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else "# youtube_ad_segment_detector\n"
    marker = "## Window and Segment Refresh v2.2"
    if marker in text:
        before, _, after = text.partition(marker)
        next_idx = after.find("\n## ")
        text = before.rstrip() + "\n\n" + section + (after[next_idx:] if next_idx >= 0 else "")
    else:
        text = text.rstrip() + "\n\n" + section
    README_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def distribution(series: pd.Series | list[Any]) -> dict[str, int]:
    values = [clean_value(value) for value in list(series)]
    return dict(sorted(Counter(values).items()))


def make_summary(report: dict[str, Any]) -> str:
    warnings = report.get("warnings", [])
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- 주요 warning 없음"
    return f"""# refresh_window_segments_v2_2_with_manual_mapping summary

## 결과

- manual mapping file: `{report['manual_mapping_file_used']}`
- manual mapping applied count: {report['manual_mapping_applied_count']}
- ambiguous title count before/after: {report['ambiguous_title_count_before']} / {report['ambiguous_title_count_after']}
- unresolved ambiguous count: {report['unresolved_ambiguous_count']}

## Scope

- before: `{report['scope_distribution_before']}`
- after: `{report['scope_distribution_after']}`

## Window Labels

- v2_1: `{report['window_label_scope_distribution_before_v2_1']}`
- v2_2: `{report['window_label_scope_distribution_after_v2_2']}`

## Segments

- ad interval before/after: {report['ad_interval_segment_count_before_v2_1']} / {report['ad_interval_segment_count_after_v2_2']}
- combined before/after: {report['combined_segment_count_before_v2_1']} / {report['combined_segment_count_after_v2_2']}
- clipped segment count v2_2: {report['clipped_segment_count_v2_2']}

## Sub Agent Results

`{report.get('sub_agent_results', {})}`

## Warning

{warning_text}

## 다음 작업

OpenCV 기반 scene-change 후보 추출을 v2_2 window/segment 기준으로 진행할 수 있다.
"""


def clear_latest_and_copy(files: list[Path]) -> tuple[bool, list[str]]:
    expected = PROJECT_ROOT / "outputs/latest_for_chatgpt"
    if LATEST_DIR.resolve() != expected.resolve():
        return False, []
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    for child in LATEST_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    copied: list[str] = []
    for src in files:
        if src.exists():
            dst = LATEST_DIR / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    descriptions = {
        "README.md": "프로젝트 README와 v2.2 기준 설명",
        "clean_ad_labels_v0_scope_refreshed_v2_2.csv": "valid interval을 abrupt-transition yes로 반영한 라벨",
        "video_manifest_v2_2.csv": "manual mapping이 적용된 video manifest",
        "manual_mapping_applied_audit_v2_2.csv": "manual mapping 적용 audit",
        "window_labels_5s_v2_2.csv": "v2_2 기준 5초 window label",
        "ad_interval_segments_v2_2.csv": "v2_2 광고 interval segment",
        "ad_context_segments_v2_2.csv": "v2_2 광고 전후 context segment",
        "label_interval_context_segments_v2_2.csv": "v2_2 통합 segment",
        "clipped_segment_audit_v2_2.csv": "v2_2 clipped segment audit",
        "refresh_window_segments_v2_2_with_manual_mapping_report.json": "v2_2 상세 report",
        "refresh_window_segments_v2_2_with_manual_mapping_summary.md": "v2_2 요약",
        "refresh_window_segments_v2_2_with_manual_mapping_run_log.txt": "v2_2 실행 log",
        "refresh_window_segments_v2_2_with_manual_mapping.py": "v2_2 재현 스크립트",
    }
    LATEST_README.write_text(
        "# latest_for_chatgpt files\n\n"
        "latest_for_chatgpt는 최신 작업 핵심 파일만 모아둔 복사본 경로이다. 원본 파일은 프로젝트 내부 원래 경로에 존재한다.\n\n"
        "이번 작업명: refresh_window_segments_v2_2_with_manual_mapping\n\n"
        "복사된 파일 목록과 목적:\n\n"
        + "\n".join(f"- `{Path(path).name}`: {descriptions.get(Path(path).name, 'v2_2 output')}" for path in copied)
        + "\n\nmp4 영상 파일, 원본 xlsx, frame, model, checkpoint, cache 파일은 복사하지 않는다.\n",
        encoding="utf-8",
    )
    copied.append(str(LATEST_README))
    return True, copied


def main() -> None:
    ensure_dirs()
    warnings: list[Any] = []
    errors: list[Any] = []
    sub_agent_results: dict[str, Any] = {}

    step(1, "cv 환경 확인")
    cv_ok, python_executable, cv_warnings = verify_cv_environment()
    warnings.extend(cv_warnings)
    if not cv_ok:
        errors.append("cv_environment_check_failed")

    step(2, "입력 파일 탐색")
    report_v2_1 = json.loads(INPUT_REPORT_V2_1.read_text(encoding="utf-8")) if INPUT_REPORT_V2_1.exists() else {}
    clean_df = read_csv(INPUT_CLEAN_LABEL)
    manifest_v2 = read_csv(INPUT_MANIFEST_V2)
    window_grid_v2 = read_csv(INPUT_WINDOW_GRID_V2)
    window_labels_v2_1 = read_csv_if_exists(INPUT_WINDOW_LABELS_V2_1)
    ad_segments_v2_1 = read_csv_if_exists(INPUT_AD_SEGMENTS_V2_1)
    combined_segments_v2_1 = read_csv_if_exists(INPUT_COMBINED_SEGMENTS_V2_1)
    ambiguous_audit_v2_1 = read_csv_if_exists(INPUT_AMBIGUOUS_AUDIT_V2_1)
    log(f"Loaded labels={len(clean_df)}, manifest_v2={len(manifest_v2)}, window_grid_v2={len(window_grid_v2)}")

    step(3, "manual mapping 로드")
    manual_file, manual_df, manual_rows = find_manual_mapping_file(warnings)
    log(f"manual_mapping_file_used={manual_file}, filled_rows={manual_rows}")

    step(4, "label scope refresh")
    refreshed_labels, scope_before, scope_after = refresh_labels_all_valid_yes(clean_df)
    refreshed_labels.to_csv(OUTPUT_LABEL, index=False, encoding="utf-8-sig")
    valid_label_count = int(label_valid_mask(refreshed_labels).sum())
    invalid_label_count = int(len(refreshed_labels) - valid_label_count)

    refreshed_manifest, manual_audit_rows, mapping_by_filename, manual_applied_count, ambiguous_before, unresolved_count = apply_manual_mapping(
        manifest_v2,
        refreshed_labels,
        manual_df,
        manual_file,
        ambiguous_audit_v2_1,
        warnings,
    )
    ambiguous_after = int((refreshed_manifest["title_match_status"].astype(str) == "ambiguous").sum())
    refreshed_manifest.to_csv(OUTPUT_MANIFEST, index=False, encoding="utf-8-sig")
    write_csv(
        OUTPUT_MANUAL_AUDIT,
        manual_audit_rows,
        [
            "video_filename",
            "previous_match_status",
            "previous_candidate_video_ids",
            "selected_video_id",
            "selected_video_title",
            "resolution_note",
            "manual_mapping_source_file",
            "mapping_applied",
            "mapping_warning",
        ],
    )
    if unresolved_count:
        warnings.append({"unresolved_ambiguous_count": unresolved_count})
    if manual_applied_count == 0:
        warnings.append("manual_mapping_applied_count_is_zero")

    step(5, "window label 재계산")
    window_labels_v2_2 = create_window_labels(window_grid_v2, refreshed_labels, refreshed_manifest, mapping_by_filename)
    window_labels_v2_2.to_csv(OUTPUT_WINDOW_LABELS, index=False, encoding="utf-8-sig")

    step(6, "segment v2_2 재생성")
    ad_rows, context_rows, combined_rows, clipped_rows, empty_segment_count = create_segments(refreshed_labels, mapping_by_filename, warnings)
    segment_columns = [
        "segment_id",
        "segment_type",
        "boundary_role",
        "video_id",
        "video_title",
        "video_filename",
        "video_path",
        "file_stem",
        "ad_interval_id",
        "source_label_row_index",
        "segment_start_sec",
        "segment_end_sec",
        "segment_duration_sec",
        "context_window_sec",
        "ad_start_sec",
        "ad_end_sec",
        "is_abrupt_transition_ad_original",
        "is_abrupt_transition_ad_refreshed",
        "scope_value_source",
        "label_valid",
        "video_duration_sec",
        "label_mapping_status",
        "segment_refresh_version",
        "segment_clipped",
        "clipping_reason",
        "segment_valid",
    ]
    write_csv(OUTPUT_AD_SEGMENTS, ad_rows, segment_columns)
    write_csv(OUTPUT_CONTEXT_SEGMENTS, context_rows, segment_columns)
    write_csv(OUTPUT_COMBINED_SEGMENTS, combined_rows, segment_columns)

    step(7, "clipped segment audit 생성")
    write_csv(
        OUTPUT_CLIPPED_AUDIT,
        clipped_rows,
        [
            "source_version",
            "segment_id",
            "segment_type",
            "video_id",
            "video_title",
            "video_filename",
            "ad_interval_id",
            "original_segment_start_sec",
            "original_segment_end_sec",
            "clipped_segment_start_sec",
            "clipped_segment_end_sec",
            "video_duration_sec",
            "clipping_applied",
            "clipping_reason",
            "needs_manual_review",
        ],
    )

    step(8, "QA 검증")
    duplicate_window_id_count = int(window_labels_v2_2["window_id"].duplicated().sum())
    duplicate_segment_id_count = sum(1 for _, count in Counter(row["segment_id"] for row in combined_rows).items() if count > 1)
    bad_window_bounds = int((window_labels_v2_2["window_start_sec"] >= window_labels_v2_2["window_end_sec"]).sum())
    bad_segment_bounds = sum(1 for row in combined_rows if float(row["segment_start_sec"]) >= float(row["segment_end_sec"]))
    if duplicate_window_id_count:
        errors.append("duplicate_window_id_detected")
    if duplicate_segment_id_count:
        errors.append("duplicate_segment_id_detected")
    if bad_window_bounds:
        errors.append("bad_window_bounds_detected")
    if bad_segment_bounds:
        errors.append("bad_segment_bounds_detected")

    resolved_keys = {
        (payload["mapped_video_id"], payload["mapped_video_title"])
        for payload in mapping_by_filename.values()
        if payload.get("mapped_video_id") and payload.get("mapped_video_title")
    }
    label_rows_valid = valid_label_rows(refreshed_labels)
    labels_without_video = [
        {
            "ad_interval_id": clean_value(row.get("ad_interval_id")),
            "video_id": clean_value(row.get("video_id")),
            "video_title": clean_value(row.get("video_title")),
        }
        for _, row in label_rows_valid.iterrows()
        if (clean_value(row.get("video_id")), clean_value(row.get("video_title"))) not in resolved_keys
    ]
    videos_without_label = [
        payload["video_filename"]
        for payload in mapping_by_filename.values()
        if not payload.get("mapped_video_id")
    ]
    if labels_without_video:
        warnings.append({"labels_without_video_count": len(labels_without_video)})
    if videos_without_label:
        warnings.append({"videos_without_label_count": len(videos_without_label)})

    window_scope_before = distribution(window_labels_v2_1["window_label_scope"]) if not window_labels_v2_1.empty and "window_label_scope" in window_labels_v2_1.columns else {}
    window_scope_after = distribution(window_labels_v2_2["window_label_scope"])
    if window_scope_after.get("not_reviewed_ad", 0):
        warnings.append({"not_reviewed_ad_remaining": window_scope_after.get("not_reviewed_ad", 0)})

    update_readme()

    generated_files = [
        OUTPUT_LABEL,
        OUTPUT_MANIFEST,
        OUTPUT_MANUAL_AUDIT,
        OUTPUT_WINDOW_LABELS,
        OUTPUT_AD_SEGMENTS,
        OUTPUT_CONTEXT_SEGMENTS,
        OUTPUT_COMBINED_SEGMENTS,
        OUTPUT_CLIPPED_AUDIT,
        REPORT_PATH,
        SUMMARY_PATH,
        LOG_PATH,
        SCRIPT_PATH,
    ]

    step(9, "report/latest 갱신")
    report = {
        "project_root": str(PROJECT_ROOT),
        "cv_environment_checked": cv_ok,
        "python_executable": python_executable,
        "old_project_modified": False,
        "manual_mapping_file_used": str(manual_file) if manual_file else "",
        "manual_mapping_rows": manual_rows,
        "manual_mapping_applied_count": manual_applied_count,
        "ambiguous_title_count_before": ambiguous_before,
        "ambiguous_title_count_after": ambiguous_after,
        "unresolved_ambiguous_count": unresolved_count,
        "input_label_row_count": int(len(clean_df)),
        "valid_label_row_count": valid_label_count,
        "invalid_label_row_count": invalid_label_count,
        "scope_distribution_before": scope_before,
        "scope_distribution_after": scope_after,
        "window_count": int(len(window_labels_v2_2)),
        "window_label_scope_distribution_before_v2_1": window_scope_before,
        "window_label_scope_distribution_after_v2_2": window_scope_after,
        "ad_interval_segment_count_before_v2_1": int(len(ad_segments_v2_1)),
        "ad_interval_segment_count_after_v2_2": int(len(ad_rows)),
        "combined_segment_count_before_v2_1": int(len(combined_segments_v2_1)),
        "combined_segment_count_after_v2_2": int(len(combined_rows)),
        "clipped_segment_count_v2_2": int(len(clipped_rows)),
        "duplicate_window_id_count": duplicate_window_id_count,
        "duplicate_segment_id_count": duplicate_segment_id_count,
        "bad_window_bounds_count": bad_window_bounds,
        "bad_segment_bounds_count": bad_segment_bounds,
        "labels_without_video": labels_without_video,
        "videos_without_label": videos_without_label,
        "empty_segment_count": empty_segment_count,
        "generated_files": [str(path) for path in generated_files],
        "sub_agent_results": sub_agent_results,
        "warnings": warnings,
        "errors": errors,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(make_summary(report), encoding="utf-8")
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")

    latest_ok, latest_files = clear_latest_and_copy([README_PATH] + generated_files)
    report["latest_for_chatgpt_updated"] = latest_ok
    report["latest_for_chatgpt_files"] = latest_files
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(make_summary(report), encoding="utf-8")
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    if latest_ok:
        shutil.copy2(REPORT_PATH, LATEST_DIR / REPORT_PATH.name)
        shutil.copy2(SUMMARY_PATH, LATEST_DIR / SUMMARY_PATH.name)
        shutil.copy2(LOG_PATH, LATEST_DIR / LOG_PATH.name)

    print(
        json.dumps(
            {
                "manual_mapping_applied_count": manual_applied_count,
                "ambiguous_title_count_before": ambiguous_before,
                "ambiguous_title_count_after": ambiguous_after,
                "unresolved_ambiguous_count": unresolved_count,
                "scope_distribution_after": scope_after,
                "window_label_scope_distribution_after_v2_2": window_scope_after,
                "ad_interval_segment_count_after_v2_2": len(ad_rows),
                "combined_segment_count_after_v2_2": len(combined_rows),
                "clipped_segment_count_v2_2": len(clipped_rows),
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
