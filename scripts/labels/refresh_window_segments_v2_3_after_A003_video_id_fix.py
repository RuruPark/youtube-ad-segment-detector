#!/usr/bin/env python3
"""A003 video_id 수정 뒤 v2.3 window label과 segment를 갱신한다."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")

INPUT_LABEL = PROJECT_ROOT / "data/labels/clean_ad_labels_v0_scope_refreshed_v2_2.csv"
INPUT_MANIFEST = PROJECT_ROOT / "data/video_metadata/video_manifest_v2_2.csv"
INPUT_WINDOW_GRID = PROJECT_ROOT / "data/windows/window_grid_5s_v2.csv"
INPUT_MANUAL_AUDIT = PROJECT_ROOT / "data/video_metadata/manual_mapping_applied_audit_v2_2.csv"
INPUT_WINDOW_LABELS_V2_2 = PROJECT_ROOT / "data/windows/window_labels_5s_v2_2.csv"
INPUT_AD_SEGMENTS_V2_2 = PROJECT_ROOT / "data/segments/ad_interval_segments_v2_2.csv"
INPUT_CONTEXT_SEGMENTS_V2_2 = PROJECT_ROOT / "data/segments/ad_context_segments_v2_2.csv"
INPUT_COMBINED_SEGMENTS_V2_2 = PROJECT_ROOT / "data/segments/label_interval_context_segments_v2_2.csv"
INPUT_REPORT_V2_2 = PROJECT_ROOT / "reports/refresh_window_segments_v2_2_with_manual_mapping_report.json"

OUTPUT_A003_LABEL_AUDIT = PROJECT_ROOT / "data/labels/A003_label_fix_audit_v2_3.csv"
OUTPUT_MAPPING_AUDIT = PROJECT_ROOT / "data/video_metadata/label_video_mapping_audit_v2_3.csv"
OUTPUT_WINDOW_LABELS = PROJECT_ROOT / "data/windows/window_labels_5s_v2_3.csv"
OUTPUT_AD_SEGMENTS = PROJECT_ROOT / "data/segments/ad_interval_segments_v2_3.csv"
OUTPUT_CONTEXT_SEGMENTS = PROJECT_ROOT / "data/segments/ad_context_segments_v2_3.csv"
OUTPUT_COMBINED_SEGMENTS = PROJECT_ROOT / "data/segments/label_interval_context_segments_v2_3.csv"
OUTPUT_A003_INCLUSION_AUDIT = PROJECT_ROOT / "data/segments/A003_inclusion_audit_v2_3.csv"
OUTPUT_CLIPPED_AUDIT = PROJECT_ROOT / "data/segments/clipped_segment_audit_v2_3.csv"
OUTPUT_OVERLAP_AUDIT = PROJECT_ROOT / "data/segments/ad_interval_overlap_audit_v2_3.csv"

REPORT_PATH = PROJECT_ROOT / "reports/refresh_window_segments_v2_3_after_A003_video_id_fix_report.json"
SUMMARY_PATH = PROJECT_ROOT / "reports/refresh_window_segments_v2_3_after_A003_video_id_fix_summary.md"
LOG_PATH = PROJECT_ROOT / "logs/refresh_window_segments_v2_3_after_A003_video_id_fix_run_log.txt"
SCRIPT_PATH = PROJECT_ROOT / "scripts/labels/refresh_window_segments_v2_3_after_A003_video_id_fix.py"
README_PATH = PROJECT_ROOT / "README.md"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt"
LATEST_README = LATEST_DIR / "README_latest_files.md"
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


def to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return read_csv(path) if path.exists() else pd.DataFrame()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_inside_project(path)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: fmt_value(row.get(column, "")) for column in columns})


def distribution(values: pd.Series | list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(clean_value(value) for value in list(values)).items()))


def label_valid_mask(df: pd.DataFrame) -> pd.Series:
    if "label_valid" in df.columns:
        return df["label_valid"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
    starts = df["ad_start_sec"].map(to_float) if "ad_start_sec" in df.columns else pd.Series([None] * len(df))
    ends = df["ad_end_sec"].map(to_float) if "ad_end_sec" in df.columns else pd.Series([None] * len(df))
    return starts.notna() & ends.notna() & (starts < ends)


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


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_mapping_by_filename(manifest: pd.DataFrame) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for _, row in manifest.iterrows():
        filename = clean_value(row.get("video_filename"))
        mapped_id = clean_value(row.get("label_mapping_video_id")) or clean_value(row.get("video_id"))
        mapped_title = (
            clean_value(row.get("label_mapping_video_title"))
            or clean_value(row.get("matched_label_video_title"))
            or clean_value(row.get("video_title"))
        )
        mapping_status = clean_value(row.get("label_mapping_status")) or clean_value(row.get("title_match_status"))
        mapping_warning = clean_value(row.get("label_mapping_warning"))
        if mapping_status.startswith("unresolved") or mapping_status in {"unmatched", "ambiguous"}:
            mapped_id = ""
            mapped_title = ""
        mapping[filename] = {
            "mapped_video_id": mapped_id,
            "mapped_video_title": mapped_title,
            "label_mapping_status": mapping_status,
            "label_mapping_warning": mapping_warning,
            "matched_manifest_video_id": clean_value(row.get("video_id")),
            "matched_video_filename": filename,
            "matched_video_path": clean_value(row.get("video_path")),
            "file_stem": clean_value(row.get("file_stem")),
            "title_match_status": clean_value(row.get("title_match_status")),
            "video_duration_sec": to_float(row.get("duration_sec")),
            "fps": to_float(row.get("fps")),
            "frame_count": to_float(row.get("frame_count")),
            "width": to_float(row.get("width")),
            "height": to_float(row.get("height")),
        }
    return mapping


def mapping_by_label_key(mapping_by_filename: dict[str, dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for payload in mapping_by_filename.values():
        key = (payload.get("mapped_video_id", ""), payload.get("mapped_video_title", ""))
        if key[0] and key[1] and key not in result:
            result[key] = payload
    return result


def create_a003_label_audit(labels: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    status = {
        "A003_video_id_before_or_current": "",
        "A003_video_id_is_2": False,
        "A003_label_valid": False,
        "A003_scope_is_yes": False,
        "A003_time_valid": False,
    }
    a003 = labels[labels["ad_interval_id"].map(clean_value).eq("A003")].copy()
    if a003.empty:
        rows.append({"ad_interval_id": "A003", "fix_status": "fail", "fix_warning": "A003 row not found"})
        return rows, status
    for _, row in a003.iterrows():
        start = to_float(row.get("ad_start_sec"))
        end = to_float(row.get("ad_end_sec"))
        valid = clean_value(row.get("label_valid")).lower() in {"true", "1", "yes", "y"}
        scope = clean_value(row.get("is_abrupt_transition_ad_refreshed"))
        warnings: list[str] = []
        if clean_value(row.get("video_id")) != "2":
            warnings.append("A003_video_id_is_not_2")
        if not valid:
            warnings.append("A003_label_valid_is_not_true")
        if scope != "yes":
            warnings.append("A003_scope_is_not_yes")
        if start is None or end is None or end <= start:
            warnings.append("A003_time_invalid")
        fix_status = "pass" if not warnings else "fail"
        rows.append(
            {
                "ad_interval_id": clean_value(row.get("ad_interval_id")),
                "video_id": clean_value(row.get("video_id")),
                "video_title": clean_value(row.get("video_title")),
                "ad_start_sec": start,
                "ad_end_sec": end,
                "ad_duration_sec": (end - start) if start is not None and end is not None else "",
                "label_valid": clean_value(row.get("label_valid")),
                "is_abrupt_transition_ad_refreshed": scope,
                "fix_status": fix_status,
                "fix_warning": ";".join(warnings),
            }
        )
        status.update(
            {
                "A003_video_id_before_or_current": clean_value(row.get("video_id")),
                "A003_video_id_is_2": clean_value(row.get("video_id")) == "2",
                "A003_label_valid": valid,
                "A003_scope_is_yes": scope == "yes",
                "A003_time_valid": start is not None and end is not None and end > start,
            }
        )
    return rows, status


def create_label_video_mapping_audit(labels: pd.DataFrame, mapping_key: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in valid_label_rows(labels).iterrows():
        key = (clean_value(row.get("video_id")), clean_value(row.get("video_title")))
        mapping = mapping_key.get(key)
        mapped = mapping is not None
        rows.append(
            {
                "ad_interval_id": clean_value(row.get("ad_interval_id")),
                "label_video_id": key[0],
                "label_video_title": key[1],
                "matched_manifest_video_id": mapping.get("matched_manifest_video_id", "") if mapping else "",
                "matched_video_filename": mapping.get("matched_video_filename", "") if mapping else "",
                "matched_video_path": mapping.get("matched_video_path", "") if mapping else "",
                "mapping_status": "mapped" if mapped else "unmapped",
                "mapping_warning": "" if mapped else "no manifest row with same video_id and video_title",
            }
        )
    return rows


def labels_grouped_by_mapping(labels: pd.DataFrame, mapping_key: dict[tuple[str, str], dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for _, row in valid_label_rows(labels).iterrows():
        key = (clean_value(row.get("video_id")), clean_value(row.get("video_title")))
        if key not in mapping_key:
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
    mapping_by_filename: dict[str, dict[str, Any]],
    mapping_key: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    grouped = labels_grouped_by_mapping(labels, mapping_key)
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
        source_window_id = clean_value(base.get("window_id"))
        if mapped_id:
            base["video_id"] = mapped_id
            base["video_title"] = mapped_title
            base["window_id"] = f"{mapped_id}_W{int(float(base['window_index'])):06d}"
        base.update(
            {
                "source_window_id_v2": source_window_id,
                "label_mapping_status": mapping.get("label_mapping_status", "mapping_missing"),
                "label_mapping_warning": mapping.get("label_mapping_warning", ""),
                "label_refresh_version": "v2_3",
                "window_label_refresh_source": "A003_video_id_fix",
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
                "source_version": "v2_3",
                "segment_id": segment_id,
                "segment_type": segment_type,
                "video_id": clean_value(label.get("video_id")),
                "video_title": clean_value(label.get("video_title")),
                "video_filename": mapping.get("matched_video_filename", ""),
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
        "video_filename": mapping.get("matched_video_filename", ""),
        "video_path": mapping.get("matched_video_path", ""),
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
        "segment_refresh_version": "v2_3",
        "segment_clipped": bool_text(segment_clipped),
        "clipping_reason": clipping_reason,
        "segment_valid": "true",
    }
    return row, clipped_audit


def create_segments(
    labels: pd.DataFrame,
    mapping_key: dict[tuple[str, str], dict[str, Any]],
    warnings: list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    ad_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    clipped_rows: list[dict[str, Any]] = []
    empty_segment_count = 0
    for _, label in valid_label_rows(labels).iterrows():
        key = (clean_value(label.get("video_id")), clean_value(label.get("video_title")))
        mapping = mapping_key.get(key)
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
                clipped_rows.append(audit)
            if row is None:
                empty_segment_count += 1
                warnings.append({"empty_segment_skipped": segment_id, "reason": "segment_start_sec_gte_segment_end_sec"})
                continue
            if segment_type == "ad_interval":
                ad_rows.append(row)
            else:
                context_rows.append(row)
    return ad_rows, context_rows, ad_rows + context_rows, clipped_rows, empty_segment_count


def create_a003_inclusion_audit(
    labels: pd.DataFrame,
    mapping_audit: list[dict[str, Any]],
    window_labels: pd.DataFrame,
    ad_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    a003_label = labels[labels["ad_interval_id"].map(clean_value).eq("A003")].copy()
    if a003_label.empty:
        return [{"ad_interval_id": "A003", "A003_inclusion_status": "missing_label", "A003_inclusion_warning": "A003 label not found"}]
    label = a003_label.iloc[0]
    mapped = next((row for row in mapping_audit if row.get("ad_interval_id") == "A003"), {})
    matched_windows = window_labels[window_labels["matched_ad_interval_ids"].astype(str).str.split(";").map(lambda values: "A003" in values)]
    a003_ad = next((row for row in ad_rows if row.get("ad_interval_id") == "A003"), None)
    a003_pre = next((row for row in context_rows if row.get("ad_interval_id") == "A003" and row.get("segment_type") == "pre_ad_start_10s"), None)
    a003_post = next((row for row in context_rows if row.get("ad_interval_id") == "A003" and row.get("segment_type") == "post_ad_end_10s"), None)
    warnings: list[str] = []
    if mapped.get("mapping_status") != "mapped":
        warnings.append("A003_label_not_mapped_to_video")
    if len(matched_windows) == 0:
        warnings.append("A003_has_no_overlap_windows")
    if not a003_ad:
        warnings.append("A003_ad_interval_segment_missing")
    if not a003_pre:
        warnings.append("A003_pre_context_segment_missing")
    if not a003_post:
        warnings.append("A003_post_context_segment_missing")
    return [
        {
            "ad_interval_id": "A003",
            "label_video_id": clean_value(label.get("video_id")),
            "label_video_title": clean_value(label.get("video_title")),
            "label_valid": clean_value(label.get("label_valid")),
            "is_abrupt_transition_ad_refreshed": clean_value(label.get("is_abrupt_transition_ad_refreshed")),
            "ad_start_sec": to_float(label.get("ad_start_sec")),
            "ad_end_sec": to_float(label.get("ad_end_sec")),
            "matched_manifest_video_id": mapped.get("matched_manifest_video_id", ""),
            "matched_video_filename": mapped.get("matched_video_filename", ""),
            "matched_video_path": mapped.get("matched_video_path", ""),
            "A003_overlap_window_count": int(len(matched_windows)),
            "A003_abrupt_ad_window_count": int((matched_windows["window_label_scope"] == "abrupt_ad").sum()) if not matched_windows.empty else 0,
            "A003_mixed_or_boundary_window_count": int((matched_windows["window_label_scope"] == "mixed_or_boundary").sum()) if not matched_windows.empty else 0,
            "A003_ad_interval_segment_exists": bool_text(a003_ad is not None),
            "A003_pre_segment_exists": bool_text(a003_pre is not None),
            "A003_post_segment_exists": bool_text(a003_post is not None),
            "A003_segment_start_sec": a003_ad.get("segment_start_sec", "") if a003_ad else "",
            "A003_segment_end_sec": a003_ad.get("segment_end_sec", "") if a003_ad else "",
            "A003_video_duration_sec": a003_ad.get("video_duration_sec", "") if a003_ad else "",
            "A003_inclusion_status": "included" if not warnings else "warning",
            "A003_inclusion_warning": ";".join(warnings),
        }
    ]


def create_overlap_audit(labels: pd.DataFrame, mapping_key: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    labels_valid = valid_label_rows(labels)
    grouped: dict[tuple[str, str], list[pd.Series]] = defaultdict(list)
    for _, row in labels_valid.iterrows():
        key = (clean_value(row.get("video_id")), clean_value(row.get("video_title")))
        if key in mapping_key:
            grouped[key].append(row)
    overlap_count = 0
    for (video_id, video_title), items in grouped.items():
        if len(items) < 2:
            continue
        items = sorted(items, key=lambda item: (float(item["_ad_start_sec_num"]), clean_value(item.get("ad_interval_id"))))
        for idx, first in enumerate(items):
            for second in items[idx + 1 :]:
                start1 = float(first["_ad_start_sec_num"])
                end1 = float(first["_ad_end_sec_num"])
                start2 = float(second["_ad_start_sec_num"])
                end2 = float(second["_ad_end_sec_num"])
                overlap = max(0.0, min(end1, end2) - max(start1, start2))
                min_duration = max(min(end1 - start1, end2 - start2), 1e-9)
                if overlap > 0:
                    overlap_count += 1
                rows.append(
                    {
                        "video_id": video_id,
                        "video_title": video_title,
                        "ad_interval_id_1": clean_value(first.get("ad_interval_id")),
                        "ad_interval_id_2": clean_value(second.get("ad_interval_id")),
                        "interval_1_start_sec": start1,
                        "interval_1_end_sec": end1,
                        "interval_2_start_sec": start2,
                        "interval_2_end_sec": end2,
                        "overlap_sec": overlap,
                        "overlap_ratio_min_duration": overlap / min_duration,
                        "overlap_warning": "overlap_detected" if overlap > 0 else "no_overlap",
                    }
                )
    return rows, overlap_count


def update_readme() -> None:
    section = """## Window and Segment Refresh v2.3

v2_3은 A003 video_id 수정 반영 버전이다. A003은 기존 v2_2에서 feature/evaluation 대상에서 제외되어 있었으나, v2_3에서 video_id=2로 연결되어 segment에 포함되었다.

이후 OpenCV/ResNet scene-change, OCR, audio feature는 v2_3 기준 window/segment를 사용해야 한다. 5초 window는 최종 광고 판정 단위가 아니라 feature 수집 단위이며, 최종 판단은 interval 단위 rule-based detector에서 수행할 예정이다.
"""
    text = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else "# youtube_ad_segment_detector\n"
    marker = "## Window and Segment Refresh v2.3"
    if marker in text:
        before, _, after = text.partition(marker)
        next_idx = after.find("\n## ")
        text = before.rstrip() + "\n\n" + section + (after[next_idx:] if next_idx >= 0 else "")
    else:
        text = text.rstrip() + "\n\n" + section
    ensure_inside_project(README_PATH).write_text(text.rstrip() + "\n", encoding="utf-8")


def make_summary(report: dict[str, Any]) -> str:
    warnings = report.get("warnings", [])
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- 주요 warning 없음"
    sub_agent_text = "\n".join(f"- {key}: {value}" for key, value in report.get("sub_agent_results", {}).items()) or "- sub agent 검증 대기"
    return f"""# refresh_window_segments_v2_3_after_A003_video_id_fix summary

## A003 반영

- A003 video_id current: `{report.get('A003_video_id_before_or_current')}`
- A003 video_id is 2: {report.get('A003_video_id_is_2')}
- A003 mapping status: `{report.get('A003_mapping_status')}`
- A003 overlap window count: {report.get('A003_overlap_window_count')}
- A003 ad/pre/post segment included: {report.get('A003_included_in_ad_segments')} / {report.get('A003_pre_context_segment_exists')} / {report.get('A003_post_context_segment_exists')}

## Mapping

- labels without video before/after: {report.get('labels_without_video_before_v2_2')} / {report.get('labels_without_video_after_v2_3')}
- mapped valid labels: {report.get('mapped_label_row_count')} / {report.get('valid_label_row_count')}

## Window Labels

- before v2_2: `{report.get('window_label_scope_distribution_before_v2_2')}`
- after v2_3: `{report.get('window_label_scope_distribution_after_v2_3')}`

## Segments

- ad interval segment count before/after: {report.get('ad_interval_segment_count_before_v2_2')} / {report.get('ad_interval_segment_count_after_v2_3')}
- combined segment count before/after: {report.get('combined_segment_count_before_v2_2')} / {report.get('combined_segment_count_after_v2_3')}
- clipped segment count v2_3: {report.get('clipped_segment_count_v2_3')}
- ad interval overlap count: {report.get('ad_interval_overlap_count')}

## Sub Agent Results

{sub_agent_text}

## Warning

{warning_text}

## 다음 작업

OpenCV 기반 scene-change 후보 추출을 v2_3 window/segment 기준으로 진행할 수 있다.
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
        if not src.exists():
            continue
        dst = LATEST_DIR / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    descriptions = {
        "README.md": "프로젝트 README와 v2.3 기준 설명",
        "A003_label_fix_audit_v2_3.csv": "A003 video_id=2 수정 반영 확인 audit",
        "label_video_mapping_audit_v2_3.csv": "수정 라벨과 manifest 연결 상태 audit",
        "window_labels_5s_v2_3.csv": "A003 수정 반영 후 5초 window label",
        "ad_interval_segments_v2_3.csv": "v2_3 광고 interval segment",
        "ad_context_segments_v2_3.csv": "v2_3 광고 전후 context segment",
        "label_interval_context_segments_v2_3.csv": "v2_3 통합 segment",
        "A003_inclusion_audit_v2_3.csv": "A003 window/segment 포함 여부 audit",
        "clipped_segment_audit_v2_3.csv": "v2_3 clipped segment audit",
        "ad_interval_overlap_audit_v2_3.csv": "같은 video_id 내 광고 interval overlap audit",
        "refresh_window_segments_v2_3_after_A003_video_id_fix_report.json": "v2_3 상세 report",
        "refresh_window_segments_v2_3_after_A003_video_id_fix_summary.md": "v2_3 요약",
        "refresh_window_segments_v2_3_after_A003_video_id_fix_run_log.txt": "v2_3 실행 log",
        "refresh_window_segments_v2_3_after_A003_video_id_fix.py": "v2_3 재현 스크립트",
    }
    LATEST_README.write_text(
        "# latest_for_chatgpt files\n\n"
        "latest_for_chatgpt는 최신 작업 핵심 파일만 모아둔 복사본 경로이다. 원본 파일은 프로젝트 내부 원래 경로에 존재한다.\n\n"
        "이번 작업명: refresh_window_segments_v2_3_after_A003_video_id_fix\n\n"
        "이번 작업에서 추가된 핵심 변경:\n\n"
        "- A003 video_id=2 수정 반영\n"
        "- A003 window/segment 포함 audit 생성\n"
        "- v2_3 window label 및 interval/context segment 재생성\n\n"
        "복사된 파일 목록과 목적:\n\n"
        + "\n".join(f"- `{Path(path).name}`: {descriptions.get(Path(path).name, 'v2_3 output')}" for path in copied)
        + "\n\nmp4 영상 파일, 원본 xlsx, frame, model, checkpoint, cache 파일은 복사하지 않는다.\n",
        encoding="utf-8",
    )
    copied.append(str(LATEST_README))
    return True, copied


def main() -> None:
    ensure_dirs()
    warnings: list[Any] = []
    errors: list[Any] = []
    sub_agent_results: dict[str, Any] = {
        "sub_agent_1_label_mapping": "PENDING_EXTERNAL_REVIEW",
        "sub_agent_2_window_segment": "PENDING_EXTERNAL_REVIEW",
        "sub_agent_3_safety_output": "PENDING_EXTERNAL_REVIEW",
    }

    step(1, "cv 환경 확인")
    cv_ok, python_executable, cv_warnings = verify_cv_environment()
    warnings.extend(cv_warnings)
    if not cv_ok:
        errors.append("cv_environment_check_failed")

    step(2, "입력 파일 로드")
    required_inputs = [INPUT_LABEL, INPUT_MANIFEST, INPUT_WINDOW_GRID, INPUT_MANUAL_AUDIT]
    missing_inputs = [str(path) for path in required_inputs if not path.exists()]
    if missing_inputs:
        errors.append({"missing_inputs": missing_inputs})
    labels = read_csv(INPUT_LABEL)
    manifest = read_csv(INPUT_MANIFEST)
    window_grid = read_csv(INPUT_WINDOW_GRID)
    window_labels_v2_2 = read_csv_if_exists(INPUT_WINDOW_LABELS_V2_2)
    ad_segments_v2_2 = read_csv_if_exists(INPUT_AD_SEGMENTS_V2_2)
    combined_segments_v2_2 = read_csv_if_exists(INPUT_COMBINED_SEGMENTS_V2_2)
    report_v2_2 = load_json_if_exists(INPUT_REPORT_V2_2)
    log(f"Loaded labels={len(labels)}, manifest={len(manifest)}, window_grid={len(window_grid)}")

    step(3, "A003 video_id 수정 반영 확인")
    a003_label_audit_rows, a003_status = create_a003_label_audit(labels)
    write_csv(
        OUTPUT_A003_LABEL_AUDIT,
        a003_label_audit_rows,
        [
            "ad_interval_id",
            "video_id",
            "video_title",
            "ad_start_sec",
            "ad_end_sec",
            "ad_duration_sec",
            "label_valid",
            "is_abrupt_transition_ad_refreshed",
            "fix_status",
            "fix_warning",
        ],
    )
    if not a003_status["A003_video_id_is_2"]:
        errors.append("A003_video_id_is_not_2_in_input_label")
    if not a003_status["A003_label_valid"]:
        errors.append("A003_label_not_valid")
    if not a003_status["A003_scope_is_yes"]:
        warnings.append("A003_scope_is_not_yes")
    if not a003_status["A003_time_valid"]:
        errors.append("A003_time_invalid")

    step(4, "label-to-video 매핑 재검증")
    mapping_by_filename = build_mapping_by_filename(manifest)
    mapping_key = mapping_by_label_key(mapping_by_filename)
    mapping_audit_rows = create_label_video_mapping_audit(labels, mapping_key)
    write_csv(
        OUTPUT_MAPPING_AUDIT,
        mapping_audit_rows,
        [
            "ad_interval_id",
            "label_video_id",
            "label_video_title",
            "matched_manifest_video_id",
            "matched_video_filename",
            "matched_video_path",
            "mapping_status",
            "mapping_warning",
        ],
    )
    mapped_label_count = sum(1 for row in mapping_audit_rows if row["mapping_status"] == "mapped")
    unmapped_label_count = len(mapping_audit_rows) - mapped_label_count
    labels_without_video = [row for row in mapping_audit_rows if row["mapping_status"] != "mapped"]
    videos_without_label_count = sum(1 for payload in mapping_by_filename.values() if not payload.get("mapped_video_id"))
    a003_mapping_status = next((row["mapping_status"] for row in mapping_audit_rows if row["ad_interval_id"] == "A003"), "missing")
    if labels_without_video:
        warnings.append({"labels_without_video_after_v2_3": labels_without_video})

    step(5, "window label v2_3 재계산")
    window_labels_v2_3 = create_window_labels(window_grid, labels, mapping_by_filename, mapping_key)
    window_labels_v2_3.to_csv(OUTPUT_WINDOW_LABELS, index=False, encoding="utf-8-sig")

    step(6, "ad/context segment v2_3 재생성")
    ad_rows, context_rows, combined_rows, clipped_rows, empty_segment_count = create_segments(labels, mapping_key, warnings)
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

    step(7, "A003 포함 여부 및 clipped/overlap audit")
    a003_inclusion_rows = create_a003_inclusion_audit(labels, mapping_audit_rows, window_labels_v2_3, ad_rows, context_rows)
    write_csv(
        OUTPUT_A003_INCLUSION_AUDIT,
        a003_inclusion_rows,
        [
            "ad_interval_id",
            "label_video_id",
            "label_video_title",
            "label_valid",
            "is_abrupt_transition_ad_refreshed",
            "ad_start_sec",
            "ad_end_sec",
            "matched_manifest_video_id",
            "matched_video_filename",
            "matched_video_path",
            "A003_overlap_window_count",
            "A003_abrupt_ad_window_count",
            "A003_mixed_or_boundary_window_count",
            "A003_ad_interval_segment_exists",
            "A003_pre_segment_exists",
            "A003_post_segment_exists",
            "A003_segment_start_sec",
            "A003_segment_end_sec",
            "A003_video_duration_sec",
            "A003_inclusion_status",
            "A003_inclusion_warning",
        ],
    )
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
    overlap_rows, ad_interval_overlap_count = create_overlap_audit(labels, mapping_key)
    write_csv(
        OUTPUT_OVERLAP_AUDIT,
        overlap_rows,
        [
            "video_id",
            "video_title",
            "ad_interval_id_1",
            "ad_interval_id_2",
            "interval_1_start_sec",
            "interval_1_end_sec",
            "interval_2_start_sec",
            "interval_2_end_sec",
            "overlap_sec",
            "overlap_ratio_min_duration",
            "overlap_warning",
        ],
    )

    step(8, "QA report/summary/log 생성")
    duplicate_window_id_count = int(window_labels_v2_3["window_id"].duplicated().sum())
    duplicate_segment_id_count = sum(1 for _, count in Counter(row["segment_id"] for row in combined_rows).items() if count > 1)
    bad_window_bounds_count = int((window_labels_v2_3["window_start_sec"].astype(float) >= window_labels_v2_3["window_end_sec"].astype(float)).sum())
    bad_segment_bounds_count = sum(1 for row in combined_rows if float(row["segment_start_sec"]) >= float(row["segment_end_sec"]))
    if duplicate_window_id_count:
        errors.append("duplicate_window_id_detected")
    if duplicate_segment_id_count:
        errors.append("duplicate_segment_id_detected")
    if bad_window_bounds_count:
        errors.append("bad_window_bounds_detected")
    if bad_segment_bounds_count:
        errors.append("bad_segment_bounds_detected")

    a003_inclusion = a003_inclusion_rows[0] if a003_inclusion_rows else {}
    a003_overlap_count = int(a003_inclusion.get("A003_overlap_window_count", 0) or 0)
    a003_in_ad = clean_value(a003_inclusion.get("A003_ad_interval_segment_exists")) == "true"
    a003_pre_exists = clean_value(a003_inclusion.get("A003_pre_segment_exists")) == "true"
    a003_post_exists = clean_value(a003_inclusion.get("A003_post_segment_exists")) == "true"
    a003_in_combined = a003_in_ad and a003_pre_exists and a003_post_exists
    if a003_mapping_status != "mapped":
        errors.append("A003_mapping_status_not_mapped")
    if a003_overlap_count == 0:
        errors.append("A003_overlap_window_count_is_zero")
    if not a003_in_ad:
        errors.append("A003_ad_interval_segment_missing")
    if not a003_in_combined:
        errors.append("A003_combined_segments_incomplete")

    valid_label_count = int(label_valid_mask(labels).sum())
    invalid_label_count = int(len(labels) - valid_label_count)
    window_scope_before = distribution(window_labels_v2_2["window_label_scope"]) if not window_labels_v2_2.empty and "window_label_scope" in window_labels_v2_2.columns else {}
    window_scope_after = distribution(window_labels_v2_3["window_label_scope"])
    labels_without_video_before = report_v2_2.get("labels_without_video_count")
    if labels_without_video_before is None:
        labels_without_video_before = len(report_v2_2.get("labels_without_video", []))
    not_reviewed_after = window_scope_after.get("not_reviewed_ad", 0)
    if not_reviewed_after:
        warnings.append({"not_reviewed_ad_remaining": not_reviewed_after})
    if empty_segment_count:
        warnings.append({"empty_segment_count": empty_segment_count})

    generated_files = [
        OUTPUT_A003_LABEL_AUDIT,
        OUTPUT_MAPPING_AUDIT,
        OUTPUT_WINDOW_LABELS,
        OUTPUT_AD_SEGMENTS,
        OUTPUT_CONTEXT_SEGMENTS,
        OUTPUT_COMBINED_SEGMENTS,
        OUTPUT_A003_INCLUSION_AUDIT,
        OUTPUT_CLIPPED_AUDIT,
        OUTPUT_OVERLAP_AUDIT,
        REPORT_PATH,
        SUMMARY_PATH,
        LOG_PATH,
        SCRIPT_PATH,
    ]

    report = {
        "project_root": str(PROJECT_ROOT),
        "cv_environment_checked": cv_ok,
        "python_executable": python_executable,
        "old_project_modified": False,
        "input_label_path": str(INPUT_LABEL),
        "input_manifest_path": str(INPUT_MANIFEST),
        "input_window_grid_path": str(INPUT_WINDOW_GRID),
        "input_manual_mapping_audit_path": str(INPUT_MANUAL_AUDIT),
        "A003_video_id_before_or_current": a003_status["A003_video_id_before_or_current"],
        "A003_video_id_is_2": a003_status["A003_video_id_is_2"],
        "A003_label_valid": a003_status["A003_label_valid"],
        "A003_scope_is_yes": a003_status["A003_scope_is_yes"],
        "A003_time_valid": a003_status["A003_time_valid"],
        "A003_mapping_status": a003_mapping_status,
        "A003_included_in_window_labels": a003_overlap_count > 0,
        "A003_overlap_window_count": a003_overlap_count,
        "A003_abrupt_ad_window_count": int(a003_inclusion.get("A003_abrupt_ad_window_count", 0) or 0),
        "A003_mixed_or_boundary_window_count": int(a003_inclusion.get("A003_mixed_or_boundary_window_count", 0) or 0),
        "A003_included_in_ad_segments": a003_in_ad,
        "A003_pre_context_segment_exists": a003_pre_exists,
        "A003_post_context_segment_exists": a003_post_exists,
        "A003_included_in_combined_segments": a003_in_combined,
        "label_row_count": int(len(labels)),
        "valid_label_row_count": valid_label_count,
        "invalid_label_row_count": invalid_label_count,
        "mapped_label_row_count": mapped_label_count,
        "unmapped_label_row_count": unmapped_label_count,
        "labels_without_video_before_v2_2": labels_without_video_before,
        "labels_without_video_after_v2_3": len(labels_without_video),
        "videos_without_label_count": videos_without_label_count,
        "window_count": int(len(window_labels_v2_3)),
        "window_label_scope_distribution_before_v2_2": window_scope_before,
        "window_label_scope_distribution_after_v2_3": window_scope_after,
        "ad_interval_segment_count_before_v2_2": int(len(ad_segments_v2_2)),
        "ad_interval_segment_count_after_v2_3": int(len(ad_rows)),
        "context_segment_count_after_v2_3": int(len(context_rows)),
        "combined_segment_count_before_v2_2": int(len(combined_segments_v2_2)),
        "combined_segment_count_after_v2_3": int(len(combined_rows)),
        "clipped_segment_count_v2_3": int(len(clipped_rows)),
        "duplicate_window_id_count": duplicate_window_id_count,
        "duplicate_segment_id_count": duplicate_segment_id_count,
        "bad_window_bounds_count": bad_window_bounds_count,
        "bad_segment_bounds_count": bad_segment_bounds_count,
        "ad_interval_overlap_count": ad_interval_overlap_count,
        "ad_interval_overlap_audit_row_count": int(len(overlap_rows)),
        "generated_files": [str(path) for path in generated_files],
        "sub_agent_results": sub_agent_results,
        "warnings": warnings,
        "errors": errors,
    }

    update_readme()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(make_summary(report), encoding="utf-8")
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")

    step(9, "README/latest_for_chatgpt 갱신")
    latest_files = [
        README_PATH,
        OUTPUT_A003_LABEL_AUDIT,
        OUTPUT_MAPPING_AUDIT,
        OUTPUT_WINDOW_LABELS,
        OUTPUT_AD_SEGMENTS,
        OUTPUT_CONTEXT_SEGMENTS,
        OUTPUT_COMBINED_SEGMENTS,
        OUTPUT_A003_INCLUSION_AUDIT,
        OUTPUT_CLIPPED_AUDIT,
        OUTPUT_OVERLAP_AUDIT,
        REPORT_PATH,
        SUMMARY_PATH,
        LOG_PATH,
        SCRIPT_PATH,
    ]
    latest_ok, copied_latest = clear_latest_and_copy(latest_files)
    report["latest_for_chatgpt_updated"] = latest_ok
    report["latest_for_chatgpt_files"] = copied_latest
    report["latest_for_chatgpt_cleared"] = latest_ok
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
                "A003_video_id": report["A003_video_id_before_or_current"],
                "A003_mapping_status": report["A003_mapping_status"],
                "A003_overlap_window_count": report["A003_overlap_window_count"],
                "labels_without_video_before_after": [
                    report["labels_without_video_before_v2_2"],
                    report["labels_without_video_after_v2_3"],
                ],
                "ad_interval_segment_count_before_after": [
                    report["ad_interval_segment_count_before_v2_2"],
                    report["ad_interval_segment_count_after_v2_3"],
                ],
                "combined_segment_count_before_after": [
                    report["combined_segment_count_before_v2_2"],
                    report["combined_segment_count_after_v2_3"],
                ],
                "window_label_scope_distribution_after_v2_3": report["window_label_scope_distribution_after_v2_3"],
                "clipped_segment_count_v2_3": report["clipped_segment_count_v2_3"],
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
