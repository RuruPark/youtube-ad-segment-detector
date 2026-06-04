#!/usr/bin/env python3
"""v2.1 label mapping QA 기준으로 v2 window label과 segment를 갱신한다."""

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
INPUT_WINDOW_LABELS_V2 = PROJECT_ROOT / "data/windows/window_labels_5s_v2.csv"
INPUT_AD_SEGMENTS_V2 = PROJECT_ROOT / "data/segments/ad_interval_segments_v2.csv"
INPUT_CONTEXT_SEGMENTS_V2 = PROJECT_ROOT / "data/segments/ad_context_segments_v2.csv"
INPUT_COMBINED_SEGMENTS_V2 = PROJECT_ROOT / "data/segments/label_interval_context_segments_v2.csv"
INPUT_REPORT_V2 = PROJECT_ROOT / "reports/create_5s_video_windows_v2_report.json"

OUTPUT_REFRESHED_LABEL = PROJECT_ROOT / "data/labels/clean_ad_labels_v0_scope_refreshed_v2_1.csv"
OUTPUT_AMBIGUOUS_AUDIT = PROJECT_ROOT / "data/video_metadata/ambiguous_title_mapping_audit_v2_1.csv"
MANUAL_OVERRIDE_PATH = PROJECT_ROOT / "data/video_metadata/manual_video_title_mapping_v2_1.csv"
MANUAL_OVERRIDE_TEMPLATE = PROJECT_ROOT / "data/video_metadata/manual_video_title_mapping_v2_1_template.csv"
OUTPUT_WINDOW_LABELS = PROJECT_ROOT / "data/windows/window_labels_5s_v2_1.csv"
OUTPUT_AD_SEGMENTS = PROJECT_ROOT / "data/segments/ad_interval_segments_v2_1.csv"
OUTPUT_CONTEXT_SEGMENTS = PROJECT_ROOT / "data/segments/ad_context_segments_v2_1.csv"
OUTPUT_COMBINED_SEGMENTS = PROJECT_ROOT / "data/segments/label_interval_context_segments_v2_1.csv"
OUTPUT_CLIPPED_AUDIT = PROJECT_ROOT / "data/segments/clipped_segment_audit_v2_1.csv"
REPORT_PATH = PROJECT_ROOT / "reports/fix_label_mapping_and_refresh_window_segments_v2_1_report.json"
SUMMARY_PATH = PROJECT_ROOT / "reports/fix_label_mapping_and_refresh_window_segments_v2_1_summary.md"
LOG_PATH = PROJECT_ROOT / "logs/fix_label_mapping_and_refresh_window_segments_v2_1_run_log.txt"
README_PATH = PROJECT_ROOT / "README.md"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt"
LATEST_README = LATEST_DIR / "README_latest_files.md"
CHECK_ENV_SCRIPT = PROJECT_ROOT / "scripts/utils/check_cv_environment.py"

ALLOWED_SCOPE_VALUES = {"yes", "no", "unclear", "not_reviewed"}
CONTEXT_WINDOW_SEC = 10.0
RUN_LOG: list[str] = []


def log(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    RUN_LOG.append(f"[{timestamp}] {message}")
    print(f"[fix_v2_1] {message}")


def ensure_inside_project(path: Path) -> Path:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and not str(resolved).startswith(str(root) + os.sep):
        raise RuntimeError(f"Refusing to write outside project root: {resolved}")
    return path


def ensure_dirs() -> None:
    for folder in [
        PROJECT_ROOT / "data/labels",
        PROJECT_ROOT / "data/video_metadata",
        PROJECT_ROOT / "data/windows",
        PROJECT_ROOT / "data/segments",
        PROJECT_ROOT / "reports",
        PROJECT_ROOT / "logs",
        LATEST_DIR,
    ]:
        ensure_inside_project(folder).mkdir(parents=True, exist_ok=True)


def normalize_title(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.strip().casefold()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^0-9a-z가-힣\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else None


def bool_text(value: bool) -> str:
    return "true" if value else "false"


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


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_inside_project(path)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: fmt_value(row.get(column, "")) for column in columns})


def verify_cv_environment() -> tuple[bool, str, list[str]]:
    warnings: list[str] = []
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


def label_valid_mask(df: pd.DataFrame) -> pd.Series:
    if "label_valid" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df["label_valid"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def load_review_scope_values(warnings: list[Any]) -> tuple[dict[str, str], bool]:
    if not INPUT_REVIEW_XLSX.exists():
        warnings.append("review_xlsx_missing")
        return {}, False
    try:
        review = pd.read_excel(INPUT_REVIEW_XLSX, sheet_name="label_review")
    except Exception as exc:
        warnings.append({"review_xlsx_read_failed": repr(exc)})
        return {}, False
    if "ad_interval_id" not in review.columns or "is_abrupt_transition_ad" not in review.columns:
        warnings.append("review_xlsx_missing_ad_interval_or_scope_column")
        return {}, True
    scope_by_interval: dict[str, str] = {}
    for _, row in review.iterrows():
        interval_id = clean_value(row.get("ad_interval_id"))
        scope_value = clean_value(row.get("is_abrupt_transition_ad"))
        if interval_id and scope_value:
            scope_by_interval[interval_id] = scope_value
    return scope_by_interval, True


def create_refreshed_labels(clean_df: pd.DataFrame, warnings: list[Any]) -> tuple[pd.DataFrame, dict[str, int], dict[str, int], list[dict[str, Any]], bool]:
    review_scope, review_found = load_review_scope_values(warnings)
    refreshed = clean_df.copy()
    original_values: list[str] = []
    refreshed_values: list[str] = []
    sources: list[str] = []
    valid_flags: list[str] = []
    scope_warnings: list[str] = []
    invalid_scope_values: list[dict[str, Any]] = []

    for idx, row in refreshed.iterrows():
        interval_id = clean_value(row.get("ad_interval_id"))
        original = clean_value(row.get("is_abrupt_transition_ad"))
        if interval_id in review_scope and review_scope[interval_id] != "":
            chosen = review_scope[interval_id]
            source = "review_xlsx"
        else:
            chosen = original
            source = "clean_csv"

        normalized = chosen.strip().lower()
        is_valid = normalized in ALLOWED_SCOPE_VALUES
        warning = ""
        if not is_valid:
            warning = "invalid_scope_value"
            invalid_scope_values.append(
                {
                    "row_index": int(idx),
                    "ad_interval_id": interval_id,
                    "raw_scope_value": chosen,
                    "scope_value_source": source,
                }
            )
            refreshed_value = chosen
        else:
            refreshed_value = normalized

        original_values.append(original)
        refreshed_values.append(refreshed_value)
        sources.append(source)
        valid_flags.append(bool_text(is_valid))
        scope_warnings.append(warning)

    refreshed["is_abrupt_transition_ad_original"] = original_values
    refreshed["is_abrupt_transition_ad_refreshed"] = refreshed_values
    refreshed["scope_value_source"] = sources
    refreshed["scope_value_valid"] = valid_flags
    refreshed["scope_warning"] = scope_warnings

    before = dict(sorted(Counter(original_values).items()))
    after = dict(sorted(Counter(refreshed_values).items()))
    return refreshed, before, after, invalid_scope_values, review_found


def candidate_labels_for_ambiguous(ambiguous_row: pd.Series, refreshed_df: pd.DataFrame, candidate_ids: list[str]) -> pd.DataFrame:
    normalized_stem = clean_value(ambiguous_row.get("normalized_file_stem"))
    df = refreshed_df.copy()
    df["_video_id_text"] = df["video_id"].map(clean_value)
    df["_normalized_title"] = df["video_title"].map(normalize_title)
    return df[df["_video_id_text"].isin(candidate_ids) & df["_normalized_title"].eq(normalized_stem)].copy()


def read_manual_overrides(warnings: list[Any]) -> tuple[dict[str, dict[str, str]], bool]:
    if not MANUAL_OVERRIDE_PATH.exists():
        return {}, False
    overrides_df = pd.read_csv(MANUAL_OVERRIDE_PATH)
    required = {"video_filename", "selected_video_id", "selected_video_title", "resolution_note"}
    missing = sorted(required - set(overrides_df.columns))
    if missing:
        warnings.append({"manual_override_missing_columns": missing})
        return {}, True
    overrides: dict[str, dict[str, str]] = {}
    for _, row in overrides_df.iterrows():
        filename = clean_value(row.get("video_filename"))
        selected = clean_value(row.get("selected_video_id"))
        if filename and selected:
            overrides[filename] = {
                "selected_video_id": selected,
                "selected_video_title": clean_value(row.get("selected_video_title")),
                "resolution_note": clean_value(row.get("resolution_note")),
            }
    return overrides, True


def create_manual_template(ambiguous_manifest: pd.DataFrame) -> bool:
    if MANUAL_OVERRIDE_PATH.exists():
        return False
    rows = [
        {
            "video_filename": clean_value(row.get("video_filename")),
            "selected_video_id": "",
            "selected_video_title": "",
            "resolution_note": "Fill this file only when manually resolving ambiguous title mapping.",
        }
        for _, row in ambiguous_manifest.iterrows()
    ]
    write_csv(MANUAL_OVERRIDE_TEMPLATE, rows, ["video_filename", "selected_video_id", "selected_video_title", "resolution_note"])
    return True


def create_ambiguous_audit(manifest_df: pd.DataFrame, refreshed_df: pd.DataFrame, report_v2: dict[str, Any], warnings: list[Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], bool, bool]:
    ambiguous_manifest = manifest_df[manifest_df["title_match_status"].astype(str).eq("ambiguous")].copy()
    overrides, override_found = read_manual_overrides(warnings)
    template_created = create_manual_template(ambiguous_manifest)
    candidate_ids_by_normalized: dict[str, list[str]] = {}
    for item in report_v2.get("ambiguous_video_title_matches", []):
        normalized = clean_value(item.get("normalized_title"))
        candidate_ids = [clean_value(value) for value in item.get("candidate_video_ids", []) if clean_value(value)]
        if normalized and candidate_ids:
            candidate_ids_by_normalized[normalized] = candidate_ids

    audit_rows: list[dict[str, Any]] = []
    resolution_by_filename: dict[str, dict[str, Any]] = {}
    for _, video in ambiguous_manifest.iterrows():
        filename = clean_value(video.get("video_filename"))
        normalized_stem = clean_value(video.get("normalized_file_stem"))
        candidate_ids = candidate_ids_by_normalized.get(normalized_stem, [])
        candidates = candidate_labels_for_ambiguous(video, refreshed_df, candidate_ids) if candidate_ids else pd.DataFrame()

        selected_video_id = ""
        resolution_status = "unresolved"
        resolution_note = "manual override not found and automatic resolution criteria not met"
        if filename in overrides:
            selected_video_id = overrides[filename]["selected_video_id"]
            resolution_status = "resolved_manual_override"
            resolution_note = overrides[filename]["resolution_note"] or "manual override selected video_id"
        elif not candidates.empty:
            valid_by_id = {
                clean_value(video_id): bool(label_valid_mask(group).any())
                for video_id, group in candidates.groupby("video_id")
            }
            valid_ids = [video_id for video_id, is_valid in valid_by_id.items() if is_valid]
            if len(valid_ids) == 1 and len(valid_by_id) > 1:
                selected_video_id = valid_ids[0]
                resolution_status = "resolved_single_valid_candidate"
                resolution_note = "only one candidate video_id has label_valid=true"
            elif len(valid_by_id) == 1:
                selected_video_id = next(iter(valid_by_id))
                resolution_status = "resolved_single_candidate"
                resolution_note = "only one candidate video_id remains in refreshed labels"

        if selected_video_id:
            selected_rows = candidates[candidates["video_id"].map(clean_value).eq(selected_video_id)] if not candidates.empty else pd.DataFrame()
            selected_title = clean_value(selected_rows.iloc[0]["video_title"]) if not selected_rows.empty else overrides.get(filename, {}).get("selected_video_title", "")
            resolution_by_filename[filename] = {
                "status": resolution_status,
                "selected_video_id": selected_video_id,
                "selected_video_title": selected_title,
                "note": resolution_note,
            }
        else:
            resolution_by_filename[filename] = {
                "status": "unresolved_ambiguous",
                "selected_video_id": "",
                "selected_video_title": "",
                "note": resolution_note,
            }

        if candidates.empty:
            audit_rows.append(
                {
                    "video_filename": filename,
                    "file_stem": clean_value(video.get("file_stem")),
                    "normalized_file_stem": normalized_stem,
                    "candidate_video_id": "",
                    "candidate_ad_interval_id": "",
                    "candidate_video_title": "",
                    "candidate_ad_start_sec": "",
                    "candidate_ad_end_sec": "",
                    "candidate_duration_sec": "",
                    "current_title_match_status": clean_value(video.get("title_match_status")),
                    "resolution_status": resolution_by_filename[filename]["status"],
                    "selected_video_id": selected_video_id,
                    "resolution_note": resolution_note,
                }
            )
            continue

        for _, label in candidates.iterrows():
            start = to_float(label.get("ad_start_sec"))
            end = to_float(label.get("ad_end_sec"))
            audit_rows.append(
                {
                    "video_filename": filename,
                    "file_stem": clean_value(video.get("file_stem")),
                    "normalized_file_stem": normalized_stem,
                    "candidate_video_id": clean_value(label.get("video_id")),
                    "candidate_ad_interval_id": clean_value(label.get("ad_interval_id")),
                    "candidate_video_title": clean_value(label.get("video_title")),
                    "candidate_ad_start_sec": start,
                    "candidate_ad_end_sec": end,
                    "candidate_duration_sec": (end - start) if start is not None and end is not None else "",
                    "current_title_match_status": clean_value(video.get("title_match_status")),
                    "resolution_status": resolution_by_filename[filename]["status"],
                    "selected_video_id": selected_video_id,
                    "resolution_note": resolution_note,
                }
            )
    return audit_rows, resolution_by_filename, override_found, template_created


def mapping_for_manifest_row(row: pd.Series, resolution_by_filename: dict[str, dict[str, Any]]) -> dict[str, Any]:
    status = clean_value(row.get("title_match_status"))
    filename = clean_value(row.get("video_filename"))
    if status in {"exact", "normalized"}:
        return {
            "label_mapping_status": "resolved_v2_title_match",
            "label_mapping_warning": "",
            "mapped_video_id": clean_value(row.get("video_id")),
            "mapped_video_title": clean_value(row.get("matched_label_video_title")),
        }
    if status == "ambiguous":
        resolution = resolution_by_filename.get(filename, {})
        if resolution.get("selected_video_id"):
            return {
                "label_mapping_status": resolution.get("status", "resolved_manual_override"),
                "label_mapping_warning": resolution.get("note", ""),
                "mapped_video_id": clean_value(resolution.get("selected_video_id")),
                "mapped_video_title": clean_value(resolution.get("selected_video_title")),
            }
        return {
            "label_mapping_status": "unresolved_ambiguous",
            "label_mapping_warning": resolution.get("note", "ambiguous title unresolved"),
            "mapped_video_id": "",
            "mapped_video_title": "",
        }
    return {
        "label_mapping_status": f"unresolved_{status or 'unknown'}",
        "label_mapping_warning": "no label title match available",
        "mapped_video_id": "",
        "mapped_video_title": "",
    }


def build_manifest_mapping(manifest_df: pd.DataFrame, resolution_by_filename: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for _, row in manifest_df.iterrows():
        filename = clean_value(row.get("video_filename"))
        payload = mapping_for_manifest_row(row, resolution_by_filename)
        payload.update(
            {
                "video_filename": filename,
                "video_path": clean_value(row.get("video_path")),
                "file_stem": clean_value(row.get("file_stem")),
                "title_match_status": clean_value(row.get("title_match_status")),
                "video_duration_sec": to_float(row.get("duration_sec")),
                "video_title_from_manifest": clean_value(row.get("video_title")),
            }
        )
        mapping[filename] = payload
    return mapping


def valid_refreshed_label_rows(refreshed_df: pd.DataFrame) -> pd.DataFrame:
    df = refreshed_df.copy()
    df["_source_label_row_index"] = df.index + 2
    df["_video_id_text"] = df["video_id"].map(clean_value)
    df["_video_title_text"] = df["video_title"].map(clean_value)
    df["_ad_start_sec_num"] = df["ad_start_sec"].map(to_float)
    df["_ad_end_sec_num"] = df["ad_end_sec"].map(to_float)
    df["_label_valid_bool"] = label_valid_mask(df)
    return df[
        df["_label_valid_bool"]
        & df["_video_id_text"].ne("")
        & df["_video_title_text"].ne("")
        & df["_ad_start_sec_num"].notna()
        & df["_ad_end_sec_num"].notna()
        & (df["_ad_start_sec_num"] < df["_ad_end_sec_num"])
    ].copy()


def labels_by_mapping_key(refreshed_df: pd.DataFrame, manifest_mapping: dict[str, dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    resolved_keys = {
        (payload["mapped_video_id"], payload["mapped_video_title"])
        for payload in manifest_mapping.values()
        if payload["mapped_video_id"] and payload["mapped_video_title"]
    }
    label_rows = valid_refreshed_label_rows(refreshed_df)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for _, row in label_rows.iterrows():
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
                "scope_value_valid": clean_value(row.get("scope_value_valid")),
            }
        )
    return grouped


def scope_to_window_label(overlaps: list[dict[str, Any]], window_duration: float) -> str:
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


def create_window_labels_v2_1(window_grid_df: pd.DataFrame, refreshed_df: pd.DataFrame, manifest_mapping: dict[str, dict[str, Any]]) -> pd.DataFrame:
    grouped = labels_by_mapping_key(refreshed_df, manifest_mapping)
    rows: list[dict[str, Any]] = []
    for _, window in window_grid_df.iterrows():
        filename = clean_value(window.get("video_filename"))
        mapping = manifest_mapping.get(filename, {})
        mapped_key = (mapping.get("mapped_video_id", ""), mapping.get("mapped_video_title", ""))
        win_start = float(window.get("window_start_sec"))
        win_end = float(window.get("window_end_sec"))
        win_duration = max(float(window.get("window_duration_sec")), 1e-9)
        overlaps: list[dict[str, Any]] = []
        if mapping.get("mapped_video_id") and mapping.get("mapped_video_title"):
            for label in grouped.get(mapped_key, []):
                overlap = max(0.0, min(win_end, label["ad_end_sec"]) - max(win_start, label["ad_start_sec"]))
                if overlap > 0:
                    overlaps.append({**label, "overlap_sec": overlap, "overlap_ratio": overlap / win_duration})

        overlap_sum = min(sum(item["overlap_sec"] for item in overlaps), win_duration)
        max_overlap = max(overlaps, key=lambda item: item["overlap_sec"], default=None)
        scopes = sorted({item["scope"] for item in overlaps})
        base = window.to_dict()
        base.update(
            {
                "label_mapping_status": mapping.get("label_mapping_status", "mapping_missing"),
                "label_mapping_warning": mapping.get("label_mapping_warning", ""),
                "label_mapping_video_id": mapping.get("mapped_video_id", ""),
                "label_mapping_video_title": mapping.get("mapped_video_title", ""),
                "overlap_any_ad": bool_text(bool(overlaps)),
                "overlap_ad_sec": overlap_sum,
                "overlap_ad_ratio": overlap_sum / win_duration,
                "matched_ad_interval_ids": ";".join(item["ad_interval_id"] for item in overlaps),
                "matched_abrupt_scope_values": ";".join(scopes),
                "max_overlap_ad_interval_id": max_overlap["ad_interval_id"] if max_overlap else "",
                "max_overlap_ad_sec": max_overlap["overlap_sec"] if max_overlap else 0,
                "max_overlap_ad_ratio": max_overlap["overlap_ratio"] if max_overlap else 0,
                "window_label_scope": scope_to_window_label(overlaps, win_duration),
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def create_segment_row(
    segment_id: str,
    segment_type: str,
    boundary_role: str,
    label: pd.Series,
    mapping: dict[str, Any],
    original_start: float,
    original_end: float,
    context_window_sec: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    start = original_start
    end = original_end
    duration = mapping.get("video_duration_sec")
    clipping = False
    warnings: list[str] = []
    clipping_audit: dict[str, Any] | None = None
    if duration is not None:
        clipped_start = max(0.0, start)
        clipped_end = min(float(duration), end)
        clipping = clipped_start != start or clipped_end != end
        start, end = clipped_start, clipped_end
        if clipping:
            warnings.append("segment_clipped_to_video_duration")
            clipping_audit = {
                "source_version": "v2_1",
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
                "clipping_reason": "segment exceeded video duration or started before zero",
                "needs_manual_review": "true",
            }
    if start >= end:
        return None, clipping_audit

    row = {
        "segment_id": segment_id,
        "segment_type": segment_type,
        "boundary_role": boundary_role,
        "video_id": clean_value(label.get("video_id")),
        "video_title": clean_value(label.get("video_title")),
        "video_filename": mapping.get("video_filename", ""),
        "video_path": mapping.get("video_path", ""),
        "file_stem": mapping.get("file_stem", ""),
        "title_match_status": mapping.get("title_match_status", ""),
        "ad_interval_id": clean_value(label.get("ad_interval_id")),
        "source_label_row_index": int(label.get("_source_label_row_index")),
        "segment_start_sec": start,
        "segment_end_sec": end,
        "segment_duration_sec": end - start,
        "context_window_sec": context_window_sec,
        "ad_start_sec": float(label.get("_ad_start_sec_num")),
        "ad_end_sec": float(label.get("_ad_end_sec_num")),
        "is_abrupt_transition_ad": clean_value(label.get("is_abrupt_transition_ad_original")),
        "is_abrupt_transition_ad_refreshed": clean_value(label.get("is_abrupt_transition_ad_refreshed")),
        "scope_value_source": clean_value(label.get("scope_value_source")),
        "label_valid": bool_text(True),
        "video_duration_sec": duration,
        "clipping_applied": bool_text(clipping),
        "segment_valid": "true",
        "segment_warning": "; ".join(warnings),
        "label_mapping_status": mapping.get("label_mapping_status", ""),
        "segment_refresh_version": "v2_1",
    }
    return row, clipping_audit


def create_segments_v2_1(refreshed_df: pd.DataFrame, manifest_mapping: dict[str, dict[str, Any]], warnings: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    mapping_by_key = {
        (payload["mapped_video_id"], payload["mapped_video_title"]): payload
        for payload in manifest_mapping.values()
        if payload.get("mapped_video_id") and payload.get("mapped_video_title")
    }
    labels = valid_refreshed_label_rows(refreshed_df)
    ad_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    clipping_audits: list[dict[str, Any]] = []
    empty_count = 0
    for _, label in labels.iterrows():
        key = (clean_value(label.get("video_id")), clean_value(label.get("video_title")))
        mapping = mapping_by_key.get(key)
        if not mapping:
            continue
        video_id = clean_value(label.get("video_id"))
        interval_id = clean_value(label.get("ad_interval_id"))
        ad_start = float(label.get("_ad_start_sec_num"))
        ad_end = float(label.get("_ad_end_sec_num"))

        segment_specs = [
            (f"{video_id}_{interval_id}_AD", "ad_interval", "", ad_start, ad_end, ""),
            (f"{video_id}_{interval_id}_PRE10", "pre_ad_start_10s", "before_ad_start", max(0.0, ad_start - CONTEXT_WINDOW_SEC), ad_start, CONTEXT_WINDOW_SEC),
            (f"{video_id}_{interval_id}_POST10", "post_ad_end_10s", "after_ad_end", ad_end, ad_end + CONTEXT_WINDOW_SEC, CONTEXT_WINDOW_SEC),
        ]
        for segment_id, segment_type, boundary_role, start, end, context_sec in segment_specs:
            row, clip = create_segment_row(segment_id, segment_type, boundary_role, label, mapping, start, end, context_sec)
            if clip:
                clipping_audits.append(clip)
            if row is None:
                empty_count += 1
                warnings.append({"empty_segment_skipped": segment_id, "reason": "segment_start_sec_gte_segment_end_sec"})
                continue
            if segment_type == "ad_interval":
                ad_rows.append(row)
            else:
                context_rows.append(row)
    combined = ad_rows + context_rows
    return ad_rows, context_rows, combined, clipping_audits, empty_count


def reconstruct_original_segment(row: pd.Series) -> tuple[float | None, float | None, str]:
    segment_type = clean_value(row.get("segment_type"))
    ad_start = to_float(row.get("ad_start_sec"))
    ad_end = to_float(row.get("ad_end_sec"))
    if ad_start is None or ad_end is None:
        return None, None, "ad_start_or_end_missing"
    if segment_type == "ad_interval":
        return ad_start, ad_end, "ad_interval_original_label_bounds"
    if segment_type == "pre_ad_start_10s":
        return max(0.0, ad_start - CONTEXT_WINDOW_SEC), ad_start, "pre_context_original_bounds"
    if segment_type == "post_ad_end_10s":
        return ad_end, ad_end + CONTEXT_WINDOW_SEC, "post_context_exceeded_video_duration"
    return to_float(row.get("segment_start_sec")), to_float(row.get("segment_end_sec")), "unknown_segment_type"


def clipped_audit_from_existing(path: Path, source_version: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "clipping_applied" not in df.columns:
        return []
    clipped = df[df["clipping_applied"].astype(str).str.strip().str.lower().eq("true")].copy()
    rows: list[dict[str, Any]] = []
    for _, row in clipped.iterrows():
        original_start, original_end, reason = reconstruct_original_segment(row)
        rows.append(
            {
                "source_version": source_version,
                "segment_id": clean_value(row.get("segment_id")),
                "segment_type": clean_value(row.get("segment_type")),
                "video_id": clean_value(row.get("video_id")),
                "video_title": clean_value(row.get("video_title")),
                "video_filename": clean_value(row.get("video_filename")),
                "ad_interval_id": clean_value(row.get("ad_interval_id")),
                "original_segment_start_sec": original_start,
                "original_segment_end_sec": original_end,
                "clipped_segment_start_sec": to_float(row.get("segment_start_sec")),
                "clipped_segment_end_sec": to_float(row.get("segment_end_sec")),
                "video_duration_sec": to_float(row.get("video_duration_sec")),
                "clipping_applied": "true",
                "clipping_reason": reason,
                "needs_manual_review": "true",
            }
        )
    return rows


def update_readme() -> None:
    section = """## Window and Segment Refresh v2.1

v2에서 발생한 ambiguous title mapping과 `not_reviewed` scope 문제를 보정하기 위한 refresh 작업이다. 원본 v2 산출물은 보존하고 v2_1 산출물을 새로 생성했다.

`is_abrupt_transition_ad=yes`는 이후 abrupt-transition 광고 탐지의 positive target으로 사용할 수 있다. `is_abrupt_transition_ad=no`는 이번 탐지 범위에서는 out-of-scope 또는 ignore로 사용한다. `unclear`는 학습/평가에서 제외하는 것을 권장한다.

이후 Scene/OCR/audio feature는 v2_1 window/segment 기준으로 생성하는 것을 권장한다.
"""
    text = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else "# youtube_ad_segment_detector\n"
    marker = "## Window and Segment Refresh v2.1"
    if marker in text:
        before, _, after = text.partition(marker)
        next_idx = after.find("\n## ")
        text = before.rstrip() + "\n\n" + section + (after[next_idx:] if next_idx >= 0 else "")
    else:
        text = text.rstrip() + "\n\n" + section
    README_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def summary_text(report: dict[str, Any]) -> str:
    warnings = report.get("warnings", [])
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- 주요 warning 없음"
    clipped_rows = pd.read_csv(OUTPUT_CLIPPED_AUDIT) if OUTPUT_CLIPPED_AUDIT.exists() else pd.DataFrame()
    clipped_text = clipped_rows.to_string(index=False) if not clipped_rows.empty else "clipped segment 없음"
    return f"""# fix_label_mapping_and_refresh_window_segments_v2_1 summary

## 결과

- ambiguous title count before/after: {report['ambiguous_title_count_before']} / {report['ambiguous_title_count_after']}
- unresolved ambiguous count: {report['unresolved_ambiguous_count']}
- review xlsx scope 반영: {report['review_xlsx_found']}
- manual override 사용: {report['manual_override_file_found']}

## Scope 분포

- before: `{report['scope_distribution_before']}`
- after: `{report['scope_distribution_after']}`

## Window Label Scope 분포

- before: `{report['window_label_scope_distribution_before']}`
- after: `{report['window_label_scope_distribution_after']}`

## Segment 수

- ad_interval before/after: {report['ad_interval_segment_count_before']} / {report['ad_interval_segment_count_after']}
- combined before/after: {report['combined_segment_count_before']} / {report['combined_segment_count_after']}

## Clipped Segment Audit

```text
{clipped_text}
```

## 아직 사람이 확인해야 할 사항

- ambiguous title 영상은 manual override가 없고 자동 판별 조건도 충족하지 않아 unresolved 상태이다.
- `is_abrupt_transition_ad` 값은 review xlsx에도 모두 `not_reviewed` 상태이다.

## Warning

{warning_text}

## 다음 작업

1. `data/video_metadata/manual_video_title_mapping_v2_1_template.csv`를 참고해 ambiguous 영상의 `selected_video_id`를 확정한다.
2. `clean_ad_labels_v0_review.xlsx`에서 `is_abrupt_transition_ad`를 yes/no/unclear로 검토한다.
3. 이후 v2_1 refresh를 다시 실행한 뒤 Scene/OCR/audio feature를 생성한다.
"""


def clear_and_refresh_latest(copied_sources: list[Path]) -> tuple[bool, list[str]]:
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
    for src in copied_sources:
        if src.exists():
            dst = LATEST_DIR / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    descriptions = {
        "README.md": "프로젝트 README와 v2.1 refresh 설명",
        "clean_ad_labels_v0_scope_refreshed_v2_1.csv": "review xlsx 우선 반영 scope refreshed label",
        "ambiguous_title_mapping_audit_v2_1.csv": "ambiguous title 후보와 resolution 상태 audit",
        "manual_video_title_mapping_v2_1_template.csv": "manual override 작성 템플릿",
        "window_labels_5s_v2_1.csv": "refreshed label 기준 5초 window overlap/scope",
        "ad_interval_segments_v2_1.csv": "refreshed label 기준 광고 interval segment",
        "ad_context_segments_v2_1.csv": "refreshed label 기준 광고 전후 10초 context segment",
        "label_interval_context_segments_v2_1.csv": "v2.1 interval/context 통합 segment",
        "clipped_segment_audit_v2_1.csv": "v2/v2.1 clipped segment audit",
        "fix_label_mapping_and_refresh_window_segments_v2_1_report.json": "v2.1 상세 report",
        "fix_label_mapping_and_refresh_window_segments_v2_1_summary.md": "v2.1 사람이 읽는 요약",
        "fix_label_mapping_and_refresh_window_segments_v2_1_run_log.txt": "v2.1 실행 log",
    }
    LATEST_README.write_text(
        "# latest_for_chatgpt files\n\n"
        "latest_for_chatgpt는 최신 작업 핵심 파일만 모아둔 복사본 경로이다. 원본 파일은 프로젝트 내부 원래 경로에 존재한다.\n\n"
        "이번 작업명: fix_label_mapping_and_refresh_window_segments_v2_1\n\n"
        "복사된 파일 목록과 목적:\n\n"
        + "\n".join(f"- `{Path(path).name}`: {descriptions.get(Path(path).name, 'v2.1 output')}" for path in copied)
        + "\n\nmp4 영상 파일, 원본 xlsx, 프레임, 모델, checkpoint, cache 파일은 복사하지 않는다.\n",
        encoding="utf-8",
    )
    copied.append(str(LATEST_README))
    return True, copied


def main() -> None:
    ensure_dirs()
    warnings: list[Any] = []
    errors: list[Any] = []
    log("Started fix_label_mapping_and_refresh_window_segments_v2_1.")
    log("No mp4 modification, clipping, frame extraction, OCR, Scene, Audio, ResNet, Gemma, threshold tuning, or evaluation will be run.")

    cv_ok, python_executable, cv_warnings = verify_cv_environment()
    warnings.extend(cv_warnings)
    if not cv_ok:
        errors.append("cv_environment_check_failed")

    report_v2 = json.loads(INPUT_REPORT_V2.read_text(encoding="utf-8")) if INPUT_REPORT_V2.exists() else {}
    before_keys = [
        "mp4_file_count",
        "valid_video_metadata_count",
        "title_match_exact_count",
        "title_match_normalized_count",
        "title_match_ambiguous_count",
        "videos_without_label_title_match",
        "label_titles_without_video_file",
        "ambiguous_video_title_matches",
        "window_label_scope_distribution",
        "abrupt_scope_distribution",
        "clipped_segment_count",
        "duplicate_window_id_count",
        "duplicate_segment_id_count",
    ]
    log("v2 before state: " + json.dumps({key: report_v2.get(key) for key in before_keys}, ensure_ascii=False))

    clean_df = pd.read_csv(INPUT_CLEAN_LABEL)
    manifest_df = pd.read_csv(INPUT_MANIFEST_V2)
    window_grid_df = pd.read_csv(INPUT_WINDOW_GRID_V2)
    window_labels_v2_df = read_csv_if_exists(INPUT_WINDOW_LABELS_V2)
    ad_segments_v2_df = read_csv_if_exists(INPUT_AD_SEGMENTS_V2)
    combined_v2_df = read_csv_if_exists(INPUT_COMBINED_SEGMENTS_V2)

    refreshed_df, scope_before, scope_after, invalid_scope_values, review_found = create_refreshed_labels(clean_df, warnings)
    if invalid_scope_values:
        warnings.append({"invalid_scope_values": invalid_scope_values})
    refreshed_df.to_csv(OUTPUT_REFRESHED_LABEL, index=False, encoding="utf-8-sig")
    log(f"Wrote refreshed labels: {OUTPUT_REFRESHED_LABEL}")

    ambiguous_audit_rows, resolution_by_filename, manual_override_found, template_created = create_ambiguous_audit(manifest_df, refreshed_df, report_v2, warnings)
    ambiguous_columns = [
        "video_filename",
        "file_stem",
        "normalized_file_stem",
        "candidate_video_id",
        "candidate_ad_interval_id",
        "candidate_video_title",
        "candidate_ad_start_sec",
        "candidate_ad_end_sec",
        "candidate_duration_sec",
        "current_title_match_status",
        "resolution_status",
        "selected_video_id",
        "resolution_note",
    ]
    write_csv(OUTPUT_AMBIGUOUS_AUDIT, ambiguous_audit_rows, ambiguous_columns)
    log(f"Wrote ambiguous title audit: {OUTPUT_AMBIGUOUS_AUDIT}")

    manifest_mapping = build_manifest_mapping(manifest_df, resolution_by_filename)
    window_labels_v2_1 = create_window_labels_v2_1(window_grid_df, refreshed_df, manifest_mapping)
    window_labels_v2_1.to_csv(OUTPUT_WINDOW_LABELS, index=False, encoding="utf-8-sig")
    log(f"Wrote v2.1 window labels: {OUTPUT_WINDOW_LABELS}")

    ad_rows, context_rows, combined_rows, clipping_audits_v2_1, empty_segment_count = create_segments_v2_1(refreshed_df, manifest_mapping, warnings)
    segment_columns = [
        "segment_id",
        "segment_type",
        "boundary_role",
        "video_id",
        "video_title",
        "video_filename",
        "video_path",
        "file_stem",
        "title_match_status",
        "ad_interval_id",
        "source_label_row_index",
        "segment_start_sec",
        "segment_end_sec",
        "segment_duration_sec",
        "context_window_sec",
        "ad_start_sec",
        "ad_end_sec",
        "is_abrupt_transition_ad",
        "is_abrupt_transition_ad_refreshed",
        "scope_value_source",
        "label_valid",
        "video_duration_sec",
        "clipping_applied",
        "segment_valid",
        "segment_warning",
        "label_mapping_status",
        "segment_refresh_version",
    ]
    write_csv(OUTPUT_AD_SEGMENTS, ad_rows, segment_columns)
    write_csv(OUTPUT_CONTEXT_SEGMENTS, context_rows, segment_columns)
    write_csv(OUTPUT_COMBINED_SEGMENTS, combined_rows, segment_columns)
    log("Wrote v2.1 segment CSV outputs.")

    clipped_audit_rows = clipped_audit_from_existing(INPUT_COMBINED_SEGMENTS_V2, "v2") + clipping_audits_v2_1
    clipped_columns = [
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
    ]
    write_csv(OUTPUT_CLIPPED_AUDIT, clipped_audit_rows, clipped_columns)
    log(f"Wrote clipped segment audit: {OUTPUT_CLIPPED_AUDIT}")

    update_readme()
    log("Updated README v2.1 section.")

    duplicate_window_id_count = int(window_labels_v2_1["window_id"].duplicated().sum()) if "window_id" in window_labels_v2_1.columns else 0
    duplicate_segment_id_count = sum(1 for _, count in Counter(row["segment_id"] for row in combined_rows).items() if count > 1)
    if duplicate_window_id_count:
        errors.append("duplicate_window_id_detected")
    if duplicate_segment_id_count:
        errors.append("duplicate_segment_id_detected")

    unresolved_ambiguous_count = sum(1 for value in resolution_by_filename.values() if value.get("status") == "unresolved_ambiguous")
    ambiguous_before = int(report_v2.get("title_match_ambiguous_count", 0) or 0)
    ambiguous_after = unresolved_ambiguous_count
    window_scope_after = dict(sorted(Counter(window_labels_v2_1["window_label_scope"].map(clean_value)).items()))
    clipped_after = sum(1 for row in combined_rows if row.get("clipping_applied") == "true")
    if unresolved_ambiguous_count:
        warnings.append({"unresolved_ambiguous_count": unresolved_ambiguous_count})
    if scope_after.get("not_reviewed", 0):
        warnings.append("is_abrupt_transition_ad still contains not_reviewed after refresh")
    if template_created:
        warnings.append(f"manual override template created: {MANUAL_OVERRIDE_TEMPLATE}")

    generated_files = [
        OUTPUT_REFRESHED_LABEL,
        OUTPUT_AMBIGUOUS_AUDIT,
        MANUAL_OVERRIDE_TEMPLATE if template_created or MANUAL_OVERRIDE_TEMPLATE.exists() else None,
        OUTPUT_WINDOW_LABELS,
        OUTPUT_AD_SEGMENTS,
        OUTPUT_CONTEXT_SEGMENTS,
        OUTPUT_COMBINED_SEGMENTS,
        OUTPUT_CLIPPED_AUDIT,
        REPORT_PATH,
        SUMMARY_PATH,
        LOG_PATH,
    ]
    generated_files_str = [str(path) for path in generated_files if path is not None]

    report = {
        "project_root": str(PROJECT_ROOT),
        "cv_environment_checked": cv_ok,
        "python_executable": python_executable,
        "old_project_modified": False,
        "input_clean_label_path": str(INPUT_CLEAN_LABEL),
        "input_review_xlsx_path": str(INPUT_REVIEW_XLSX),
        "input_manifest_v2_path": str(INPUT_MANIFEST_V2),
        "input_window_grid_v2_path": str(INPUT_WINDOW_GRID_V2),
        "input_report_v2_path": str(INPUT_REPORT_V2),
        "review_xlsx_found": review_found,
        "manual_override_file_found": manual_override_found,
        "manual_override_template_created": template_created,
        "ambiguous_title_count_before": ambiguous_before,
        "ambiguous_title_count_after": ambiguous_after,
        "unresolved_ambiguous_count": unresolved_ambiguous_count,
        "label_row_count": int(len(refreshed_df)),
        "label_valid_count": int(label_valid_mask(refreshed_df).sum()),
        "scope_distribution_before": scope_before,
        "scope_distribution_after": scope_after,
        "window_label_scope_distribution_before": report_v2.get("window_label_scope_distribution", {}),
        "window_label_scope_distribution_after": window_scope_after,
        "ad_interval_segment_count_before": int(report_v2.get("ad_interval_segment_count", len(ad_segments_v2_df)) or 0),
        "ad_interval_segment_count_after": len(ad_rows),
        "combined_segment_count_before": int(report_v2.get("combined_segment_count", len(combined_v2_df)) or 0),
        "combined_segment_count_after": len(combined_rows),
        "clipped_segment_count_before": int(report_v2.get("clipped_segment_count", 0) or 0),
        "clipped_segment_count_after": clipped_after,
        "clipped_segment_audit_path": str(OUTPUT_CLIPPED_AUDIT),
        "duplicate_window_id_count": duplicate_window_id_count,
        "duplicate_segment_id_count": duplicate_segment_id_count,
        "empty_segment_count": empty_segment_count,
        "invalid_scope_values": invalid_scope_values,
        "ambiguous_title_mapping_audit_path": str(OUTPUT_AMBIGUOUS_AUDIT),
        "refreshed_label_path": str(OUTPUT_REFRESHED_LABEL),
        "window_labels_v2_1_path": str(OUTPUT_WINDOW_LABELS),
        "ad_interval_segments_v2_1_path": str(OUTPUT_AD_SEGMENTS),
        "ad_context_segments_v2_1_path": str(OUTPUT_CONTEXT_SEGMENTS),
        "label_interval_context_segments_v2_1_path": str(OUTPUT_COMBINED_SEGMENTS),
        "generated_files": generated_files_str,
        "warnings": warnings,
        "errors": errors,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(summary_text(report), encoding="utf-8")
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")

    latest_sources = [
        README_PATH,
        OUTPUT_REFRESHED_LABEL,
        OUTPUT_AMBIGUOUS_AUDIT,
        MANUAL_OVERRIDE_TEMPLATE if MANUAL_OVERRIDE_TEMPLATE.exists() else None,
        OUTPUT_WINDOW_LABELS,
        OUTPUT_AD_SEGMENTS,
        OUTPUT_CONTEXT_SEGMENTS,
        OUTPUT_COMBINED_SEGMENTS,
        OUTPUT_CLIPPED_AUDIT,
        REPORT_PATH,
        SUMMARY_PATH,
        LOG_PATH,
    ]
    latest_ok, latest_files = clear_and_refresh_latest([path for path in latest_sources if path is not None])
    report["latest_for_chatgpt_updated"] = latest_ok
    report["latest_for_chatgpt_files"] = latest_files
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(summary_text(report), encoding="utf-8")
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    if latest_ok:
        shutil.copy2(REPORT_PATH, LATEST_DIR / REPORT_PATH.name)
        shutil.copy2(SUMMARY_PATH, LATEST_DIR / SUMMARY_PATH.name)
        shutil.copy2(LOG_PATH, LATEST_DIR / LOG_PATH.name)

    log("Finished fix_label_mapping_and_refresh_window_segments_v2_1.")
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    if latest_ok:
        shutil.copy2(LOG_PATH, LATEST_DIR / LOG_PATH.name)

    print(
        json.dumps(
            {
                "ambiguous_title_count_before": ambiguous_before,
                "ambiguous_title_count_after": ambiguous_after,
                "unresolved_ambiguous_count": unresolved_ambiguous_count,
                "scope_distribution_before": scope_before,
                "scope_distribution_after": scope_after,
                "window_label_scope_distribution_after": window_scope_after,
                "clipped_segment_count_after": clipped_after,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
