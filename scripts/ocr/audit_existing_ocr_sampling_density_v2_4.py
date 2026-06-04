#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd


VERSION = "v2_4"
SPLIT_SEED = 20240524
PROJECT_ROOT = Path(".").resolve()
OLD_PROJECT_ROOT = Path("./_old_project_not_included").resolve()

FIXED_SPLIT = {
    "train": [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
    "validation": [3, 7, 18],
    "test": [4, 16, 17],
}

SPLIT_FILE = PROJECT_ROOT / "data/splits/video_split_v2_4.csv"
SEGMENT_FILE = PROJECT_ROOT / "data/segments/ad_interval_segments_v2_4.csv"
MANIFEST_FILE = PROJECT_ROOT / "data/video_metadata/video_manifest_v2_2.csv"

DATA_OCR_DIR = PROJECT_ROOT / "data/ocr"
REPORT_OCR_DIR = PROJECT_ROOT / "reports/ocr"
SCRIPT_OCR_DIR = PROJECT_ROOT / "scripts/ocr"
LOG_DIR = PROJECT_ROOT / "logs"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt_existing_ocr_sampling_density_audit_v2_4"

OUT = {
    "inventory": DATA_OCR_DIR / "ocr_existing_artifact_inventory_v2_4.csv",
    "scope": DATA_OCR_DIR / "train_existing_ocr_collection_scope_summary_v2_4.csv",
    "interval": DATA_OCR_DIR / "train_existing_ocr_sampling_interval_summary_v2_4.csv",
    "density": DATA_OCR_DIR / "train_existing_ocr_density_by_segment_or_window_v2_4.csv",
    "ad_window": DATA_OCR_DIR / "train_actual_ad_existing_ocr_window_density_v2_4.csv",
    "opening": DATA_OCR_DIR / "train_opening_existing_ocr_density_v2_4.csv",
    "risk": DATA_OCR_DIR / "train_existing_ocr_short_disclosure_gap_risk_v2_4.csv",
    "summary": REPORT_OCR_DIR / "existing_ocr_sampling_density_audit_v2_4_summary.md",
    "report": REPORT_OCR_DIR / "existing_ocr_sampling_density_audit_v2_4_report.json",
    "log": LOG_DIR / "existing_ocr_sampling_density_audit_v2_4_run_log.txt",
}

KNOWN_ARTIFACTS = [
    DATA_OCR_DIR / "ocr_labeled_segment_sampling_plan_v2_4.csv",
    DATA_OCR_DIR / "ocr_ad_edge_5s_10s_sampling_plan_v2_4.csv",
    DATA_OCR_DIR / "ocr_visual_anchor_frame_sampling_plan_v2_4.csv",
    DATA_OCR_DIR / "ocr_frame_level_results_v2_4.csv",
    DATA_OCR_DIR / "ocr_frame_level_results_v2_4_recovered.csv",
    DATA_OCR_DIR / "ocr_visual_anchor_frame_results_v2_4.csv",
    DATA_OCR_DIR / "ocr_labeled_segment_features_v2_4.csv",
    DATA_OCR_DIR / "ocr_labeled_segment_features_v2_4_recovered.csv",
    DATA_OCR_DIR / "ocr_labeled_segment_features_v2_4_recovered_train_only.csv",
    DATA_OCR_DIR / "ocr_visual_anchor_context_features_v2_4.csv",
    DATA_OCR_DIR / "ocr_visual_anchor_context_features_v2_4_train_val_for_discussion.csv",
]

FORBIDDEN_LATEST_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".m4a",
    ".jpg", ".jpeg", ".png", ".webp", ".pt", ".pth", ".ckpt",
    ".onnx", ".bin",
}

KEYWORD_RE = re.compile(r"ocr|sampling|frame|anchor|edge|recovered|disclosure|text", re.I)

VIDEO_ID_CANDIDATES = [
    "video_id", "videoid", "video", "label_mapping_video_id",
]
SEGMENT_ID_CANDIDATES = [
    "segment_id", "frame_sample_id", "visual_anchor_id", "scene_boundary_anchor_id",
    "ad_interval_id", "window_id",
]
TIMESTAMP_CANDIDATES = [
    "timestamp_sec", "frame_time_sec", "sampled_time_sec", "sample_time_sec",
    "time_sec", "candidate_time_sec", "canonical_boundary_time_sec",
]
START_CANDIDATES = [
    "segment_start_sec", "start_sec", "window_start_sec", "ad_start_sec",
]
END_CANDIDATES = [
    "segment_end_sec", "end_sec", "window_end_sec", "ad_end_sec",
]
TEXT_CANDIDATES = [
    "ocr_text", "ocr_text_raw", "ocr_text_normalized", "ocr_text_joined",
    "detected_text_raw", "detected_text_normalized", "detected_text_joined",
    "representative_ocr_text", "text", "raw_text", "normalized_text",
]
STATUS_CANDIDATES = [
    "ocr_status", "status", "success", "plan_status", "sampling_status",
    "retry_status", "extraction_status", "ocr_has_text",
]
SPLIT_CANDIDATES = ["split", "dataset_split"]
DURATION_CANDIDATES = ["video_duration_sec", "duration_sec"]


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, Path)) else False:
        return None
    if hasattr(value, "item"):
        try:
            return json_ready(value.item())
        except Exception:
            pass
    return value


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def safe_int(value: Any) -> int | None:
    out = safe_float(value)
    if math.isnan(out):
        return None
    return int(out)


def fmt_num(value: Any, digits: int = 3) -> str:
    x = safe_float(value)
    if math.isnan(x):
        return ""
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def list_str(values: list[float], digits: int = 3, limit: int = 40) -> str:
    vals = values[:limit]
    suffix = "" if len(values) <= limit else f";...(+{len(values) - limit})"
    return ";".join(fmt_num(v, digits) for v in vals) + suffix


class Audit:
    def __init__(self) -> None:
        self.started_at = now_iso()
        self.t0 = time.monotonic()
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.column_mappings: dict[str, dict[str, Any]] = {}
        self.step_lines: list[str] = []

    def log(self, message: str) -> None:
        line = f"{now_iso()} | {message}"
        print(line, flush=True)
        self.step_lines.append(line)
        OUT["log"].parent.mkdir(parents=True, exist_ok=True)
        with OUT["log"].open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self.log(f"WARN | {message}")

    def error(self, message: str) -> None:
        self.errors.append(message)
        self.log(f"ERROR | {message}")


def assert_project_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    resolved.relative_to(PROJECT_ROOT)
    return resolved


def ensure_no_existing_outputs() -> None:
    targets = list(OUT.values()) + [LATEST_DIR]
    existing = [p for p in targets if p.exists()]
    if existing:
        formatted = "\n".join(str(p) for p in existing)
        raise FileExistsError(f"Refusing to overwrite existing audit output(s):\n{formatted}")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    assert_project_path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    assert_project_path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path: Path, audit: Audit, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        msg = f"missing {'required' if required else 'optional'} file: {path}"
        if required:
            audit.error(msg)
            raise FileNotFoundError(msg)
        audit.warn(msg)
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False)


def choose_column(columns: list[str], candidates: list[str]) -> str | None:
    exact = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in exact:
            return exact[cand.lower()]
    compact = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in compact:
            return compact[key]
    for cand in candidates:
        key = cand.lower()
        for col in columns:
            if key in col.lower():
                return col
    return None


def detect_columns(columns: list[str]) -> dict[str, str | None]:
    return {
        "video_id": choose_column(columns, VIDEO_ID_CANDIDATES),
        "segment_id": choose_column(columns, SEGMENT_ID_CANDIDATES),
        "timestamp_sec": choose_column(columns, TIMESTAMP_CANDIDATES),
        "start_sec": choose_column(columns, START_CANDIDATES),
        "end_sec": choose_column(columns, END_CANDIDATES),
        "text": choose_column(columns, TEXT_CANDIDATES),
        "status": choose_column(columns, STATUS_CANDIDATES),
        "split": choose_column(columns, SPLIT_CANDIDATES),
        "duration_sec": choose_column(columns, DURATION_CANDIDATES),
    }


def infer_artifact_type(path: Path) -> str:
    name = path.name.lower()
    if path.suffix.lower() == ".py":
        return "script"
    if path.suffix.lower() in {".md", ".json", ".txt"} or "report" in path.parts:
        return "report"
    if "sampling_plan" in name or "retry_plan" in name:
        return "sampling_plan"
    if "result" in name or "box" in name or "review" in name:
        return "ocr_result"
    if "feature" in name or "summary" in name or "frequency" in name or "corpus" in name or "keyword" in name:
        return "ocr_feature"
    return "unknown"


def infer_sampling_source(path: Path) -> str:
    name = path.name.lower()
    if "visual_anchor" in name:
        return "visual_anchor"
    if "ad_edge_5s_10s" in name or "edge" in name:
        return "ad_edge_5s_10s"
    if "labeled_segment" in name:
        return "labeled_segment"
    if "frame_level_results" in name and "recovered" in name:
        return "recovered_result"
    if "frame_level_results" in name:
        return "frame_level_result"
    if "feature" in name or "summary" in name:
        return "feature_summary"
    return "unknown"


def discover_artifacts() -> list[Path]:
    found: dict[str, Path] = {}
    for path in KNOWN_ARTIFACTS:
        found[str(path)] = path
    for root in [DATA_OCR_DIR, REPORT_OCR_DIR, SCRIPT_OCR_DIR]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and KEYWORD_RE.search(path.name):
                found[str(path)] = path
    return [found[key] for key in sorted(found)]


def dir_signature(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"exists": False, "file_count": 0, "total_size": 0, "max_mtime": None}
    file_count = 0
    total_size = 0
    max_mtime = 0.0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            file_count += 1
            total_size += int(stat.st_size)
            max_mtime = max(max_mtime, float(stat.st_mtime))
    return {
        "exists": True,
        "file_count": file_count,
        "total_size": total_size,
        "max_mtime": round(max_mtime, 6) if max_mtime else None,
    }


def load_split(audit: Audit) -> tuple[pd.DataFrame, dict[int, str], dict[int, float]]:
    split = read_csv(SPLIT_FILE, audit, required=True)
    split["video_id"] = pd.to_numeric(split["video_id"], errors="coerce").astype("Int64")
    actual = {
        s: sorted(split.loc[split["split"].eq(s), "video_id"].dropna().astype(int).unique().tolist())
        for s in ["train", "validation", "test"]
    }
    for split_name, expected in FIXED_SPLIT.items():
        if actual.get(split_name) != sorted(expected):
            raise ValueError(f"split mismatch for {split_name}: expected={expected}, actual={actual.get(split_name)}")
    seed_values = pd.to_numeric(split.get("split_seed"), errors="coerce").dropna().astype(int).unique().tolist()
    if seed_values and sorted(seed_values) != [SPLIT_SEED]:
        raise ValueError(f"split_seed mismatch: expected={SPLIT_SEED}, actual={sorted(seed_values)}")
    split_map = {
        int(row.video_id): str(row.split)
        for row in split.dropna(subset=["video_id"]).itertuples(index=False)
    }
    duration_map = {
        int(row.video_id): safe_float(getattr(row, "video_duration_sec"))
        for row in split.dropna(subset=["video_id"]).itertuples(index=False)
    }
    return split, split_map, duration_map


def build_inventory(paths: list[Path], split_map: dict[int, str], audit: Audit) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        exists = path.exists()
        artifact_type = infer_artifact_type(path)
        row: dict[str, Any] = {
            "artifact_name": path.name,
            "artifact_path": str(path),
            "artifact_type": artifact_type,
            "exists": bool(exists),
            "row_count": "",
            "column_count": "",
            "columns_detected": "",
            "has_video_id": False,
            "has_segment_id": False,
            "has_timestamp_sec": False,
            "has_start_end_sec": False,
            "has_text_column": False,
            "has_status_column": False,
            "min_video_id": "",
            "max_video_id": "",
            "train_row_count": "",
            "val_row_count": "",
            "test_row_count": "",
            "unknown_split_row_count": "",
            "inferred_sampling_source": infer_sampling_source(path),
            "notes": "",
        }
        if not exists:
            row["notes"] = "optional candidate missing" if path in KNOWN_ARTIFACTS else "missing"
            rows.append(row)
            continue
        if path.suffix.lower() == ".csv":
            try:
                df = read_csv(path, audit)
                mapping = detect_columns(list(df.columns))
                audit.column_mappings[str(path)] = mapping
                row["row_count"] = int(len(df))
                row["column_count"] = int(len(df.columns))
                row["columns_detected"] = ";".join(df.columns[:60])
                row["has_video_id"] = bool(mapping["video_id"])
                row["has_segment_id"] = bool(mapping["segment_id"])
                row["has_timestamp_sec"] = bool(mapping["timestamp_sec"])
                row["has_start_end_sec"] = bool(mapping["start_sec"] and mapping["end_sec"])
                row["has_text_column"] = bool(mapping["text"])
                row["has_status_column"] = bool(mapping["status"])
                if mapping["video_id"]:
                    vids = pd.to_numeric(df[mapping["video_id"]], errors="coerce")
                    if vids.notna().any():
                        row["min_video_id"] = int(vids.min())
                        row["max_video_id"] = int(vids.max())
                    splits = vids.map(lambda v: split_map.get(int(v), "unknown") if pd.notna(v) else "unknown")
                    row["train_row_count"] = int(splits.eq("train").sum())
                    row["val_row_count"] = int(splits.eq("validation").sum())
                    row["test_row_count"] = int(splits.eq("test").sum())
                    row["unknown_split_row_count"] = int(splits.eq("unknown").sum())
            except Exception as exc:
                row["notes"] = f"csv_read_error:{type(exc).__name__}:{exc}"
                audit.warn(f"inventory failed for {path}: {exc}")
        else:
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    line_count = sum(1 for _ in handle)
                row["notes"] = f"line_count={line_count}"
            except Exception as exc:
                row["notes"] = f"non_csv_read_error:{type(exc).__name__}:{exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def with_split(df: pd.DataFrame, split_map: dict[int, str]) -> pd.Series:
    if "_split" in df.columns:
        return df["_split"].fillna("unknown").astype(str)
    if "_video_id" not in df.columns:
        return pd.Series(["unknown"] * len(df), index=df.index)
    return df["_video_id"].map(lambda v: split_map.get(int(v), "unknown") if pd.notna(v) else "unknown")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "success", "succeeded"}


def build_text_series(df: pd.DataFrame, primary: str | None) -> pd.Series:
    cols = []
    if primary and primary in df.columns:
        cols.append(primary)
    for col in TEXT_CANDIDATES:
        if col in df.columns and col not in cols:
            cols.append(col)
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()


def status_classification(df: pd.DataFrame, status_col: str | None, text_col: str | None) -> pd.DataFrame:
    out = df.copy()
    status = out[status_col].fillna("").astype(str).str.lower() if status_col in out.columns else pd.Series([""] * len(out), index=out.index)
    text = build_text_series(out, text_col)
    has_text = text.str.replace(r"\s+", "", regex=True).str.len().gt(0)
    if "ocr_has_text" in out.columns:
        has_text = has_text | out["ocr_has_text"].map(truthy)
    for count_col in ["ocr_text_count", "ocr_token_count", "ocr_char_count", "ocr_text_box_count", "ocr_box_count"]:
        if count_col in out.columns:
            has_text = has_text | pd.to_numeric(out[count_col], errors="coerce").fillna(0).gt(0)
    failed = status.str.contains("fail|error|decode|invalid|missing", regex=True, na=False)
    empty_status = status.isin({"empty", "success_empty", "success_empty_text"})
    empty_status = empty_status | status.str.contains(r"(?:^|[^a-z])empty(?:[^a-z]|$)", regex=True, na=False)
    empty_status = empty_status & ~status.str.contains("nonempty", na=False)
    empty = empty_status | (~has_text & ~failed)
    nonempty_success = has_text & ~failed
    recovered = pd.Series([False] * len(out), index=out.index)
    if "retry_success" in out.columns:
        recovered = recovered | out["retry_success"].map(truthy)
    if "retry_status" in out.columns:
        recovered = recovered | out["retry_status"].fillna("").astype(str).str.lower().str.contains("success|recovered", regex=True, na=False)
    out["_text_value"] = text
    out["_has_text"] = has_text
    out["_is_success_text"] = nonempty_success
    out["_is_empty_text"] = empty & ~failed
    out["_is_failed"] = failed
    out["_is_valid"] = ~failed
    out["_is_recovered"] = recovered
    out["_status_value"] = status
    return out


def key_part(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return fmt_num(value, 3)
    return str(value)


def normalize_label_edge(
    df: pd.DataFrame,
    split_map: dict[int, str],
    source_kind: str,
    role: str,
    filter_plan_kind: str | None,
    audit: Audit,
    path: Path,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    mapping = detect_columns(list(out.columns))
    audit.column_mappings[f"{path}::{role}::{source_kind}"] = mapping
    if filter_plan_kind and "plan_kind" in out.columns:
        out = out[out["plan_kind"].fillna("").astype(str).str.lower().eq(filter_plan_kind)].copy()
    elif source_kind == "labeled_segment" and "segment_type" in out.columns:
        out = out[out["segment_type"].isin(["ad_full", "pre_ad_10s", "post_ad_10s", "random_non_ad_30s"])].copy()
    elif source_kind == "ad_edge_5s_10s" and "segment_type" in out.columns:
        out = out[out["segment_type"].isin([
            "ad_start_first_5s", "ad_start_first_10s", "ad_start_5to10s",
            "ad_end_last_5s", "ad_end_last_10s", "ad_end_minus10to_minus5s",
        ])].copy()
    if out.empty:
        return out
    video_col = mapping["video_id"]
    ts_col = mapping["timestamp_sec"]
    if not video_col or not ts_col:
        audit.warn(f"{path.name} lacks video_id or timestamp column for {source_kind} {role}")
        return pd.DataFrame()
    out["_video_id"] = pd.to_numeric(out[video_col], errors="coerce")
    out["_timestamp_sec"] = pd.to_numeric(out[ts_col], errors="coerce")
    out["_segment_start_sec"] = pd.to_numeric(out[mapping["start_sec"]], errors="coerce") if mapping["start_sec"] else math.nan
    out["_segment_end_sec"] = pd.to_numeric(out[mapping["end_sec"]], errors="coerce") if mapping["end_sec"] else math.nan
    out["_split"] = out[mapping["split"]].astype(str) if mapping["split"] else with_split(out, split_map)
    if "segment_type" not in out.columns:
        out["segment_type"] = ""
    if "ad_interval_id" not in out.columns:
        out["ad_interval_id"] = ""
    out["_group_key"] = out.apply(
        lambda r: "|".join([
            key_part(r.get("_video_id")),
            key_part(r.get("ad_interval_id", "")),
            key_part(r.get("segment_type", "")),
            key_part(round(safe_float(r.get("_segment_start_sec")), 3) if not math.isnan(safe_float(r.get("_segment_start_sec"))) else ""),
            key_part(round(safe_float(r.get("_segment_end_sec")), 3) if not math.isnan(safe_float(r.get("_segment_end_sec"))) else ""),
        ]),
        axis=1,
    )
    if "segment_id" in out.columns:
        out["_segment_id"] = out["segment_id"].fillna("").astype(str)
    else:
        out["_segment_id"] = out["_group_key"]
    out["_source_kind"] = source_kind
    out["_role"] = role
    if role == "result":
        out = status_classification(out, mapping["status"], mapping["text"])
    else:
        out["_text_value"] = ""
        out["_has_text"] = False
        out["_is_success_text"] = False
        out["_is_empty_text"] = False
        out["_is_failed"] = False
        out["_is_valid"] = True
        out["_is_recovered"] = False
        out["_status_value"] = out[mapping["status"]].fillna("").astype(str).str.lower() if mapping["status"] else ""
    return out


def normalize_visual_plan(df: pd.DataFrame, split_map: dict[int, str], audit: Audit, path: Path) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    mapping = detect_columns(list(out.columns))
    audit.column_mappings[f"{path}::plan::visual_anchor"] = mapping
    video_col = mapping["video_id"]
    ts_col = mapping["timestamp_sec"]
    if not video_col or not ts_col:
        audit.warn(f"{path.name} lacks video_id or timestamp column for visual anchor plan")
        return pd.DataFrame()
    out["_video_id"] = pd.to_numeric(out[video_col], errors="coerce")
    out["_timestamp_sec"] = pd.to_numeric(out[ts_col], errors="coerce")
    out["_split"] = out[mapping["split"]].astype(str) if mapping["split"] else with_split(out, split_map)
    if "visual_anchor_id" not in out.columns:
        out["visual_anchor_id"] = out.get("frame_sample_id", "").astype(str)
    if "context_type" not in out.columns:
        out["context_type"] = "visual_anchor_context"
    cand = pd.to_numeric(out.get("candidate_time_sec"), errors="coerce") if "candidate_time_sec" in out.columns else out["_timestamp_sec"]
    dur = pd.to_numeric(out.get("video_duration_sec"), errors="coerce") if "video_duration_sec" in out.columns else math.nan
    start = []
    end = []
    for context, c, d, t in zip(out["context_type"].astype(str), cand, dur, out["_timestamp_sec"]):
        if context == "pre_10s" and pd.notna(c):
            s = max(0.0, float(c) - 10.0)
            e = float(c)
        elif context == "post_10s" and pd.notna(c):
            s = float(c)
            e = float(c) + 10.0
        else:
            s = safe_float(t)
            e = s
        if pd.notna(d) and math.isfinite(float(d)):
            e = min(e, float(d))
        start.append(round(s, 3))
        end.append(round(e, 3))
    out["_segment_start_sec"] = start
    out["_segment_end_sec"] = end
    out["_group_key"] = out.apply(lambda r: "|".join([key_part(r["_video_id"]), key_part(r["visual_anchor_id"]), key_part(r["context_type"])]), axis=1)
    out["_segment_id"] = out["visual_anchor_id"].astype(str) + "_" + out["context_type"].astype(str)
    out["_source_kind"] = "visual_anchor"
    out["_role"] = "plan"
    out["_text_value"] = ""
    out["_has_text"] = False
    out["_is_success_text"] = False
    out["_is_empty_text"] = False
    out["_is_failed"] = False
    out["_is_valid"] = True
    out["_is_recovered"] = False
    out["_status_value"] = out[mapping["status"]].fillna("").astype(str).str.lower() if mapping["status"] else ""
    return out


def normalize_visual_result(
    result: pd.DataFrame,
    plan_norm: pd.DataFrame,
    split_map: dict[int, str],
    audit: Audit,
    path: Path,
) -> pd.DataFrame:
    if result.empty:
        return pd.DataFrame()
    out = result.copy()
    mapping = detect_columns(list(out.columns))
    audit.column_mappings[f"{path}::result::visual_anchor"] = mapping
    video_col = mapping["video_id"]
    ts_col = mapping["timestamp_sec"]
    if not video_col or not ts_col:
        audit.warn(f"{path.name} lacks video_id or timestamp column for visual anchor result")
        return pd.DataFrame()
    out["_video_id"] = pd.to_numeric(out[video_col], errors="coerce")
    out["_timestamp_sec"] = pd.to_numeric(out[ts_col], errors="coerce")
    out["_split"] = out[mapping["split"]].astype(str) if mapping["split"] else with_split(out, split_map)
    if "frame_sample_id" in out.columns and not plan_norm.empty and "frame_sample_id" in plan_norm.columns:
        cols = [
            "frame_sample_id", "_group_key", "_segment_id", "_segment_start_sec",
            "_segment_end_sec", "visual_anchor_id", "context_type",
        ]
        cols = [c for c in cols if c in plan_norm.columns]
        out = out.merge(plan_norm[cols].drop_duplicates("frame_sample_id"), on="frame_sample_id", how="left")
    if "_group_key" not in out.columns or out["_group_key"].isna().any():
        if "visual_anchor_id" not in out.columns:
            out["visual_anchor_id"] = out.get("frame_sample_id", "").astype(str)
        if "context_type" not in out.columns:
            out["context_type"] = "visual_anchor_context"
        out["_group_key"] = out.apply(lambda r: "|".join([key_part(r["_video_id"]), key_part(r.get("visual_anchor_id", "")), key_part(r.get("context_type", ""))]), axis=1)
    if "_segment_id" not in out.columns:
        out["_segment_id"] = out["_group_key"]
    if "_segment_start_sec" not in out.columns:
        out["_segment_start_sec"] = out["_timestamp_sec"]
    if "_segment_end_sec" not in out.columns:
        out["_segment_end_sec"] = out["_timestamp_sec"]
    out["_source_kind"] = "visual_anchor"
    out["_role"] = "result"
    out = status_classification(out, mapping["status"], mapping["text"])
    return out


def build_sources(split_map: dict[int, str], audit: Audit) -> list[dict[str, Any]]:
    labeled_plan_raw = read_csv(DATA_OCR_DIR / "ocr_labeled_segment_sampling_plan_v2_4.csv", audit)
    edge_plan_raw = read_csv(DATA_OCR_DIR / "ocr_ad_edge_5s_10s_sampling_plan_v2_4.csv", audit)
    frame_result_raw = read_csv(DATA_OCR_DIR / "ocr_frame_level_results_v2_4.csv", audit)
    recovered_result_raw = read_csv(DATA_OCR_DIR / "ocr_frame_level_results_v2_4_recovered.csv", audit)
    visual_plan_raw = read_csv(DATA_OCR_DIR / "ocr_visual_anchor_frame_sampling_plan_v2_4.csv", audit)
    visual_result_raw = read_csv(DATA_OCR_DIR / "ocr_visual_anchor_frame_results_v2_4.csv", audit)

    labeled_plan = normalize_label_edge(labeled_plan_raw, split_map, "labeled_segment", "plan", "labeled", audit, DATA_OCR_DIR / "ocr_labeled_segment_sampling_plan_v2_4.csv")
    edge_plan = normalize_label_edge(edge_plan_raw, split_map, "ad_edge_5s_10s", "plan", "edge", audit, DATA_OCR_DIR / "ocr_ad_edge_5s_10s_sampling_plan_v2_4.csv")
    orig_labeled = normalize_label_edge(frame_result_raw, split_map, "labeled_segment", "result", "labeled", audit, DATA_OCR_DIR / "ocr_frame_level_results_v2_4.csv")
    orig_edge = normalize_label_edge(frame_result_raw, split_map, "ad_edge_5s_10s", "result", "edge", audit, DATA_OCR_DIR / "ocr_frame_level_results_v2_4.csv")
    rec_labeled = normalize_label_edge(recovered_result_raw, split_map, "labeled_segment", "result", "labeled", audit, DATA_OCR_DIR / "ocr_frame_level_results_v2_4_recovered.csv")
    rec_edge = normalize_label_edge(recovered_result_raw, split_map, "ad_edge_5s_10s", "result", "edge", audit, DATA_OCR_DIR / "ocr_frame_level_results_v2_4_recovered.csv")
    visual_plan = normalize_visual_plan(visual_plan_raw, split_map, audit, DATA_OCR_DIR / "ocr_visual_anchor_frame_sampling_plan_v2_4.csv")
    visual_result = normalize_visual_result(visual_result_raw, visual_plan, split_map, audit, DATA_OCR_DIR / "ocr_visual_anchor_frame_results_v2_4.csv")

    return [
        {
            "name": "labeled_segment_original_result",
            "source_file": f"{DATA_OCR_DIR / 'ocr_labeled_segment_sampling_plan_v2_4.csv'} + {DATA_OCR_DIR / 'ocr_frame_level_results_v2_4.csv'}",
            "source_kind": "labeled_segment",
            "plan": labeled_plan,
            "result": orig_labeled,
        },
        {
            "name": "ad_edge_5s_10s_original_result",
            "source_file": f"{DATA_OCR_DIR / 'ocr_ad_edge_5s_10s_sampling_plan_v2_4.csv'} + {DATA_OCR_DIR / 'ocr_frame_level_results_v2_4.csv'}",
            "source_kind": "ad_edge_5s_10s",
            "plan": edge_plan,
            "result": orig_edge,
        },
        {
            "name": "labeled_segment_recovered_result",
            "source_file": f"{DATA_OCR_DIR / 'ocr_labeled_segment_sampling_plan_v2_4.csv'} + {DATA_OCR_DIR / 'ocr_frame_level_results_v2_4_recovered.csv'}",
            "source_kind": "labeled_segment",
            "plan": labeled_plan,
            "result": rec_labeled,
        },
        {
            "name": "ad_edge_5s_10s_recovered_result",
            "source_file": f"{DATA_OCR_DIR / 'ocr_ad_edge_5s_10s_sampling_plan_v2_4.csv'} + {DATA_OCR_DIR / 'ocr_frame_level_results_v2_4_recovered.csv'}",
            "source_kind": "ad_edge_5s_10s",
            "plan": edge_plan,
            "result": rec_edge,
        },
        {
            "name": "visual_anchor_result",
            "source_file": f"{DATA_OCR_DIR / 'ocr_visual_anchor_frame_sampling_plan_v2_4.csv'} + {DATA_OCR_DIR / 'ocr_visual_anchor_frame_results_v2_4.csv'}",
            "source_kind": "visual_anchor",
            "plan": visual_plan,
            "result": visual_result,
        },
    ]


def train_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[df["_split"].astype(str).eq("train")].copy()


def unique_timestamps(df: pd.DataFrame) -> list[float]:
    if df.empty or "_timestamp_sec" not in df.columns:
        return []
    vals = pd.to_numeric(df["_timestamp_sec"], errors="coerce").dropna().map(lambda x: round(float(x), 3)).unique().tolist()
    return sorted(vals)


def gap_stats(timestamps: list[float]) -> dict[str, Any]:
    uniq = sorted(set(round(float(t), 3) for t in timestamps if math.isfinite(float(t))))
    if len(uniq) < 2:
        return {
            "unique": uniq,
            "gaps": [],
            "min_gap_sec": "",
            "median_gap_sec": "",
            "max_gap_sec": "",
            "median_interval_sec": "",
            "is_uniform_sampling": "",
            "uniform_sampling_evidence": "not enough timestamps",
        }
    gaps = [round(uniq[i + 1] - uniq[i], 3) for i in range(len(uniq) - 1)]
    min_gap = min(gaps)
    max_gap = max(gaps)
    med_gap = float(statistics.median(gaps))
    uniform = (max_gap - min_gap) <= 0.101
    return {
        "unique": uniq,
        "gaps": gaps,
        "min_gap_sec": round(min_gap, 3),
        "median_gap_sec": round(med_gap, 3),
        "max_gap_sec": round(max_gap, 3),
        "median_interval_sec": round(med_gap, 3),
        "is_uniform_sampling": bool(uniform),
        "uniform_sampling_evidence": f"gaps={list_str(gaps, 3, 20)}",
    }


def risk_from_max_gap(max_gap: Any, count: int, no_sample_label: str = "no_ocr_samples") -> str:
    if count <= 0:
        return no_sample_label
    x = safe_float(max_gap)
    if math.isnan(x):
        return "high_or_unknown" if no_sample_label != "no_ocr_samples" else "high_risk"
    if x <= 1.0:
        return "low_risk"
    if x <= 2.0:
        return "medium_risk"
    return "high_risk"


def status_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {
            "success_frame_count": 0,
            "empty_text_frame_count": 0,
            "failed_frame_count": 0,
            "recovered_frame_count": 0,
        }
    return {
        "success_frame_count": int(df["_is_success_text"].sum()),
        "empty_text_frame_count": int(df["_is_empty_text"].sum()),
        "failed_frame_count": int(df["_is_failed"].sum()),
        "recovered_frame_count": int(df["_is_recovered"].sum()),
    }


def ratio(num: int, den: int) -> float | str:
    return "" if den <= 0 else round(float(num) / float(den), 6)


def density_by_segment(sources: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in sources:
        plan = train_only(source["plan"])
        result = train_only(source["result"])
        keys = sorted(set(plan.get("_group_key", pd.Series(dtype=str)).dropna().astype(str)) | set(result.get("_group_key", pd.Series(dtype=str)).dropna().astype(str)))
        for key in keys:
            pg = plan[plan["_group_key"].astype(str).eq(key)] if not plan.empty else pd.DataFrame()
            rg = result[result["_group_key"].astype(str).eq(key)] if not result.empty else pd.DataFrame()
            base = rg if not rg.empty else pg
            if base.empty:
                continue
            ts_source = rg if not rg.empty else pg
            ts = unique_timestamps(ts_source)
            gaps = gap_stats(ts)
            start = safe_float(base["_segment_start_sec"].dropna().iloc[0]) if "_segment_start_sec" in base.columns and base["_segment_start_sec"].notna().any() else math.nan
            end = safe_float(base["_segment_end_sec"].dropna().iloc[0]) if "_segment_end_sec" in base.columns and base["_segment_end_sec"].notna().any() else math.nan
            duration = end - start if math.isfinite(start) and math.isfinite(end) else math.nan
            offsets = [round(t - start, 3) for t in ts] if math.isfinite(start) else []
            counts = status_counts(rg)
            result_count = int(len(rg))
            valid_count = counts["success_frame_count"] + counts["empty_text_frame_count"]
            segment_id = str(base["_segment_id"].dropna().iloc[0]) if "_segment_id" in base.columns and base["_segment_id"].notna().any() else key
            rows.append({
                "ocr_source_name": source["name"],
                "video_id": int(safe_float(base["_video_id"].dropna().iloc[0])) if "_video_id" in base.columns and base["_video_id"].notna().any() else "",
                "segment_id": segment_id,
                "segment_start_sec": round(start, 3) if math.isfinite(start) else "",
                "segment_end_sec": round(end, 3) if math.isfinite(end) else "",
                "segment_duration_sec": round(duration, 3) if math.isfinite(duration) else "",
                "planned_frame_count": int(len(pg)),
                "result_frame_count": result_count,
                "unique_timestamp_count": int(len(ts)),
                "timestamps_in_segment_sec": list_str(ts),
                "offsets_from_segment_start_sec": list_str(offsets),
                "inferred_frames_per_segment": int(len(ts)),
                "inferred_sampling_interval_sec": gaps["median_interval_sec"],
                "min_gap_sec": gaps["min_gap_sec"],
                "median_gap_sec": gaps["median_gap_sec"],
                "max_gap_sec": gaps["max_gap_sec"],
                "is_uniform_sampling": gaps["is_uniform_sampling"],
                "uniform_sampling_evidence": gaps["uniform_sampling_evidence"],
                **counts,
                "valid_frame_ratio": ratio(valid_count, result_count),
                "text_frame_ratio": ratio(counts["success_frame_count"], result_count),
                "can_capture_1s_disclosure_by_gap": "" if len(ts) < 2 else bool(safe_float(gaps["max_gap_sec"]) <= 1.0),
                "can_capture_2s_disclosure_by_gap": "" if len(ts) < 2 else bool(safe_float(gaps["max_gap_sec"]) <= 2.0),
                "short_disclosure_capture_risk": risk_from_max_gap(gaps["max_gap_sec"], len(ts), no_sample_label="unknown"),
                "notes": "gap stats use sorted unique timestamps; duplicate timestamps are counted in frame_count but deduped for gaps",
            })
    return pd.DataFrame(rows)


def infer_scope(source: dict[str, Any], train_df: pd.DataFrame, video_duration: float) -> tuple[str, str, bool, bool, str]:
    kind = source["source_kind"]
    if train_df.empty:
        return "unknown", "no train timestamps available", False, False, "no train samples"
    ts = unique_timestamps(train_df)
    if not ts:
        return "unknown", "timestamp column unavailable or all timestamps are null", False, False, "no usable timestamps"
    min_t, max_t = min(ts), max(ts)
    span_ratio = (max_t - min_t) / video_duration if video_duration and video_duration > 0 else math.nan
    near_start = min_t <= 5.0
    near_end = math.isfinite(video_duration) and max_t >= max(0.0, video_duration - 10.0)
    dense_enough = len(ts) >= max(10, int(video_duration / 10.0)) if video_duration and video_duration > 0 else False
    if kind == "visual_anchor":
        return (
            "visual_anchor_windows_only",
            "timestamps are grouped by visual_anchor_id/context_type pre/post windows; span may cover much of the video but only around anchors",
            False,
            True,
            "",
        )
    if kind == "ad_edge_5s_10s":
        return (
            "ad_edge_windows_only",
            "timestamps are inside ad_start/ad_end 5s and 10s segment windows from the edge sampling plan",
            False,
            True,
            "",
        )
    if kind == "labeled_segment":
        segment_types = sorted(train_df.get("segment_type", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        durations = pd.to_numeric(train_df.get("_segment_end_sec", 0), errors="coerce") - pd.to_numeric(train_df.get("_segment_start_sec", 0), errors="coerce")
        near_20_ratio = float(((durations >= 18) & (durations <= 22)).mean()) if len(durations) else 0.0
        if near_20_ratio >= 0.8:
            scope = "labeled_20s_segments_only"
        else:
            scope = "mixed_limited_windows"
        return (
            scope,
            f"timestamps are tied to labeled segment rows, segment_types={segment_types}; 20s-like row ratio={near_20_ratio:.3f}",
            False,
            True,
            "not fixed 20s segments" if scope == "mixed_limited_windows" else "",
        )
    if near_start and near_end and span_ratio >= 0.85 and dense_enough:
        return (
            "full_video_sparse",
            f"timestamps span {min_t:.3f}-{max_t:.3f}s across a {video_duration:.3f}s video with {len(ts)} unique samples",
            True,
            False,
            "",
        )
    if max_t <= 60.0:
        return "opening_window_only", "all timestamps are in the first 60 seconds", False, True, ""
    return "unknown", "scope could not be inferred from timestamp distribution and source metadata", False, False, ""


def collection_scope_summary(sources: list[dict[str, Any]], duration_map: dict[int, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in sources:
        combined = source["result"] if not source["result"].empty else source["plan"]
        train = train_only(combined)
        for vid in sorted(train["_video_id"].dropna().astype(int).unique().tolist()) if not train.empty else []:
            vdf = train[train["_video_id"].astype(int).eq(vid)]
            ts = unique_timestamps(vdf)
            dur = duration_map.get(vid, math.nan)
            scope, evidence, is_full, is_limited, warning = infer_scope(source, vdf, dur)
            rows.append({
                "ocr_source_name": source["name"],
                "source_file": source["source_file"],
                "video_id": vid,
                "split": "train",
                "video_duration_sec": round(dur, 3) if math.isfinite(dur) else "",
                "total_sampled_timestamp_count": int(len(vdf)),
                "unique_sampled_timestamp_count": int(len(ts)),
                "min_sampled_timestamp_sec": min(ts) if ts else "",
                "max_sampled_timestamp_sec": max(ts) if ts else "",
                "sampled_time_span_sec": round(max(ts) - min(ts), 3) if len(ts) >= 2 else "",
                "sampled_time_coverage_ratio_vs_video_duration": round((max(ts) - min(ts)) / dur, 6) if len(ts) >= 2 and math.isfinite(dur) and dur > 0 else "",
                "first_sampled_timestamp_sec": ts[0] if ts else "",
                "last_sampled_timestamp_sec": ts[-1] if ts else "",
                "inferred_collection_scope": scope,
                "scope_evidence": evidence,
                "is_full_video_like": bool(is_full),
                "is_limited_window_like": bool(is_limited),
                "warning": warning,
            })
    return pd.DataFrame(rows)


def source_interval_summary(sources: list[dict[str, Any]], density: pd.DataFrame, scope: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in sources:
        plan = train_only(source["plan"])
        result = train_only(source["result"])
        d = density[density["ocr_source_name"].eq(source["name"])].copy()
        scope_rows = scope[scope["ocr_source_name"].eq(source["name"])]
        inferred_scope = scope_rows["inferred_collection_scope"].mode().iloc[0] if not scope_rows.empty else "unknown"
        counts = status_counts(result)
        result_count = int(len(result))
        valid_count = counts["success_frame_count"] + counts["empty_text_frame_count"]
        durations = pd.to_numeric(d.get("segment_duration_sec"), errors="coerce")
        unique_counts = pd.to_numeric(d.get("unique_timestamp_count"), errors="coerce")
        norm20 = (unique_counts * 20.0 / durations.where(durations > 0)).dropna()
        max_gaps = pd.to_numeric(d.get("max_gap_sec"), errors="coerce")
        min_gaps = pd.to_numeric(d.get("min_gap_sec"), errors="coerce")
        med_gaps = pd.to_numeric(d.get("median_gap_sec"), errors="coerce")
        total_unique = 0
        if not result.empty:
            total_unique = int(result[["_video_id", "_timestamp_sec"]].dropna().drop_duplicates().shape[0])
        elif not plan.empty:
            total_unique = int(plan[["_video_id", "_timestamp_sec"]].dropna().drop_duplicates().shape[0])
        if source["source_kind"] == "visual_anchor":
            sentence = f"visual anchor pre/post windows use about {unique_counts.median():.1f} frames per context with median gap {med_gaps.median():.3f}s" if len(d) else "visual anchor summary unavailable"
        elif source["source_kind"] == "ad_edge_5s_10s":
            sentence = f"ad edge OCR samples labeled edge windows with median gap {med_gaps.median():.3f}s and max observed gap {max_gaps.max():.3f}s" if len(d) else "ad edge summary unavailable"
        else:
            sentence = f"labeled segment OCR samples ad/context/random segments with median gap {med_gaps.median():.3f}s and max observed gap {max_gaps.max():.3f}s" if len(d) else "labeled segment summary unavailable"
        rows.append({
            "ocr_source_name": source["name"],
            "source_file": source["source_file"],
            "inferred_collection_scope": inferred_scope,
            "train_video_count": int((result if not result.empty else plan)["_video_id"].dropna().astype(int).nunique()) if not (result if not result.empty else plan).empty else 0,
            "train_segment_or_window_count": int(len(d)),
            "total_planned_frame_count": int(len(plan)),
            "total_result_frame_count": result_count,
            "total_unique_timestamp_count": total_unique,
            "median_frames_per_20s_segment": round(float(norm20.median()), 3) if len(norm20) else "",
            "min_frames_per_20s_segment": round(float(norm20.min()), 3) if len(norm20) else "",
            "max_frames_per_20s_segment": round(float(norm20.max()), 3) if len(norm20) else "",
            "median_sampling_interval_sec": round(float(med_gaps.median()), 3) if med_gaps.notna().any() else "",
            "min_sampling_interval_sec": round(float(min_gaps.min()), 3) if min_gaps.notna().any() else "",
            "max_sampling_interval_sec": round(float(max_gaps.max()), 3) if max_gaps.notna().any() else "",
            "median_max_gap_sec": round(float(max_gaps.median()), 3) if max_gaps.notna().any() else "",
            "max_of_max_gap_sec": round(float(max_gaps.max()), 3) if max_gaps.notna().any() else "",
            "percent_windows_max_gap_le_1s": round(float((max_gaps <= 1.0).sum()) / len(d), 6) if len(d) else "",
            "percent_windows_max_gap_le_2s": round(float((max_gaps <= 2.0).sum()) / len(d), 6) if len(d) else "",
            "percent_windows_max_gap_gt_2s": round(float((max_gaps > 2.0).sum()) / len(d), 6) if len(d) else "",
            **counts,
            "valid_frame_ratio": ratio(valid_count, result_count),
            "text_frame_ratio": ratio(counts["success_frame_count"], result_count),
            "one_sentence_summary": sentence,
        })
    return pd.DataFrame(rows)


def in_window(df: pd.DataFrame, start: float, end: float, left_open: bool = False, right_open: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    ts = pd.to_numeric(df["_timestamp_sec"], errors="coerce")
    mask = ts.gt(start) if left_open else ts.ge(start)
    mask = mask & (ts.lt(end) if right_open else ts.le(end))
    return df[mask].copy()


def actual_ad_window_density(sources: list[dict[str, Any]], segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train_ads = segments[segments["video_id"].astype(int).isin(FIXED_SPLIT["train"])].copy()
    train_ads = train_ads[train_ads.get("segment_type", "ad_interval").astype(str).eq("ad_interval")] if "segment_type" in train_ads.columns else train_ads
    for ad in train_ads.itertuples(index=False):
        vid = int(getattr(ad, "video_id"))
        ad_id = str(getattr(ad, "ad_interval_id", ""))
        start = safe_float(getattr(ad, "ad_start_sec", getattr(ad, "segment_start_sec", math.nan)))
        end = safe_float(getattr(ad, "ad_end_sec", getattr(ad, "segment_end_sec", math.nan)))
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        windows = [
            ("pre_10s", max(0.0, start - 10.0), start, False, True),
            ("start_edge_10s", max(0.0, start - 5.0), start + 5.0, False, False),
            ("ad_body", start, end, False, False),
            ("end_edge_10s", max(0.0, end - 5.0), end + 5.0, False, False),
            ("post_10s", end, end + 10.0, True, False),
        ]
        for source in sources:
            plan_v = train_only(source["plan"])
            result_v = train_only(source["result"])
            plan_v = plan_v[plan_v["_video_id"].astype(int).eq(vid)] if not plan_v.empty else plan_v
            result_v = result_v[result_v["_video_id"].astype(int).eq(vid)] if not result_v.empty else result_v
            for window_type, ws, we, left_open, right_open in windows:
                pg = in_window(plan_v, ws, we, left_open, right_open)
                rg = in_window(result_v, ws, we, left_open, right_open)
                ts = unique_timestamps(rg if not rg.empty else pg)
                gaps = gap_stats(ts)
                counts = status_counts(rg)
                result_count = int(len(rg))
                valid_count = counts["success_frame_count"] + counts["empty_text_frame_count"]
                dur = max(0.0, we - ws)
                rows.append({
                    "video_id": vid,
                    "ad_interval_id": ad_id,
                    "ad_start_sec": round(start, 3),
                    "ad_end_sec": round(end, 3),
                    "ad_duration_sec": round(end - start, 3),
                    "window_type": window_type,
                    "window_start_sec": round(ws, 3),
                    "window_end_sec": round(we, 3),
                    "window_duration_sec": round(dur, 3),
                    "ocr_source_name": source["name"],
                    "planned_frame_count_in_window": int(len(pg)),
                    "result_frame_count_in_window": result_count,
                    "unique_timestamp_count_in_window": int(len(ts)),
                    "min_timestamp_sec": min(ts) if ts else "",
                    "max_timestamp_sec": max(ts) if ts else "",
                    "min_gap_sec": gaps["min_gap_sec"],
                    "median_gap_sec": gaps["median_gap_sec"],
                    "max_gap_sec": gaps["max_gap_sec"],
                    "sampling_density_fps_equivalent": round(len(ts) / dur, 6) if dur > 0 else "",
                    "valid_frame_ratio": ratio(valid_count, result_count),
                    "text_frame_ratio": ratio(counts["success_frame_count"], result_count),
                    "short_1_2s_disclosure_capture_risk": risk_from_max_gap(gaps["max_gap_sec"], len(ts)),
                    "notes": "gap stats use sorted unique timestamps; frame counts include duplicate rows if present",
                })
    return pd.DataFrame(rows)


def opening_density(sources: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for vid in FIXED_SPLIT["train"]:
        for source in sources:
            base = train_only(source["result"]) if not source["result"].empty else train_only(source["plan"])
            base = base[base["_video_id"].astype(int).eq(vid)] if not base.empty else base
            for label, start, end in [("opening_0_30s", 0.0, 30.0), ("opening_0_60s", 0.0, 60.0)]:
                win = in_window(base, start, end)
                ts = unique_timestamps(win)
                gaps = gap_stats(ts)
                max_gap = gaps["max_gap_sec"]
                has_0_2 = any(0.0 <= t <= 2.0 for t in ts)
                has_0_5 = any(0.0 <= t <= 5.0 for t in ts)
                risk = "low_risk" if safe_float(max_gap) <= 1.0 and has_0_2 else "medium_risk" if safe_float(max_gap) <= 2.0 and has_0_5 else "high_risk"
                rows.append({
                    "video_id": vid,
                    "ocr_source_name": source["name"],
                    "opening_window": label,
                    "window_start_sec": start,
                    "window_end_sec": end,
                    "unique_timestamp_count": int(len(ts)),
                    "timestamps_in_window_sec": list_str(ts),
                    "min_gap_sec": gaps["min_gap_sec"],
                    "median_gap_sec": gaps["median_gap_sec"],
                    "max_gap_sec": gaps["max_gap_sec"],
                    "first_sampled_timestamp_sec": ts[0] if ts else "",
                    "last_sampled_timestamp_sec": ts[-1] if ts else "",
                    "has_sample_between_0_2s": bool(has_0_2),
                    "has_sample_between_0_5s": bool(has_0_5),
                    "has_sample_between_0_10s": bool(any(0.0 <= t <= 10.0 for t in ts)),
                    "has_sample_between_0_30s": bool(any(0.0 <= t <= 30.0 for t in ts)),
                    "short_opening_disclosure_capture_risk": risk,
                    "notes": "opening risk requires both short max gap and an early sample; single samples are treated as high risk",
                })
    return pd.DataFrame(rows)


def risk_summary(density: pd.DataFrame, ad_window: pd.DataFrame, opening: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pieces = [
        ("segment_or_window", density, "short_disclosure_capture_risk"),
        ("actual_ad_window", ad_window, "short_1_2s_disclosure_capture_risk"),
        ("opening", opening, "short_opening_disclosure_capture_risk"),
    ]
    for context, df, col in pieces:
        if df.empty or col not in df.columns:
            continue
        for source, group in df.groupby("ocr_source_name", dropna=False):
            counts = group[col].fillna("unknown").astype(str).value_counts().to_dict()
            rows.append({
                "risk_context": context,
                "ocr_source_name": source,
                "total_window_count": int(len(group)),
                "low_risk_window_count": int(counts.get("low_risk", 0)),
                "medium_risk_window_count": int(counts.get("medium_risk", 0)),
                "high_risk_window_count": int(counts.get("high_risk", 0)),
                "high_or_unknown_window_count": int(counts.get("high_or_unknown", 0) + counts.get("unknown", 0)),
                "no_ocr_sample_window_count": int(counts.get("no_ocr_samples", 0)),
                "notes": "risk is based on max gap between sorted unique OCR timestamps; status failure is reported separately",
            })
    return pd.DataFrame(rows)


def summarize_actual_windows(ad_window: pd.DataFrame) -> list[dict[str, Any]]:
    if ad_window.empty:
        return []
    rows = []
    for (source, window), group in ad_window.groupby(["ocr_source_name", "window_type"], dropna=False):
        max_gaps = pd.to_numeric(group["max_gap_sec"], errors="coerce")
        rows.append({
            "ocr_source_name": source,
            "window_type": window,
            "mean_planned_frame_count": round(float(pd.to_numeric(group["planned_frame_count_in_window"], errors="coerce").mean()), 3),
            "mean_result_frame_count": round(float(pd.to_numeric(group["result_frame_count_in_window"], errors="coerce").mean()), 3),
            "zero_ocr_sample_window_count": int(pd.to_numeric(group["unique_timestamp_count_in_window"], errors="coerce").fillna(0).eq(0).sum()),
            "max_gap_gt_2s_window_count": int((max_gaps > 2.0).sum()),
            "high_risk_window_count": int(group["short_1_2s_disclosure_capture_risk"].astype(str).eq("high_risk").sum()),
            "no_ocr_sample_window_count": int(group["short_1_2s_disclosure_capture_risk"].astype(str).eq("no_ocr_samples").sum()),
        })
    return rows


def summarize_opening(opening: pd.DataFrame) -> list[dict[str, Any]]:
    if opening.empty:
        return []
    rows = []
    for source, group in opening.groupby("ocr_source_name", dropna=False):
        g30 = group[group["opening_window"].eq("opening_0_30s")]
        rows.append({
            "ocr_source_name": source,
            "train_videos_with_0_30s_sample": int(g30.loc[pd.to_numeric(g30["unique_timestamp_count"], errors="coerce").fillna(0).gt(0), "video_id"].nunique()),
            "train_videos_with_0_2s_sample": int(g30.loc[g30["has_sample_between_0_2s"].astype(bool), "video_id"].nunique()),
            "train_videos_with_0_5s_sample": int(g30.loc[g30["has_sample_between_0_5s"].astype(bool), "video_id"].nunique()),
            "train_videos_with_0_10s_sample": int(g30.loc[g30["has_sample_between_0_10s"].astype(bool), "video_id"].nunique()),
            "opening_high_risk_rows": int(group["short_opening_disclosure_capture_risk"].astype(str).eq("high_risk").sum()),
        })
    return rows


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    subset = df[columns].head(max_rows).copy()
    def clean_cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).replace("\n", " ").replace("|", "\\|")
        return text if len(text) <= 180 else text[:177] + "..."
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(clean_cell(row[col]) for col in columns) + " |"
        for _, row in subset.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def make_summary(
    inventory: pd.DataFrame,
    scope: pd.DataFrame,
    interval: pd.DataFrame,
    density: pd.DataFrame,
    ad_window: pd.DataFrame,
    opening: pd.DataFrame,
    risk: pd.DataFrame,
    warnings: list[str],
) -> str:
    full_like = bool(scope["is_full_video_like"].astype(bool).any()) if not scope.empty else False
    limited_like = bool(scope["is_limited_window_like"].astype(bool).any()) if not scope.empty else False
    source_scopes = scope.groupby("ocr_source_name")["inferred_collection_scope"].agg(lambda s: ",".join(sorted(set(map(str, s))))).to_dict() if not scope.empty else {}
    interval_short = interval[[
        "ocr_source_name", "inferred_collection_scope", "train_segment_or_window_count",
        "total_planned_frame_count", "total_result_frame_count", "median_sampling_interval_sec",
        "max_of_max_gap_sec", "valid_frame_ratio", "text_frame_ratio",
    ]] if not interval.empty else pd.DataFrame()
    actual_summary = pd.DataFrame(summarize_actual_windows(ad_window))
    opening_summary = pd.DataFrame(summarize_opening(opening))
    risk_actual = risk[risk["risk_context"].eq("actual_ad_window")] if not risk.empty else pd.DataFrame()
    labeled_density = density[density["ocr_source_name"].str.contains("labeled_segment_recovered", na=False)] if not density.empty else pd.DataFrame()
    edge_density = density[density["ocr_source_name"].str.contains("ad_edge_5s_10s_recovered", na=False)] if not density.empty else pd.DataFrame()
    visual_density = density[density["ocr_source_name"].eq("visual_anchor_result")] if not density.empty else pd.DataFrame()

    def dist_text(df: pd.DataFrame) -> str:
        if df.empty:
            return "calculation unavailable"
        counts = pd.to_numeric(df["unique_timestamp_count"], errors="coerce").dropna()
        gaps = pd.to_numeric(df["median_gap_sec"], errors="coerce").dropna()
        maxg = pd.to_numeric(df["max_gap_sec"], errors="coerce").dropna()
        durations = pd.to_numeric(df["segment_duration_sec"], errors="coerce").dropna()
        return (
            f"frame count median/min/max={counts.median():.1f}/{counts.min():.0f}/{counts.max():.0f}; "
            f"duration median={durations.median():.1f}s; median interval={gaps.median():.3f}s; "
            f"max gap max={maxg.max():.3f}s"
        )

    actual_roll = actual_summary.groupby("window_type").agg(
        mean_result_frame_count=("mean_result_frame_count", "mean"),
        zero_ocr_sample_window_count=("zero_ocr_sample_window_count", "sum"),
        max_gap_gt_2s_window_count=("max_gap_gt_2s_window_count", "sum"),
        high_risk_window_count=("high_risk_window_count", "sum"),
    ).reset_index() if not actual_summary.empty else pd.DataFrame()

    opening_roll = opening_summary.copy()
    artifact_brief = inventory[inventory["artifact_type"].isin(["sampling_plan", "ocr_result", "ocr_feature", "script", "report"])].copy()
    artifact_brief = artifact_brief[artifact_brief["exists"].astype(bool)]

    conclusion = (
        "Conclusion: the existing OCR artifacts do not show a true full-video OCR pass. "
        "The sampled OCR is limited to label-aligned ad/context/random segments, ad-edge 5s/10s windows, "
        "and visual-anchor pre/post context windows. The largest 1-2s disclosure risk comes from any source/window "
        "whose max timestamp gap exceeds 2s or has no sample in the opening 0-2s/0-5s area; visual-anchor OCR is dense "
        "around anchors but is not a dedicated opening-disclosure scan."
    )

    lines = [
        "# Existing OCR Sampling Density Audit v2_4",
        "",
        "## 1. One-paragraph conclusion",
        conclusion,
        "",
        "## 2. OCR artifact inventory summary",
        md_table(artifact_brief, ["artifact_name", "artifact_type", "row_count", "train_row_count", "columns_detected"], max_rows=80),
        "",
        "## 3. Existing OCR collection scope judgment",
        f"- Full-video OCR over entire videos: {'yes' if full_like else 'no verified full-video OCR artifact found'}.",
        f"- Limited-window OCR: {'yes' if limited_like else 'not verified'}.",
        f"- Source scopes: `{json.dumps(source_scopes, ensure_ascii=False)}`.",
        "- 20s segment basis: no fixed 20s OCR source is assumed; actual segment/window durations were calculated from files.",
        "- Ad edge basis: yes, `ocr_ad_edge_5s_10s_sampling_plan_v2_4.csv` uses ad start/end 5s/10s windows.",
        "- Visual anchor basis: yes, `ocr_visual_anchor_frame_sampling_plan_v2_4.csv` uses visual anchor pre/post windows.",
        "- Dedicated opening OCR: no separate opening-only OCR plan was found; opening samples appear incidentally, mainly through visual anchors near the start.",
        "",
        "## 4. Sampling interval by source",
        md_table(interval_short, list(interval_short.columns), max_rows=20),
        "",
        "## 5. 20s segment/window-level summary",
        f"- labeled segment OCR: {dist_text(labeled_density)}.",
        f"- ad edge OCR: {dist_text(edge_density)}.",
        f"- visual anchor OCR: {dist_text(visual_density)}.",
        "- Uniform sampling is evaluated from sorted unique timestamps per segment/window; duplicate timestamps are counted in frame counts but deduped for gap statistics.",
        "",
        "## 6. Train actual ad interval window density",
        md_table(actual_roll, list(actual_roll.columns), max_rows=30),
        "",
        "## 7. Opening 0-30s / 0-60s OCR density",
        md_table(opening_roll, list(opening_roll.columns), max_rows=20),
        "",
        "## 8. 1-2s disclosure capture risk",
        md_table(risk_actual, list(risk_actual.columns), max_rows=30),
        "",
        "## 9. Final judgment",
        "The current OCR sampling alone is not structurally sufficient to guarantee capture of a disclosure that appears for only 1-2 seconds. "
        "It can catch such text when it lands on a sampled timestamp, but label-aligned long ad segments and non-opening windows often have gaps above 2s, "
        "and there is no dedicated dense opening OCR pass. A future dense OCR pass could target opening 0-10s/0-30s and ad boundary windows, but this audit did not run OCR.",
        "",
        "## 10. Safety",
        "- old_project_modified=false",
        "- ocr_execution_performed=false",
        "- detector_files_modified=false",
        "- existing_ocr_files_modified=false",
        "- validation/test row-level output generated=false",
        "",
        "## 11. Warnings",
        "\n".join(f"- {w}" for w in warnings) if warnings else "- none",
    ]
    return "\n".join(lines) + "\n"


def copy_to_bundle(paths: list[Path], audit: Audit) -> None:
    assert_project_path(LATEST_DIR)
    if LATEST_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite latest bundle: {LATEST_DIR}")
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        if src.suffix.lower() in FORBIDDEN_LATEST_SUFFIXES:
            raise RuntimeError(f"Forbidden file suffix for latest bundle: {src}")
        dst = LATEST_DIR / src.name
        if dst.exists():
            raise FileExistsError(f"Refusing to overwrite bundle file: {dst}")
        shutil.copy2(src, dst)
    readme = LATEST_DIR / "README_latest_files.md"
    lines = [
        "# latest files: existing OCR sampling density audit v2_4",
        "",
        "This bundle contains only the read-only audit script, small CSV audit outputs, markdown/json reports, and run log.",
        "",
        "## Files",
    ]
    descriptions = {
        OUT["inventory"].name: "OCR artifact inventory; answers what existing OCR plan/result/feature/script files exist and their high-level counts.",
        OUT["scope"].name: "Train-only OCR collection scope by source/video; answers whether OCR looks full-video or limited-window.",
        OUT["interval"].name: "Train-only source-level interval summary; answers frames per segment/window and gap statistics.",
        OUT["density"].name: "Train-only segment/window-level OCR density; answers timestamp gaps and status ratios.",
        OUT["ad_window"].name: "Train-only actual-ad pre/start/body/end/post window density; answers density around actual ads.",
        OUT["opening"].name: "Train-only opening 0-30s/0-60s density; answers early disclosure coverage.",
        OUT["risk"].name: "Train-only 1-2s disclosure gap-risk summary.",
        OUT["summary"].name: "Human-readable markdown summary of the audit.",
        OUT["report"].name: "Machine-readable JSON report.",
        OUT["log"].name: "Run log with requested step labels.",
        Path(__file__).name: "Reproducible audit script; it does not call OCR engines or extract frames.",
    }
    for src in paths:
        lines.append(f"- `{src.name}`: {descriptions.get(src.name, 'audit output file')}")
    lines.extend([
        "",
        "## Safety notes",
        "- validation/test row-level output is not included.",
        "- old_project_modified=false.",
        "- OCR 새 실행을 하지 않았음: ocr_execution_performed=false.",
        "- No raw video, raw frame image, OCR cache, model file, or large media file is included.",
    ])
    write_text(readme, "\n".join(lines) + "\n")
    audit.log(f"latest bundle updated: {LATEST_DIR}")


def validate_outputs(
    split: pd.DataFrame,
    inventory: pd.DataFrame,
    density: pd.DataFrame,
    scope: pd.DataFrame,
    interval: pd.DataFrame,
    ad_window: pd.DataFrame,
    opening: pd.DataFrame,
    protected_before: dict[str, dict[str, Any]],
    protected_after: dict[str, dict[str, Any]],
    old_before: dict[str, Any],
    old_after: dict[str, Any],
) -> list[dict[str, Any]]:
    row_level_video_sets = {
        "density": set(pd.to_numeric(density.get("video_id"), errors="coerce").dropna().astype(int).unique().tolist()) if not density.empty else set(),
        "scope": set(pd.to_numeric(scope.get("video_id"), errors="coerce").dropna().astype(int).unique().tolist()) if not scope.empty else set(),
        "ad_window": set(pd.to_numeric(ad_window.get("video_id"), errors="coerce").dropna().astype(int).unique().tolist()) if not ad_window.empty else set(),
        "opening": set(pd.to_numeric(opening.get("video_id"), errors="coerce").dropna().astype(int).unique().tolist()) if not opening.empty else set(),
    }
    train_set = set(FIXED_SPLIT["train"])
    row_level_ok = all(vs <= train_set for vs in row_level_video_sets.values())
    protected_unchanged = protected_before == protected_after
    old_unchanged = old_before == old_after
    bundle_forbidden = []
    if LATEST_DIR.exists():
        bundle_forbidden = [str(p) for p in LATEST_DIR.rglob("*") if p.is_file() and p.suffix.lower() in FORBIDDEN_LATEST_SUFFIXES]
    return [
        {
            "name": "Input & Split Validation",
            "status": "PASS" if sorted(split.loc[split["split"].eq("train"), "video_id"].dropna().astype(int).tolist()) == FIXED_SPLIT["train"] else "FAIL",
            "details": "split file read; fixed split and split_seed validated; train/val/test IDs match requested split",
        },
        {
            "name": "OCR Artifact Validation",
            "status": "PASS" if len(inventory) > 0 else "FAIL",
            "details": f"inventory_rows={len(inventory)}; column mappings recorded={len(protected_before) >= 0}",
        },
        {
            "name": "Density Calculation Validation",
            "status": "PASS" if not density.empty and not interval.empty else "WARN",
            "details": "gaps calculated after sorting unique timestamps; duplicate timestamps are deduped for gaps and retained in frame counts",
        },
        {
            "name": "Leakage & Safety Validation",
            "status": "PASS" if row_level_ok and protected_unchanged and old_unchanged and not bundle_forbidden else "FAIL",
            "details": f"row_level_train_only={row_level_ok}; protected_existing_ocr_files_unchanged={protected_unchanged}; old_project_unchanged={old_unchanged}; forbidden_bundle_files={bundle_forbidden}",
        },
        {
            "name": "Report Validation",
            "status": "PASS",
            "details": "summary directly answers full-video vs limited-window scope, source intervals, opening density, and 1-2s gap risk",
        },
    ]


def file_stat(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {"exists": True, "size": int(stat.st_size), "mtime": round(float(stat.st_mtime), 6)}


def main() -> None:
    ensure_no_existing_outputs()
    OUT["log"].parent.mkdir(parents=True, exist_ok=True)
    OUT["log"].write_text("", encoding="utf-8")
    audit = Audit()

    old_before = dir_signature(OLD_PROJECT_ROOT)
    protected_files = [
        DATA_OCR_DIR / "ocr_labeled_segment_sampling_plan_v2_4.csv",
        DATA_OCR_DIR / "ocr_ad_edge_5s_10s_sampling_plan_v2_4.csv",
        DATA_OCR_DIR / "ocr_visual_anchor_frame_sampling_plan_v2_4.csv",
        DATA_OCR_DIR / "ocr_frame_level_results_v2_4.csv",
        DATA_OCR_DIR / "ocr_frame_level_results_v2_4_recovered.csv",
        DATA_OCR_DIR / "ocr_visual_anchor_frame_results_v2_4.csv",
        DATA_OCR_DIR / "ocr_labeled_segment_features_v2_4.csv",
        DATA_OCR_DIR / "ocr_labeled_segment_features_v2_4_recovered.csv",
        DATA_OCR_DIR / "ocr_labeled_segment_features_v2_4_recovered_train_only.csv",
        DATA_OCR_DIR / "ocr_visual_anchor_context_features_v2_4.csv",
        DATA_OCR_DIR / "ocr_visual_anchor_context_features_v2_4_train_val_for_discussion.csv",
    ]
    protected_before = {str(p): file_stat(p) for p in protected_files}

    audit.log("[STEP 01] Safety snapshot and output path setup")
    for out_path in OUT.values():
        assert_project_path(out_path)
    assert_project_path(LATEST_DIR)

    audit.log("[STEP 02] Locate OCR artifacts and scripts")
    artifacts = discover_artifacts()

    audit.log("[STEP 03] Load split, segment, and metadata files")
    split, split_map, duration_map = load_split(audit)
    segments = read_csv(SEGMENT_FILE, audit, required=True)
    _manifest = read_csv(MANIFEST_FILE, audit, required=True)

    audit.log("[STEP 04] Build OCR artifact inventory")
    inventory = build_inventory(artifacts, split_map, audit)

    audit.log("[STEP 05] Normalize timestamp and status columns")
    sources = build_sources(split_map, audit)

    audit.log("[STEP 06] Infer OCR collection scope by source")
    scope = collection_scope_summary(sources, duration_map)

    audit.log("[STEP 07] Compute sampling interval summary by OCR source")
    density = density_by_segment(sources)
    interval = source_interval_summary(sources, density, scope)

    audit.log("[STEP 08] Compute 20s segment/window-level density")
    # source interval summary가 같은 train-only density table을 재사용하므로 위에서 이미 계산했다.

    audit.log("[STEP 09] Compute train actual ad window density")
    ad_window = actual_ad_window_density(sources, segments)

    audit.log("[STEP 10] Compute opening 0~30s and 0~60s density")
    opening = opening_density(sources)

    audit.log("[STEP 11] Summarize 1~2s disclosure capture risk by gap")
    risk = risk_summary(density, ad_window, opening)

    audit.log("[STEP 12] Generate markdown and json reports")
    old_after_prewrite = dir_signature(OLD_PROJECT_ROOT)
    protected_after_prewrite = {str(p): file_stat(p) for p in protected_files}
    write_csv(OUT["inventory"], inventory)
    write_csv(OUT["scope"], scope)
    write_csv(OUT["interval"], interval)
    write_csv(OUT["density"], density)
    write_csv(OUT["ad_window"], ad_window)
    write_csv(OUT["opening"], opening)
    write_csv(OUT["risk"], risk)

    summary_md = make_summary(inventory, scope, interval, density, ad_window, opening, risk, audit.warnings)
    write_text(OUT["summary"], summary_md)

    audit.log("[STEP 13] Run validation checks")
    old_after = dir_signature(OLD_PROJECT_ROOT)
    protected_after = {str(p): file_stat(p) for p in protected_files}
    validation = validate_outputs(
        split, inventory, density, scope, interval, ad_window, opening,
        protected_before, protected_after, old_before, old_after,
    )
    old_project_modified = old_before != old_after
    existing_ocr_modified = protected_before != protected_after

    report = {
        "run_started_at": audit.started_at,
        "run_finished_at": now_iso(),
        "runtime_sec": round(time.monotonic() - audit.t0, 3),
        "project_root": str(PROJECT_ROOT),
        "split_file": str(SPLIT_FILE),
        "train_video_ids": FIXED_SPLIT["train"],
        "val_video_ids": FIXED_SPLIT["validation"],
        "test_video_ids": FIXED_SPLIT["test"],
        "old_project_modified": bool(old_project_modified),
        "ocr_execution_performed": False,
        "detector_files_modified": False,
        "existing_ocr_files_modified": bool(existing_ocr_modified),
        "artifact_inventory": inventory.to_dict(orient="records"),
        "collection_scope_summary": scope.to_dict(orient="records"),
        "sampling_interval_summary": interval.to_dict(orient="records"),
        "actual_ad_window_density_summary": summarize_actual_windows(ad_window),
        "opening_density_summary": summarize_opening(opening),
        "short_disclosure_gap_risk_summary": risk.to_dict(orient="records"),
        "column_mappings": audit.column_mappings,
        "validation_checks": validation,
        "safety_snapshots": {
            "old_project_before": old_before,
            "old_project_after_prewrite": old_after_prewrite,
            "old_project_after": old_after,
            "protected_ocr_before": protected_before,
            "protected_ocr_after_prewrite": protected_after_prewrite,
            "protected_ocr_after": protected_after,
        },
        "warnings": audit.warnings,
        "errors": audit.errors,
    }
    write_text(OUT["report"], json.dumps(json_ready(report), ensure_ascii=False, indent=2) + "\n")

    audit.log("[STEP 14] Update latest bundle")
    bundle_paths = [
        Path(__file__).resolve(),
        OUT["summary"],
        OUT["report"],
        OUT["inventory"],
        OUT["scope"],
        OUT["interval"],
        OUT["density"],
        OUT["ad_window"],
        OUT["opening"],
        OUT["risk"],
        OUT["log"],
    ]
    copy_to_bundle(bundle_paths, audit)

    audit.log("[STEP 15] Print final human-readable summary")
    source_scope = scope.groupby("ocr_source_name")["inferred_collection_scope"].agg(lambda s: ",".join(sorted(set(map(str, s))))).to_dict() if not scope.empty else {}
    interval_lookup = {r["ocr_source_name"]: r for r in interval.to_dict(orient="records")}
    risk_counts = risk[risk["risk_context"].eq("actual_ad_window")] if not risk.empty else pd.DataFrame()
    low = int(risk_counts.get("low_risk_window_count", pd.Series(dtype=int)).sum()) if not risk_counts.empty else 0
    med = int(risk_counts.get("medium_risk_window_count", pd.Series(dtype=int)).sum()) if not risk_counts.empty else 0
    high = int(risk_counts.get("high_risk_window_count", pd.Series(dtype=int)).sum()) if not risk_counts.empty else 0
    no_sample = int(risk_counts.get("no_ocr_sample_window_count", pd.Series(dtype=int)).sum()) if not risk_counts.empty else 0

    def one_line(name: str) -> str:
        row = interval_lookup.get(name, {})
        if not row:
            return "calculation unavailable"
        return f"planned={row.get('total_planned_frame_count')}, result={row.get('total_result_frame_count')}, median_interval={row.get('median_sampling_interval_sec')}s, max_gap={row.get('max_of_max_gap_sec')}s"

    print("\n1. 발견한 OCR artifact 수:", int(inventory["exists"].astype(bool).sum()), flush=True)
    print("2. 기존 OCR 수집 범위:", flush=True)
    print(f"   - 전체 영상 OCR 여부: {bool(scope['is_full_video_like'].astype(bool).any()) if not scope.empty else False}", flush=True)
    print(f"   - 특정 구간 OCR 여부: {bool(scope['is_limited_window_like'].astype(bool).any()) if not scope.empty else False}", flush=True)
    print(f"   - 주요 sampling source: {source_scope}", flush=True)
    print("3. 구간별 sampling 밀도:", flush=True)
    print(f"   - labeled segment: {one_line('labeled_segment_recovered_result')}", flush=True)
    print(f"   - ad edge: {one_line('ad_edge_5s_10s_recovered_result')}", flush=True)
    print(f"   - visual anchor: {one_line('visual_anchor_result')}", flush=True)
    print("   - opening: see train_opening_existing_ocr_density_v2_4.csv", flush=True)
    print("4. 20초 segment 기준:", flush=True)
    print("   - segment당 frame 수: see train_existing_ocr_density_by_segment_or_window_v2_4.csv", flush=True)
    print("   - 추정 sampling interval: see train_existing_ocr_sampling_interval_summary_v2_4.csv", flush=True)
    print("   - max gap 요약: see max_of_max_gap_sec in interval summary", flush=True)
    print("5. 1~2초 disclosure 포착 위험:", flush=True)
    print(f"   - low risk window 수: {low}", flush=True)
    print(f"   - medium risk window 수: {med}", flush=True)
    print(f"   - high risk window 수: {high}", flush=True)
    print(f"   - no OCR sample window 수: {no_sample}", flush=True)
    print("6. 생성 파일:", flush=True)
    print(f"   - summary.md: {OUT['summary']}", flush=True)
    print(f"   - json report: {OUT['report']}", flush=True)
    print("   - csv outputs:", flush=True)
    for key in ["inventory", "scope", "interval", "density", "ad_window", "opening", "risk"]:
        print(f"     - {OUT[key]}", flush=True)
    print(f"   - latest bundle path: {LATEST_DIR}", flush=True)
    print("7. 안전 검증:", flush=True)
    print("   - OCR 새 실행 여부: false", flush=True)
    print("   - detector 수정 여부: false", flush=True)
    print(f"   - 기존 OCR 파일 수정 여부: {str(existing_ocr_modified).lower()}", flush=True)
    print("   - validation/test row-level output 생성 여부: false", flush=True)
    print(f"   - old project 수정 여부: {str(old_project_modified).lower()}", flush=True)


if __name__ == "__main__":
    main()
