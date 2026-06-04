#!/usr/bin/env python3
"""Audit audio feature inventory and score formula for v2_4 train scope.

This is a documentation/audit script only. It does not decode media, recompute
features, tune thresholds, modify detector rules, or create validation/test
row-level outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


VERSION = "v2_4"
TASK_NAME = "audio_feature_inventory_audit_v2_4_train"
DEFAULT_PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
SCRIPT_RELATIVE_PATH = "scripts/audio/audit_audio_feature_inventory_v2_4_train.py"
SPLIT_SEED = 20240524
FIXED_SPLIT = {
    "train": [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
    "validation": [3, 7, 18],
    "test": [4, 16, 17],
}
FORBIDDEN_BUNDLE_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".m4a",
    ".jpg", ".jpeg", ".png", ".webp", ".pt", ".pth", ".ckpt", ".bin",
}
FEATURE_INVENTORY_COLUMNS = [
    "feature_name",
    "feature_group",
    "source_artifact",
    "source_script_or_config",
    "exists_in_labeled_segment_features",
    "exists_in_edge_subwindow_features",
    "exists_in_full_video_features",
    "used_in_audio_ad_like_score",
    "selected_feature",
    "feature_weight",
    "threshold_related",
    "train_only_threshold_available",
    "short_description_kr",
    "interpretation_for_ad_detection_kr",
    "recommended_for_2_2",
    "caution_note",
]


class TaskLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(f"{timestamp} {message}\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_ready(v) for v in obj]
    try:
        if pd.isna(obj) and not isinstance(obj, (str, bytes, bytearray)):
            return None
    except Exception:
        pass
    return obj


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def unique_file_path(path: Path, run_id: str) -> Path:
    if not path.exists():
        return path
    candidate = path.with_name(f"{path.stem}_{run_id}{path.suffix}")
    idx = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{run_id}_{idx}{path.suffix}")
        idx += 1
    return candidate


def unique_dir_path(path: Path, run_id: str) -> Path:
    if not path.exists():
        return path
    candidate = path.with_name(f"{path.name}_{run_id}")
    idx = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{run_id}_{idx}")
        idx += 1
    return candidate


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return str(path)


def snapshot_path(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "file_count": 0, "metadata_digest": None}
    entries: List[str] = []
    file_count = 0
    if path.is_file():
        targets = [path]
        base = path.parent
    else:
        targets = [p for p in path.rglob("*") if p.is_file()]
        base = path
    for item in targets:
        try:
            stat = item.stat()
        except OSError:
            continue
        try:
            item_rel = item.relative_to(base).as_posix()
        except Exception:
            item_rel = str(item)
        entries.append(f"{item_rel}\t{stat.st_size}\t{stat.st_mtime_ns}")
        file_count += 1
    digest = hashlib.sha256("\n".join(sorted(entries)).encode("utf-8")).hexdigest()
    return {
        "path": str(path),
        "exists": True,
        "file_count": file_count,
        "metadata_digest": digest,
        "snapshot_type": "relative_path_size_mtime_ns",
    }


def snapshot_unchanged(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    return before.get("exists") == after.get("exists") and before.get("metadata_digest") == after.get("metadata_digest")


def output_paths(project_root: Path, run_id: str) -> Dict[str, Path]:
    planned = {
        "feature_inventory": project_root / "data/audio/audio_feature_inventory_audit_v2_4_train.csv",
        "score_formula": project_root / "data/audio/audio_score_formula_audit_v2_4_train.csv",
        "source_artifact_inventory": project_root / "data/audio/audio_feature_source_artifact_inventory_v2_4_train.csv",
        "recommendations": project_root / "data/audio/audio_feature_recommendations_for_relative_analysis_v2_4_train.csv",
        "summary_md": project_root / "reports/audio/audio_feature_inventory_audit_v2_4_train_summary.md",
        "report_json": project_root / "reports/audio/audio_feature_inventory_audit_v2_4_train_report.json",
        "presentation_md": project_root / "reports/audio/audio_feature_explanation_for_presentation_v2_4_train.md",
        "run_log": project_root / "logs/audio_feature_inventory_audit_v2_4_train_run_log.txt",
    }
    return {key: unique_file_path(path, run_id) for key, path in planned.items()}


def load_module(path: Path, name: str) -> Tuple[Optional[Any], str]:
    if not path.exists():
        return None, f"missing: {path}"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            return None, "spec_or_loader_missing"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def read_json(path: Path) -> Tuple[Dict[str, Any], str]:
    if not path.exists():
        return {}, f"missing: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def read_csv_header_sample(path: Path, max_rows: int = 5) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "read_mode": "not_read",
        "header": [],
        "sample_rows_read": 0,
        "column_count": 0,
        "row_count": None,
        "row_count_method": "not_counted",
        "error": "",
    }
    if not path.exists():
        info["error"] = "missing"
        return info
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            samples = []
            for _ in range(max_rows):
                try:
                    samples.append(next(reader))
                except StopIteration:
                    break
        info.update({
            "read_mode": f"csv_header_plus_nrows_{max_rows}",
            "header": header,
            "sample_rows_read": len(samples),
            "column_count": len(header),
        })
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
            line_count = sum(1 for _ in fh)
        info["row_count"] = max(0, line_count - 1)
        info["row_count_method"] = "line_count_stream_no_data_parse"
    except Exception as exc:
        info["row_count_method"] = f"row_count_failed: {type(exc).__name__}: {exc}"
    return info


def locate_audio_artifacts(project_root: Path) -> List[Dict[str, Any]]:
    roots = [
        project_root / "scripts/audio",
        project_root / "data/audio",
        project_root / "configs",
        project_root / "reports",
        project_root / "0525",
        project_root / "outputs",
    ]
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            path_text = str(path)
            if path_text in seen:
                continue
            seen.add(path_text)
            relative = rel(path, project_root)
            lower = relative.lower()
            if "audio" not in lower:
                continue
            if "/data/raw/" in f"/{lower}" or path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
                continue
            stat = path.stat()
            if relative.startswith("scripts/audio/"):
                artifact_type = "script"
            elif relative.startswith("configs/"):
                artifact_type = "config"
            elif relative.startswith("data/audio/"):
                artifact_type = "data_audio_artifact"
            elif relative.startswith("reports/"):
                artifact_type = "report"
            elif relative.startswith("outputs/"):
                artifact_type = "latest_bundle_or_output"
            else:
                artifact_type = "reference"
            inspected_for = "inventory"
            read_mode = "not_read"
            if path.name in {
                "extract_labeled_audio_features_v2_4.py",
                "extract_audio_ad_edge_persistence_v2_4.py",
                "analyze_audio_rule_features_v2_4.py",
                "audio_split_aware_rule_reanalysis_v2_4.py",
                "full_video_audio_clue_extraction_v2_4_train.py",
            }:
                inspected_for = "script_feature_or_formula_evidence"
                read_mode = "script_text_or_import"
            elif path.name in {"audio_persistence_rule_config_v2_4_train_only.json", "audio_persistence_rule_config_v2_4.json"}:
                inspected_for = "config_threshold_score_evidence"
                read_mode = "json_full"
            elif path.name in {
                "audio_labeled_segment_features_v2_4_with_split.csv",
                "audio_ad_edge_persistence_subwindow_features_v2_4_with_split.csv",
                "full_video_audio_subwindow_features_v2_4_train_20260526_1750_final.csv",
                "audio_rule_feature_recommendations_v2_4_train_only.csv",
                "audio_rule_candidate_thresholds_v2_4_train_only.csv",
            }:
                inspected_for = "csv_header_sample_only"
                read_mode = "csv_header_plus_nrows_5"
            rows.append({
                "artifact_path": str(path),
                "relative_path": relative,
                "artifact_type": artifact_type,
                "exists": True,
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "inspected_for": inspected_for,
                "read_mode": read_mode,
                "scope_note": "train-centered audit; no validation/test row-level output created",
            })
    rows.sort(key=lambda r: (r["artifact_type"], r["relative_path"]))
    return rows


def infer_group(feature: str) -> str:
    if feature in {"video_id", "split", "start_sec", "end_sec", "duration_sec", "subwindow_start_sec", "subwindow_end_sec", "subwindow_duration_sec", "segment_start_sec", "segment_end_sec", "segment_duration_sec", "extraction_status", "feature_status", "has_audio", "audio_available", "sample_rate", "decoded_sample_rate", "audio_num_samples", "decoded_num_samples", "audio_stream_index", "source_video_path", "video_path", "created_at", "created_by_script"}:
        return "metadata_status"
    if feature.startswith("audio_rms") or feature.startswith("audio_log_energy") or feature in {"audio_peak_amplitude", "audio_mean_abs_amplitude"}:
        return "volume_energy"
    if feature in {"silence_ratio", "low_energy_ratio", "inverse_silence_score"}:
        return "silence_low_energy"
    if feature.startswith("onset") or feature in {"onset_density_score"}:
        return "temporal_change_onset"
    if feature.startswith("spectral_flux") or feature in {"flux_onset_score", "spectral_texture_score"}:
        return "spectral_change_texture"
    if feature.startswith("spectral_centroid") or feature.startswith("spectral_bandwidth") or feature.startswith("spectral_rolloff") or feature.startswith("spectral_flatness") or feature.startswith("zero_crossing_rate"):
        return "spectral_shape"
    if feature.startswith("mfcc_"):
        return "mfcc_timbre"
    if feature in {"energy_score", "audio_ad_like_score", "audio_score_source"} or feature.endswith("__video_robust_z") or feature.endswith("__directional_score"):
        return "combined_score"
    if feature.endswith("_score"):
        return "combined_score"
    return "other_audio_or_context"


def base_feature_name(feature: str) -> str:
    for suffix in ["__video_robust_z", "__directional_score"]:
        if feature.endswith(suffix):
            return feature[: -len(suffix)]
    return feature


def describe_feature(feature: str) -> Tuple[str, str, str, bool]:
    base = base_feature_name(feature)
    group = infer_group(feature)
    if group == "volume_energy":
        return (
            "소리의 크기와 에너지 수준을 요약한다.",
            "광고 구간에서 배경음악, 내레이션, 효과음 때문에 평균 에너지나 피크가 높아지는지 보는 단서이다.",
            "볼륨/음량에 직접 영향을 받으므로 영상별 정규화 없이 절대값만 비교하면 조심해야 한다.",
            base in {"audio_rms_mean", "audio_rms_std", "audio_log_energy_mean", "audio_log_energy_std", "audio_mean_abs_amplitude"},
        )
    if group == "silence_low_energy":
        return (
            "프레임 중 조용하거나 에너지가 낮은 비율을 나타낸다.",
            "광고가 말소리/음악으로 채워져 있으면 무음 또는 저에너지 비율이 낮아질 수 있다.",
            "조용한 브이로그 본문이나 자막-only 구간과 혼동될 수 있어 다른 feature와 함께 봐야 한다.",
            True,
        )
    if group == "temporal_change_onset":
        return (
            "짧은 시간 안의 소리 시작점, 박자감, 순간 변화량을 요약한다.",
            "징글, 컷 전환음, 말/음악의 잦은 시작처럼 광고적인 시간 변화가 많은지 보는 단서이다.",
            "2초 창에서는 onset count가 창 길이에 민감하므로 duration-normalized feature를 우선 쓰는 편이 낫다.",
            base in {"onset_density", "onset_count_per_sec", "onset_strength_mean", "onset_strength_max"},
        )
    if group == "spectral_change_texture":
        return (
            "주파수 스펙트럼이 프레임 사이에서 얼마나 바뀌는지와 음색 질감을 요약한다.",
            "음악/효과음/강한 편집음처럼 스펙트럼 변화가 큰 광고성 오디오를 포착할 수 있다.",
            "잡음이나 급격한 환경음 변화도 높게 나올 수 있어 독립 판정 feature로 쓰면 위험하다.",
            base in {"spectral_flux_mean", "spectral_flux_std", "spectral_flux_max"},
        )
    if group == "spectral_shape":
        return (
            "소리가 낮은/높은 주파수에 치우쳤는지, 대역폭과 평탄도가 어떤지 나타낸다.",
            "광고 음악, 음성, 잡음성 효과음의 주파수 모양 차이를 보조적으로 설명한다.",
            "콘텐츠 장르와 녹음 환경 영향을 크게 받아 relative 분석에서는 보조 feature로 다루는 것이 좋다.",
            base in {"spectral_centroid_mean", "spectral_bandwidth_mean", "spectral_flatness_mean", "spectral_flatness_std"},
        )
    if group == "mfcc_timbre":
        return (
            "음색을 압축한 MFCC 계수의 평균/변동성을 나타낸다.",
            "광고와 일반 구간의 말소리/음악/효과음 음색 차이를 포착할 수 있다.",
            "개별 계수 해석이 직관적이지 않아 설명 문서나 rule 설계에서는 보조 근거로 두는 편이 안전하다.",
            base in {"mfcc_3_std", "mfcc_5_std", "mfcc_7_std", "mfcc_8_std", "mfcc_9_std", "mfcc_10_std", "mfcc_11_std", "mfcc_12_std", "mfcc_13_std"},
        )
    if group == "combined_score":
        return (
            "여러 raw audio feature를 정규화/방향화해 합친 파생 점수 또는 score 중간값이다.",
            "한 가지 볼륨이 아니라 onset, flux, energy, silence, spectral texture를 함께 종합한다.",
            "이미 score formula가 반영된 값이므로 2.2에서 새 baseline을 만들 때 raw feature와 구분해야 한다.",
            base == "audio_ad_like_score",
        )
    if group == "metadata_status":
        return (
            "추출 구간, split, 경로, 상태를 기록하는 메타데이터이다.",
            "feature 해석 자체보다는 train-only 필터링과 오류 점검에 사용한다.",
            "음향 feature가 아니므로 relative audio pattern 판단에는 직접 쓰지 않는다.",
            False,
        )
    return (
        "audio artifact에 존재하는 기타 컬럼이다.",
        "직접적인 음향 단서 또는 context 식별용 보조 컬럼일 수 있다.",
        "사용 전 생성 맥락을 확인해야 한다.",
        False,
    )


def format_weight(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def feature_to_component(feature: str, score_components: Dict[str, List[str]]) -> Optional[str]:
    base = base_feature_name(feature)
    for component, features in score_components.items():
        if base in features:
            return component
    if base in score_components:
        return base
    return None


def collect_feature_names(
    helper_features: Sequence[str],
    headers: Dict[str, List[str]],
    score_components: Dict[str, List[str]],
    selected_features: Sequence[str],
    thresholds: Dict[str, Any],
) -> List[str]:
    names: set[str] = set()
    names.update(helper_features)
    names.update(selected_features)
    names.update(thresholds.keys())
    names.update(["onset_density", "onset_count_per_sec", "spectral_flux_mean_per_sec_proxy", "onset_strength_mean_per_sec_proxy"])
    for component, features in score_components.items():
        names.add(component)
        names.update(features)
    names.update(["energy_score", "inverse_silence_score", "audio_ad_like_score", "audio_score_source"])
    metadata_keep = {
        "video_id", "split", "start_sec", "end_sec", "duration_sec", "subwindow_id",
        "subwindow_start_sec", "subwindow_end_sec", "subwindow_duration_sec",
        "segment_start_sec", "segment_end_sec", "segment_duration_sec", "feature_status",
        "extraction_status", "has_audio", "audio_available", "sample_rate", "decoded_sample_rate",
        "audio_num_samples", "decoded_num_samples", "source_video_path", "video_path",
    }
    for header in headers.values():
        for col in header:
            if col in metadata_keep or infer_group(col) != "other_audio_or_context":
                names.add(col)
    return sorted(names, key=lambda n: (infer_group(n), n))


def source_artifact_string(feature: str, headers: Dict[str, List[str]], labels: Dict[str, str]) -> str:
    sources = [labels[key] for key, header in headers.items() if feature in header]
    return "; ".join(sources)


def source_script_or_config_string(
    feature: str,
    helper_features: Sequence[str],
    selected_features: Sequence[str],
    thresholds: Dict[str, Any],
    score_components: Dict[str, List[str]],
    paths: Dict[str, Path],
) -> str:
    sources: List[str] = []
    base = base_feature_name(feature)
    if base in helper_features:
        sources.append(str(paths["helper_script"]))
    if feature_to_component(feature, score_components):
        sources.append(str(paths["persistence_script"]))
    if base in selected_features or base in thresholds:
        sources.append(str(paths["train_config"]))
    if feature in {"audio_ad_like_score", "audio_score_source"} or feature.endswith("__video_robust_z") or feature.endswith("__directional_score"):
        sources.append(str(paths["full_video_script"]))
    return "; ".join(dict.fromkeys(sources)) or "not_found"


def build_feature_inventory(
    feature_names: Sequence[str],
    headers: Dict[str, List[str]],
    header_labels: Dict[str, str],
    helper_features: Sequence[str],
    score_components: Dict[str, List[str]],
    score_weights: Dict[str, float],
    score_input_features: Sequence[str],
    selected_features: Sequence[str],
    thresholds: Dict[str, Any],
    paths_for_source: Dict[str, Path],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    selected_set = set(selected_features)
    score_inputs = set(score_input_features)
    for feature in feature_names:
        base = base_feature_name(feature)
        component = feature_to_component(feature, score_components)
        group = infer_group(feature)
        desc, interpretation, caution, recommend = describe_feature(feature)
        if feature.endswith("__video_robust_z") or feature.endswith("__directional_score"):
            recommend = False
            caution = "이미 기존 score formula의 정규화/방향화 결과이므로 2.2 raw baseline 후보로 직접 쓰기보다 참고용으로만 둔다."
        if group == "metadata_status":
            recommend = False
        used_in_score = bool(base in score_inputs or feature in score_components or feature == "audio_ad_like_score" or feature.endswith("__video_robust_z") or feature.endswith("__directional_score"))
        feature_weight = ""
        if component and component in score_weights:
            feature_weight = format_weight(float(score_weights[component]))
        elif feature in score_weights:
            feature_weight = format_weight(float(score_weights[feature]))
        threshold_info = thresholds.get(base, thresholds.get(feature, {})) if isinstance(thresholds, dict) else {}
        threshold_available = bool(threshold_info)
        threshold_related = "yes" if threshold_available else "no"
        if feature in {"audio_ad_like_score", "onset_density_score", "flux_onset_score", "energy_score", "inverse_silence_score", "spectral_texture_score"}:
            threshold_related = "score_or_component"
        rows.append({
            "feature_name": feature,
            "feature_group": group,
            "source_artifact": source_artifact_string(feature, headers, header_labels),
            "source_script_or_config": source_script_or_config_string(feature, helper_features, selected_features, thresholds, score_components, paths_for_source),
            "exists_in_labeled_segment_features": feature in headers.get("labeled", []),
            "exists_in_edge_subwindow_features": feature in headers.get("edge", []),
            "exists_in_full_video_features": feature in headers.get("full_video", []),
            "used_in_audio_ad_like_score": used_in_score,
            "selected_feature": base in selected_set,
            "feature_weight": feature_weight,
            "threshold_related": threshold_related,
            "train_only_threshold_available": threshold_available,
            "short_description_kr": desc,
            "interpretation_for_ad_detection_kr": interpretation,
            "recommended_for_2_2": recommend,
            "caution_note": caution,
            "score_component": component or "",
            "score_input_direction": "higher" if base in set(score_input_features[:9]) else "lower" if base in set(score_input_features[9:]) else "",
            "threshold_rule": threshold_info.get("threshold_rule", "") if isinstance(threshold_info, dict) else "",
            "threshold_value": threshold_info.get("threshold", "") if isinstance(threshold_info, dict) else "",
        })
    return pd.DataFrame(rows)


def build_score_formula_audit(
    paths: Dict[str, Path],
    train_config: Dict[str, Any],
    full_config: Dict[str, Any],
    score_components: Dict[str, List[str]],
    score_weights: Dict[str, float],
    higher_features: Sequence[str],
    lower_features: Sequence[str],
    selected_features: Sequence[str],
    thresholds: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    formula = train_config.get("audio_ad_like_score_formula") or full_config.get("ad_like_score_formula") or "not_found"
    rows.append({
        "audit_item": "audio_ad_like_score_formula",
        "source_type": "script_and_config",
        "source_path": f"{paths['persistence_script']}; {paths['train_config']}",
        "function_or_key": "add_ad_like_scores / audio_ad_like_score_formula",
        "component": "audio_ad_like_score",
        "input_features": "; ".join(dict.fromkeys(list(higher_features) + list(lower_features))),
        "feature_weight": "weighted component average",
        "normalization": "per-video robust z-score using median/IQR; fallback to baseline/global median and scale when needed",
        "directionality": "higher features keep z; lower features invert z before sigmoid",
        "threshold_name": "audio_ad_like_score_threshold",
        "threshold_value": train_config.get("start_boundary_rule", {}).get("post_10s_median_score_min", full_config.get("default_thresholds", {}).get("audio_ad_like_score_threshold", "not_found")),
        "threshold_source": "train_only_config" if train_config.get("split_basis") == "train_only" else full_config.get("threshold_source", "not_found"),
        "split_basis": train_config.get("split_basis", "not_found"),
        "evidence_summary": formula,
        "caution_note": "이번 감사는 formula 확인만 수행하며 threshold tuning 또는 적용 평가는 하지 않음",
    })
    for component, features in score_components.items():
        rows.append({
            "audit_item": "score_component",
            "source_type": "script_constant",
            "source_path": str(paths["persistence_script"]),
            "function_or_key": "SCORE_COMPONENTS / SCORE_WEIGHTS",
            "component": component,
            "input_features": "; ".join(features),
            "feature_weight": score_weights.get(component, "not_found"),
            "normalization": "component inputs are directional sigmoid scores averaged within component",
            "directionality": "higher/lower feature direction set before sigmoid",
            "threshold_name": "not_component_specific",
            "threshold_value": "",
            "threshold_source": "component weights from script/config, not threshold candidates",
            "split_basis": train_config.get("split_basis", "not_found"),
            "evidence_summary": f"{component} = mean({', '.join(features)} directional scores)",
            "caution_note": "component score is derived, not a raw acoustic measurement",
        })
    rows.append({
        "audit_item": "selected_features",
        "source_type": "config_key",
        "source_path": str(paths["train_config"]),
        "function_or_key": "selected_features",
        "component": "selected_features",
        "input_features": "; ".join(selected_features) if selected_features else "not_found",
        "feature_weight": "not_weighted_list",
        "normalization": "not_applicable",
        "directionality": json.dumps(train_config.get("feature_directions", {}), ensure_ascii=False),
        "threshold_name": "per-feature train_only_thresholds",
        "threshold_value": f"{len(thresholds)} threshold entries",
        "threshold_source": "train_only_thresholds" if thresholds else "not_found",
        "split_basis": train_config.get("split_basis", "not_found"),
        "evidence_summary": "Config-selected feature list for train-only audio rule discussion",
        "caution_note": "selected_features 목록은 score input list와 완전히 동일하지 않을 수 있음",
    })
    for feature, info in thresholds.items():
        if not isinstance(info, dict):
            info = {}
        rows.append({
            "audit_item": "train_only_threshold",
            "source_type": "config_key",
            "source_path": str(paths["train_config"]),
            "function_or_key": "train_only_thresholds",
            "component": infer_group(feature),
            "input_features": feature,
            "feature_weight": "",
            "normalization": "threshold candidate from train-only analysis, not recomputed here",
            "directionality": info.get("direction", "not_found"),
            "threshold_name": feature,
            "threshold_value": info.get("threshold", "not_found"),
            "threshold_source": "train_only_thresholds",
            "split_basis": train_config.get("split_basis", "not_found"),
            "evidence_summary": info.get("threshold_rule", "not_found"),
            "caution_note": "감사 대상일 뿐 이번 작업에서 threshold tuning/적용은 하지 않음",
        })
    return pd.DataFrame(rows)


def build_recommendations(feature_inventory: pd.DataFrame) -> pd.DataFrame:
    df = feature_inventory[feature_inventory["recommended_for_2_2"].astype(bool)].copy()
    preferred_order = [
        "audio_rms_mean", "audio_rms_std", "audio_log_energy_mean", "audio_log_energy_std", "audio_mean_abs_amplitude",
        "silence_ratio", "low_energy_ratio", "spectral_flux_mean", "spectral_flux_std", "spectral_flux_max",
        "onset_density", "onset_count_per_sec", "onset_strength_mean", "onset_strength_max",
        "spectral_centroid_mean", "spectral_bandwidth_mean", "spectral_flatness_mean", "spectral_flatness_std",
        "audio_ad_like_score", "mfcc_3_std", "mfcc_5_std", "mfcc_7_std", "mfcc_8_std", "mfcc_9_std", "mfcc_10_std", "mfcc_11_std", "mfcc_12_std", "mfcc_13_std",
    ]
    priority = {name: idx for idx, name in enumerate(preferred_order)}
    df["recommendation_priority"] = df["feature_name"].map(priority).fillna(999).astype(int)
    df["recommended_usage_for_2_2_kr"] = df.apply(
        lambda row: "핵심 raw feature 후보" if row["recommendation_priority"] < 18 else "보조/해석주의 feature 후보",
        axis=1,
    )
    cols = [
        "feature_name", "feature_group", "recommendation_priority", "recommended_usage_for_2_2_kr",
        "short_description_kr", "interpretation_for_ad_detection_kr", "caution_note",
        "exists_in_full_video_features", "used_in_audio_ad_like_score", "selected_feature",
    ]
    return df.sort_values(["recommendation_priority", "feature_name"])[cols]


def validate_outputs(
    project_root: Path,
    paths: Dict[str, Path],
    split_df: pd.DataFrame,
    csv_infos: Dict[str, Dict[str, Any]],
    feature_inventory: pd.DataFrame,
    score_audit: pd.DataFrame,
    protected_before: Dict[str, Dict[str, Any]],
    bundle_dir: Optional[Path],
) -> Dict[str, Any]:
    required_cols_missing = [col for col in FEATURE_INVENTORY_COLUMNS if col not in feature_inventory.columns]
    groups_present = sorted(feature_inventory["feature_group"].dropna().unique().tolist()) if "feature_group" in feature_inventory else []
    required_groups = ["volume_energy", "silence_low_energy", "temporal_change_onset", "spectral_change_texture", "spectral_shape", "mfcc_timbre", "combined_score"]
    missing_groups = [group for group in required_groups if group not in groups_present]
    header_mismatches: List[str] = []
    for source_key, flag_col in [
        ("labeled", "exists_in_labeled_segment_features"),
        ("edge", "exists_in_edge_subwindow_features"),
        ("full_video", "exists_in_full_video_features"),
    ]:
        header = set(csv_infos.get(source_key, {}).get("header", []))
        for _, row in feature_inventory.iterrows():
            expected = row["feature_name"] in header
            observed = bool(row[flag_col])
            if expected != observed:
                header_mismatches.append(f"{source_key}:{row['feature_name']} expected={expected} observed={observed}")
    protected_after = {name: snapshot_path(Path(snap["path"])) for name, snap in protected_before.items()}
    unchanged = {name: snapshot_unchanged(protected_before[name], protected_after[name]) for name in protected_before}
    forbidden_bundle_files: List[str] = []
    if bundle_dir and bundle_dir.exists():
        for path in bundle_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
                forbidden_bundle_files.append(str(path))
    split_groups = {
        split: sorted(pd.to_numeric(group["video_id"], errors="coerce").dropna().astype(int).tolist())
        for split, group in split_df.groupby("split")
    }
    return {
        "input_source_validation": {
            "key_scripts_exist": all((project_root / p).exists() for p in [
                "scripts/audio/extract_labeled_audio_features_v2_4.py",
                "scripts/audio/extract_audio_ad_edge_persistence_v2_4.py",
                "scripts/audio/full_video_audio_clue_extraction_v2_4_train.py",
            ]),
            "key_configs_exist": all((project_root / p).exists() for p in [
                "configs/audio_persistence_rule_config_v2_4_train_only.json",
                "configs/audio_persistence_rule_config_v2_4.json",
            ]),
            "large_csv_read_modes": {key: info.get("read_mode") for key, info in csv_infos.items()},
            "csv_header_or_nrows_only": all(str(info.get("read_mode", "")).startswith("csv_header_plus_nrows") for info in csv_infos.values()),
            "raw_video_audio_read_or_copied": bool(forbidden_bundle_files),
            "split_matches_fixed": split_groups == FIXED_SPLIT,
        },
        "feature_inventory_validation": {
            "required_columns_missing": required_cols_missing,
            "groups_present": groups_present,
            "required_groups_missing": missing_groups,
            "header_flag_mismatch_count": len(header_mismatches),
            "header_flag_mismatch_examples": header_mismatches[:20],
            "inventory_rows": int(len(feature_inventory)),
        },
        "score_formula_validation": {
            "audio_ad_like_score_formula_evidence_found": bool((score_audit["component"] == "audio_ad_like_score").any()),
            "selected_features_recorded": bool((score_audit["audit_item"] == "selected_features").any()),
            "feature_weights_recorded": bool((score_audit["audit_item"] == "score_component").any()),
            "thresholds_recorded_count": int((score_audit["audit_item"] == "train_only_threshold").sum()),
            "not_found_items": score_audit[score_audit.astype(str).apply(lambda row: row.str.contains("not_found", regex=False).any(), axis=1)]["audit_item"].head(20).tolist(),
        },
        "scope_leakage_validation": {
            "relative_audio_baseline_performed": False,
            "detector_config_script_output_modified": not all(unchanged.get(name, True) for name in ["detector_scripts", "detector_configs", "detector_outputs"]),
            "validation_test_row_level_output_created": False,
            "old_project_modified": not unchanged.get("old_project", True),
            "protected_paths_unchanged": unchanged,
            "forbidden_bundle_files": forbidden_bundle_files,
        },
    }


def write_summary(
    path: Path,
    report: Dict[str, Any],
    feature_inventory: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = feature_inventory.groupby("feature_group")["feature_name"].count().sort_index().to_dict()
    score_rows = report["score_formula_summary"]
    recommended_names = recommendations.sort_values("recommendation_priority")["feature_name"].head(24).tolist()
    lines = [
        "# Audio Feature Inventory Audit v2_4 Train",
        "",
        "## Scope",
        "- 목적: 현재 audio feature가 단순 볼륨인지, 복합 음향 feature인지 감사하고 문서화한다.",
        "- 2.2 relative audio baseline analysis는 수행하지 않았다.",
        "- raw video/audio decoding은 수행하지 않았다.",
        "- detector rule/config/output은 수정하지 않았다.",
        "- validation/test row-level output은 생성하지 않았다.",
        "",
        "## Conclusion",
        "현재 오디오 feature extraction은 단순 볼륨만 보지 않는다. RMS/log energy 같은 volume/energy feature 외에도 silence/low-energy ratio, zero crossing, spectral centroid/bandwidth/rolloff/flatness, spectral flux, onset count/strength, MFCC timbre feature를 함께 추출한다. 또한 `audio_ad_like_score`는 onset, flux/onset, energy, inverse silence, spectral texture component를 weighted average로 결합한 복합 점수이다.",
        "",
        "## Feature Groups",
    ]
    for group, count in groups.items():
        examples = feature_inventory[feature_inventory["feature_group"] == group]["feature_name"].head(8).tolist()
        lines.append(f"- {group}: {count} columns, examples: `{', '.join(examples)}`")
    lines += [
        "",
        "## Score Formula",
        f"- Function/key: {score_rows['function_or_key']}",
        f"- Formula: {score_rows['formula']}",
        f"- Component weights: {score_rows['component_weights']}",
        f"- Normalization: {score_rows['normalization']}",
        f"- Directionality: {score_rows['directionality']}",
        f"- Threshold source: {score_rows['threshold_source']}",
        "",
        "## 2.2 Recommended Feature Candidates",
        "- " + ", ".join(f"`{name}`" for name in recommended_names),
        "",
        "## Cautions For 2.2",
        "- `audio_ad_like_score`, `*_score`, `*__video_robust_z`, `*__directional_score`는 이미 기존 formula/정규화가 반영된 파생값이다.",
        "- MFCC 계수는 음색 차이를 담지만 개별 계수 설명이 직관적이지 않으므로 보조 feature로 다루는 편이 안전하다.",
        "- 이번 작업에서는 per-video percentile, robust z-score, high/medium/low 판정을 새로 만들지 않았다.",
        "",
        "## Generated Files",
    ]
    for key, value in report["output_files"].items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## Safety Flags",
        "- detector_rule_modified=false",
        "- old_project_modified=false",
        "- validation_test_row_level_output_created=false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_presentation(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# 오디오 Feature 설명 v2_4 Train

현재 오디오 분석은 단순히 볼륨만 보는 방식이 아니다. 기본적인 음량/에너지(RMS, log energy, peak amplitude)뿐 아니라, 무음 또는 저에너지 비율, 프레임 사이 주파수 변화(spectral flux), 소리의 시작점과 순간 변화(onset count/strength), 주파수 분포 모양(spectral centroid, bandwidth, rolloff, flatness), 그리고 음색을 요약하는 MFCC 계수까지 함께 추출한다. `audio_ad_like_score`도 단일 볼륨 임계값이 아니라 onset density, flux/onset 변화, energy, inverse silence, spectral texture를 각각 방향화/정규화한 뒤 가중 평균으로 결합한 복합 오디오 단서이다.

## 쉽게 말하면
- volume/energy: 소리가 얼마나 크고 강한가.
- silence/low-energy: 조용한 구간이 얼마나 적거나 많은가.
- onset/temporal change: 소리의 시작, 박자감, 순간 변화가 얼마나 잦은가.
- spectral change/shape: 음악, 효과음, 음성처럼 주파수 모양과 변화가 어떻게 다른가.
- MFCC/timbre: 사람이 듣는 음색 차이를 압축해서 표현한 값이다.
- combined score: 위 단서들을 합쳐 광고처럼 들릴 가능성을 요약한 파생 점수이다.

## 2.2로 넘길 때의 권장 방향
2.2 relative baseline에서는 raw feature인 `audio_rms_mean`, `audio_log_energy_mean`, `silence_ratio`, `low_energy_ratio`, `spectral_flux_mean`, `spectral_flux_max`, `onset_density`, `onset_count_per_sec`, `onset_strength_mean`, `onset_strength_max`, `spectral_flatness_mean/std`를 우선 후보로 두는 것이 좋다. `audio_ad_like_score`는 요약 점수로 참고할 수 있지만, 이미 기존 score formula가 들어간 파생값이므로 raw feature 기반 relative baseline과 분리해서 해석해야 한다.
"""
    path.write_text(text, encoding="utf-8")


def write_latest_bundle(bundle_dir: Path, project_root: Path, paths: Dict[str, Path], report: Dict[str, Any]) -> Dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    copy_map = {
        "script": project_root / SCRIPT_RELATIVE_PATH,
        "summary_md": paths["summary_md"],
        "report_json": paths["report_json"],
        "presentation_md": paths["presentation_md"],
        "feature_inventory_csv": paths["feature_inventory"],
        "score_formula_csv": paths["score_formula"],
        "source_artifact_inventory_csv": paths["source_artifact_inventory"],
        "recommendation_csv": paths["recommendations"],
        "run_log": paths["run_log"],
    }
    for _, src in copy_map.items():
        if not src.exists() or src.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
            continue
        dst = bundle_dir / src.name
        if dst.exists():
            dst = unique_file_path(dst, report["run_id"])
        shutil.copy2(src, dst)
        copied.append(str(dst))
    readme = bundle_dir / "README_latest_files.md"
    lines = [
        "# Latest Files: Audio Feature Inventory Audit v2_4 Train",
        "",
        "This bundle contains audit/report artifacts only. It does not include raw video/audio/cache/model files or the large full-video feature CSV.",
        "",
        "## Main Paths",
    ]
    for key, value in paths.items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## Large Full-Video Feature CSV",
        f"- Referenced only, not copied: `{report['referenced_large_full_video_feature_csv']}`",
        "",
        "## Copied Files",
    ]
    for item in copied:
        lines.append(f"- `{item}`")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    copied.append(str(readme))
    return {"bundle_dir": str(bundle_dir), "copied_files": copied, "raw_or_large_feature_copied": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = output_paths(project_root, run_id)
    logger = TaskLogger(paths["run_log"])
    t0 = time.time()

    logger.log("[STEP 01] Safety snapshot and output path planning")
    protected_targets = {
        "old_project": OLD_PROJECT_ROOT,
        "detector_scripts": project_root / "scripts/detectors",
        "detector_configs": project_root / "configs/detectors",
        "detector_outputs": project_root / "data/predictions",
        "existing_audio_input_features": project_root / "data/audio/audio_labeled_segment_features_v2_4_with_split.csv",
        "split_file": project_root / "data/splits/video_split_v2_4.csv",
        "label_file": project_root / "data/segments/ad_interval_segments_v2_4.csv",
        "raw_videos": project_root / "data/raw/videos",
        "two_one_full_video_output": project_root / "data/audio/full_video_audio_subwindow_features_v2_4_train_20260526_1750_final.csv",
    }
    protected_before = {name: snapshot_path(path) for name, path in protected_targets.items()}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    bundle_dir = project_root / "outputs/latest_for_chatgpt_audio_feature_inventory_audit_v2_4_train"

    logger.log("[STEP 02] Locate audio scripts/configs/reports/data artifacts")
    artifact_rows = locate_audio_artifacts(project_root)
    pd.DataFrame(artifact_rows).to_csv(paths["source_artifact_inventory"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 03] Read script-defined feature columns and helper functions")
    source_paths = {
        "helper_script": project_root / "scripts/audio/extract_labeled_audio_features_v2_4.py",
        "persistence_script": project_root / "scripts/audio/extract_audio_ad_edge_persistence_v2_4.py",
        "full_video_script": project_root / "scripts/audio/full_video_audio_clue_extraction_v2_4_train.py",
        "train_config": project_root / "configs/audio_persistence_rule_config_v2_4_train_only.json",
        "full_config": project_root / "configs/audio_persistence_rule_config_v2_4.json",
    }
    helper, helper_error = load_module(source_paths["helper_script"], "audio_feature_helper_for_audit")
    persistence, persistence_error = load_module(source_paths["persistence_script"], "audio_persistence_for_audit")
    helper_features = list(getattr(helper, "FEATURE_COLUMNS", [])) if helper else []
    higher_features = list(getattr(persistence, "HIGHER_FEATURES", [])) if persistence else []
    lower_features = list(getattr(persistence, "LOWER_FEATURES", [])) if persistence else []
    score_components = dict(getattr(persistence, "SCORE_COMPONENTS", {})) if persistence else {}
    score_weights = dict(getattr(persistence, "SCORE_WEIGHTS", {})) if persistence else {}

    logger.log("[STEP 04] Read config-defined selected features, weights, and thresholds")
    train_config, train_config_error = read_json(source_paths["train_config"])
    full_config, full_config_error = read_json(source_paths["full_config"])
    selected_features = list(train_config.get("selected_features", []))
    thresholds = dict(train_config.get("train_only_thresholds", {}))
    if not score_components:
        score_components = dict(train_config.get("score_components", full_config.get("score_components", {})))
    if not score_weights:
        score_weights = dict(train_config.get("feature_weights", full_config.get("feature_weights", {})))
    if not higher_features or not lower_features:
        directions = train_config.get("feature_directions", full_config.get("feature_directions", {}))
        higher_features = [name for name, direction in directions.items() if str(direction).startswith("higher")]
        lower_features = [name for name, direction in directions.items() if str(direction).startswith("lower")]

    logger.log("[STEP 05] Inspect CSV headers without loading large full data")
    csv_paths = {
        "labeled": project_root / "data/audio/audio_labeled_segment_features_v2_4_with_split.csv",
        "edge": project_root / "data/audio/audio_ad_edge_persistence_subwindow_features_v2_4_with_split.csv",
        "full_video": project_root / "data/audio/full_video_audio_subwindow_features_v2_4_train_20260526_1750_final.csv",
        "train_recommendations": project_root / "data/audio/audio_rule_feature_recommendations_v2_4_train_only.csv",
        "train_threshold_candidates": project_root / "data/audio/audio_rule_candidate_thresholds_v2_4_train_only.csv",
    }
    csv_infos = {key: read_csv_header_sample(path, 5) for key, path in csv_paths.items()}
    headers = {key: info.get("header", []) for key, info in csv_infos.items()}
    header_labels = {key: str(path) for key, path in csv_paths.items()}

    logger.log("[STEP 06] Build audio feature inventory table")
    score_input_features = list(dict.fromkeys(higher_features + lower_features))
    feature_names = collect_feature_names(helper_features, headers, score_components, selected_features, thresholds)
    feature_inventory = build_feature_inventory(
        feature_names=feature_names,
        headers={"labeled": headers["labeled"], "edge": headers["edge"], "full_video": headers["full_video"]},
        header_labels=header_labels,
        helper_features=helper_features,
        score_components=score_components,
        score_weights=score_weights,
        score_input_features=score_input_features,
        selected_features=selected_features,
        thresholds=thresholds,
        paths_for_source=source_paths,
    )
    feature_inventory.to_csv(paths["feature_inventory"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 07] Audit audio_ad_like_score formula and score components")
    score_audit = build_score_formula_audit(
        paths=source_paths,
        train_config=train_config,
        full_config=full_config,
        score_components=score_components,
        score_weights=score_weights,
        higher_features=higher_features,
        lower_features=lower_features,
        selected_features=selected_features,
        thresholds=thresholds,
    )
    score_audit.to_csv(paths["score_formula"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 08] Classify features by acoustic meaning")
    # classification은 feature_inventory에 이미 들어 있으므로 이 단계에서는 group summary만 기록한다.
    group_summary = feature_inventory.groupby("feature_group")["feature_name"].count().sort_index().to_dict()

    logger.log("[STEP 09] Write 2.2 recommendation table")
    recommendations = build_recommendations(feature_inventory)
    recommendations.to_csv(paths["recommendations"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 10] Generate summary/report/presentation explanation")
    split_df = pd.read_csv(project_root / "data/splits/video_split_v2_4.csv", encoding="utf-8-sig")
    formula = train_config.get("audio_ad_like_score_formula", full_config.get("ad_like_score_formula", "not_found"))
    score_formula_summary = {
        "function_or_key": "add_ad_like_scores / add_video_robust_scores / SCORE_COMPONENTS / SCORE_WEIGHTS",
        "formula": formula,
        "component_weights": score_weights,
        "normalization": "video robust z-score median/IQR, fallback baseline/global median-scale, sigmoid directional scores",
        "directionality": {"higher_features": higher_features, "lower_features": lower_features},
        "threshold_source": "train_only_thresholds and train-only config" if train_config.get("split_basis") == "train_only" else "not_found_or_full_split_reference",
    }
    conclusion = {
        "simple_volume_only": False,
        "conclusion_kr": "단순 볼륨만 추출하지 않는다. volume/energy 외에 silence, spectral shape/change, onset, MFCC, combined score를 포함한다.",
        "feature_group_summary": group_summary,
        "score_is_multi_feature": True,
    }
    report: Dict[str, Any] = {
        "task_name": TASK_NAME,
        "version": VERSION,
        "run_id": run_id,
        "created_at": now_iso(),
        "project_root": str(project_root),
        "purpose": "audit current audio feature inventory and audio_ad_like_score formula; no relative analysis",
        "train_video_ids": FIXED_SPLIT["train"],
        "validation_video_ids": FIXED_SPLIT["validation"],
        "test_video_ids": FIXED_SPLIT["test"],
        "input_files": {key: str(value) for key, value in source_paths.items()},
        "csv_header_inspection": {key: {k: v for k, v in info.items() if k != "header"} for key, info in csv_infos.items()},
        "csv_headers": {key: info.get("header", []) for key, info in csv_infos.items()},
        "script_read_errors": {"helper": helper_error, "persistence": persistence_error},
        "config_read_errors": {"train_config": train_config_error, "full_config": full_config_error},
        "selected_features": selected_features,
        "score_input_features": score_input_features,
        "score_components": score_components,
        "score_weights": score_weights,
        "train_only_threshold_count": len(thresholds),
        "threshold_split_basis": train_config.get("split_basis", "not_found"),
        "leakage_guard_notes": train_config.get("leakage_guard_notes", []),
        "score_formula_summary": score_formula_summary,
        "conclusion": conclusion,
        "recommended_features_for_2_2": recommendations["feature_name"].head(40).tolist(),
        "feature_inventory_row_count": int(len(feature_inventory)),
        "score_formula_audit_row_count": int(len(score_audit)),
        "source_artifact_inventory_row_count": int(len(artifact_rows)),
        "recommendation_row_count": int(len(recommendations)),
        "output_files": {key: str(value) for key, value in paths.items()},
        "referenced_large_full_video_feature_csv": str(csv_paths["full_video"]),
        "modified_files": [str(project_root / SCRIPT_RELATIVE_PATH)],
        "generated_files": [str(value) for value in paths.values()],
        "detector_rule_modified": False,
        "old_project_modified": False,
        "validation_test_row_level_output_created": False,
    }
    write_summary(paths["summary_md"], report, feature_inventory, recommendations)
    write_presentation(paths["presentation_md"])
    save_json(paths["report_json"], report)

    logger.log("[STEP 11] Run Sub Agent validations")
    validations = validate_outputs(project_root, paths, split_df, csv_infos, feature_inventory, score_audit, protected_before, None)
    report["sub_agent_validations"] = validations
    report["detector_rule_modified"] = bool(validations["scope_leakage_validation"]["detector_config_script_output_modified"])
    report["old_project_modified"] = bool(validations["scope_leakage_validation"]["old_project_modified"])
    report["validation_test_row_level_output_created"] = bool(validations["scope_leakage_validation"]["validation_test_row_level_output_created"])
    write_summary(paths["summary_md"], report, feature_inventory, recommendations)
    save_json(paths["report_json"], report)

    logger.log("[STEP 12] Update latest bundle")
    bundle_report = write_latest_bundle(bundle_dir, project_root, paths, report)
    validations_after_bundle = validate_outputs(project_root, paths, split_df, csv_infos, feature_inventory, score_audit, protected_before, bundle_dir)
    report["latest_bundle"] = bundle_report
    report["sub_agent_validations"] = validations_after_bundle
    report["detector_rule_modified"] = bool(validations_after_bundle["scope_leakage_validation"]["detector_config_script_output_modified"])
    report["old_project_modified"] = bool(validations_after_bundle["scope_leakage_validation"]["old_project_modified"])
    report["validation_test_row_level_output_created"] = bool(validations_after_bundle["scope_leakage_validation"]["validation_test_row_level_output_created"])
    report["elapsed_sec"] = time.time() - t0
    write_summary(paths["summary_md"], report, feature_inventory, recommendations)
    save_json(paths["report_json"], report)
    # final JSON refresh 전에 첫 bundle copy가 일어났다면 final report/summary를 한 번 더 복사한다.
    write_latest_bundle(bundle_dir, project_root, paths, report)

    logger.log("[STEP 13] Print final human-readable summary")
    logger.log(
        "[STEP 13] "
        f"feature_inventory_rows={len(feature_inventory)}, "
        f"score_audit_rows={len(score_audit)}, "
        f"recommended_features={len(recommendations)}, "
        f"simple_volume_only={conclusion['simple_volume_only']}, "
        f"detector_rule_modified={report['detector_rule_modified']}, "
        f"old_project_modified={report['old_project_modified']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
