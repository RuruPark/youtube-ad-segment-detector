#!/usr/bin/env python3
"""pretrained ResNet embedding을 이용해 v2.3 장면 전환 후보를 추출한다."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("YASD_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
OLD_PROJECT_ROOT = Path(os.environ.get("YASD_OLD_PROJECT_ROOT", PROJECT_ROOT / "_old_project_not_included")).resolve()
VIDEO_INPUT_DIR = PROJECT_ROOT / "data/raw/videos"
MANIFEST_PATH = PROJECT_ROOT / "data/video_metadata/video_manifest_v2_2.csv"
AD_SEGMENTS_PATH = PROJECT_ROOT / "data/segments/ad_interval_segments_v2_3.csv"
ALL_SEGMENTS_PATH = PROJECT_ROOT / "data/segments/label_interval_context_segments_v2_3.csv"
WINDOWS_PATH = PROJECT_ROOT / "data/windows/window_labels_5s_v2_3.csv"
CHECK_ENV_SCRIPT = PROJECT_ROOT / "scripts/utils/check_cv_environment.py"

SCENE_DIR = PROJECT_ROOT / "data/scene"
FRAME_DECODE_SUMMARY_PATH = SCENE_DIR / "resnet_frame_decode_summary_v2_3.csv"
SCORES_PATH = SCENE_DIR / "resnet_scene_scores_v2_3.csv"
CANDIDATES_PATH = SCENE_DIR / "resnet_scene_candidates_v2_3.csv"
CANDIDATES_MMSS_PATH = SCENE_DIR / "resnet_scene_candidates_v2_3_mmss.csv"
VIDEO_SUMMARY_PATH = SCENE_DIR / "resnet_scene_video_summary_v2_3.csv"
BOUNDARY_AUDIT_PATH = SCENE_DIR / "resnet_scene_candidate_boundary_audit_v2_3.csv"
COMPARISON_PATH = SCENE_DIR / "opencv_vs_resnet_scene_comparison_v2_3.csv"
OVERLAP_PATH = SCENE_DIR / "opencv_resnet_candidate_overlap_v2_3.csv"

OPENCV_CANDIDATES_PATH = SCENE_DIR / "scene_candidates_v2_3_merged_ffmpeg_fallback.csv"
OPENCV_CANDIDATES_MMSS_PATH = SCENE_DIR / "scene_candidates_v2_3_merged_ffmpeg_fallback_mmss.csv"
OPENCV_AUDIT_PATH = SCENE_DIR / "scene_candidate_boundary_audit_v2_3_merged_ffmpeg_fallback.csv"

REPORT_PATH = PROJECT_ROOT / "reports/extract_resnet_embedding_scene_change_v2_3_report.json"
SUMMARY_PATH = PROJECT_ROOT / "reports/extract_resnet_embedding_scene_change_v2_3_summary.md"
LOG_PATH = PROJECT_ROOT / "logs/extract_resnet_embedding_scene_change_v2_3_run_log.txt"
SCRIPT_PATH = PROJECT_ROOT / "scripts/scene/extract_resnet_embedding_scene_change_v2_3.py"
README_PATH = PROJECT_ROOT / "README.md"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt"
LATEST_README = LATEST_DIR / "README_latest_files.md"

ESTIMATED_RUNTIME = "약 45분"
RUNTIME_ESTIMATION_REASON = "18개 영상 전체에 대해 ffmpeg 1fps frame decode, ResNet embedding 추출, candidate 생성, OpenCV 비교 audit을 수행해야 함."
SAMPLE_FPS = 1.0
RESIZE_SHORTER_SIDE = 256
DEDUP_TOLERANCE_SEC = 2.0
RUN_LOG: list[str] = []


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    RUN_LOG.append(f"[{stamp}] {message}")
    print(message, flush=True)


def step(number: int, total: int, message: str) -> None:
    log(f"[STEP {number}/{total}] {message}")


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def to_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        rounded = round(float(value), 6)
        if abs(rounded - round(rounded)) < 1e-9:
            return int(round(rounded))
        return rounded
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def is_true(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y"}


def mmss_floor(seconds: Any) -> str:
    sec = to_float(seconds)
    if sec is None:
        return ""
    total = max(0, int(math.floor(sec)))
    return f"{total // 60:02d}분 {total % 60:02d}초"


def mmss_round(seconds: Any) -> str:
    sec = to_float(seconds)
    if sec is None:
        return ""
    total = max(0, int(round(sec)))
    return f"{total // 60:02d}분 {total % 60:02d}초"


def readable(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}분 {total % 60}초"


def ensure_inside_project(path: Path) -> None:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and not str(resolved).startswith(str(root) + os.sep):
        raise RuntimeError(f"Refusing to write outside project root: {resolved}")
    old_root = OLD_PROJECT_ROOT.resolve()
    if resolved == old_root or str(resolved).startswith(str(old_root) + os.sep):
        raise RuntimeError(f"Refusing to write inside old project root: {resolved}")


def ensure_dirs() -> None:
    for path in [SCENE_DIR, PROJECT_ROOT / "reports", PROJECT_ROOT / "logs", PROJECT_ROOT / "scripts/scene", LATEST_DIR]:
        ensure_inside_project(path)
        path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_inside_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: fmt(row.get(column, "")) for column in columns})


def read_csv(path: Path, required: bool, warnings: list[Any], errors: list[Any]) -> pd.DataFrame:
    if not path.exists():
        message = {"missing_file": str(path)}
        if required:
            errors.append(message)
        else:
            warnings.append(message)
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def run_cmd(args: list[str]) -> tuple[bool, str, str]:
    log("Command: " + " ".join(args))
    try:
        result = subprocess.run(args, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        return result.returncode == 0, result.stdout, result.stderr
    except FileNotFoundError as exc:
        return False, "", str(exc)


def verify_cv_environment(warnings: list[Any]) -> tuple[bool, str]:
    executable = sys.executable
    in_cv = "/envs/cv/" in executable or executable.endswith("/envs/cv/bin/python")
    if CHECK_ENV_SCRIPT.exists():
        ok, out, err = run_cmd(["conda", "run", "-n", "cv", "python", str(CHECK_ENV_SCRIPT)])
        log("cv check stdout: " + out.strip().replace("\n", " | "))
        if err.strip():
            log("cv check stderr: " + err.strip().replace("\n", " | "))
        if not ok:
            warnings.append("cv_environment_check_script_failed")
            return False, executable
    else:
        ok, out, err = run_cmd(
            [
                "conda",
                "run",
                "-n",
                "cv",
                "python",
                "-c",
                "import sys; print(sys.executable); import pandas as pd; import numpy as np; print('pandas', pd.__version__); print('numpy', np.__version__)",
            ]
        )
        log("cv fallback stdout: " + out.strip().replace("\n", " | "))
        if err.strip():
            log("cv fallback stderr: " + err.strip().replace("\n", " | "))
        if not ok:
            warnings.append("cv_environment_fallback_check_failed")
            return False, executable
    if not in_cv:
        warnings.append({"current_python_executable_not_in_cv": executable})
    return in_cv, executable


def tool_version(tool: str) -> tuple[bool, str, str]:
    exe = shutil.which(tool)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if exe is None and conda_prefix:
        candidate = Path(conda_prefix) / "bin" / tool
        if candidate.exists():
            exe = str(candidate)
    if exe is None:
        return False, "", ""
    ok, out, err = run_cmd([exe, "-version"])
    first = (out or err).strip().splitlines()
    return ok, first[0] if first else "", exe


def import_torch_stack() -> tuple[bool, dict[str, Any], dict[str, Any]]:
    details: dict[str, Any] = {}
    modules: dict[str, Any] = {}
    try:
        import torch
        import torch.nn as nn
        import torchvision
        from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, resnet18, resnet34, resnet50

        details = {
            "torch_available": True,
            "torchvision_available": True,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        }
        modules = {
            "torch": torch,
            "nn": nn,
            "torchvision": torchvision,
            "resnet50": resnet50,
            "resnet34": resnet34,
            "resnet18": resnet18,
            "ResNet50_Weights": ResNet50_Weights,
            "ResNet34_Weights": ResNet34_Weights,
            "ResNet18_Weights": ResNet18_Weights,
        }
        return True, details, modules
    except Exception as exc:
        return False, {"torch_available": False, "torchvision_available": False, "torch_import_error": str(exc)}, {}


def checkpoint_path(torch_module: Any, weights: Any) -> Path:
    filename = Path(urlparse(weights.url).path).name
    return Path(torch_module.hub.get_dir()) / "checkpoints" / filename


def load_pretrained_resnet(modules: dict[str, Any], warnings: list[Any]) -> dict[str, Any]:
    torch = modules["torch"]
    nn = modules["nn"]
    model_specs = [
        ("resnet50", modules["resnet50"], modules["ResNet50_Weights"].DEFAULT),
        ("resnet34", modules["resnet34"], modules["ResNet34_Weights"].DEFAULT),
        ("resnet18", modules["resnet18"], modules["ResNet18_Weights"].DEFAULT),
    ]
    cache_status: list[dict[str, Any]] = []
    for model_name, constructor, weights in model_specs:
        cache_path = checkpoint_path(torch, weights)
        cache_status.append({"model_name": model_name, "weight": str(weights), "cache_path": str(cache_path), "exists": cache_path.exists()})
        if not cache_path.exists():
            warnings.append({"pretrained_weight_cache_missing": model_name, "expected_path": str(cache_path)})
            continue
        try:
            model = constructor(weights=None)
            state_dict = torch.load(cache_path, map_location="cpu")
            model.load_state_dict(state_dict)
            feature_dim = int(model.fc.in_features)
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
            preprocess = weights.transforms()
            return {
                "available": True,
                "model": model,
                "feature_extractor": feature_extractor,
                "preprocess": preprocess,
                "model_name": model_name,
                "pretrained_weight_used": str(weights),
                "pretrained_weight_source": f"local_torch_cache:{cache_path}",
                "feature_dim": feature_dim,
                "cache_status": cache_status,
            }
        except Exception as exc:
            warnings.append({"pretrained_weight_load_failed": model_name, "cache_path": str(cache_path), "error": str(exc)})
    return {
        "available": False,
        "model_name": "",
        "pretrained_weight_used": "",
        "pretrained_weight_source": "",
        "feature_dim": 0,
        "cache_status": cache_status,
    }


def load_videos(manifest: pd.DataFrame, warnings: list[Any]) -> list[dict[str, Any]]:
    if manifest.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        filename = clean(row.get("video_filename"))
        path = Path(clean(row.get("video_path")))
        if not path.exists() and filename:
            fallback = VIDEO_INPUT_DIR / filename
            if fallback.exists():
                path = fallback
        video_id = clean(row.get("label_mapping_video_id")) or clean(row.get("video_id"))
        video_title = clean(row.get("label_mapping_video_title")) or clean(row.get("matched_label_video_title")) or clean(row.get("video_title"))
        if not path.exists():
            warnings.append({"missing_video_file": filename, "video_path": str(path)})
        rows.append(
            {
                "video_id": video_id,
                "video_title": video_title,
                "video_filename": filename,
                "video_path": str(path),
                "duration_sec_manifest": to_float(row.get("duration_sec")),
                "width_manifest": to_float(row.get("width")),
                "height_manifest": to_float(row.get("height")),
            }
        )
    return sorted(rows, key=lambda r: (int(r["video_id"]) if str(r["video_id"]).isdigit() else 999999, r["video_filename"]))


def ffprobe_metadata(video: dict[str, Any], ffprobe_exe: str) -> dict[str, Any]:
    path = clean(video.get("video_path"))
    ok, out, err = run_cmd([ffprobe_exe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path])
    row = {
        "ffprobe_success": ok,
        "ffprobe_warning": "" if ok else err.strip(),
        "duration_sec": video.get("duration_sec_manifest") or "",
        "width": video.get("width_manifest") or "",
        "height": video.get("height_manifest") or "",
    }
    if not ok:
        return row
    try:
        payload = json.loads(out)
        fmt = payload.get("format", {})
        stream = next((s for s in payload.get("streams", []) if s.get("codec_type") == "video"), {})
        row.update(
            {
                "duration_sec": to_float(fmt.get("duration")) or to_float(stream.get("duration")) or row["duration_sec"],
                "width": to_float(stream.get("width")) or row["width"],
                "height": to_float(stream.get("height")) or row["height"],
                "codec_name": clean(stream.get("codec_name")),
                "pix_fmt": clean(stream.get("pix_fmt")),
            }
        )
    except Exception as exc:
        row["ffprobe_warning"] = f"ffprobe_json_parse_failed:{exc}"
    return row


def even_int(value: float) -> int:
    rounded = max(2, int(round(value)))
    return rounded if rounded % 2 == 0 else rounded + 1


def scaled_dimensions(width: Any, height: Any) -> tuple[int, int]:
    w = to_float(width)
    h = to_float(height)
    if not w or not h:
        return RESIZE_SHORTER_SIDE, RESIZE_SHORTER_SIDE
    if w >= h:
        scaled_h = RESIZE_SHORTER_SIDE
        scaled_w = even_int(w * RESIZE_SHORTER_SIDE / h)
    else:
        scaled_w = RESIZE_SHORTER_SIDE
        scaled_h = even_int(h * RESIZE_SHORTER_SIDE / w)
    return scaled_w, scaled_h


def read_exact(stream: Any, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def infer_batch_with_recovery(
    feature_extractor: Any,
    batch: list[Any],
    torch_module: Any,
    device: Any,
    warnings: list[Any],
    model_name: str,
) -> np.ndarray:
    try:
        inputs = torch_module.stack(batch, dim=0).to(device, non_blocking=True)
        with torch_module.no_grad():
            features = feature_extractor(inputs).flatten(1)
        return features.detach().cpu().numpy().astype(np.float32)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" not in message or len(batch) == 1:
            raise
        if str(device).startswith("cuda"):
            torch_module.cuda.empty_cache()
        warnings.append({"cuda_oom_batch_split": len(batch), "model_name": model_name})
        midpoint = len(batch) // 2
        left = infer_batch_with_recovery(feature_extractor, batch[:midpoint], torch_module, device, warnings, model_name)
        right = infer_batch_with_recovery(feature_extractor, batch[midpoint:], torch_module, device, warnings, model_name)
        return np.concatenate([left, right], axis=0)


def extract_video_embeddings(
    video: dict[str, Any],
    probe: dict[str, Any],
    ffmpeg_exe: str,
    feature_extractor: Any,
    preprocess: Any,
    torch_module: Any,
    device: Any,
    batch_size: int,
    model_name: str,
    warnings: list[Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    from PIL import Image

    width, height = scaled_dimensions(probe.get("width"), probe.get("height"))
    frame_size = width * height * 3
    duration_sec = to_float(probe.get("duration_sec")) or to_float(video.get("duration_sec_manifest")) or 0.0
    expected_count = int(math.floor(duration_sec)) + 1 if duration_sec > 0 else 0
    vf = f"fps={SAMPLE_FPS:g},scale={width}:{height}"
    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        clean(video.get("video_path")),
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    log("Command: " + " ".join(cmd[:7]) + " <video> " + " ".join(cmd[8:]))
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    feature_chunks: list[np.ndarray] = []
    batch: list[Any] = []
    decoded_count = 0
    decode_warning = ""
    batch_size_effective = batch_size
    started = time.time()
    try:
        while True:
            raw = read_exact(proc.stdout, frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                decode_warning = f"incomplete_raw_frame_bytes:{len(raw)}/{frame_size}"
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            image = Image.fromarray(frame, mode="RGB")
            batch.append(preprocess(image))
            decoded_count += 1
            if len(batch) >= batch_size_effective:
                features = infer_batch_with_recovery(feature_extractor, batch, torch_module, device, warnings, model_name)
                feature_chunks.append(features)
                batch.clear()
        if batch:
            features = infer_batch_with_recovery(feature_extractor, batch, torch_module, device, warnings, model_name)
            feature_chunks.append(features)
            batch.clear()
    finally:
        if proc.stdout:
            proc.stdout.close()
    stderr_text = proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else ""
    return_code = proc.wait()
    if stderr_text:
        decode_warning = (decode_warning + "; " if decode_warning else "") + stderr_text[:1000]
    if return_code != 0:
        decode_warning = (decode_warning + "; " if decode_warning else "") + f"ffmpeg_returncode:{return_code}"
    if expected_count and abs(decoded_count - expected_count) > 2:
        decode_warning = (decode_warning + "; " if decode_warning else "") + f"decoded_expected_mismatch:{decoded_count}/{expected_count}"
    embeddings = np.concatenate(feature_chunks, axis=0) if feature_chunks else np.empty((0, 0), dtype=np.float32)
    meta = {
        "duration_sec": duration_sec,
        "decoded_frame_count": int(decoded_count),
        "expected_1fps_frame_count": int(expected_count),
        "decode_success": decoded_count > 0 and return_code == 0,
        "decode_warning": decode_warning,
        "scaled_width": width,
        "scaled_height": height,
        "embedding_count": int(embeddings.shape[0]),
        "feature_dim_observed": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] else 0,
        "embedding_success": embeddings.shape[0] > 0,
        "embedding_elapsed_sec": time.time() - started,
        "batch_size_effective": batch_size_effective,
    }
    return embeddings, meta


def score_rows_for_video(video: dict[str, Any], embeddings: np.ndarray, meta: dict[str, Any], model_info: dict[str, Any]) -> list[dict[str, Any]]:
    if embeddings.shape[0] < 2:
        return []
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = embeddings / norms
    cosine_similarity = np.sum(normalized[:-1] * normalized[1:], axis=1)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    cosine_distance = 1.0 - cosine_similarity
    l2_distance = np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=1)
    duration = to_float(meta.get("duration_sec")) or 0.0
    rows: list[dict[str, Any]] = []
    for idx, (cos_dist, l2_dist) in enumerate(zip(cosine_distance, l2_distance)):
        prev_time = float(idx)
        next_time = float(idx + 1)
        if duration > 0:
            prev_time = min(prev_time, duration)
            next_time = min(next_time, duration)
        rows.append(
            {
                "video_id": clean(video.get("video_id")),
                "video_title": clean(video.get("video_title")),
                "video_filename": clean(video.get("video_filename")),
                "video_path": clean(video.get("video_path")),
                "prev_sample_time_sec": prev_time,
                "next_sample_time_sec": next_time,
                "score_time_sec": next_time,
                "score_time_mmss": mmss_floor(next_time),
                "cosine_distance": float(cos_dist),
                "l2_distance": float(l2_dist),
                "scene_change_score": float(cos_dist),
                "feature_dim": int(model_info.get("feature_dim") or embeddings.shape[1]),
                "model_name": model_info.get("model_name", ""),
                "method_used": "resnet_embedding_1fps",
                "candidate_source": "resnet_embedding_v2_3",
            }
        )
    return rows


def score_stats(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {
            "score_mean": "",
            "score_std": "",
            "score_p50": "",
            "score_p90": "",
            "score_p95": "",
            "score_p99": "",
            "threshold": "",
        }
    values = np.array([float(row["scene_change_score"]) for row in scores], dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    p95 = float(np.percentile(values, 95))
    threshold = max(median + 3.0 * mad, p95)
    return {
        "score_mean": float(np.mean(values)),
        "score_std": float(np.std(values)),
        "score_p50": median,
        "score_p90": float(np.percentile(values, 90)),
        "score_p95": p95,
        "score_p99": float(np.percentile(values, 99)),
        "threshold": float(threshold),
    }


def nearest_window(windows: pd.DataFrame, video_id: str, timestamp: float) -> str:
    if windows.empty or "video_id" not in windows.columns:
        return ""
    subset = windows[windows["video_id"].map(clean).eq(video_id)]
    if subset.empty:
        return ""
    starts = subset["window_start_sec"].astype(float)
    ends = subset["window_end_sec"].astype(float)
    hit = subset[(starts <= timestamp) & (timestamp < ends)]
    if hit.empty:
        hit = subset[(starts <= timestamp) & (timestamp <= ends)]
    return clean(hit.iloc[0].get("window_id")) if not hit.empty else ""


def boundary_context(ad_segments: pd.DataFrame, video_id: str, timestamp: float) -> dict[str, Any]:
    subset = ad_segments[ad_segments["video_id"].map(clean).eq(video_id)] if not ad_segments.empty else pd.DataFrame()
    best: tuple[float, str, float, pd.Series] | None = None
    start_dists: list[float] = []
    end_dists: list[float] = []
    for _, row in subset.iterrows():
        start = to_float(row.get("ad_start_sec")) or to_float(row.get("segment_start_sec"))
        end = to_float(row.get("ad_end_sec")) or to_float(row.get("segment_end_sec"))
        for btype, sec in [("ad_start", start), ("ad_end", end)]:
            if sec is None:
                continue
            dist = abs(timestamp - sec)
            if btype == "ad_start":
                start_dists.append(dist)
            else:
                end_dists.append(dist)
            if best is None or dist < best[0]:
                best = (dist, btype, sec, row)
    if best is None:
        return {
            "nearest_ad_interval_id": "",
            "nearest_ad_boundary_type": "",
            "nearest_ad_boundary_sec": "",
            "nearest_ad_boundary_mmss": "",
            "distance_to_nearest_ad_boundary_sec": "",
            "is_near_ad_start_3s": "false",
            "is_near_ad_start_5s": "false",
            "is_near_ad_end_3s": "false",
            "is_near_ad_end_5s": "false",
            "is_near_any_ad_boundary_5s": "false",
        }
    min_start = min(start_dists) if start_dists else math.inf
    min_end = min(end_dists) if end_dists else math.inf
    return {
        "nearest_ad_interval_id": clean(best[3].get("ad_interval_id")),
        "nearest_ad_boundary_type": best[1],
        "nearest_ad_boundary_sec": best[2],
        "nearest_ad_boundary_mmss": mmss_floor(best[2]),
        "distance_to_nearest_ad_boundary_sec": best[0],
        "is_near_ad_start_3s": bool_text(min_start <= 3),
        "is_near_ad_start_5s": bool_text(min_start <= 5),
        "is_near_ad_end_3s": bool_text(min_end <= 3),
        "is_near_ad_end_5s": bool_text(min_end <= 5),
        "is_near_any_ad_boundary_5s": bool_text(min(min_start, min_end) <= 5),
    }


def dedup_candidates(candidates: list[dict[str, Any]], tolerance_sec: float = DEDUP_TOLERANCE_SEC) -> tuple[list[dict[str, Any]], int]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_video[clean(row.get("video_id"))].append(row)
    deduped: list[dict[str, Any]] = []
    removed = 0
    for video_id, rows in by_video.items():
        ranked = sorted(rows, key=lambda r: (-float(r.get("scene_change_score") or 0.0), float(r.get("candidate_time_sec") or 0.0)))
        kept: list[dict[str, Any]] = []
        for row in ranked:
            t = float(row.get("candidate_time_sec") or 0.0)
            if any(abs(t - float(prev.get("candidate_time_sec") or 0.0)) <= tolerance_sec for prev in kept):
                removed += 1
                continue
            kept.append(row)
        deduped.extend(sorted(kept, key=lambda r: float(r.get("candidate_time_sec") or 0.0)))
    return sorted(deduped, key=lambda r: (int(clean(r.get("video_id"))) if clean(r.get("video_id")).isdigit() else 999999, float(r.get("candidate_time_sec") or 0.0))), removed


def make_candidates(
    video: dict[str, Any],
    scores: list[dict[str, Any]],
    stats: dict[str, Any],
    windows: pd.DataFrame,
    ad_segments: pd.DataFrame,
) -> tuple[list[dict[str, Any]], int]:
    threshold = to_float(stats.get("threshold"))
    if threshold is None or not scores:
        return [], 0
    values = np.array([float(row["scene_change_score"]) for row in scores], dtype=np.float64)
    order = np.argsort(-values)
    ranks = {int(idx): int(rank) for rank, idx in enumerate(order, start=1)}
    candidates: list[dict[str, Any]] = []
    for idx, row in enumerate(scores):
        score = float(row["scene_change_score"])
        if score < threshold:
            continue
        timestamp = float(row["score_time_sec"])
        boundary = boundary_context(ad_segments, clean(video.get("video_id")), timestamp)
        candidates.append(
            {
                "video_id": clean(video.get("video_id")),
                "video_title": clean(video.get("video_title")),
                "video_filename": clean(video.get("video_filename")),
                "video_path": clean(video.get("video_path")),
                "candidate_time_sec": timestamp,
                "candidate_time_mmss": mmss_floor(timestamp),
                "candidate_time_mmss_floor": mmss_floor(timestamp),
                "candidate_time_mmss_round": mmss_round(timestamp),
                "scene_change_score": score,
                "threshold": threshold,
                "cosine_distance": float(row["cosine_distance"]),
                "l2_distance": float(row["l2_distance"]),
                "score_rank_in_video": ranks[idx],
                "score_percentile_in_video": float((values <= score).mean()),
                "candidate_source": "resnet_embedding_v2_3",
                "method_used": "resnet_embedding_1fps",
                "model_name": row.get("model_name", ""),
                "feature_dim": row.get("feature_dim", ""),
                "nearest_window_id": nearest_window(windows, clean(video.get("video_id")), timestamp),
                **boundary,
                "review_note": "",
            }
        )
    return dedup_candidates(candidates)


def count_between(candidates: pd.DataFrame, video_id: str, start: float, end: float) -> int:
    if candidates.empty or "candidate_time_sec" not in candidates.columns:
        return 0
    subset = candidates[candidates["video_id"].map(clean).eq(video_id)]
    if subset.empty:
        return 0
    times = subset["candidate_time_sec"].astype(float)
    return int(((times >= start) & (times <= end)).sum())


def nearest_candidate(candidates: pd.DataFrame, video_id: str, boundary_sec: float) -> tuple[str, str, str]:
    if candidates.empty or "candidate_time_sec" not in candidates.columns:
        return "", "", ""
    subset = candidates[candidates["video_id"].map(clean).eq(video_id)].copy()
    if subset.empty:
        return "", "", ""
    subset["_distance"] = (subset["candidate_time_sec"].astype(float) - boundary_sec).abs()
    row = subset.sort_values(["_distance", "scene_change_score"], ascending=[True, False]).iloc[0]
    sec = float(row["candidate_time_sec"])
    dist = float(row["_distance"])
    return str(fmt(sec)), mmss_floor(sec), str(fmt(dist))


def boundary_audit(candidates: list[dict[str, Any]], ad_segments: pd.DataFrame, all_segments: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cands = pd.DataFrame(candidates)
    rows: list[dict[str, Any]] = []
    sh3 = sh5 = eh3 = eh5 = both5 = 0
    if ad_segments.empty:
        return rows, {
            "total_ad_intervals": 0,
            "total_boundaries": 0,
            "start_hit_3s_count": 0,
            "start_hit_5s_count": 0,
            "end_hit_3s_count": 0,
            "end_hit_5s_count": 0,
            "boundary_hit_5s_count": 0,
            "boundary_hit_5s_rate": 0.0,
            "both_boundary_hit_5s_count": 0,
        }
    for _, ad in ad_segments.iterrows():
        video_id = clean(ad.get("video_id"))
        ad_interval_id = clean(ad.get("ad_interval_id"))
        start = to_float(ad.get("ad_start_sec")) or to_float(ad.get("segment_start_sec")) or 0.0
        end = to_float(ad.get("ad_end_sec")) or to_float(ad.get("segment_end_sec")) or 0.0
        subset = cands[cands["video_id"].map(clean).eq(video_id)].copy() if not cands.empty else pd.DataFrame()
        times = subset["candidate_time_sec"].astype(float) if not subset.empty else pd.Series(dtype=float)
        start3 = int(((times - start).abs() <= 3).sum())
        start5 = int(((times - start).abs() <= 5).sum())
        end3 = int(((times - end).abs() <= 3).sum())
        end5 = int(((times - end).abs() <= 5).sum())
        sh3 += int(start3 > 0)
        sh5 += int(start5 > 0)
        eh3 += int(end3 > 0)
        eh5 += int(end5 > 0)
        both5 += int(start5 > 0 and end5 > 0)
        ns_sec, ns_mmss, ns_dist = nearest_candidate(cands, video_id, start)
        ne_sec, ne_mmss, ne_dist = nearest_candidate(cands, video_id, end)
        context = all_segments[(all_segments["video_id"].map(clean).eq(video_id)) & (all_segments["ad_interval_id"].map(clean).eq(ad_interval_id))] if not all_segments.empty else pd.DataFrame()
        pre = context[context["segment_type"].map(clean).eq("pre_ad_start_10s")] if not context.empty else pd.DataFrame()
        post = context[context["segment_type"].map(clean).eq("post_ad_end_10s")] if not context.empty else pd.DataFrame()
        rows.append(
            {
                "ad_interval_id": ad_interval_id,
                "video_id": video_id,
                "video_title": clean(ad.get("video_title")),
                "ad_start_sec": start,
                "ad_start_mmss": mmss_floor(start),
                "ad_end_sec": end,
                "ad_end_mmss": mmss_floor(end),
                "start_hit_3s": bool_text(start3 > 0),
                "start_hit_5s": bool_text(start5 > 0),
                "end_hit_3s": bool_text(end3 > 0),
                "end_hit_5s": bool_text(end5 > 0),
                "both_boundary_hit_5s": bool_text(start5 > 0 and end5 > 0),
                "nearest_candidate_to_start_sec": ns_sec,
                "nearest_candidate_to_start_mmss": ns_mmss,
                "distance_to_start_candidate_sec": ns_dist,
                "nearest_candidate_to_end_sec": ne_sec,
                "nearest_candidate_to_end_mmss": ne_mmss,
                "distance_to_end_candidate_sec": ne_dist,
                "candidate_count_in_ad_interval": count_between(cands, video_id, start, end),
                "candidate_count_in_pre10": count_between(cands, video_id, float(pre.iloc[0]["segment_start_sec"]), float(pre.iloc[0]["segment_end_sec"])) if not pre.empty else 0,
                "candidate_count_in_post10": count_between(cands, video_id, float(post.iloc[0]["segment_start_sec"]), float(post.iloc[0]["segment_end_sec"])) if not post.empty else 0,
                "candidate_count_near_start_5s": start5,
                "candidate_count_near_end_5s": end5,
            }
        )
    total_boundaries = len(ad_segments) * 2
    summary = {
        "total_ad_intervals": int(len(ad_segments)),
        "total_boundaries": int(total_boundaries),
        "start_hit_3s_count": int(sh3),
        "start_hit_5s_count": int(sh5),
        "end_hit_3s_count": int(eh3),
        "end_hit_5s_count": int(eh5),
        "boundary_hit_5s_count": int(sh5 + eh5),
        "boundary_hit_5s_rate": float((sh5 + eh5) / total_boundaries) if total_boundaries else 0.0,
        "both_boundary_hit_5s_count": int(both5),
    }
    return rows, summary


def hit_boundary_set(audit: pd.DataFrame) -> set[str]:
    hits: set[str] = set()
    if audit.empty:
        return hits
    for _, row in audit.iterrows():
        ad_id = clean(row.get("ad_interval_id"))
        if is_true(row.get("start_hit_5s")):
            hits.add(f"{ad_id}:start")
        if is_true(row.get("end_hit_5s")):
            hits.add(f"{ad_id}:end")
    return hits


def comparison_row(method: str, candidates: pd.DataFrame, audit: pd.DataFrame, total_video_count: int, notes: str) -> dict[str, Any]:
    total_candidate_count = int(len(candidates))
    denominator = total_video_count if total_video_count else max(1, int(candidates["video_id"].nunique()) if not candidates.empty and "video_id" in candidates.columns else 1)
    start3 = int(audit["start_hit_3s"].map(is_true).sum()) if not audit.empty and "start_hit_3s" in audit.columns else 0
    start5 = int(audit["start_hit_5s"].map(is_true).sum()) if not audit.empty and "start_hit_5s" in audit.columns else 0
    end3 = int(audit["end_hit_3s"].map(is_true).sum()) if not audit.empty and "end_hit_3s" in audit.columns else 0
    end5 = int(audit["end_hit_5s"].map(is_true).sum()) if not audit.empty and "end_hit_5s" in audit.columns else 0
    both5 = int(audit["both_boundary_hit_5s"].map(is_true).sum()) if not audit.empty and "both_boundary_hit_5s" in audit.columns else 0
    total_ad_intervals = int(len(audit))
    total_boundaries = total_ad_intervals * 2
    boundary_hit = start5 + end5
    return {
        "method": method,
        "total_candidate_count": total_candidate_count,
        "candidate_count_per_video_mean": total_candidate_count / denominator if denominator else 0.0,
        "start_hit_3s_count": start3,
        "start_hit_5s_count": start5,
        "end_hit_3s_count": end3,
        "end_hit_5s_count": end5,
        "boundary_hit_5s_count": boundary_hit,
        "boundary_hit_5s_rate": boundary_hit / total_boundaries if total_boundaries else 0.0,
        "both_boundary_hit_5s_count": both5,
        "total_ad_intervals": total_ad_intervals,
        "total_boundaries": total_boundaries,
        "notes": notes,
    }


def candidate_times(df: pd.DataFrame, video_id: str) -> list[float]:
    if df.empty or "candidate_time_sec" not in df.columns:
        return []
    subset = df[df["video_id"].map(clean).eq(video_id)]
    return sorted(float(x) for x in subset["candidate_time_sec"].dropna().astype(float).tolist())


def count_overlap(source: list[float], target: list[float], tolerance: float) -> int:
    return sum(1 for value in source if any(abs(value - other) <= tolerance for other in target))


def count_only_near_boundary(candidates: pd.DataFrame, other: pd.DataFrame, video_id: str, tolerance: float = 5.0) -> int:
    if candidates.empty or "candidate_time_sec" not in candidates.columns:
        return 0
    subset = candidates[candidates["video_id"].map(clean).eq(video_id)]
    if subset.empty:
        return 0
    other_times = candidate_times(other, video_id)
    count = 0
    for _, row in subset.iterrows():
        t = float(row["candidate_time_sec"])
        near_boundary = is_true(row.get("is_near_any_ad_boundary_5s")) or (to_float(row.get("distance_to_nearest_ad_boundary_sec")) is not None and float(row.get("distance_to_nearest_ad_boundary_sec")) <= 5)
        overlaps_other = any(abs(t - other_time) <= tolerance for other_time in other_times)
        if near_boundary and not overlaps_other:
            count += 1
    return count


def candidate_overlap_rows(videos: list[dict[str, Any]], opencv_candidates: pd.DataFrame, resnet_candidates: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video in videos:
        video_id = clean(video.get("video_id"))
        ocv_times = candidate_times(opencv_candidates, video_id)
        res_times = candidate_times(resnet_candidates, video_id)
        rows.append(
            {
                "video_id": video_id,
                "video_title": clean(video.get("video_title")),
                "unique_opencv_candidate_count": len(ocv_times),
                "unique_resnet_candidate_count": len(res_times),
                "overlap_candidate_count_3s": count_overlap(res_times, ocv_times, 3.0),
                "overlap_candidate_count_5s": count_overlap(res_times, ocv_times, 5.0),
                "resnet_only_near_ad_boundary_count": count_only_near_boundary(resnet_candidates, opencv_candidates, video_id, 5.0),
                "opencv_only_near_ad_boundary_count": count_only_near_boundary(opencv_candidates, resnet_candidates, video_id, 5.0),
            }
        )
    return rows


def qa_embedding_score(
    videos: list[dict[str, Any]],
    decode_rows: list[dict[str, Any]],
    video_summaries: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    feature_dim: int,
    warnings: list[Any],
) -> dict[str, Any]:
    status = "PASS"
    details: list[Any] = []
    score_by_video = defaultdict(int)
    for row in score_rows:
        score_by_video[clean(row.get("video_id"))] += 1
    cand_by_video = defaultdict(int)
    for row in candidates:
        cand_by_video[clean(row.get("video_id"))] += 1
        duration = to_float(next((d.get("duration_sec") for d in decode_rows if clean(d.get("video_id")) == clean(row.get("video_id"))), "")) or math.inf
        t = to_float(row.get("candidate_time_sec")) or 0.0
        if t > duration:
            status = "FAIL"
            details.append({"candidate_time_exceeds_duration": row})
    for summary in video_summaries:
        decoded = int(summary.get("decoded_frame_count") or 0)
        embedding = int(summary.get("embedding_count") or 0)
        scores = int(summary.get("score_count") or 0)
        candidate_count = int(summary.get("candidate_count") or 0)
        if decoded <= 0 or embedding <= 0:
            status = "FAIL"
            details.append({"embedding_missing": clean(summary.get("video_id"))})
        if scores != max(0, embedding - 1):
            status = "FAIL"
            details.append({"score_count_mismatch": clean(summary.get("video_id")), "score_count": scores, "embedding_count": embedding})
        if int(summary.get("feature_dim") or 0) != feature_dim:
            status = "FAIL"
            details.append({"feature_dim_mismatch": clean(summary.get("video_id")), "observed": summary.get("feature_dim"), "expected": feature_dim})
        if scores > 0 and candidate_count == 0:
            if status == "PASS":
                status = "WARN"
            details.append({"candidate_count_zero": clean(summary.get("video_id"))})
        if scores > 0 and candidate_count > max(20, int(scores * 0.12)):
            if status == "PASS":
                status = "WARN"
            details.append({"candidate_count_high": clean(summary.get("video_id")), "candidate_count": candidate_count, "score_count": scores})
    if len(video_summaries) != len(videos):
        status = "FAIL"
        details.append({"video_summary_count_mismatch": [len(video_summaries), len(videos)]})
    if details:
        warnings.append({"sub_agent_2_embedding_score_details": details})
    return {"status": status, "details": details[:20]}


def qa_comparison_output(
    ad_segments: pd.DataFrame,
    resnet_audit: pd.DataFrame,
    comparison_rows: list[dict[str, Any]],
    latest_files: list[str],
    warnings: list[Any],
) -> dict[str, Any]:
    status = "PASS"
    details: list[Any] = []
    expected_intervals = len(ad_segments)
    if expected_intervals != 21:
        status = "WARN"
        details.append({"ad_interval_count_not_21": expected_intervals})
    if len(resnet_audit) != expected_intervals:
        status = "FAIL"
        details.append({"resnet_audit_row_count_mismatch": [len(resnet_audit), expected_intervals]})
    if not comparison_rows or len(comparison_rows) < 2:
        status = "FAIL"
        details.append("comparison_table_missing_or_incomplete")
    forbidden_suffixes = (".mp4", ".xlsx", ".pt", ".pth", ".ckpt", ".bin")
    forbidden_tokens = ("frame", "checkpoint", "cache", "model_weight")
    bad_latest = [
        path
        for path in latest_files
        if Path(path).suffix.lower() in forbidden_suffixes or any(token in Path(path).name.lower() for token in forbidden_tokens if token != "frame")
    ]
    # resnet_frame_decode_summary는 허용하고, 실제 frame dump로 보이는 파일만 제외한다.
    bad_latest.extend([path for path in latest_files if "frame_" in Path(path).name.lower() and Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    if bad_latest:
        status = "FAIL"
        details.append({"forbidden_latest_files": bad_latest})
    if details:
        warnings.append({"sub_agent_3_comparison_output_details": details})
    return {"status": status, "details": details[:20]}


def write_summary(report: dict[str, Any]) -> None:
    warning_text = "\n".join(f"- {warning}" for warning in report.get("warnings", [])) if report.get("warnings") else "- 주요 warning 없음"
    unique_resnet = report.get("unique_resnet_boundary_hits", 0)
    unique_opencv = report.get("unique_opencv_boundary_hits", 0)
    SUMMARY_PATH.write_text(
        f"""# extract_resnet_embedding_scene_change_v2_3 summary

## 작업 시간

- estimated runtime: {report.get('estimated_runtime')}
- start time: {report.get('start_time')}
- end time: {report.get('end_time')}
- actual runtime: {report.get('actual_runtime_readable')}

## ResNet Embedding

- pretrained ResNet available: {report.get('pretrained_resnet_available')}
- model_name: {report.get('model_name')}
- pretrained weight: {report.get('pretrained_weight_used')}
- weight source: {report.get('pretrained_weight_source')}
- device: {report.get('device_used')}
- feature_dim: {report.get('feature_dim')}
- video_count: {report.get('video_count')}
- decode_success_count: {report.get('decode_success_count')}
- embedding_success_count: {report.get('embedding_success_count')}
- score_count_total: {report.get('score_count_total')}
- resnet_candidate_count_total: {report.get('resnet_candidate_count_total')}

## 영상별 Candidate 수

`{report.get('resnet_candidate_count_by_video')}`

## Boundary Audit

- ResNet ±5s boundary hit: {report.get('resnet_boundary_hit_5s')} / rate {report.get('resnet_boundary_hit_5s_rate')}
- OpenCV/ffmpeg ±5s boundary hit: {report.get('opencv_boundary_hit_5s')} / rate {report.get('opencv_boundary_hit_5s_rate')}
- ResNet-only boundary hits: {unique_resnet}
- OpenCV-only boundary hits: {unique_opencv}

이 값은 scene-change 후보 audit용 수치이며 final performance claim이 아니다.

## OpenCV/ffmpeg 비교

- comparison CSV: `{COMPARISON_PATH}`
- overlap CSV: `{OVERLAP_PATH}`

## 다음 작업 추천

ResNet 후보는 OpenCV/ffmpeg 후보와 boundary hit 및 overlap을 함께 보고, 상호보완적인 후보만 rule-based detector의 visual evidence 후보로 연결하는 것을 권장한다.

## Warning

{warning_text}
""",
        encoding="utf-8",
    )


def update_readme() -> None:
    section = """## ResNet Embedding Scene-change v2.3

ResNet은 광고 분류 모델로 학습하지 않았고 fine-tuning도 수행하지 않았다. pretrained ResNet을 1fps frame embedding extractor로만 사용했으며, 인접 frame embedding 간 cosine distance를 scene-change score로 사용했다.

이 결과는 OpenCV/ffmpeg 기반 scene-change 후보와 비교하기 위한 boundary evidence audit이다. 광고 label은 ResNet 학습에 사용하지 않았고, boundary hit audit 수치는 final 광고 탐지 성능 claim이 아니다.
"""
    text = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else "# youtube_ad_segment_detector\n"
    marker = "## ResNet Embedding Scene-change v2.3"
    if marker in text:
        before, _, after = text.partition(marker)
        next_idx = after.find("\n## ")
        text = before.rstrip() + "\n\n" + section + (after[next_idx:] if next_idx >= 0 else "")
    else:
        text = text.rstrip() + "\n\n" + section
    README_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def latest_copy(files: list[Path]) -> tuple[bool, list[str]]:
    ensure_inside_project(LATEST_DIR)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    forbidden_suffixes = {".mp4", ".xlsx", ".jpg", ".jpeg", ".png", ".webp", ".pt", ".pth", ".ckpt", ".bin"}
    for src in files:
        if not src.exists():
            continue
        if src.suffix.lower() in forbidden_suffixes:
            raise RuntimeError(f"Refusing to copy forbidden latest file: {src}")
        dst = LATEST_DIR / src.name
        ensure_inside_project(dst)
        shutil.copy2(src, dst)
        copied.append(str(dst))
    LATEST_README.write_text(
        "# latest_for_chatgpt files\n\n"
        "이번 작업명: extract_resnet_embedding_scene_change_v2_3\n\n"
        "ResNet embedding scene-change 후보, score, audit, comparison, report, summary, log, script만 복사했다. mp4, 원본 xlsx, frame image, model weight, checkpoint, cache 파일은 복사하지 않는다.\n",
        encoding="utf-8",
    )
    copied.append(str(LATEST_README))
    return True, copied


def write_report(report: dict[str, Any]) -> None:
    ensure_inside_project(REPORT_PATH)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(report)
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    start_dt = datetime.now().astimezone()
    start_time = start_dt.isoformat(timespec="seconds")
    started = time.time()
    warnings: list[Any] = []
    errors: list[Any] = []
    log(f"작업 시작 전 예상 작업 시간: {ESTIMATED_RUNTIME}")
    log(f"예상 근거: {RUNTIME_ESTIMATION_REASON}")
    log(f"작업 시작 시각: {start_time}")

    report: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "estimated_runtime": ESTIMATED_RUNTIME,
        "runtime_estimation_reason": RUNTIME_ESTIMATION_REASON,
        "start_time": start_time,
        "end_time": "",
        "actual_runtime_seconds": 0,
        "actual_runtime_readable": "",
        "cv_environment_checked": False,
        "python_executable": sys.executable,
        "torch_available": False,
        "torchvision_available": False,
        "cuda_available": False,
        "device_used": "",
        "ffmpeg_available": False,
        "ffprobe_available": False,
        "pretrained_resnet_available": False,
        "pretrained_resnet_unavailable": False,
        "model_name": "",
        "pretrained_weight_used": "",
        "pretrained_weight_source": "",
        "feature_dim": 0,
        "video_count": 0,
        "decode_success_count": 0,
        "embedding_success_count": 0,
        "score_count_total": 0,
        "resnet_candidate_count_total": 0,
        "opencv_candidate_count_total": 0,
        "resnet_boundary_hit_5s": 0,
        "opencv_boundary_hit_5s": 0,
        "resnet_boundary_hit_5s_rate": 0.0,
        "opencv_boundary_hit_5s_rate": 0.0,
        "unique_resnet_boundary_hits": 0,
        "unique_opencv_boundary_hits": 0,
        "generated_files": [],
        "sub_agent_results": {},
        "warnings": warnings,
        "errors": errors,
        "old_project_modified": False,
    }

    try:
        step(1, 12, "cv/torch/torchvision/ffmpeg 환경 확인")
        cv_ok, python_executable = verify_cv_environment(warnings)
        report["cv_environment_checked"] = cv_ok
        report["python_executable"] = python_executable
        torch_ok, torch_details, modules = import_torch_stack()
        report.update(torch_details)
        ffmpeg_ok, ffmpeg_version, ffmpeg_exe = tool_version("ffmpeg")
        ffprobe_ok, ffprobe_version, ffprobe_exe = tool_version("ffprobe")
        report.update(
            {
                "ffmpeg_available": ffmpeg_ok,
                "ffprobe_available": ffprobe_ok,
                "ffmpeg_version": ffmpeg_version,
                "ffprobe_version": ffprobe_version,
                "ffmpeg_executable": ffmpeg_exe,
                "ffprobe_executable": ffprobe_exe,
            }
        )
        if not (cv_ok and torch_ok and ffmpeg_ok and ffprobe_ok):
            errors.append("required_environment_unavailable")
            raise RuntimeError("Required cv/torch/ffmpeg environment is unavailable")

        step(2, 12, "pretrained ResNet weight cache 확인")
        model_info = load_pretrained_resnet(modules, warnings)
        report.update(
            {
                "pretrained_resnet_available": bool(model_info.get("available")),
                "pretrained_resnet_unavailable": not bool(model_info.get("available")),
                "model_name": model_info.get("model_name", ""),
                "pretrained_weight_used": model_info.get("pretrained_weight_used", ""),
                "pretrained_weight_source": model_info.get("pretrained_weight_source", ""),
                "feature_dim": model_info.get("feature_dim", 0),
                "pretrained_weight_cache_status": model_info.get("cache_status", []),
            }
        )
        if not model_info.get("available"):
            errors.append("pretrained_resnet_unavailable")
            raise RuntimeError("No cached pretrained ResNet weight is available; random initialized ResNet is forbidden")

        torch_module = modules["torch"]
        device = torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
        batch_size = 32 if str(device).startswith("cuda") else 16
        feature_extractor = model_info["feature_extractor"].to(device)
        feature_extractor.eval()
        report["device_used"] = str(device)
        report["batch_size"] = batch_size

        step(3, 12, "입력 CSV 로드")
        manifest = read_csv(MANIFEST_PATH, True, warnings, errors)
        ad_segments = read_csv(AD_SEGMENTS_PATH, True, warnings, errors)
        all_segments = read_csv(ALL_SEGMENTS_PATH, False, warnings, errors)
        windows = read_csv(WINDOWS_PATH, False, warnings, errors)
        opencv_candidates = read_csv(OPENCV_CANDIDATES_PATH, False, warnings, errors)
        opencv_audit = read_csv(OPENCV_AUDIT_PATH, False, warnings, errors)
        if manifest.empty:
            errors.append("video_manifest_missing_or_empty")
            raise RuntimeError("video_manifest is required")
        videos = load_videos(manifest, warnings)
        report["video_count"] = len(videos)
        report["ad_interval_count"] = int(len(ad_segments))
        if ad_segments.empty:
            warnings.append("ad_interval_segments_missing_or_empty_boundary_audit_warn")

        step(4, 12, "ffprobe metadata 확인")
        probes = [ffprobe_metadata(video, ffprobe_exe) for video in videos]

        step(5, 12, "ffmpeg 1fps frame decode 및 ResNet embedding 추출")
        decode_rows: list[dict[str, Any]] = []
        all_scores: list[dict[str, Any]] = []
        all_candidates: list[dict[str, Any]] = []
        video_summaries: list[dict[str, Any]] = []
        duplicate_removed_total = 0
        for idx, (video, probe) in enumerate(zip(videos, probes), start=1):
            log(f"video {idx}/{len(videos)}: {video['video_filename']}")
            embeddings, meta = extract_video_embeddings(
                video,
                probe,
                ffmpeg_exe,
                model_info["feature_extractor"],
                model_info["preprocess"],
                torch_module,
                device,
                batch_size,
                model_info["model_name"],
                warnings,
            )
            scores = score_rows_for_video(video, embeddings, meta, model_info)
            stats = score_stats(scores)
            candidates, duplicate_removed = make_candidates(video, scores, stats, windows, ad_segments)
            duplicate_removed_total += duplicate_removed
            all_scores.extend(scores)
            all_candidates.extend(candidates)
            decode_rows.append(
                {
                    "video_id": clean(video.get("video_id")),
                    "video_filename": clean(video.get("video_filename")),
                    "video_path": clean(video.get("video_path")),
                    "duration_sec": meta.get("duration_sec", ""),
                    "decoded_frame_count": meta.get("decoded_frame_count", 0),
                    "expected_1fps_frame_count": meta.get("expected_1fps_frame_count", 0),
                    "decode_success": bool_text(bool(meta.get("decode_success"))),
                    "decode_warning": meta.get("decode_warning", ""),
                }
            )
            warning_parts = [clean(meta.get("decode_warning"))]
            if int(meta.get("decoded_frame_count") or 0) <= 0:
                warning_parts.append("decoded_frame_count_zero")
            if int(meta.get("embedding_count") or 0) <= 0:
                warning_parts.append("embedding_count_zero")
            if scores and not candidates:
                warning_parts.append("score_count_positive_candidate_count_zero")
            if len(candidates) > max(20, int(max(1, len(scores)) * 0.12)):
                warning_parts.append("candidate_count_high")
            video_summaries.append(
                {
                    "video_id": clean(video.get("video_id")),
                    "video_title": clean(video.get("video_title")),
                    "video_filename": clean(video.get("video_filename")),
                    "duration_sec": meta.get("duration_sec", ""),
                    "decoded_frame_count": meta.get("decoded_frame_count", 0),
                    "embedding_count": meta.get("embedding_count", 0),
                    "score_count": len(scores),
                    **stats,
                    "threshold": stats.get("threshold", ""),
                    "candidate_count": len(candidates),
                    "candidate_count_near_ad_start_3s": sum(1 for row in candidates if is_true(row.get("is_near_ad_start_3s"))),
                    "candidate_count_near_ad_start_5s": sum(1 for row in candidates if is_true(row.get("is_near_ad_start_5s"))),
                    "candidate_count_near_ad_end_3s": sum(1 for row in candidates if is_true(row.get("is_near_ad_end_3s"))),
                    "candidate_count_near_ad_end_5s": sum(1 for row in candidates if is_true(row.get("is_near_ad_end_5s"))),
                    "decode_success": bool_text(bool(meta.get("decode_success"))),
                    "embedding_success": bool_text(bool(meta.get("embedding_success"))),
                    "feature_dim": meta.get("feature_dim_observed") or model_info.get("feature_dim"),
                    "warning": "; ".join(part for part in warning_parts if part),
                }
            )
            log(
                "decoded={decoded} embeddings={embeddings} scores={scores} candidates={candidates} elapsed={elapsed}".format(
                    decoded=meta.get("decoded_frame_count", 0),
                    embeddings=meta.get("embedding_count", 0),
                    scores=len(scores),
                    candidates=len(candidates),
                    elapsed=readable(float(meta.get("embedding_elapsed_sec") or 0.0)),
                )
            )

        step(6, 12, "ResNet score/candidate/mm:ss/video summary CSV 생성")
        score_cols = [
            "video_id",
            "video_title",
            "video_filename",
            "video_path",
            "prev_sample_time_sec",
            "next_sample_time_sec",
            "score_time_sec",
            "score_time_mmss",
            "cosine_distance",
            "l2_distance",
            "scene_change_score",
            "feature_dim",
            "model_name",
            "method_used",
            "candidate_source",
        ]
        candidate_cols = [
            "video_id",
            "video_title",
            "video_filename",
            "video_path",
            "candidate_time_sec",
            "candidate_time_mmss",
            "candidate_time_mmss_floor",
            "candidate_time_mmss_round",
            "scene_change_score",
            "threshold",
            "cosine_distance",
            "l2_distance",
            "score_rank_in_video",
            "score_percentile_in_video",
            "candidate_source",
            "method_used",
            "model_name",
            "feature_dim",
            "nearest_window_id",
            "nearest_ad_interval_id",
            "nearest_ad_boundary_type",
            "nearest_ad_boundary_sec",
            "nearest_ad_boundary_mmss",
            "distance_to_nearest_ad_boundary_sec",
            "is_near_ad_start_3s",
            "is_near_ad_start_5s",
            "is_near_ad_end_3s",
            "is_near_ad_end_5s",
            "is_near_any_ad_boundary_5s",
            "review_note",
        ]
        summary_cols = [
            "video_id",
            "video_title",
            "video_filename",
            "duration_sec",
            "decoded_frame_count",
            "embedding_count",
            "score_count",
            "score_mean",
            "score_std",
            "score_p50",
            "score_p90",
            "score_p95",
            "score_p99",
            "threshold",
            "candidate_count",
            "candidate_count_near_ad_start_3s",
            "candidate_count_near_ad_start_5s",
            "candidate_count_near_ad_end_3s",
            "candidate_count_near_ad_end_5s",
            "decode_success",
            "embedding_success",
            "warning",
        ]
        decode_cols = ["video_id", "video_filename", "video_path", "duration_sec", "decoded_frame_count", "expected_1fps_frame_count", "decode_success", "decode_warning"]
        write_csv(FRAME_DECODE_SUMMARY_PATH, decode_rows, decode_cols)
        write_csv(SCORES_PATH, all_scores, score_cols)
        write_csv(CANDIDATES_PATH, all_candidates, candidate_cols)
        write_csv(CANDIDATES_MMSS_PATH, all_candidates, candidate_cols)
        write_csv(VIDEO_SUMMARY_PATH, video_summaries, summary_cols)

        step(7, 12, "광고 boundary audit 생성")
        audit_rows, audit_summary = boundary_audit(all_candidates, ad_segments, all_segments)
        audit_cols = [
            "ad_interval_id",
            "video_id",
            "video_title",
            "ad_start_sec",
            "ad_start_mmss",
            "ad_end_sec",
            "ad_end_mmss",
            "start_hit_3s",
            "start_hit_5s",
            "end_hit_3s",
            "end_hit_5s",
            "both_boundary_hit_5s",
            "nearest_candidate_to_start_sec",
            "nearest_candidate_to_start_mmss",
            "distance_to_start_candidate_sec",
            "nearest_candidate_to_end_sec",
            "nearest_candidate_to_end_mmss",
            "distance_to_end_candidate_sec",
            "candidate_count_in_ad_interval",
            "candidate_count_in_pre10",
            "candidate_count_in_post10",
            "candidate_count_near_start_5s",
            "candidate_count_near_end_5s",
        ]
        write_csv(BOUNDARY_AUDIT_PATH, audit_rows, audit_cols)
        resnet_audit = pd.DataFrame(audit_rows)

        step(8, 12, "OpenCV/ffmpeg 방식과 ResNet 비교표 생성")
        resnet_candidates_df = pd.DataFrame(all_candidates)
        comparison_rows = [
            comparison_row("opencv_ffmpeg_merged_v2_3", opencv_candidates, opencv_audit, len(videos), "OpenCV v2.3 candidates plus ffmpeg fallback; scene boundary evidence audit only."),
            comparison_row("resnet_embedding_v2_3", resnet_candidates_df, resnet_audit, len(videos), "Pretrained ResNet embedding cosine distance; no ad classifier training or fine-tuning."),
        ]
        comparison_cols = [
            "method",
            "total_candidate_count",
            "candidate_count_per_video_mean",
            "start_hit_3s_count",
            "start_hit_5s_count",
            "end_hit_3s_count",
            "end_hit_5s_count",
            "boundary_hit_5s_count",
            "boundary_hit_5s_rate",
            "both_boundary_hit_5s_count",
            "total_ad_intervals",
            "total_boundaries",
            "notes",
        ]
        write_csv(COMPARISON_PATH, comparison_rows, comparison_cols)
        overlap_rows = candidate_overlap_rows(videos, opencv_candidates, resnet_candidates_df)
        overlap_cols = [
            "video_id",
            "video_title",
            "unique_opencv_candidate_count",
            "unique_resnet_candidate_count",
            "overlap_candidate_count_3s",
            "overlap_candidate_count_5s",
            "resnet_only_near_ad_boundary_count",
            "opencv_only_near_ad_boundary_count",
        ]
        write_csv(OVERLAP_PATH, overlap_rows, overlap_cols)

        step(9, 12, "Sub Agent QA status 계산")
        resnet_hits = hit_boundary_set(resnet_audit)
        opencv_hits = hit_boundary_set(opencv_audit)
        unique_resnet_hits = sorted(resnet_hits - opencv_hits)
        unique_opencv_hits = sorted(opencv_hits - resnet_hits)
        qa2 = qa_embedding_score(videos, decode_rows, video_summaries, all_scores, all_candidates, int(model_info.get("feature_dim") or 0), warnings)

        generated_files = [
            FRAME_DECODE_SUMMARY_PATH,
            SCORES_PATH,
            CANDIDATES_PATH,
            CANDIDATES_MMSS_PATH,
            VIDEO_SUMMARY_PATH,
            BOUNDARY_AUDIT_PATH,
            COMPARISON_PATH,
            OVERLAP_PATH,
            REPORT_PATH,
            SUMMARY_PATH,
            LOG_PATH,
            SCRIPT_PATH,
        ]
        step(10, 12, "README 및 latest_for_chatgpt 갱신")
        update_readme()
        latest_ok, latest_files = latest_copy(generated_files)
        qa3 = qa_comparison_output(ad_segments, resnet_audit, comparison_rows, latest_files, warnings)

        end_time = datetime.now().astimezone().isoformat(timespec="seconds")
        elapsed = time.time() - started
        report.update(
            {
                "end_time": end_time,
                "actual_runtime_seconds": elapsed,
                "actual_runtime_readable": readable(elapsed),
                "decode_success_count": sum(1 for row in decode_rows if is_true(row.get("decode_success"))),
                "embedding_success_count": sum(1 for row in video_summaries if is_true(row.get("embedding_success"))),
                "score_count_total": len(all_scores),
                "resnet_candidate_count_total": len(all_candidates),
                "opencv_candidate_count_total": int(len(opencv_candidates)),
                "resnet_candidate_count_by_video": {clean(row.get("video_id")): int(row.get("candidate_count") or 0) for row in video_summaries},
                "duplicate_removed_total": duplicate_removed_total,
                "resnet_boundary_hit_5s": int(audit_summary.get("boundary_hit_5s_count", 0)),
                "opencv_boundary_hit_5s": int(comparison_rows[0]["boundary_hit_5s_count"]),
                "resnet_boundary_hit_5s_rate": float(audit_summary.get("boundary_hit_5s_rate", 0.0)),
                "opencv_boundary_hit_5s_rate": float(comparison_rows[0]["boundary_hit_5s_rate"]),
                "unique_resnet_boundary_hits": len(unique_resnet_hits),
                "unique_opencv_boundary_hits": len(unique_opencv_hits),
                "unique_resnet_boundary_hit_ids": unique_resnet_hits,
                "unique_opencv_boundary_hit_ids": unique_opencv_hits,
                "total_ad_intervals": int(audit_summary.get("total_ad_intervals", 0)),
                "total_boundaries": int(audit_summary.get("total_boundaries", 0)),
                "generated_files": [str(path) for path in generated_files],
                "latest_for_chatgpt_updated": latest_ok,
                "latest_for_chatgpt_files": latest_files,
                "sub_agent_results": {
                    "sub_agent_1_model_environment": {
                        "status": "PASS" if report["pretrained_resnet_available"] and report["ffmpeg_available"] and report["torch_available"] else "FAIL",
                        "details": {
                            "torch_version": report.get("torch_version"),
                            "torchvision_version": report.get("torchvision_version"),
                            "cuda_available": report.get("cuda_available"),
                            "device_used": report.get("device_used"),
                            "model_name": report.get("model_name"),
                            "pretrained_weight_source": report.get("pretrained_weight_source"),
                        },
                    },
                    "sub_agent_2_embedding_score": qa2,
                    "sub_agent_3_comparison_output": qa3,
                },
                "warnings": warnings,
                "errors": errors,
                "old_project_modified": False,
            }
        )
        step(11, 12, "report/summary/log 생성")
        write_report(report)
        step(12, 12, "완료")
        log(f"작업 종료 시각: {end_time}")
        log(f"실제 작업 시간: {readable(elapsed)}")
        write_report(report)
        print(
            json.dumps(
                {
                    "model_name": report["model_name"],
                    "device_used": report["device_used"],
                    "video_count": report["video_count"],
                    "resnet_candidate_count_total": report["resnet_candidate_count_total"],
                    "resnet_boundary_hit_5s": report["resnet_boundary_hit_5s"],
                    "opencv_boundary_hit_5s": report["opencv_boundary_hit_5s"],
                    "actual_runtime_readable": report["actual_runtime_readable"],
                    "errors": errors,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as exc:
        errors.append({"fatal_error": str(exc)})
        end_time = datetime.now().astimezone().isoformat(timespec="seconds")
        elapsed = time.time() - started
        report.update(
            {
                "end_time": end_time,
                "actual_runtime_seconds": elapsed,
                "actual_runtime_readable": readable(elapsed),
                "warnings": warnings,
                "errors": errors,
                "old_project_modified": False,
            }
        )
        write_report(report)
        log(f"작업 종료 시각: {end_time}")
        log(f"실제 작업 시간: {readable(elapsed)}")
        raise


if __name__ == "__main__":
    main()
