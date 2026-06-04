from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
PRED_PATH = ROOT / "notebooks/rule_lab_v2_3_modular/lastest_csv/05_manual_predictions.csv"
LABEL_PATH = ROOT / "data/labels/clean_ad_labels_v0_scope_refreshed_v2_4.csv"
VIDEO_META_PATH = ROOT / "data/video_metadata/video_manifest_v2_2.csv"
OUT_DIR = ROOT / "outputs/demo/development_ad_interval_timeline_images"
REPORT_PATH = ROOT / "reports/demo/development_ad_interval_timeline_images_report.json"
LOG_PATH = ROOT / "logs/development_ad_interval_timeline_images_run_log.txt"


def clean_interval_rows(df: pd.DataFrame) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for _, row in df.iterrows():
        try:
            start = float(row["ad_start_sec"])
            end = float(row["ad_end_sec"])
        except Exception:
            continue
        if math.isnan(start) or math.isnan(end) or end <= start:
            continue
        rows.append({"start_sec": start, "end_sec": end})
    rows.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    return rows


def fmt_time(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}:{secs:02d}"


def duration_for_video(video_id: int, pred_rows: pd.DataFrame, video_meta: pd.DataFrame, labels: pd.DataFrame) -> float:
    meta_rows = video_meta[video_meta["video_id"] == video_id]
    if len(meta_rows) and "duration_sec" in meta_rows.columns:
        value = float(meta_rows.iloc[0]["duration_sec"])
        if not math.isnan(value) and value > 0:
            return value

    if "video_duration_sec" in pred_rows.columns and pred_rows["video_duration_sec"].notna().any():
        value = float(pred_rows["video_duration_sec"].dropna().iloc[0])
        if not math.isnan(value) and value > 0:
            return value

    max_end = 0.0
    for frame in [pred_rows, labels[labels["video_id"] == video_id]]:
        if "ad_end_sec" in frame.columns and frame["ad_end_sec"].notna().any():
            max_end = max(max_end, float(frame["ad_end_sec"].max()))
    return max(max_end, 1.0)


def draw_timeline(
    video_id: int,
    duration: float,
    pred_intervals: list[dict[str, float]],
    actual_intervals: list[dict[str, float]],
    out_path: Path,
    font_prop: font_manager.FontProperties,
) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 1.35), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bg_y = 0.30
    bg_h = 0.43
    ax.add_patch(
        FancyBboxPatch(
            (0, bg_y),
            duration,
            bg_h,
            boxstyle=f"round,pad=0.004,rounding_size={max(duration * 0.004, 3.0)}",
            linewidth=0.8,
            edgecolor="#cbd5e1",
            facecolor="#eef2f7",
            mutation_aspect=0.02,
            zorder=1,
        )
    )

    for i in range(1, 10):
        x = duration * i / 10
        ax.plot([x, x], [bg_y, bg_y + bg_h], color="#dbe3ee", linewidth=0.65, zorder=2)

    pred_color = "#1f7ae0"
    actual_color = "#e34242"

    for interval in pred_intervals:
        start = max(0.0, min(duration, interval["start_sec"]))
        end = max(0.0, min(duration, interval["end_sec"]))
        if end > start:
            ax.add_patch(
                FancyBboxPatch(
                    (start, 0.48),
                    end - start,
                    0.18,
                    boxstyle=f"round,pad=0.002,rounding_size={max(duration * 0.0025, 2.0)}",
                    linewidth=0,
                    facecolor=pred_color,
                    zorder=4,
                    mutation_aspect=0.02,
                )
            )

    for interval in actual_intervals:
        start = max(0.0, min(duration, interval["start_sec"]))
        end = max(0.0, min(duration, interval["end_sec"]))
        if end > start:
            ax.add_patch(
                FancyBboxPatch(
                    (start, 0.37),
                    end - start,
                    0.065,
                    boxstyle=f"round,pad=0.002,rounding_size={max(duration * 0.002, 1.8)}",
                    linewidth=0,
                    facecolor=actual_color,
                    zorder=5,
                    mutation_aspect=0.02,
                )
            )

    ax.text(0, 0.86, "00:00", ha="left", va="center", fontsize=10.5, color="#27476f", fontproperties=font_prop)
    ax.text(duration, 0.86, fmt_time(duration), ha="right", va="center", fontsize=10.5, color="#4a2e20", fontproperties=font_prop)

    ax.add_patch(Rectangle((0.000, 0.085), 0.014, 0.055, transform=ax.transAxes, color=pred_color, clip_on=False, zorder=8))
    ax.text(0.019, 0.112, "예측 광고 구간", transform=ax.transAxes, ha="left", va="center", fontsize=9.2, color="#33506f", fontproperties=font_prop)
    ax.add_patch(Rectangle((0.100, 0.085), 0.014, 0.055, transform=ax.transAxes, color=actual_color, clip_on=False, zorder=8))
    ax.text(0.119, 0.112, "실제 광고 구간", transform=ax.transAxes, ha="left", va="center", fontsize=9.2, color="#33506f", fontproperties=font_prop)

    fig.subplots_adjust(left=0.004, right=0.996, top=0.98, bottom=0.02)
    fig.savefig(out_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(PRED_PATH)
    labels = pd.read_csv(LABEL_PATH)
    video_meta = pd.read_csv(VIDEO_META_PATH)

    if "video_id" not in pred.columns:
        raise RuntimeError(f"prediction CSV missing video_id: {PRED_PATH}")

    missing_pred_cols = sorted({"ad_start_sec", "ad_end_sec"} - set(pred.columns))
    if missing_pred_cols:
        raise RuntimeError(f"prediction CSV missing columns: {missing_pred_cols}")

    if "label_valid" in labels.columns:
        label_valid = labels["label_valid"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        labels = labels[label_valid].copy()

    noto_cjk_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_path = str(noto_cjk_path) if noto_cjk_path.exists() else font_manager.findfont("Noto Sans CJK KR", fallback_to_default=True)
    font_prop = font_manager.FontProperties(fname=font_path)

    pred["video_id"] = pred["video_id"].astype(int)
    labels["video_id"] = labels["video_id"].astype(int)
    video_meta["video_id"] = video_meta["video_id"].astype(int)

    dev_video_ids = sorted(pred["video_id"].dropna().astype(int).unique().tolist())
    image_records = []

    for video_id in dev_video_ids:
        pred_rows = pred[pred["video_id"] == video_id].copy()
        actual_rows = labels[labels["video_id"] == video_id].copy()
        duration = duration_for_video(video_id, pred_rows, video_meta, labels)
        pred_intervals = clean_interval_rows(pred_rows)
        actual_intervals = clean_interval_rows(actual_rows)
        out_path = OUT_DIR / f"development_video_{video_id:02d}_ad_timeline.png"
        draw_timeline(video_id, duration, pred_intervals, actual_intervals, out_path, font_prop)
        image_records.append(
            {
                "video_id": video_id,
                "image_path": str(out_path),
                "duration_sec": duration,
                "duration_label": fmt_time(duration),
                "predicted_interval_count": len(pred_intervals),
                "actual_interval_count": len(actual_intervals),
                "predicted_intervals": pred_intervals,
                "actual_intervals": actual_intervals,
            }
        )

    report = {
        "task": "development_ad_interval_timeline_image_export",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "viewer_modified": False,
        "prediction_source": str(PRED_PATH),
        "actual_interval_source": str(LABEL_PATH),
        "duration_source": str(VIDEO_META_PATH),
        "output_dir": str(OUT_DIR),
        "video_ids": dev_video_ids,
        "image_count": len(image_records),
        "images": image_records,
        "style": {
            "predicted_ad_interval_label": "예측 광고 구간",
            "actual_ad_interval_label": "실제 광고 구간",
            "predicted_color": "#1f7ae0",
            "actual_color": "#e34242",
            "image_size_px": [1120, 135],
        },
        "warnings": [],
        "errors": [],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG_PATH.write_text(
        "\n".join(
            [
                "[development_ad_interval_timeline_image_export]",
                f"created_at={report['created_at']}",
                f"prediction_source={PRED_PATH}",
                f"actual_interval_source={LABEL_PATH}",
                f"duration_source={VIDEO_META_PATH}",
                f"output_dir={OUT_DIR}",
                f"video_ids={dev_video_ids}",
                f"image_count={len(image_records)}",
                "viewer_modified=false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(OUT_DIR),
                "report_path": str(REPORT_PATH),
                "log_path": str(LOG_PATH),
                "video_ids": dev_video_ids,
                "image_count": len(image_records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
