from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRED_PATH = ROOT / "notebooks/rule_lab_v2_3_modular/lastest_csv/05_manual_predictions.csv"
TEST_PRED_PATH = ROOT / "notebooks/rule_lab_v2_3_modular/lastest_csv_test_set/05_manual_predictions.csv"
LABEL_PATH = ROOT / "data/labels/clean_ad_labels_v0_scope_refreshed_v2_4.csv"
VIDEO_META_PATH = ROOT / "data/video_metadata/video_manifest_v2_2.csv"

OUT_CSV = ROOT / "outputs/demo/development_set_metrics_by_video.csv"
TEST_OUT_CSV = ROOT / "outputs/demo/test_set_metrics_by_video.csv"
REPORT_JSON = ROOT / "reports/demo/development_set_metrics_report.json"
SUMMARY_MD = ROOT / "reports/demo/development_set_metrics_summary.md"
RUN_LOG = ROOT / "logs/development_set_metrics_run_log.txt"


def clean_intervals(df: pd.DataFrame) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for _, row in df.iterrows():
        try:
            start = float(row["ad_start_sec"])
            end = float(row["ad_end_sec"])
        except Exception:
            continue
        if math.isnan(start) or math.isnan(end) or end <= start:
            continue
        intervals.append((start, end))
    return sorted(intervals)


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return merged


def interval_total(intervals: list[tuple[float, float]]) -> float:
    return sum(max(0.0, end - start) for start, end in intervals)


def overlap_total(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    i = 0
    j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def boundary_errors(actual: list[tuple[float, float]], predicted: list[tuple[float, float]]) -> tuple[float | None, float | None, int]:
    if not actual or not predicted:
        return None, None, 0

    start_errors: list[float] = []
    end_errors: list[float] = []
    for actual_start, actual_end in actual:
        best_pred = None
        best_overlap = -1.0
        best_distance = float("inf")
        actual_center = (actual_start + actual_end) / 2.0

        for pred_start, pred_end in predicted:
            overlap = max(0.0, min(actual_end, pred_end) - max(actual_start, pred_start))
            pred_center = (pred_start + pred_end) / 2.0
            distance = abs(actual_center - pred_center)
            if overlap > best_overlap or (math.isclose(overlap, best_overlap) and distance < best_distance):
                best_overlap = overlap
                best_distance = distance
                best_pred = (pred_start, pred_end)

        if best_pred is None:
            continue
        start_errors.append(abs(best_pred[0] - actual_start))
        end_errors.append(abs(best_pred[1] - actual_end))

    if not start_errors:
        return None, None, 0
    return sum(start_errors) / len(start_errors), sum(end_errors) / len(end_errors), len(start_errors)


def pct(value: float | None) -> str:
    return "계산 불가" if value is None else f"{value:.1f}%"


def sec(value: float | None) -> str:
    return "계산 불가" if value is None else f"{value:.1f}초"


def mmss(value: float | None) -> str:
    if value is None:
        return "계산 불가"
    total = max(0, int(round(value)))
    return f"{total // 60}:{total % 60:02d} ({value:.1f}초)"


def dense_rank(series: pd.Series, ascending: bool) -> pd.Series:
    return series.rank(method="min", ascending=ascending, na_option="bottom")


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    TEST_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(PRED_PATH)
    labels = pd.read_csv(LABEL_PATH)
    video_meta = pd.read_csv(VIDEO_META_PATH)

    if "label_valid" in labels.columns:
        labels = labels[labels["label_valid"].astype(str).str.lower().isin(["true", "1", "yes", "y"])].copy()

    pred["video_id"] = pred["video_id"].astype(int)
    labels["video_id"] = labels["video_id"].astype(int)
    video_meta["video_id"] = video_meta["video_id"].astype(int)

    dev_video_ids = sorted(pred["video_id"].dropna().astype(int).unique().tolist())
    rows = []
    overall_pred_duration = 0.0
    overall_actual_duration = 0.0
    overall_overlap = 0.0
    overall_fp = 0.0
    overall_missed = 0.0
    false_positive_by_video: list[float] = []
    all_start_errors: list[float] = []
    all_end_errors: list[float] = []

    for video_id in dev_video_ids:
        pred_intervals = merge_intervals(clean_intervals(pred[pred["video_id"] == video_id]))
        actual_intervals = merge_intervals(clean_intervals(labels[labels["video_id"] == video_id]))

        pred_duration = interval_total(pred_intervals)
        actual_duration = interval_total(actual_intervals)
        overlap = overlap_total(pred_intervals, actual_intervals)
        false_positive = max(0.0, pred_duration - overlap)
        missed = max(0.0, actual_duration - overlap)
        capture_rate = None if actual_duration <= 0 else overlap / actual_duration * 100.0
        precision = None if pred_duration <= 0 else overlap / pred_duration * 100.0
        start_error, end_error, matched_count = boundary_errors(actual_intervals, pred_intervals)

        overall_pred_duration += pred_duration
        overall_actual_duration += actual_duration
        overall_overlap += overlap
        overall_fp += false_positive
        overall_missed += missed
        false_positive_by_video.append(false_positive)
        if start_error is not None:
            all_start_errors.extend([start_error] * matched_count)
        if end_error is not None:
            all_end_errors.extend([end_error] * matched_count)

        duration_rows = video_meta[video_meta["video_id"] == video_id]
        video_duration = float(duration_rows.iloc[0]["duration_sec"]) if len(duration_rows) else None

        rows.append(
            {
                "video_id": video_id,
                "video_duration_sec": video_duration,
                "actual_ad_duration_sec": actual_duration,
                "prediction_duration_sec": pred_duration,
                "overlap_sec": overlap,
                "missed_actual_duration_sec": missed,
                "false_positive_duration_sec": false_positive,
                "ad_capture_rate_pct": capture_rate,
                "predicted_ad_precision_pct": precision,
                "mean_start_error_sec": start_error,
                "mean_end_error_sec": end_error,
                "boundary_matched_actual_interval_count": matched_count,
                "predicted_interval_count": len(pred_intervals),
                "actual_interval_count": len(actual_intervals),
            }
        )

    by_video = pd.DataFrame(rows)
    by_video["rank_capture_rate"] = dense_rank(by_video["ad_capture_rate_pct"], ascending=False)
    by_video["rank_precision"] = dense_rank(by_video["predicted_ad_precision_pct"], ascending=False)
    by_video["rank_start_error"] = dense_rank(by_video["mean_start_error_sec"], ascending=True)
    by_video["rank_end_error"] = dense_rank(by_video["mean_end_error_sec"], ascending=True)
    by_video["four_metric_rank_sum"] = (
        by_video["rank_capture_rate"]
        + by_video["rank_precision"]
        + by_video["rank_start_error"]
        + by_video["rank_end_error"]
    )

    by_video = by_video.sort_values(
        [
            "four_metric_rank_sum",
            "rank_capture_rate",
            "rank_precision",
            "rank_start_error",
            "rank_end_error",
            "false_positive_duration_sec",
            "video_id",
        ],
        ascending=[True, True, True, True, True, True, True],
    )
    best_row = by_video.iloc[0].to_dict()
    by_video = by_video.sort_values("video_id")
    by_video.to_csv(OUT_CSV, index=False)

    false_positive_mean_per_video = (
        None if not false_positive_by_video else sum(false_positive_by_video) / len(false_positive_by_video)
    )

    overall = {
        "video_count": len(dev_video_ids),
        "actual_ad_duration_sec": overall_actual_duration,
        "prediction_duration_sec": overall_pred_duration,
        "overlap_sec": overall_overlap,
        "missed_actual_duration_sec": overall_missed,
        "false_positive_duration_sec": false_positive_mean_per_video,
        "false_positive_total_duration_sec": overall_fp,
        "false_positive_mean_duration_sec_per_video": false_positive_mean_per_video,
        "ad_capture_rate_pct": None if overall_actual_duration <= 0 else overall_overlap / overall_actual_duration * 100.0,
        "predicted_ad_precision_pct": None if overall_pred_duration <= 0 else overall_overlap / overall_pred_duration * 100.0,
        "mean_start_error_sec": None if not all_start_errors else sum(all_start_errors) / len(all_start_errors),
        "mean_end_error_sec": None if not all_end_errors else sum(all_end_errors) / len(all_end_errors),
        "boundary_error_actual_interval_count": len(all_start_errors),
    }

    test_pred = pd.read_csv(TEST_PRED_PATH)
    test_pred["video_id"] = test_pred["video_id"].astype(int)
    test_video_ids = sorted(test_pred["video_id"].dropna().astype(int).unique().tolist())
    test_rows = []

    for video_id in test_video_ids:
        pred_intervals = merge_intervals(clean_intervals(test_pred[test_pred["video_id"] == video_id]))
        actual_intervals = merge_intervals(clean_intervals(labels[labels["video_id"] == video_id]))

        pred_duration = interval_total(pred_intervals)
        actual_duration = interval_total(actual_intervals)
        overlap = overlap_total(pred_intervals, actual_intervals)
        false_positive = max(0.0, pred_duration - overlap)
        missed = max(0.0, actual_duration - overlap)
        capture_rate = None if actual_duration <= 0 else overlap / actual_duration * 100.0
        precision = None if pred_duration <= 0 else overlap / pred_duration * 100.0
        start_error, end_error, matched_count = boundary_errors(actual_intervals, pred_intervals)

        duration_rows = video_meta[video_meta["video_id"] == video_id]
        video_duration = float(duration_rows.iloc[0]["duration_sec"]) if len(duration_rows) else None

        test_rows.append(
            {
                "video_id": video_id,
                "video_duration_sec": video_duration,
                "actual_ad_duration_sec": actual_duration,
                "prediction_duration_sec": pred_duration,
                "overlap_sec": overlap,
                "missed_actual_duration_sec": missed,
                "false_positive_duration_sec": false_positive,
                "ad_capture_rate_pct": capture_rate,
                "predicted_ad_precision_pct": precision,
                "mean_start_error_sec": start_error,
                "mean_end_error_sec": end_error,
                "boundary_matched_actual_interval_count": matched_count,
                "predicted_interval_count": len(pred_intervals),
                "actual_interval_count": len(actual_intervals),
            }
        )

    test_by_video = pd.DataFrame(test_rows)
    test_by_video["rank_capture_rate"] = dense_rank(test_by_video["ad_capture_rate_pct"], ascending=False)
    test_by_video["rank_precision"] = dense_rank(test_by_video["predicted_ad_precision_pct"], ascending=False)
    test_by_video["rank_start_error"] = dense_rank(test_by_video["mean_start_error_sec"], ascending=True)
    test_by_video["rank_end_error"] = dense_rank(test_by_video["mean_end_error_sec"], ascending=True)
    test_by_video["four_metric_rank_sum"] = (
        test_by_video["rank_capture_rate"]
        + test_by_video["rank_precision"]
        + test_by_video["rank_start_error"]
        + test_by_video["rank_end_error"]
    )
    test_best_row = test_by_video.sort_values(
        [
            "four_metric_rank_sum",
            "rank_capture_rate",
            "rank_precision",
            "rank_start_error",
            "rank_end_error",
            "false_positive_duration_sec",
            "video_id",
        ],
        ascending=[True, True, True, True, True, True, True],
    ).iloc[0].to_dict()
    test_by_video = test_by_video.sort_values("video_id")
    test_by_video.to_csv(TEST_OUT_CSV, index=False)
    test_four_metric_means = {
        "ad_capture_rate_pct": None if test_by_video["ad_capture_rate_pct"].dropna().empty else float(test_by_video["ad_capture_rate_pct"].mean()),
        "predicted_ad_precision_pct": None if test_by_video["predicted_ad_precision_pct"].dropna().empty else float(test_by_video["predicted_ad_precision_pct"].mean()),
        "mean_start_error_sec": None if test_by_video["mean_start_error_sec"].dropna().empty else float(test_by_video["mean_start_error_sec"].mean()),
        "mean_end_error_sec": None if test_by_video["mean_end_error_sec"].dropna().empty else float(test_by_video["mean_end_error_sec"].mean()),
    }
    test_false_positive_mean_per_video = (
        None
        if test_by_video["false_positive_duration_sec"].dropna().empty
        else float(test_by_video["false_positive_duration_sec"].mean())
    )

    report = {
        "task": "development_set_metrics_summary",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "viewer_modified": False,
        "prediction_source": str(PRED_PATH),
        "test_prediction_source": str(TEST_PRED_PATH),
        "actual_interval_source": str(LABEL_PATH),
        "duration_source": str(VIDEO_META_PATH),
        "development_video_ids": dev_video_ids,
        "metric_definitions": {
            "광고 구간 포착률": "실제 광고와 예측 광고가 겹친 시간 / 실제 광고 전체 시간 * 100",
            "예측 광고 정밀도": "실제 광고와 예측 광고가 겹친 시간 / 예측 광고 전체 시간 * 100",
            "평균 시작 오차": "각 실제 광고 구간과 가장 잘 맞는 예측 구간의 시작 시점 차이 절댓값 평균",
            "평균 종료 오차": "각 실제 광고 구간과 가장 잘 맞는 예측 구간의 종료 시점 차이 절댓값 평균",
            "비광고 오탐 시간": "전체 성능에서는 Development 영상별 비광고 오탐 시간의 산술평균, 영상별 성능에서는 해당 영상의 비광고 오탐 시간",
        },
        "overall_metrics": overall,
        "best_video_selection": {
            "basis": "rank_sum_of_4_metrics_excluding_false_positive_time",
            "metrics": ["ad_capture_rate_pct", "predicted_ad_precision_pct", "mean_start_error_sec", "mean_end_error_sec"],
            "higher_is_better": ["ad_capture_rate_pct", "predicted_ad_precision_pct"],
            "lower_is_better": ["mean_start_error_sec", "mean_end_error_sec"],
            "best_video_id": int(best_row["video_id"]),
            "best_video_metrics": best_row,
        },
        "test_set_four_metric_summary": {
            "test_set_definition": "Test Set",
            "prediction_source": str(TEST_PRED_PATH),
            "test_video_ids": test_video_ids,
            "four_metric_mean_by_video": test_four_metric_means,
            "false_positive_mean_duration_sec_per_video": test_false_positive_mean_per_video,
            "best_video_selection": {
                "basis": "rank_sum_of_4_metrics_excluding_false_positive_time",
                "metrics": ["ad_capture_rate_pct", "predicted_ad_precision_pct", "mean_start_error_sec", "mean_end_error_sec"],
                "higher_is_better": ["ad_capture_rate_pct", "predicted_ad_precision_pct"],
                "lower_is_better": ["mean_start_error_sec", "mean_end_error_sec"],
                "best_video_id": int(test_best_row["video_id"]),
                "best_video_metrics": test_best_row,
            },
            "by_video_csv": str(TEST_OUT_CSV),
        },
        "by_video_csv": str(OUT_CSV),
        "warnings": [],
        "errors": [],
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = "\n".join(
        [
            "# Development Set Metrics Summary",
            "",
            f"- prediction_source: `{PRED_PATH}`",
            f"- actual_interval_source: `{LABEL_PATH}`",
            f"- by_video_csv: `{OUT_CSV}`",
            "",
            "## Development Set 전체 성능",
            "",
            f"- 광고 구간 포착률: {pct(overall['ad_capture_rate_pct'])}",
            f"- 예측 광고 정밀도: {pct(overall['predicted_ad_precision_pct'])}",
            f"- 평균 시작 오차: {sec(overall['mean_start_error_sec'])}",
            f"- 평균 종료 오차: {sec(overall['mean_end_error_sec'])}",
            f"- 비광고 오탐 시간: {mmss(overall['false_positive_duration_sec'])} / 영상별 평균",
            f"- 비광고 오탐 시간 총합: {mmss(overall['false_positive_total_duration_sec'])}",
            "",
            "## Development Set 4개 지표 종합 최상위 영상",
            "",
            f"- video_id: {int(best_row['video_id'])}",
            f"- 광고 구간 포착률: {pct(best_row['ad_capture_rate_pct'])}",
            f"- 예측 광고 정밀도: {pct(best_row['predicted_ad_precision_pct'])}",
            f"- 평균 시작 오차: {sec(best_row['mean_start_error_sec'])}",
            f"- 평균 종료 오차: {sec(best_row['mean_end_error_sec'])}",
            f"- 비광고 오탐 시간: {mmss(best_row['false_positive_duration_sec'])}",
            f"- 4개 지표 rank sum: {best_row['four_metric_rank_sum']:.0f}",
            "",
            "## Test Set 4개 지표 평균(영상별 평균)",
            "",
            "- test set: Test Set",
            f"- test videos: {test_video_ids}",
            f"- 평균 광고 구간 포착률: {pct(test_four_metric_means['ad_capture_rate_pct'])}",
            f"- 평균 예측 광고 정밀도: {pct(test_four_metric_means['predicted_ad_precision_pct'])}",
            f"- 평균 시작 오차: {sec(test_four_metric_means['mean_start_error_sec'])}",
            f"- 평균 종료 오차: {sec(test_four_metric_means['mean_end_error_sec'])}",
            f"- 평균 비광고 오탐 시간: {mmss(test_false_positive_mean_per_video)} / 영상별 평균",
            "",
            "## Test Set 4개 지표 종합 최상위 영상",
            "",
            f"- video_id: {int(test_best_row['video_id'])}",
            f"- 광고 구간 포착률: {pct(test_best_row['ad_capture_rate_pct'])}",
            f"- 예측 광고 정밀도: {pct(test_best_row['predicted_ad_precision_pct'])}",
            f"- 평균 시작 오차: {sec(test_best_row['mean_start_error_sec'])}",
            f"- 평균 종료 오차: {sec(test_best_row['mean_end_error_sec'])}",
            f"- 비광고 오탐 시간: {mmss(test_best_row['false_positive_duration_sec'])}",
            f"- 4개 지표 rank sum: {test_best_row['four_metric_rank_sum']:.0f}",
            "",
            "선정 기준: 비광고 오탐 시간을 제외하고, 포착률/정밀도는 높을수록 좋게, 시작/종료 오차는 낮을수록 좋게 순위를 매긴 뒤 순위 합이 가장 낮은 영상을 선택했습니다.",
            "",
        ]
    )
    SUMMARY_MD.write_text(summary, encoding="utf-8")
    RUN_LOG.write_text(
        "\n".join(
            [
                "[development_set_metrics_summary]",
                f"created_at={report['created_at']}",
                f"prediction_source={PRED_PATH}",
                f"actual_interval_source={LABEL_PATH}",
                f"duration_source={VIDEO_META_PATH}",
                f"video_ids={dev_video_ids}",
                f"best_video_id={int(best_row['video_id'])}",
                f"test_video_ids={test_video_ids}",
                f"test_best_video_id={int(test_best_row['video_id'])}",
                "viewer_modified=false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(summary)


if __name__ == "__main__":
    main()
