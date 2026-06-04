# Repository Scope

이 문서는 저장소에 포함한 범위와 제외한 항목을 정리합니다.

## 포함 범위

- 장면 전환, OCR, 오디오, 규칙 기반 탐지 관련 핵심 source와 config
- 샘플 viewer를 구성하는 HTML, CSS, JavaScript
- 스키마 확인용 샘플 CSV/JSON
- 공개용 성능 요약 CSV
- 원본 영상 프레임을 쓰지 않는 demo 설명 이미지
- 파이프라인, 데이터 정책, 규칙 설계, 오류 분석, 재현 방법 문서

## 제외 범위

- 원본 영상과 오디오, 프레임 덤프
- private label과 OCR 원문 결과
- model weight/cache
- 실행 로그, backup, generated report, notebook output
- 외부 PDF와 외부 repository 복사본

## 성능 지표 요약

최종 결과는 광고 구간을 놓치지 않는 방향을 우선한 설정에서 정리했습니다. 광고 구간 포착률은 85.0%로, 실제 광고 구간의 상당 부분을 예측 구간이 덮는 것을 목표로 했습니다. 예측 광고 정밀도는 67.8%로, 일부 비광고 구간이 광고 구간에 포함되는 오탐이 남아 있습니다.

경계 기준으로는 평균 시작 오차가 38.4초, 평균 종료 오차가 43.4초로 나타났습니다. 이는 광고 구간의 존재 여부를 찾는 데는 의미가 있었지만, 정확한 시작·종료 시점 정밀화에는 추가 개선이 필요하다는 점을 보여줍니다. 영상 하나당 비광고 오탐 시간은 평균 55.8초로 집계되었습니다.

| 지표 | 값 | 의미 |
| --- | ---: | --- |
| 광고 구간 포착률(Recall) | 85.0% | 각 영상에서 실제 광고 구간을 얼마나 덮었는지 평균 |
| 예측 광고 정밀도(Precision) | 67.8% | 각 영상에서 광고라고 예측한 구간이 실제 광고와 얼마나 겹쳤는지 평균 |
| 평균 시작 오차 | 38.4초 | 실제 광고 시작 시점과 가장 잘 맞는 예측 구간 시작 시점의 평균 차이 |
| 평균 종료 오차 | 43.4초 | 실제 광고 종료 시점과 가장 잘 맞는 예측 구간 종료 시점의 평균 차이 |
| 비광고 오탐 시간 | 55.8초 | 영상 하나당 평균적으로 비광고를 광고로 잘못 잡은 시간 |

CSV 형식의 요약은 `results/final_metrics_summary.csv`에 정리했습니다. `results/metrics_by_video_anonymized.csv`는 원본 영상 식별자를 제거하고 공개 가능한 숫자 지표만 남긴 영상별 예시입니다.

## 결과 기록

Development Set 사후 진단 기준의 주요 기록은 다음과 같습니다. 이 수치는 포함된 샘플 데이터로 재계산한 값이 아닙니다.

| metric | value |
| --- | ---: |
| final_prediction_count_after_targeted_patch | 13 |
| prediction_good | 9 |
| prediction_partial_too_short | 3 |
| candidate_exists_but_not_selected | 3 |
| false_positive_candidate_count | 0 |
| overextended_prediction_count | 1 |
| middle_gap_case_count | 0 |
| review_candidate_count_after_targeted_patch | 224 |

## Demo viewer

`outputs/demo/final_presentation_ad_skip_viewer/`에는 실제 영상 없이 타임라인과 지표 구조를 보여주는 샘플 manifest/metrics가 들어 있습니다. 원본 영상은 포함하지 않았습니다.

`assets/demo_screenshots/demo_viewer_overview.svg`는 실제 영상 프레임 없이 viewer 구조를 설명하는 공개용 도식 이미지입니다.
