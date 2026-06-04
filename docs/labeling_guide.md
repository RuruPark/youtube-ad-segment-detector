# Labeling Guide

## 기본 interval schema

| column | 의미 |
| --- | --- |
| `video_id` | 영상 식별자 |
| `ad_interval_id` | 광고 구간 식별자 |
| `ad_start_sec` | 광고 시작 초 |
| `ad_end_sec` | 광고 종료 초 |
| `label_valid` | label 사용 가능 여부 |
| `is_abrupt_transition_ad` | 일반 콘텐츠와 분리되는 전환형 광고 여부 |

## 해석 기준

주요 대상은 전체 홍보성 발언이 아니라, 일반 콘텐츠 흐름과 분리되어 삽입되는 광고 블록입니다. 영상 초반의 “유료광고 포함” 고지는 실제 광고 구간 시작이 아닐 수 있으므로 별도 guard를 둡니다.

## Leakage guard

- 규칙과 threshold는 Development Set 기준으로만 조정합니다.
- Test Set의 row-level feature는 규칙 판단에 사용하지 않습니다.
- `label`, `true`, `actual`, `nearest_true_boundary` 계열 컬럼은 판단 feature에서 제외합니다.
