# Rule Design

## 상태 정의

| 상태 | 의미 |
| --- | --- |
| `non_ad` | 광고 구간 밖 |
| `start_pending` | 시작 단서는 있으나 확정 전 |
| `in_ad` | 광고 구간 내부 |
| `end_pending` | 종료 후보가 있으나 확인 필요 |

## 핵심 원칙

- 장면 전환만으로 광고 시작/종료를 확정하지 않습니다.
- 오디오만으로 광고 시작/종료를 확정하지 않습니다.
- 영상 초반 유료광고 고지는 단독 시작 근거로 쓰지 않습니다.
- OCR 실패나 빈 결과는 비광고 근거가 아니라 unknown으로 처리합니다.
- 정답 구간은 판단 feature가 아니라 평가와 오류 분석에만 사용합니다.
- 영상별 예측량 제한으로 과도한 탐지 결과를 review 후보로 내립니다.

## Pending 규칙

| 규칙 | 값 | 의미 |
| --- | ---: | --- |
| `start_pending_max_anchor_count` | 1 | 시작 후보는 다음 기준점까지만 확인 |
| `start_pending_max_duration_sec` | 15 | 약한 시작 후보가 너무 앞쪽으로 확장되지 않도록 제한 |
| `end_pending_max_anchor_count` | 2 | 종료 후보는 내부 전환과 구분하기 위해 두 기준점까지 확인 |
| `end_pending_max_duration_sec` | 20 | 종료 후보가 길게 방치되지 않도록 제한 |
| `minimum_ad_duration_sec` | 20 | 너무 짧은 광고 구간 생성을 줄이는 prior |

## 후처리 규칙

- `bridge`: 가까운 후보를 단서 조건에 따라 연결
- `promotion`: 강한 review 후보를 제한적으로 최종 결과로 승격
- `guarded end extension`: 짧은 예측 구간의 종료를 보수적으로 확장
- `final boundary trim`: 너무 긴 예측 구간을 내부 기준점으로 축소
- `weak false-positive gate`: 단서가 약한 후보를 review로 이동
- `black/end supported restore`: 블랙 화면과 종료 단서가 함께 있는 후보를 보존
