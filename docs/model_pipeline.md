# Model Pipeline

## 1. 장면 전환 기준점

OpenCV/FFmpeg, ResNet embedding, TransNetV2 conservative 후보를 결합해 광고 상태가 바뀔 수 있는 시점을 찾습니다. 이 단계는 광고 여부를 직접 판단하지 않고, 이후 OCR/오디오 단서를 붙일 시간 기준을 만듭니다.

## 2. OCR 단서

EasyOCR 결과에서 유료광고 고지, 협찬 표현, 제품명, 구매 유도 문구, 링크 안내, 광고 후 텍스트 감소 흐름을 추출합니다. 영상 초반 고지는 전체 영상 안내일 수 있으므로 단독 광고 시작 근거로 쓰지 않습니다.

## 3. 오디오 단서

오디오는 같은 영상 안에서 평소보다 활발해지거나 조용해지는 흐름을 봅니다. RMS/log energy, silence, spectral flux, onset, timbre 계열 feature를 사용하지만 오디오만으로 시작/종료를 확정하지 않습니다.

## 4. 규칙 기반 상태 전이

탐지기는 `non_ad`, `start_pending`, `in_ad`, `end_pending` 상태를 사용합니다. OCR 단서가 주 근거이고, 오디오 단서와 블랙 화면 단서는 신뢰도를 보강합니다.

## 5. 후처리

가까운 후보 연결, 강한 후보 승격, 종료 확장, 경계 축소, 약한 후보 demotion, 영상별 예측량 제한을 적용합니다. 이 단계에서 최종 탐지 결과와 review 후보를 분리합니다.

## 6. Review viewer

최종 `ad_start_sec`, `ad_end_sec`는 viewer에서 정답 구간과 함께 확인합니다. 이 저장소의 viewer는 실제 영상 없이 샘플 타임라인만 제공합니다.
