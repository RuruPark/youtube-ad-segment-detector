# Reproducibility

## 이 저장소에서 확인할 수 있는 것

- 코드, 설정, 문서 구조
- 샘플 manifest 기반 viewer 화면
- 공개용 성능 요약 CSV와 익명화된 영상별 지표 예시
- Python 문법과 JSON 설정 유효성
- private 데이터 배치 후 재실행에 필요한 파일 스키마

## 이 저장소만으로 재현할 수 없는 것

- 원본 실험 성능 재계산
- OCR frame-level 결과 재생성
- raw video 기반 장면/오디오 feature 추출
- TransNetV2 weight를 사용한 shot-boundary 실험

`results/final_metrics_summary.csv`의 수치는 새로 계산한 값이 아니라 제공된 최종 평가 요약값을 문서화한 것입니다.

## 전체 재현에 필요한 입력

- `data/video_metadata/video_manifest_*.csv`
- `data/splits/video_split_*.csv`
- `data/segments/ad_interval_segments_*.csv`
- `data/features/visual_scene_boundary_anchors_*.csv`
- `data/audio/*features*.csv`
- `data/ocr/*features*.csv`
- `data/raw/videos/` 아래 원본 영상

## 환경 예시

```bash
cd youtube-ad-segment-detector-github
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU OCR/PyTorch 환경은 시스템마다 다릅니다. 이 저장소 정리 과정에서는 패키지 설치, 학습, OCR 재실행, 데이터 다운로드를 수행하지 않았습니다.
