# Data Policy

## 포함한 항목

- 구현 코드와 설정 파일
- 프로젝트 설명 문서
- 스키마 확인용 샘플 CSV/JSON
- 공개용 성능 요약 CSV
- 원본 영상 프레임을 쓰지 않는 demo 설명 이미지
- 실제 영상 없이 동작하는 샘플 viewer

## 제외한 항목

- 원본 YouTube 영상, 프레임 덤프, 오디오 덤프, proxy media
- private label, 실제 OCR 원문, frame-level OCR 결과
- 사람이 검토한 review output, notebook output
- cache, backup, logs, generated reports, latest bundles
- model weight/cache: `.pt`, `.pth`, `.onnx`, `.ckpt`
- 외부 repository 복사본, 논문 PDF, 대용량 binary artifact

## 정답 구간 사용 원칙

정답 구간은 후보 생성이나 규칙 판단에 넣지 않습니다. 탐지 결과가 나온 뒤 평가, 오류 분석, viewer reference 용도로만 사용합니다.

## 공개용 결과 파일

`results/final_metrics_summary.csv`는 제공된 최종 평가 요약값을 CSV로 정리한 파일입니다. `results/metrics_by_video_anonymized.csv`는 원본 영상 식별자를 제거하고 숫자 지표만 남긴 공개용 예시입니다.

## 업로드 전 점검 예시

```bash
find . -type f \( -iname '*.mp4' -o -iname '*.pth' -o -iname '*.pdf' -o -iname '*.xlsx' \)
rg -n '/home/|<sensitive-patterns>' .
du -sh .
```
