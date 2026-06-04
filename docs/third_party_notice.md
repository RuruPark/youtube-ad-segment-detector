# Third-party Notice

## Runtime dependencies

- OpenCV: 프레임 decode와 장면 전환 feature 추출
- NumPy/pandas: tabular data 처리
- PyTorch/torchvision: pretrained ResNet embedding feature 추출
- EasyOCR: 한국어/영어 OCR 단서 추출
- librosa/soundfile/scipy: 오디오 feature 추출
- scikit-learn: 분석 유틸리티

## External models/repos

TransNetV2는 선택적인 shot-boundary detector로 참조됩니다. 외부 코드와 model weight는 이 저장소에 포함하지 않았습니다. EasyOCR와 PyTorch model cache도 제외했습니다.

정식 라이선스를 붙여 배포하기 전에는 각 dependency의 upstream license와 attribution 조건을 확인해야 합니다. 이 저장소에는 외부 repository 복사본, 외부 논문 PDF, model weight를 포함하지 않습니다.
