
# Split Terminology Note

공개용 문서에서는 내부 실험 단계에서 사용한 세부 분할명을 단순화해 `Development Set`과 `Test Set` 중심으로 설명합니다.

## 공개용 표기

| 공개용 표기 | 기존 v2.4 split | 사용 목적 |
| --- | --- | --- |
| Development Set | `train` | 규칙 설계, cue 분석, 오류 진단 |
| Test Set | `validation` + `test` | 규칙을 고정한 뒤 최종 평가와 데모 확인 |

## 내부 호환

기존 파일의 `split` 값은 `train`, `validation`, `test`로 유지합니다. 일부 helper와 historical output에는 `split_role_v2_5`, `evaluation_subset_v2_5`, `is_*` 계열 column이 남아 있을 수 있습니다. 이 값들은 과거 산출물과 호환하기 위한 내부 key이며, GitHub 공개 문서에서는 `Development Set`과 `Test Set`으로 통합해 설명합니다.

## Helper

```python
from split_terminology_v2_5 import add_v2_5_split_columns, assert_development_only
```

repo root에서 import할 때는 `src/utils`를 `PYTHONPATH`에 추가하거나 파일 경로로 helper를 불러옵니다.
