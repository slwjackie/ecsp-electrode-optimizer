# Five-topology geometry validation

대상: E058, E114, R038, R050, R091.

이 패치는 `python/ecsp_image_design/geometry.py`의 5개 source-id 전용 fitter만 추가한다. 기존 143개 성공 형상은 기존 generic fitter 경로를 그대로 사용하며, preflame/postflame physics 및 solver는 수정하지 않는다.

로컬 geometry generation + audit 결과:

| ID | status | domain (mm) | Aa=Ac (mm²) | CAD gap (mm) | minimum width (mm) | final components A/C |
|---|---|---:|---:|---:|---:|---:|
| E058 | geometry_accepted | 50 | 271.4263 | 3.6500 | 8.9563 | 4 / 4 |
| E114 | geometry_accepted | 237 | 12739.0235 | 3.6500 | 6.0842 | 1 / 8 |
| R038 | geometry_accepted | 133 | 856.4109 | 16.6667 | 2.3000 | 1 / 1 |
| R050 | geometry_accepted | 286 | 27097.4231 | 12.3825 | 2.3000 | 9 / 1 |
| R091 | geometry_accepted | 181 | 2490.1332 | 7.6773 | 2.6450 | 1 / 9 |

`audit` 재검증 결과는 5/5 통과했다. PDE/preflame/postflame 성능 계산은 이 검증에서 실행하지 않았다.

추가 확인:
- 기존 v5 generic fitter 함수 본문과 v6 `_fit_reference_generic()` 본문은 함수명 외 동일하다.
- `objectives.py`와 `runner.py`는 v5와 byte-identical이다.
- `pytest -q tests/image_design`: 38 passed.
- E058은 사용자가 새로 제공한 큰 4+4 pad 이미지를 OpenCV로 추출한 mask로 `references.json`의 E058 record를 교체했다. 동일 domain에서 기존 four-finger staggered witness가 존재한다.
