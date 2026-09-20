# v8.0.0 파일 보존 감사

기준 파일 173개 / 동일 170개 / 수정 3개 / 삭제 0개

| 파일 | 변경 사유 |
|---|---|
| `README_KR.md` | v8.1 안내문 링크를 앞에 추가. 기존 v8.0 문서는 본문에 그대로 보존. |
| `VERSION.json` | v8.1 버전·새 backend 및 이번 검증 상태를 기록. 과거 검증 상태는 inherited_v8_0_0_test_status로 구분. |
| `python/ecsp_nsga2/evaluator.py` | 기존 기본 실행경로를 유지하고 native backend factory 및 _run_physics dispatch hook만 추가. |

해시는 파일 보존을 검증합니다. 기능·수치 동등성은 회귀시험과 parity 결과를 별도로 확인해야 합니다.
