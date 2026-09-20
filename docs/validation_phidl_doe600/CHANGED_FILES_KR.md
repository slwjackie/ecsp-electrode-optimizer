# 변경 파일 목록

핵심 파일 17개: 신규 16개, 기존 파일 수정 1개입니다.

- 추가: [config/phidl_doe600.yaml](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/config/phidl_doe600.yaml)
- 추가: [docs/PHIDL_DOE600_KR.md](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/docs/PHIDL_DOE600_KR.md)
- 추가: [python/ecsp_doe/__init__.py](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/python/ecsp_doe/__init__.py)
- 추가: [python/ecsp_doe/geometry.py](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/python/ecsp_doe/geometry.py)
- 추가: [python/ecsp_doe/physics.py](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/python/ecsp_doe/physics.py)
- 추가: [python/ecsp_doe/sampling.py](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/python/ecsp_doe/sampling.py)
- 추가: [python/ecsp_doe/selection.py](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/python/ecsp_doe/selection.py)
- 추가: [python/ecsp_doe/storage.py](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/python/ecsp_doe/storage.py)
- 추가: [python/ecsp_doe/topology.py](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/python/ecsp_doe/topology.py)
- 추가: [python/ecsp_doe/workflow.py](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/python/ecsp_doe/workflow.py)
- 추가: [python/requirements-phidl-doe.txt](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/python/requirements-phidl-doe.txt)
- 추가: [python/tests/test_phidl_doe_geometry.py](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/python/tests/test_phidl_doe_geometry.py)
- 추가: [python/tests/test_phidl_doe_physics.py](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/python/tests/test_phidl_doe_physics.py)
- 추가: [python/tests/test_phidl_doe_workflow.py](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/python/tests/test_phidl_doe_workflow.py)
- 추가: [python/tests/test_phidl_handoff_reuse.py](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/python/tests/test_phidl_handoff_reuse.py)
- 수정: [tools/resume_final_only.py](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/tools/resume_final_only.py)
- 추가: [tools/run_phidl_doe600.py](/Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe/tools/run_phidl_doe600.py)

## 검증 및 실행 준비 자료

- `docs/validation_phidl_doe600/`: 테스트 로그/XML, 원본 비교, hash 검증, contact sheet, 생성 요약.
- `runs/doe600_phidl_geometry_validation/`: 100개 topology 및 Stage 1 300개 PNG/JSON/NPZ. physics는 실행하지 않았습니다.
- `.doe-deps/`: 검증한 PHIDL/CAD dependency 격리 설치. 기존 ecsp-m2 환경을 변경하지 않습니다.
