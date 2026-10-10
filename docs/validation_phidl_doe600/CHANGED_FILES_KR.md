# 변경 파일 목록

핵심 파일 17개: 신규 16개, 기존 파일 수정 1개입니다.

- 추가: [config/phidl_doe600.yaml](../../config/phidl_doe600.yaml)
- 추가: [docs/PHIDL_DOE600_KR.md](../../docs/PHIDL_DOE600_KR.md)
- 추가: [python/ecsp_doe/__init__.py](../../python/ecsp_doe/__init__.py)
- 추가: [python/ecsp_doe/geometry.py](../../python/ecsp_doe/geometry.py)
- 추가: [python/ecsp_doe/physics.py](../../python/ecsp_doe/physics.py)
- 추가: [python/ecsp_doe/sampling.py](../../python/ecsp_doe/sampling.py)
- 추가: [python/ecsp_doe/selection.py](../../python/ecsp_doe/selection.py)
- 추가: [python/ecsp_doe/storage.py](../../python/ecsp_doe/storage.py)
- 추가: [python/ecsp_doe/topology.py](../../python/ecsp_doe/topology.py)
- 추가: [python/ecsp_doe/workflow.py](../../python/ecsp_doe/workflow.py)
- 추가: [python/requirements-phidl-doe.txt](../../python/requirements-phidl-doe.txt)
- 추가: [python/tests/test_phidl_doe_geometry.py](../../python/tests/test_phidl_doe_geometry.py)
- 추가: [python/tests/test_phidl_doe_physics.py](../../python/tests/test_phidl_doe_physics.py)
- 추가: [python/tests/test_phidl_doe_workflow.py](../../python/tests/test_phidl_doe_workflow.py)
- 추가: [python/tests/test_phidl_handoff_reuse.py](../../python/tests/test_phidl_handoff_reuse.py)
- 수정: [tools/resume_final_only.py](../../tools/resume_final_only.py)
- 추가: [tools/run_phidl_doe600.py](../../tools/run_phidl_doe600.py)

## 검증 및 실행 준비 자료

- `docs/validation_phidl_doe600/`: 테스트 로그/XML, 원본 비교, hash 검증, contact sheet, 생성 요약.
- `runs/doe600_phidl_geometry_validation/`: 100개 topology 및 Stage 1 300개 PNG/JSON/NPZ. physics는 실행하지 않았습니다.
- CAD dependency는 당시 로컬에 격리 설치했습니다. 현재 저장소에는 설치 복사본을
  포함하지 않으며 `python/requirements-phidl-doe.txt`로 별도 가상환경에 설치합니다.
