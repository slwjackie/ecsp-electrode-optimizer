# ECSP preflame electrode analysis

전극 형상에 따른 **점화 전 전기화학–열전달–분해반응**을 해석하는 코드입니다.
공통 해석기는 `python/ecsp_preflame/`에 있고, C++/CUDA 구현은
`python/ecsp_cpp_cuda/`와 `cpp/preflame_cpp_cuda/`에 있습니다.
사용하지 않는 NSGA-II 진화 최적화 실행 경로는 제거했습니다.

현재 실행 경로는 다음과 같습니다.

| 목적 | 실행 파일 | 안내 |
|---|---|---|
| 25 mm 직선 전극 8개 조건의 parameter study | `python/run_electrode_parameter_study.py` | [직선 전극 안내](docs/ELECTRODE_PARAMETER_STUDY_25MM_KR.md) |
| 고정 148개 형상과 각 형상의 면적 대응 staggered 비교 | `python/run_paired_preflame_screening.py` | [paired screening 안내](docs/image_design/PAIRED_SCREENING_KR.md) |
| 기존 5개 형상 생성·감사·평가 | `python/run_five_topology_preflame.py` | [5개 형상 검증 기록](docs/image_design/FIVE_TOPOLOGY_VALIDATION_KR.md) |

직선 전극 study와 148개 paired screening은 **같은 공통 물리 해석기**를 호출합니다.
전극 mask를 만드는 방법과 결과 비교 방식이 다릅니다. 기존 148개 형상 정의와
직선 전극의 치수·면적 조건, 물리식·반응계수는 이번 구조 변경의 수정 대상이 아닙니다.

## 이름의 의미

- `preflame`: 응축상 분해 onset까지의 점화 전 모델입니다. 가시 기체 화염의 점화·전파를 뜻하지 않습니다.
- `electrochemical_thermal_decomposition`: 전위/이온 수송·전극반응, 열전달, 두 채널 분해반응을 결합한 모델입니다. 이전의 모호한 `B/C` 모델 이름을 대체합니다.
- `cpp_cuda`: C++로 컴파일한 CPU 연산과 CUDA GPU 연산입니다. 이전 `native` 이름을 대체하며, 새로운 물리 모델을 뜻하지 않습니다.

[파일·설정 이름 변경표와 읽는 순서](docs/PREFLAME_REFACTOR_KR.md)에 기존 이름과 새 이름의 대응을 정리했습니다.

## 최소 실행 예시

저장소 루트에서 필요한 Python 의존성을 설치합니다. 수치 해석에는 PyTorch와 C++
컴파일 환경이 필요하며, CUDA 실행에는 CUDA 지원 PyTorch와 CUDA toolkit도 필요합니다.

```bash
python -m pip install -r python/requirements.txt
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

RUN="$PWD/runs/straight25_preflame"
python python/run_electrode_parameter_study.py generate --out "$RUN"
python python/run_electrode_parameter_study.py audit --out "$RUN"
python python/run_electrode_parameter_study.py evaluate \
  --out "$RUN" --device cuda --batch-size 8 \
  --physics-config config/preflame_electrochemical_thermal_decomposition_a100_cpu8.yaml
```

CPU 실행은 `--device cpu`로 선택합니다. `generate`와 `audit`는 PDE를 풀지 않습니다.
리팩터링 이전 실행의 소스 fingerprint는 달라지므로 **새 output 경로**를 사용하십시오.
동일 소스·설정·형상으로 실행한 결과만 `--resume`으로 재개합니다.

## 물리식과 실험보정 계수부터 읽기

1. `config/preflame_electrochemical_thermal_decomposition_a100_cpu8.yaml`: 실제 선택한 물리 입력과 `preflame_model` 블록.
2. `config/default_lp_pva.yaml`: 위 설정의 `base_config`로 상속하는 전극 계면반응·기본 물성.
3. `python/ecsp_preflame/evaluator.py`: 설정을 합치고 onset/Vmin 평가를 연결하는 `ElectrochemicalThermalDecompositionEvaluator`.
4. `python/ecsp_preflame/electrochemical_thermal_decomposition.py`: 수송계수, 열·분해반응과 시간적분의 기준 구현.
5. `python/ecsp_v6/physics/electrochem.py`와 `species.py`: 핵심 모델에서 호출하는 전위·Butler–Volmer 반응과 종 수송.

실험으로 확인·교체할 계수는 [파라미터 출처표](docs/PARAMETER_PROVENANCE_PREFLAME_KR.md)를
함께 보십시오. 현재 `literature_nominal_uncalibrated` 설정은 해당 시료로 실험보정됐다는 뜻이 아닙니다.
컴파일된 계산식까지 대조할 때만 `cpp/preflame_cpp_cuda/preflame_cell_physics.h`와
`preflame_engine.cpp`를 이어서 읽으면 됩니다.

현재 CPU 검증은 저장소 루트에서 `bash tools/validate_preflame_cpu.sh`로 실행합니다.

`README_V*_KR.md`, 버전별 검증 보고서와 기존 audit 결과는 당시 구현의 기록입니다.
과거 NSGA-II 실행 명령과 옛 모듈 이름은 현재 실행 안내로 사용하지 마십시오.

로컬 백업, 설치된 라이브러리 복사본, 컴파일 캐시와 임시 실행 결과는 Git에 포함하지 않습니다.
선택적 CAD/DOE 환경은 [의존성 설치 안내](docs/PHIDL_DOE600_KR.md#dependency)를 따릅니다.
과거 검증 보고서의 일부 개인 환경 절대경로는 상대경로 또는 `VALIDATION_WORKSPACE`로
정리했으며, 당시 테스트의 성공·실패·건너뜀 결과와 소스 해시는 보존했습니다.
