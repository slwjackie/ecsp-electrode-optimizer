# 공통 preflame 해석기 분리와 이름 변경

공통 점화 전 해석 코드가 `ecsp_nsga2`에서 `ecsp_preflame`으로 분리되었습니다.
현재 직선 전극 study와 고정 형상 screening에 필요하지 않은 NSGA-II 세대 반복,
부모 선택, 교차·돌연변이 및 최적화 전용 실행기를 제거했습니다.
기존 물리 모델을 새로운 물리 모델로 교체하는 변경은 아닙니다.

## 이름 대응표

| 이전 이름 | 현재 이름·위치 | 역할 |
|---|---|---|
| `python/ecsp_nsga2/`의 공통 해석 코드 | `python/ecsp_preflame/` | 형상 입력, 평가, onset/Vmin 및 후속 해석 연결 |
| `ecsp_v6.physics.bc_global` | `ecsp_preflame.electrochemical_thermal_decomposition` | 점화 전 전기화학–열전달–분해반응 기준 구현 |
| `BCGlobalPreflameEvaluator` | `ElectrochemicalThermalDecompositionEvaluator` | 설정 병합과 Python/Torch 모델 평가 |
| `ecsp_nsga2.bc_native` | `ecsp_preflame.cpp_cuda_evaluator` | C++/CUDA 해석 실행 및 CPU/GPU 배치 관리 |
| `NativeBCGlobalEvaluator` | `CppCudaPreflameEvaluator` | C++/CUDA 평가기 |
| `NativeBCHybridEvaluator` | `CppCudaHybridPreflameEvaluator` | CPU/GPU 작업 분배 |
| `ecsp_nsga2.bc_vmin` | `ecsp_preflame.voltage_search` | 동일 onset 기준의 최소전압 탐색 |
| `python/ecsp_native/` | `python/ecsp_cpp_cuda/` | Python 입력과 컴파일된 엔진 사이의 변환·빌드 |
| `cpp/bc_native/` | `cpp/preflame_cpp_cuda/` | C++ CPU/CUDA GPU 계산 구현 |
| `bc_engine.*`, `bc_scalar.h`, `bc_ops*` | `preflame_engine.*`, `preflame_cell_physics.h`, `preflame_ops*` | 시간적분, 셀별 물리식, CPU/GPU 연산 |
| `bc_global` 설정 블록 | `preflame_model` | 모델 입력 |
| resolved 설정의 `bcGlobal` | `preflameModel` | 실제 해석기에 전달된 모델 입력 |
| `evaluator.native` | `evaluator.cpp_cuda` | 컴파일된 실행의 장치·배치·작업자 설정 |

기본 물리 설정 파일은
`config/preflame_electrochemical_thermal_decomposition_a100_cpu8.yaml`입니다.
이전 `nsga2_bc_reactive_a100_cpu8_poweroff_200x3.yaml`의 물리 설정을 계승합니다.
`200x3`은 과거 최적화 예산을 가리켰으므로 현재 공통 물리 입력의 이름에서 제거했습니다.

현재 backend 이름은 `preflame_torch`, `preflame_cpp_cuda`,
`preflame_cpp_cuda_hybrid`, `preflame_cpp_cpu_pool`입니다.
`cpp_cuda`는 구현 방식이며, CPU와 CUDA 중 실제 장치는 실행 설정으로 선택합니다.
CUDA 요청이 실패했을 때 자동으로 CPU 결과로 바꾸지 않습니다.

`configuration.py`가 기존 `bc_global`/`bcGlobal`/`preflameModel` 설정을
`preflame_model`로, `evaluator.native`를 `evaluator.cpp_cuda`로 정규화합니다.
기존 backend 이름과 `optimization` 설정 블록도 입력 호환 경로에서 읽지만,
최적화 루프를 복구하지는 않습니다. 옛 이름과 새 이름에 서로 다른 값을 동시에
주면 충돌로 거부합니다. `ecsp_nsga2` import를 유지하는 호환 패키지는 없으므로
외부 스크립트의 import는 새 경로로 수정해야 합니다.

## 무엇을 남겼는가

고정 DOE 후보의 결과에 쓰이는 `CandidateEvaluation`과 Pareto/crowding 기반
결과 정리는 `candidate_ranking.py`에 남아 있습니다. 후보 결과의 정렬은 NSGA-II
진화 최적화 반복과 별개입니다. 공통 baseline·평가·후속 해석 기능은
`evaluation_workflow.py`의 `EvaluationWorkflow`로 분리했습니다.

148개 catalogue의 정의, 외부 frozen143 원본을 검증·복사하는 절차, 기존 5개 형상의
생성 규칙과 직선 전극 8개 조건은 유지합니다. 외부 frozen143 archive는 여전히
사용자가 준비해야 하며 저장소가 그 형상을 이름으로 재생성하지 않습니다.

전위/전류와 Nernst–Planck 수송, Butler–Volmer 계면반응, 전도·대류/복사 경계손실,
Joule·전기화학·분해 발열, 두 채널 LP/PVA 분해반응의 식과 계수는 유지합니다.
여기서 대류는 경계 열손실이며, 내부 유체 대류 해석을 추가했다는 뜻이 아닙니다.
응축상 onset과 가시 기체 화염 점화의 구분도 그대로입니다.

## 최소 파일 읽기 순서

| 순서 | 파일 | 먼저 확인할 부분 |
|---|---|---|
| 1 | `config/preflame_electrochemical_thermal_decomposition_a100_cpu8.yaml` | `physics`, `preflame_model.transport/kinetics/thermal/onsetCriterion`, `minimum_ignition_voltage_search`, `evaluator` |
| 2 | `config/default_lp_pva.yaml` | 상속되는 `interface`의 전극반응 계수와 기본 물성 |
| 3 | `python/ecsp_preflame/evaluator.py` | `ElectrochemicalThermalDecompositionEvaluator.__init__`: 최종 설정 조립 및 모델 선택 |
| 4 | `python/ecsp_preflame/electrochemical_thermal_decomposition.py` | `preflame_model_contract`, `preflame_transport_fields`, `_channel_rate`, `_reaction_inventory`, `run_preflame_batch` |
| 5 | `python/ecsp_v6/physics/electrochem.py` | 핵심 모델이 호출하는 전위·전극반응 해법 |
| 6 | `python/ecsp_v6/physics/species.py` | 종별 flux와 재고 제한 업데이트 |

이후 조성에서 물성을 만드는 과정이 필요하면 `ecsp_v6/physics/composition_model.py`,
Vmin 이분법만 필요하면 `ecsp_preflame/voltage_search.py`를 읽습니다.
실제 C++/CUDA 실행을 추적하려면 `cpp_cuda_evaluator.py` →
`ecsp_cpp_cuda/adapter.py` → `preflame_cell_physics.h`·`preflame_engine.cpp`
순서로 확인하십시오. `loader.py`는 컴파일/캐시를 담당하므로 물리식을 읽을 때는
뒤로 미뤄도 됩니다. CPU 빌드 스크립트는 `tools/build_preflame_cpp_cpu.sh`,
A100 benchmark 스크립트는 `tools/benchmark_preflame_cpp_cuda_a100.sh`입니다.
저장된 onset 상태에서 후속 해석만 실행하는 CLI는 `python/run_preflame_post_onset.py`입니다.
resolved 입력 옵션은 `--resolved-preflame-config`이며, 이전 `--resolved-bc-config`도
호환 옵션으로 받습니다.

[파라미터 출처와 실험보정 위치](PARAMETER_PROVENANCE_PREFLAME_KR.md)는 현재 설정
이름을 사용합니다. 이전 버전 문서의 nominal 값이나 가정은 실제 실행에서 저장한
`resolved_physics_config.json`과 함께 대조해야 합니다.

## 기존 결과·빌드와의 관계

- 구조 변경으로 소스 fingerprint가 바뀝니다. 이전 실행을 `--resume`으로 이어 붙이지 말고 새 output 디렉터리를 사용합니다. 기존 결과 파일은 삭제하지 않습니다.
- 빌드 캐시 기본 경로는 `.cpp_cuda_build`이며 `ECSP_CPP_CUDA_BUILD_ROOT`로 지정합니다. 기존 환경의 `ECSP_NATIVE_BUILD_ROOT`도 fallback으로 허용합니다.
- 기존 handoff의 `ecsp_bc_surface_onset_v8.2.0` schema와 `bc_handoff_fields.npz` 파일명 등 직렬화 계약은 기존 자료 판독을 위해 유지합니다. 남아 있는 이 식별자는 현재 모델·모듈 이름이 아닙니다.
- standalone 입출력의 binary magic도 호환성을 위해 보존합니다.
- 버전별 README·audit·검증 산출물과 소스 SHA-256 manifest는 당시 기록으로 남깁니다. 그 안의 옛 import나 최적화 실행 명령은 현재 사용법을 의미하지 않습니다. 이전 manifest는 이번 소스의 무결성 목록이 아닙니다.

현재 CPU 검증 진입점은 저장소 루트에서 실행하는
`bash tools/validate_preflame_cpu.sh`입니다. 버전별 `validate_v8*` 스크립트는
과거 배포의 검증 기록용으로 보존합니다.

이번 변경은 실험보정이나 실험 검증을 수행한 변경이 아닙니다. CUDA 실행·성능은
CUDA 장치가 있는 환경에서 별도로 확인해야 합니다.
