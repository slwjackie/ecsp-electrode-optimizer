# v8.1 요구사항–구현–검증 대조

| 요구사항 | 구현 위치 | 확인 방법 / 상태 |
|---|---|---|
| 기존 네 목적함수 | 새 native YAML + `ecsp_nsga2/bc_vmin.py` | objective 목록 보존 시험, 원본 설정 byte identity |
| C++17 FP64 CPU 핵심 solver | `cpp/bc_native/bc_engine.cpp`, `bc_ops_cpu.cpp`, `bc_scalar.h` | 실제 compiler 빌드, Python reference field/history parity |
| CPU standalone | `cpp/bc_native/standalone.cpp`, `python/ecsp_native/standalone.py` | 실제 binary 실행, truncated protocol 실패 시험, extension과 bitwise equality |
| CUDA FP64 커널 | `cpp/bc_native/bc_ops_cuda.cu` | shared FP64 산술·stream·guard·launch error 정적 점검. 실제 nvcc/GPU 시험은 대상 환경 필요 |
| 핵심 timestep Python loop 제거 | `bc_engine.cpp` | 적분/전위 반복이 C++에 위치. Python은 adapter/metrics/workflow/scheduling |
| A100 physics batch32/64 | `config/nsga2_bc_global_native_a100_batch{32,64}.yaml` | 프로필 자동시험. 실측 최적 배치 미확정 |
| Batched parallel Vmin | `ecsp_nsga2/bc_vmin.py`, `bc_native.py` | 혼합 전압 배치와 독립 trial 비교, fake threshold 상태기계 시험 |
| 성공 trial early-stop | `bc_engine.cpp` active mask | 10/10/1 vs 10/10/10 active steps와 동일 onset 확인 |
| 기존 2초 reference metric 유지 | adapter + evaluator | 부분 trial의 2초값 NaN, reference/handoff는 full horizon 유지 시험 |
| 최종 Vmin 상한 확인 | `verify_final_upper_full_horizon` | 최종 trial 역할·유효성 검사 시험 |
| CPU8/A100 hybrid | `ecsp_nsga2/bc_native.py` | spawned CPU pool 실제 실행, cgroup/affinity 예산 제한. GPU 동시 실행은 미검증 |
| OOM batch fallback | `NativeBCGlobalEvaluator._safe_run` | 합성 OOM으로 17개 후보 누락/중복/순서 변경 없음을 확인 |
| 실패 분류 | `_candidate_failure` / voltage search | 수치 실패는 후보 invalid; compiler/device 오류는 명시적 실패 |
| 원본 backend 보존 | `ecsp_nsga2/evaluator.py` 작은 dispatch 확장 | 원본 56 tests + 전체 80 tests 통과 |
| Pareto / handoff / propagation / staggered | 원본 workflow 및 propagation 보존 | native 단축 end-to-end, standalone CPU pool end-to-end 시험 |
| CPU/Python parity | `python/compare_bc_native_python.py` | 동일 strict tolerance 다중-step 8 trajectories 통과; 기본 tolerance 탐색 차이 별도 기록 |
| CUDA CPU parity / B32·64 | `test_bc_native.py` CUDA tests | 시험 코드 포함, 이번 환경 no CUDA로 2 skip |
| 배치 benchmark | `benchmark_bc_native_batches.py` | 동일 후보 generation helper CPU시험. 실제 A100 처리율 미측정 |
| 기존 모든 파일 보존 감사 | `audit_v800_preservation.py` | 원본 ZIP 해시와 각 파일별 비교; 별도 audit.json/md 및 unified diff |
| 릴리스 ZIP | 최종 archive + 새 SHA manifest | ZIP CRC 검사 및 내용별 SHA256 확인 |

## 원본 기능의 보존 범위

원본 구성 파일·물성·chemistry 표·기하학·post-onset propagation·기존 C++ solver·기존 launch script는 삭제하지 않았고 byte identity로 검사합니다. 수정된 원본 Python evaluator는 native backend를 선택한 경우만 새 엔진으로 dispatch합니다. 파일 동일성과 회귀시험 통과는 실행한 범위의 근거이며 모든 임의 형상·설정에서의 과학적 정확성이나 수렴 보증은 아닙니다.

## 의도적으로 변경하지 않은 것

초기 조성, nominal 계수, global kinetics, onset 기준, 시간간격, 배치 외 geometry 설정, 수치 cap, 주/보조 목적함수, staggered 독립 비교 원칙을 바꾸지 않았습니다. 원본 B/C 전극 셀 제외 의미도 그대로 보존했습니다. surface-contact 설명과의 차이는 README에 공개했습니다. 이 변경은 성능/실행 구조 포팅이며 실험보정이나 새로운 물리모델 구현이 아닙니다.

## 추가하지 않은 최적화

부모 기반 초기 bracket, surrogate voltage 예측, 느슨한 물리 tolerance, FP32 대체, 물리 시간간격 확대는 사용하지 않았습니다. CUDA 성공 후보를 매 timestep 메모리 compaction하지는 않으며 active mask로 갱신을 건너뜁니다. GPU 단일 후보 OOM을 자동 CPU 성공으로 숨기지 않습니다.
