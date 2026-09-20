> **역사적 참고문서:** 이 문서는 v7.8.1 포팅 기록입니다. 현재 production semantics는 v7.9.0의 `SURFACE_CONTACT_MULTICOMPONENT_V7_9_0_KR.md`를 따릅니다.

# v7.8.1 C++ FP64 포팅 변경내역

## 새 파일

- `cpp/ecsp_cpp_solver.cpp`
- `python/ecsp_cpp/__init__.py`
- `python/ecsp_cpp/backend.py`
- `config/nsga2_condensed_phase_no_f_m2_cpp_fp64.yaml`
- `tools/build_cpp_cpu.sh`
- `tools/run_m2_cpp_fp64_preflight.sh`
- `tools/benchmark_m2_cpp_fp64.sh`
- `tools/run_m2_cpp_fp64_100x2.sh`
- `tools/run_nsga2_m2_cpp_fp64.sh`
- `tools/validate_cpp_vs_python_fp64.sh`
- `python/compare_cpp_python_fp64.py`

## `python/ecsp_nsga2/evaluator.py`

- `CppCondensedFp64Evaluator` 추가
- C++ backend를 `cpp_fp64_cpu`, `cpp_cpu_fp64`, `m2_cpp_fp64`로 선택 가능
- geometry resize/간격 검사는 NumPy/SciPy에서 후보당 한 번 수행
- physics state를 Torch tensor로 만들지 않고 binary mask로 전달
- ThreadPoolExecutor를 제거하고 main-thread `Popen` scheduler로 교체
- 최대 4개 standalone C++ process 동시 실행
- candidate별 native stdout/stderr, input mask/config, metrics 보존
- C++ 실패를 다른 후보와 격리해 `physics_rejection.txt`로 기록

## `python/ecsp_cpp/backend.py`

- anode/cathode binary layout 구현
- resolved YAML → flat C++ config 직렬화
- static/coupled PCG tolerance 분리 전달
- native executable architecture mismatch 시 자동 rebuild 지원
- `prepare_case`, `launch_prepared_case`, `collect_prepared_case` 수명주기 분리

## `cpp/ecsp_cpp_solver.cpp`

- 기존 Python no-F condensed physics를 standalone C++17로 포팅
- 모든 주요 state array를 `std::vector<double>` FP64로 유지
- harmonic-face finite-volume 전위 연산자
- PCG/Jacobi와 deterministic red-black SOR fallback
- static `rtol=1e-10`, coupled `rtol=1e-9`, `atol=1e-12`
- local BV를 절대 260 V 대신 interface voltage-gap 변수로 계산
- mass-transfer-saturation derivative를 underflow-resistant form으로 계산
- global anode/cathode current gauge balance 구현
- Nernst–Planck species, passivation, reduced gas coverage, thermal/phase/decomposition 통합
- 4-lane FP64 dot reduction으로 long-double scalar loop 제거
- electrical/linear/Robin/gauge 누적 iteration diagnostics 추가
- config 및 mask fail-fast validation 추가

## 설정과 실행

- M2 primary backend를 `cpp_fp64_cpu`로 변경
- C++ physics에서 MPS/Metal 사용 없음
- 기본 candidate parallelism 4
- 기존 Python CPU/CUDA/MPS backend는 비교와 regression을 위해 보존
- 기존 workdir 재사용을 거부하여 이전 rejection/fitness 혼합 방지
