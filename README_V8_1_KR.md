# ECSP v8.1.0 — B/C Native FP64 + A100/CPU hybrid

v8.0.0 원본에 **새 native backend를 추가**한 소스 배포본입니다. 기존 v8.0.0/v7.9.5 backend, 설정 파일 및 실행 스크립트는 계속 사용할 수 있습니다. 새 실행기는 아래 `run_bc_native_*`입니다. 기존 `run_bc_global_a100.sh`를 실행하면 기존 v8.0.0 경로로 동작합니다.

## 이번 버전의 범위

| 항목 | 구현 |
|---|---|
| B/C 핵심 계산 | C++17 FP64 시간적분, 수송·열·계면 BV·전위 반복 풀이 |
| CPU | 같은 C++ 엔진을 링크한 standalone 실행파일. Python 독립 worker가 후보를 배분하며 실행파일 자체는 Python runtime에 링크하지 않음 |
| GPU | C++/ATen 시간 루프 + FP64 CUDA 커널. 여전히 ATen reduction 및 C++ 호스트 수렴검사가 있어 완전 단일 커널은 아님 |
| 배치 | 새 A100 프로필 기본 32, 대안 64. OOM 시 후보 순서를 보존하며 배치를 분할. 한 후보도 안 들어가면 명시적으로 실패 |
| Vmin | 한 배치 안에서 후보별 전압을 다르게 설정하는 dependency-wave bisection. 모든 전압 trial은 원래 초기상태에서 새로 시작 |
| early-stop | 성공한 **Vmin trial**의 상태 갱신을 active mask로 중단. 텐서 자체를 매번 압축하는 구현은 아님 |
| 정확도 보호 | 기준전압 해석과 handoff 생성은 전체 horizon 유지. 최종 upper voltage도 기본적으로 전체 horizon 재확인 |
| CPU8 | 기본 6 worker + host reserve 2. 실제 cgroup quota/affinity가 작으면 자동 감축 |
| 목적함수 | `t_onset, U_rem, V_min, C_J` 유지. 에너지는 기존 diagnostic 유지 |
| 후단 | 기존 NumPy condensed propagation, Pareto/diversity 선정 및 독립 staggered 비교 경로 보존 |

GPU용 물리계수·시간간격·격자·onset 정의는 v8.0.0과 같게 유지했습니다. 선형해법은 native Jacobi-PCG입니다. 포팅과 물리모델 변경을 혼동하지 않도록, 기존 구현의 실제 mask 의미도 보존했습니다.

## 검증 결과

**CPU 환경에서 전체 시험 80 passed / 2 skipped**입니다. 두 skip은 CUDA 하드웨어 시험입니다. 원본 v8.0.0 시험 56개도 수정 전에 실행해 전부 통과했습니다. C++ standalone/extension 컴파일 및 실행, 단축 전체 workflow, legacy C++ parity, 기존 요구사항 감사 24/24를 확인했습니다.

**실제 CUDA 컴파일·CPU/GPU parity·A100 batch 처리율/메모리는 미검증**입니다. 검증 환경은 CPU-only PyTorch이고 CUDA toolkit/nvcc와 GPU가 없습니다. CUDA 소스 정적 검사 7/7 통과는 CUDA 컴파일 성공을 뜻하지 않습니다. 실제 GPU preflight가 통과하기 전 장시간 production 실행은 하지 마십시오.

정확한 시험 범위와 수치 허용오차는 [검증 보고서](docs/V8_1_VALIDATION_REPORT_KR.md), 요청별 대응은 [구현 대조표](docs/V8_1_IMPLEMENTATION_AUDIT_KR.md), 원본 파일 보존은 [SHA 감사](docs/v8_1_validation/preservation/audit.md)를 참조하십시오. 실제 추진제 실험보정은 이번 소프트웨어 포팅의 범위가 아닙니다.

## 1. 설치와 CPU 동작 확인

기존 CUDA PyTorch가 설치된 학교 환경에서는 그것을 그대로 사용하십시오. native extension에는 PyTorch와 호환되는 C++17 compiler, ninja가 필요하고 CUDA 경로에는 추가로 CUDA-enabled PyTorch와 nvcc/toolkit이 필요합니다. CPU standalone도 LibTorch/ATen 공유 라이브러리를 사용하므로 빌드 시 사용한 PyTorch 설치를 유지해야 합니다.

```bash
cd ECSP_v8_1_0_BCNative_Hybrid_A100_CPU8
python -m pip install -r python/requirements-native.txt
bash tools/build_bc_native_cpu.sh
bash tools/run_bc_native_debug.sh "$PWD/runs/native_debug"
bash tools/validate_v8_1_cpu.sh "$PWD/runs/native_validation"
```

첫 실행은 로컬 컴파일 시간이 포함됩니다. `.native_build/`는 플랫폼별 캐시이며 ZIP에는 바이너리나 컴파일 캐시를 넣지 않았습니다. `ECSP_NATIVE_BUILD_ROOT=/다른/캐시경로`로 위치를 바꿀 수 있습니다. 위 debug는 배관 확인용 짧은 시간·시험 조건이며 물리적 production 결과가 아닙니다.

CPU 8코어 production 실행:

```bash
bash tools/run_bc_native_cpu8.sh "$PWD/runs/native_cpu8" 1000 5
```

먼저 소수 후보를 실행해 수렴과 실제 시간을 확인하십시오. 이번 검증 컨테이너 CPU quota는 4코어이므로 실제 8코어 처리율이나 M2 Pro 속도는 측정하지 않았습니다. macOS CPU 경로는 제공하지만 이번 검증은 Linux x86_64에서 수행했습니다.

## 2. A100 빌드·parity·배치 선택

```bash
export TORCH_CUDA_ARCH_LIST=8.0
bash tools/run_bc_native_a100_preflight.sh "$PWD/runs/native_a100_preflight"
bash tools/benchmark_bc_native_a100.sh "$PWD/runs/native_batches_short" 64 0.05
```

preflight는 CUDA extension을 실제로 빌드하고 GPU parity 시험을 실행합니다. benchmark는 동일 후보를 4/8/16/32/64 배치로 비교해 유효 후보 처리율, 배치간 결과 일치, 메모리를 기록합니다. 짧은 horizon 결과만으로 near-onset/2초 production 최적값을 확정하지 않습니다. 전체 Vmin까지 포함하는 측정은 별도 빈 경로에서:

```bash
PYTHONPATH="$PWD/python" python python/benchmark_bc_native_batches.py \
  --output-dir "$PWD/runs/native_batches_full" --cases 64 \
  --batches 16 32 64 --horizon 2.0 --include-vmin
```

A100+CPU8 production:

```bash
ECSP_BC_CUDA_BATCH=32 bash tools/run_bc_native_a100_cpu8.sh \
  "$PWD/runs/native_a100_b32" 1000 5

# 64는 preflight와 실측 benchmark 확인 후 선택
ECSP_BC_CUDA_BATCH=64 bash tools/run_bc_native_a100_cpu8.sh \
  "$PWD/runs/native_a100_b64" 1000 5
```

배치 32/64는 **검증할 운용점**이지 이미 확인된 최적값이 아닙니다. GPU4→32가 8배 빨라진다는 보장도 없습니다. `handoff_batch_size=4`는 큰 전체 공간·시간장 저장의 메모리를 제어하기 위한 별도 설정으로, physics/Vmin batch32/64와 구분됩니다.

## 3. Vmin / early-stop 의미

각 후보는 원래 reference 전압에서 끝까지 해석해 `U_rem`, `C_J`, 에너지, 이후 handoff 입력을 보존합니다. Vmin의 lower-bound 및 midpoint trial에서만 onset 성공 시 해당 후보의 갱신을 중단합니다. 실패 후보는 horizon까지 계산합니다. 부분 시간해석 결과를 2초 metric으로 표시하지 않으며 `trialOnly`, `validity_window`, `stepsExecutedPerCandidate` 등을 기록합니다.

기본 `verify_final_upper_full_horizon: true`는 최종 upper bracket을 전체 시간 재실행합니다. 정확한 원본 상태의 cap/수치 정상성을 더 확인하는 비용이며 최대 이론 가속률과 실제 가속률이 다를 수 있습니다. 수치 실패는 물리적 미점화로 바꾸지 않습니다. 단조 onset 가정 아래 시험된 상한을 보고하며 이분탐색 자체가 비단조 전압응답의 전역최소를 보증하지 않습니다.

## 4. 수치 일치의 범위와 기존 모델 한계

- Python/C++는 동일 보존·구성식의 수치 포팅이지만 iterative solver 연산순서가 달라 bitwise equality를 보장하지 않습니다. 동일한 엄격 수렴 tolerance에서 17/33 격자, 두 방향, 20/260 V, 10-step 공간장/시간장 비교를 통과했습니다.
- 기본의 느슨한 nonlinear tolerance에서의 탐색 보고서도 보존했습니다. 일부 거의 0인 handoff 열원의 매우 엄격한 점별 오차검사를 통과하지 못해, 양쪽의 수렴 tolerance를 동일하게 강화한 별도 검사를 수행했습니다. 그 차이를 숨기거나 모든 항목이 기본 tolerance에서 동일하다고 주장하지 않습니다.
- 원본 B/C는 문서의 `surface-contact overlay` 표현과 달리 실제 `propellant = ~(anode | cathode)`로 전극 셀을 제외합니다. 이 포팅은 **실제 원본 동작을 유지**합니다. 진짜 overlay로 바꾸는 것은 별도 물리 변경과 검증이 필요한 작업입니다. legacy v7.9.5 path는 그대로 별도 보존됩니다.
- 논문 기반 nominal kinetics/물성, 임계값, 수치 cap, `Xg`–inventory 연결을 새 실험값으로 바꾸지 않았습니다. `U_rem`은 기존 reduced model의 계산량이며 실제 미연소 질량의 검증값이라고 단정할 수 없습니다.
- 후단 propagation은 기상 화염 CFD가 아니며, 이번 포팅이 Tait/Euler 기체상 계산을 새로 추가한 것은 아닙니다.
- 기존 backend에서 가능한 모든 선택 구성식을 native가 지원한다고 가정하지 않습니다. 미지원 saturation/correction/solver option은 묵살하지 않고 오류를 내며 기존 backend를 사용하도록 안내합니다.

## 5. 감사 파일과 무결성

`docs/V800_BASELINE_MANIFEST.json`은 업로드 ZIP의 모든 원본 파일 해시입니다. `docs/V810_EXPECTED_CHANGES.json`은 허용된 원본 변경 세 파일과 이유입니다.

```bash
PYTHONPATH="$PWD/python" python python/audit_v800_preservation.py \
  --output-dir "$PWD/runs/preservation_recheck"
sha256sum -c PACKAGE_SHA256_MANIFEST_V8_1.txt
```

기존 `PACKAGE_SHA256_MANIFEST.txt`는 v8.0.0의 **역사적 원본 파일**로 보존한 것이며 현재 릴리스 검사에는 새 `PACKAGE_SHA256_MANIFEST_V8_1.txt`를 사용하십시오. 이 패키지의 `docs/` 아래 과거 버전 검증보고서도 현재 시험으로 오해하지 않도록, 이번 결과는 `docs/v8_1_validation/`에 분리했습니다.
