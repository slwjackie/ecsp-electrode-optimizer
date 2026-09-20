> **역사적 참고문서:** 이 문서는 v7.8.1 포팅 기록입니다. 현재 production semantics는 v7.9.0의 `SURFACE_CONTACT_MULTICOMPONENT_V7_9_0_KR.md`를 따릅니다.

# v7.8.1 C++ CPU FP64 검증 보고서

## 1. 자동시험

- Python unit/regression tests: **32/32 통과**
- C++17 native release build/version smoke: 통과
- Clang `-Wall -Wextra -Wpedantic` 검사: 치명적 경고 없음; 남은 경고는 명시적 index signedness 변환
- AddressSanitizer + UndefinedBehaviorSanitizer short evolving-state case: 통과
- Python FP64 ↔ C++ FP64 port-parity gate: 통과
- 193×193 서로 다른 4개 C++ FP64 one-step case: **4/4 성공, 0 rejection**

현재 컨테이너는 x86_64 Linux이다. M2/arm64 native build와 실제 시간은 사용자 Mac의 preflight/benchmark로 최종 확인해야 한다.

## 2. Python FP64 ↔ C++ FP64 교차검증

조건:

- 동일 deterministic geometry `G000_T00_b5964f_V000`
- grid 33×33
- `dt=2.5e-4 s`
- `t_end=0.003 s` — 12 timestep, 중간 electrical update 포함
- 양쪽 모두 IEEE-754 FP64

| metric | relative error |
|---|---:|
| peakMaximumTemperature_K | 6.84e-8 |
| peakCurrent_A | 1.05e-6 |
| peakCurrentCongestion | 1.96e-7 |
| inputElectricalEnergyAt2s_J | 1.04e-6 |
| areaAveragedUndecomposedFractionAt2s | 0 |
| finalEffectiveResistance_ohm | 1.05e-6 |
| finalNonlinearRobinGaugeOffset_V | 1.99e-7 |

최대 상대오차는 약 **1.05e-6**이다. 자동 재현:

```bash
bash tools/validate_cpp_vs_python_fp64.sh "$PWD/runs/cpp_python_fp64_parity"
```

이는 짧은 evolving-state 포팅 검사다. full 2 s 실험 validation을 뜻하지 않는다.

## 3. 193×193 C++ FP64 preflight

서로 다른 네 grammar variant를 193×193에서 initial nonlinear BV–potential solve와 한 timestep까지 계산했다.

```text
successful = 4
rejected   = 0
```

대표 최종 relative potential residual은 약 `9.3e-10 ~ 1.0e-9`였고, 모든 후보가 설정된 anode/cathode current-balance 기준을 통과했다.

자동 재현:

```bash
bash tools/run_m2_cpp_fp64_preflight.sh "$PWD/runs/m2_cpp_fp64_preflight"
```

## 4. C++ sanitizer 검증

동일 33×33 evolving-state 입력을 다음 옵션으로 별도 빌드해 실행했다.

```text
-fsanitize=address,undefined
-fno-omit-frame-pointer
```

out-of-bounds, use-after-free, integer/float undefined behavior 또는 leak 보고 없이 종료했다.

## 5. short-horizon runtime characterization

검증 컨테이너에서 193×193 대표 V000의 측정값은 다음과 같았다.

| horizon | electrical solves | C++ process wall time |
|---:|---:|---:|
| 0.00025 s | 1 | 약 17 s |
| 0.025 s | 10 | 약 20 s |
| 0.2 s | 80 | 약 66 s |

초기 nonlinear BV–potential solve가 가장 비싸고, warm-start 이후 update는 훨씬 싸다. 따라서 전체 시간을 단순히 `0.2 s 측정값 × 10`으로 계산하면 과대추정된다. 배포 benchmark는 one-step 비용을 분리한 뒤 post-startup interval 비용만 2 s까지 외삽한다.

M2의 실제 SIMD, memory bandwidth, candidate parallelism 및 thermal throttling은 이 컨테이너와 다르므로 사용자 Mac의 다음 결과를 기준으로 한다.

```bash
bash tools/benchmark_m2_cpp_fp64.sh "$PWD/runs/m2_cpp_fp64_benchmark"
```

## 6. 검증한 execution boundary

- Python에서 physics-grid mask 생성과 제조성 검사를 후보당 1회 수행
- binary mask와 flat config를 C++ process에 1회 전달
- electrical/BV/species/thermal/decomposition/time integration은 전부 C++ process 내부 수행
- Python으로는 scalar metrics와 diagnostics JSON만 복귀
- timestep별 callback 또는 MPS/CPU transfer 없음

## 7. 남는 제한

- full 2 s Python↔C++ 전수 parity는 계산비용 때문에 모든 형상에 대해 수행하지 않았다.
- C++ 2 s 단독 수치 완주는 대표형상으로만 확인하며, 여러 topology/voltage에 대한 regression은 실제 연구 run에서 축적해야 한다.
- C++ 포팅은 LP/PVA/GLY/BA parameter calibration을 제공하지 않는다.
- 기상 CFD, flame spread 및 extinction은 모델 범위 밖이다.
