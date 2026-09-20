# 검증 보고서 — v7.7.5 F 제거형 Condensed-Phase NSGA-II

검증일: 2026-08-30

## 1. 정적 검사

- `python -m compileall -q python`: 통과
- 모든 shell 실행 스크립트 `bash -n`: 통과
- 활성 solver에 flame-progress `F`, OpenFOAM 및 기상 CFD 실행 경로가 없음을 확인
- MPS 명시적 PCG가 legacy 비대칭 multigrid로 조용히 바뀌지 않도록 startup/runtime validation 추가
- active MPS 경로에서 `scatter_add_`, `scatter_reduce_`, `index_add_`, `index_put_(accumulate=True)` 제거 확인
- MPS multigrid 예비 경로에는 CPU 왕복 fallback이 없음을 확인

## 2. 전체 단위·회귀시험

```text
28 passed
```

주요 검증 범위:

1. no-F solver/config 계약과 네 목적함수 mapping
2. NSGA-II constraint-domination, Pareto sorting, crowding 및 70+20+10 mating
3. MPS device/FP32 profile 검증
4. 대칭 Jacobi equilibration + correction-form PCG
5. explicit `PCG + legacy multigrid` 조합 거부
6. true-residual 수렴, row별 restart, iterative refinement 진단
7. 고정형 batch sum/max/quantile
8. dtype 독립 physical floor
9. mass-transfer 포화 BV 미분의 FP32 underflow 회귀시험
10. 직접 condensed-phase CPU FP32/FP64 smoke

원자료: `docs/validation_data/pytest_summary.txt`

## 3. 193×193 대표 전위계 수치시험

양·음극 면적이 각각 약 7.9%인 맞물린 빗형 전극과 `1e-4~0.5 S/m` 전도도장을 사용했다.

- 대칭성 gate: 통과
- 무작위 SPD Rayleigh probe 64개: 통과
- 평형화 후 active diagonal 오차: 0
- 알려진 해 CPU FP32 true relative residual: 약 `5.1e-7`
- 알려진 해 최대 전위오차: 약 `2.4e-5 V`
- 실제 method: `pcg_symmetric_equilibrated_correction_fp32_true_residual`

원자료: `docs/validation_data/mps_numerical_preflight_193_cpu_fp32.json`

## 4. CPU FP32–FP64 전위 비교

동일한 193×193 대표 전위문제에서:

- FP32 true relative residual: 약 `4.7e-7`
- FP64 true relative residual: 약 `1.5e-13`
- FP32–FP64 최대 전위차: 약 `4.8e-4 V`
- RMS 전위차: 약 `7.6e-5 V`
- RMS 차이를 단순 BV 지수증폭으로 환산한 값: 약 `1.0015×`

원자료: `docs/validation_data/potential_precision_193_comb_cpu.json`

## 5. 193×193 전체 nonlinear one-step

실제 grammar 후보 4개를 CPU FP32에서 MPS와 동일한 코드경로로 1 timestep(`2.5e-4 s`) 계산했다.

- successful: 4
- rejected: 0
- 최종 선형 true relative residual: 약 `6.2e-7~6.9e-7`
- nonlinear BV Robin: 모두 수렴
- global anode/cathode current gauge: 모두 수렴
- PCG fallback: 4개 모두 미사용

원자료:

- `docs/validation_data/nonlinear_193_four_case_one_step_cpu_fp32.json`
- `docs/validation_data/nonlinear_193_four_case_one_step_cpu_fp32.log`

## 6. 아직 필요한 검증

빌드 환경에는 Apple MPS 장치가 없다. 따라서 실제 Metal kernel 검증은 사용자의 M2 Pro에서 다음 명령이 `successful=4 rejected=0`으로 끝나야 완료된다.

```bash
bash tools/run_m2_mps_preflight.sh "$PWD/runs/m2_v775_preflight"
```

또한 다음은 아직 수행하지 않았다.

- M2 Pro 2초 full-horizon batch benchmark
- M2 Pro 100개 × 2세대 pilot
- A100 1,000개 × 최대 5세대 production
- 실험 기반 retained-water, transport, BV 및 solid-kinetics 보정

따라서 이 배포본은 수치구조와 CPU FP32 reference가 검증된 MPS 후보이며, 실제 M2 preflight 전에는 Apple MPS end-to-end 검증 완료로 표현하면 안 된다. 출력 형상도 실험 보정 전에는 `nominal-model recommendation`이다.
