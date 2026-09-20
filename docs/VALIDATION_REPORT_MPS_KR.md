# v7.7.5 MPS 수치검증 보고서

## 검증된 범위

- 전체 pytest: 28/28 통과
- MPS 프로필 config: PCG + Jacobi, rtol `1e-6`, atol `1e-8`
- 대칭 Jacobi equilibration 및 correction-form PCG
- true-residual convergence와 iterative refinement
- 상대 곡률 guard, 최량 iterate 복귀 및 row별 restart
- MPS-resident 최소잔차 fallback
- local BV 전위강하 변수화 및 global gauge balance
- MPS-safe batch sum/max/quantile
- dtype-independent physical floors
- no-F condensed-phase 계약과 NSGA-II 핵심 로직

## 193×193 대표 빗형 선형계

전극 면적은 양·음극 각각 약 7.9%, 전도도는 설정 범위 `1e-4~0.5 S/m`를 사용했다.

CPU FP32 preflight 결과:

- 대칭성 gate: 통과
- 무작위 SPD probe: 통과
- 평형화 대각 오차: 0
- 알려진 해 true relative residual: 약 `5.1e-7`
- solver method: `pcg_symmetric_equilibrated_correction_fp32_true_residual`
- fallback: 미사용

## FP32/FP64 전위 비교

동일한 193×193 계에서:

- FP32 true relative residual: 약 `4.7e-7`
- FP64 true relative residual: 약 `1.5e-13`
- 최대 전위차: 약 `4.8e-4 V`
- RMS 전위차: 약 `7.6e-5 V`
- RMS 전위차로부터 계산한 단순 BV 전류 증폭 추정: 약 `1.0015×`

원자료: `docs/validation_data/potential_precision_193_comb_cpu.json`

## 전체 nonlinear one-step

193×193 자유형 후보 4개, FP32, 1 timestep(`0.00025 s`) CPU reference:

- successful: 4
- rejected: 0
- 최종 선형해 true residual: 약 `6.2e-7~6.9e-7`
- 실제 method: 모두 equilibrated correction-form PCG
- nonlinear Robin 및 global gauge balance: 모두 수렴

이 시험은 MPS와 동일한 FP32 물리경로를 CPU에서 실행한 reference다.

## 미검증 범위

빌드 환경에는 Apple MPS 장치가 없으므로 실제 Metal 커널의 end-to-end 결과는 생성하지 못했다. 사용자의 M2 Pro에서 `tools/run_m2_mps_preflight.sh`가 수치 gate와 실제 4형상을 모두 통과해야 MPS 실행검증이 완료된다.

또한 2초 전체 horizon 및 100×2/1000×5 계산은 빌드 환경에서 실행하지 않았다. 물성은 여전히 `nominal_unvalidated`이며 정량 예측에는 실험 보정이 필요하다.

## 포함된 원자료

- `docs/validation_data/mps_numerical_preflight_193_cpu_fp32.json`
- `docs/validation_data/potential_precision_193_comb_cpu.json`
- `docs/validation_data/nonlinear_193_four_case_one_step_cpu_fp32.json`
- `docs/validation_data/nonlinear_193_four_case_one_step_cpu_fp32.log`
- `docs/validation_data/pytest_summary.txt`
