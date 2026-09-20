# v7.9.3 변경내역 — Energy-to-Ignition objective

## 핵심 변경

기존 NSGA-II 제3 목적함수 `input_energy_j_at_2s = ∫_0^{2s} V I dt`를 제거하고 다음으로 교체했다.

\[
E_{ign}=\int_0^{t_{ign,cond}} V(t)I(t)\,dt
\]

활성 목적함수는 다음 네 개다.

1. `ignition_delay_s`
2. `area_undecomposed_fraction_at_2s`
3. `input_energy_j_to_ignition`
4. `current_congestion`

## 구현

- C++ FP64 solver가 매 timestep `V*I`를 trapezoidal integration한다.
- 응축상 점화 기준이 최초 충족된 timestep의 누적 에너지를 `inputElectricalEnergyToIgnition_J`로 저장한다.
- `inputElectricalEnergyAt2s_J` 및 `inputElectricalEnergyAtEvaluationTime_J`는 diagnostic으로 보존한다.
- 점화하지 못한 후보는 `E_ign`이 관찰되지 않은 censored case이다. 이때 evaluation-horizon energy를 NSGA-II 수치 placeholder로만 사용하며 ignition constraint 때문에 infeasible이다.
- 기존 v7.9.0/v7.9.2 결과의 `inputElectricalEnergyAt2s_J`를 점화에너지로 자동 재해석하지 않는다.
- `evaluate_area_matched_staggered_existing_run.sh`를 기존 run에 적용할 때 `recommended_design.npz`가 있으면 기존 추천 AI 형상 1개를 v7.9.3으로 자동 재평가하여 `E_ign`을 새로 계산한 뒤 staggered와 비교한다. 이는 기존 NSGA-II 선택을 다시 최적화하는 것이 아니라 이미 선택된 형상의 새 metric 재평가이다.

## Baseline 비교 출력 수정

Post-optimization area-matched staggered 비교는 이제 NSGA-II penalty가 더해진 값이 아니라 raw physical objective를 출력한다. `numerical_constraint_violation`과 `numerically_valid`를 별도 기록하므로, limiter/cap 위반 baseline에서 `1e10`급 penalty가 실제 Joule 단위 에너지처럼 표시되는 문제를 방지한다.

## 검증

- Python pytest: 44/44 passed
- C++17 FP64 build: passed
- C++ first-step ignition regression: `inputElectricalEnergyToIgnition_J == inputElectricalEnergyAt2s_J` for a one-step forced-onset test
