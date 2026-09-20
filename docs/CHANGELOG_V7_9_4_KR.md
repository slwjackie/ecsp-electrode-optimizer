# ECSP v7.9.4 변경사항 — 최소 점화전압 목적함수

## 핵심 변경

v7.9.3의 제3 NSGA-II 목적함수 `input_energy_j_to_ignition`을 `minimum_ignition_voltage_V`로 교체했다.
따라서 production objective vector는 다음 네 항목을 모두 최소화한다.

\[
\min\left[
 t_{\mathrm{ign,cond}}(V_{\mathrm{ref}}),\;
 U_\alpha(2\,\mathrm{s};V_{\mathrm{ref}}),\;
 V_{\min,\mathrm{ign}},\;
 C_J(V_{\mathrm{ref}})
\right]
\]

기본 reference voltage는 260 V이다. 점화까지의 전기에너지 `E_ign`과 2 s 누적 입력에너지는 삭제하지 않고 진단값으로만 저장한다.

## V_min 산출 방식

`V_min`은 geometry genome의 직접 변수로 두지 않는다. 각 형상에 대해 동일한 C++ FP64 condensed-phase solver를 여러 전압에서 실행하여 점화 threshold를 찾는다.

기본 설정은 다음과 같다.

```yaml
minimum_ignition_voltage_search:
  enabled: true
  lower_bound_V: 20.0
  upper_bound_V: 260.0
  tolerance_V: 5.0
  maximum_bisection_iterations: 8
  right_censor_objective_penalty_V: 50.0
  invalid_search_objective_penalty_V: 100.0
  stop_successful_trials_at_ignition: true
  reported_value: conservative_upper_igniting_bracket
```

1. 260 V reference run은 목적함수 1, 2, 4 계산과 V_min의 상한 bracket으로 동시에 재사용한다.
2. 20 V lower-bound trial을 계산한다.
3. 20 V에서 비점화, 260 V에서 점화이면 bisection으로 점화/비점화 경계를 좁힌다.
4. 최종 bracket에서 점화가 확인된 쪽의 전압(upper igniting bound)을 보수적인 `V_min` objective로 사용한다.
5. 20 V에서도 점화하면 `V_min <= 20 V` left-censored로 기록한다.
6. 260 V에서도 점화하지 않으면 `V_min > 260 V` right-censored로 기록하고 ignition infeasibility를 유지한다.
7. trial이 numerical cap/limiter 기준을 위반하거나 sampled ignition response가 비단조이면 V_min search를 invalid로 표시한다.

## 계산시간 최적화

V_min threshold trial에서 점화가 발생한 경우 C++ solver는 그 시점에 즉시 종료할 수 있다. 비점화 trial은 설정된 2 s horizon까지 계산한다. 후보 간 V_min trial은 기존 candidate-level subprocess 병렬화 구조를 이용한다.

20–260 V, 5 V tolerance에서는 전형적으로 한 형상당 reference 1회 + lower-bound 1회 + 약 6회의 bisection이 필요하므로, v7.9.3보다 physics 계산량이 크게 증가한다.

## 기존 run 비교

기존 v7.9.0–v7.9.3 추천형상은 V_min이 저장되어 있지 않다. `evaluate_area_matched_staggered_existing_run.sh`를 사용하면 가능한 경우 기존 추천 mask를 v7.9.4 solver로 재평가하여 V_min을 산출하고, area-matched hidden-bus staggered와 동일 objective contract로 비교한다. NSGA-II 전체를 다시 돌리지는 않는다.

단, 새 V_min 목적함수로 최종 Pareto front와 최종 추천형상을 다시 최적화하려면 NSGA-II 전체를 v7.9.4로 재실행해야 한다.
