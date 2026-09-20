# ECSP v8.3.0 논문 결과 재현 보고서

## 결론

공개 논문만으로 정량 case를 닫을 수 없어 **논문 결과 재현은 수행 불가**로
판정했다. 이는 코드가 예외를 내서 숨긴 실패가 아니라, strict preflight가
미공개 필수 입력을 명시적으로 거부한 결과다. runnable assumed 설정은 solver
수치 검증용 synthetic case이며 아래 논문값에 맞춰 보정하지 않았다. 현재 solver
자체도 Eq.(1)/Eq.(2)/material interface가 완전히 결합된 논문 case가 아니므로,
필수 숫자 일부만 추정해 채우는 것으로 재현 가능 상태가 되지 않는다.

## 정량 비교

| 검증량 | 논문값 | 독립 재현값 | 절대오차 | 상대오차 | 판정 |
|---|---:|---:|---:|---:|---|
| 계산 평균 온도 | 1804.8 K | N/A | N/A | N/A | `BLOCKED_MISSING_INPUTS` |
| 실험 평균 온도 | 1788.93 K | 실험 재수행 없음 | N/A | N/A | reference only |
| 계산 평균 후퇴율 | 3.6094 mm/s | N/A | N/A | N/A | `BLOCKED_MISSING_INPUTS` |
| 실험 평균 후퇴율 | 3.3696 mm/s | 실험 재수행 없음 | N/A | N/A | reference only |
| 연소 시작 위치 | 그림으로 제시 | N/A | N/A | N/A | 원시 좌표/criterion 없음 |
| 시간별 Species 분포 | 그림으로 제시 | N/A | N/A | N/A | 원시 field/time 없음 |
| 추진제 경계 이동 | 그림/설명으로 제시 | N/A | N/A | N/A | initial/interface closure 없음 |
| Joule/electrochemical/reaction heat 상대 영향 | 정성 그림/설명 | N/A | N/A | N/A | V, sigma, H rate, Q가 없음 |

N/A를 0으로 바꾸거나, assumed 상수를 조절해 3.6094 mm/s를 맞추거나, 기존
`Delta area/(front length*Delta t)`를 논문의 Species 전선 속도로 대체하지 않았다.

## 재현을 막는 직접 원인

1. Tait A/B/N/rho0와 caloric EOS가 없어 pressure-temperature-energy가 닫히지 않는다.
2. `dot(omega)`, Q, A_alpha, E_alpha, f(alpha), Q_r가 없다.
3. 논문의 H 단위와 Eq.(2) r_alpha 단위가 인쇄 식의 체적 에너지 차원과 맞지 않는다.
4. V와 sigma의 raw 2-D field, 또는 이를 구할 potential PDE/BC가 없다.
5. initial/external/material-interface conditions와 ghost-fluid jump가 없다.
6. WENO/Riemann/RK/CFL/source split/reinitialization/overlap-grid가 유일하게 정해지지 않는다.
7. 후퇴율의 Species threshold, subcell 위치, front 대응, 회귀 시간구간이 없다.
8. paper figure를 만든 raw field와 실험 time series가 없다.

## 현재 구현의 구조적 재현 장벽

공개 입력 부족과 별개로, 현재 v8.3 reference에는 다음 구현 간극이 남아 있다.

- `material_level_set`은 수송되지만 phase/EOS/물성 선택, source mask,
  ghost-fluid 또는 interface jump condition에 연결되지 않는다.
- Eq.(1) Euler와 Eq.(2) condensed thermal/alpha는 같은 전체 격자에서 각각
  전진하며 열·질량·운동량을 교환하지 않는다.
- 전기 열은 double counting을 피하기 위해 Eq.(1)에만 투입된다. 이것은 논문에서
  유도된 interface partition law가 아니다.
- 전선 속도는 fixed positive-`x` ray 교점의 이동 진단이며 true local-normal
  propagation speed가 아니다. 다중 교점 ray도 제외한다.
- row partition/halo API는 있으나 실제 MPI time loop가 없고, reactive CUDA/A100
  backend도 없다.
- `geometry_ranking_eligible=false`이며 concrete BC adapter의 convergence,
  current balance 및 BV residual gate가 없다.

따라서 저자의 입력 deck을 확보하더라도 먼저 interface coupling, 재료별 EOS/물성,
source partition과 동일한 postprocessing을 구현·검증해야 paper result reproduction을
시도할 수 있다.

## 구현 검증과 논문 재현의 구분

v8.3.0의 해석해·보존성·수렴성 시험은 구현이 선택한 CPU FP64 reference의 수치
성질을 확인한다. WENO는 보존 셀 평균에서 얻은 primitive 값을 trace별 5-cell
normalization으로 재구성하고 HLL/SSPRK(3,3)을 쓰는 **비논문 선택**이다. Tait
`N≈1` `expm1` 처리, tiny-epsilon stable weight, thermal headroom, 음의 전기 총열원
거부는 소프트웨어 안전성 검사이지 특정 ECSP 물성 또는 예측력 검증이 아니다.

비유한 conservation residual은 실패하지만 유한 residual은 calibrated
absolute/scale-aware tolerance가 없어 diagnostic-only다. history/working-set
preflight도 versioned planning bound이며 실제 RSS/allocator peak 보증이 아니다.
따라서 이 시험들이 통과해도 논문 재현 성공 또는 production ranking 적합성을
의미하지 않는다. 다음 증거를 확보한 뒤에만 위 표의 N/A를 실제 재현값으로
바꿀 수 있다.

- 저자 입력 deck 또는 supplementary data
- 측정 Tait/caloric/thermal/electrical properties
- 반응 및 전기화학 kinetics와 단위가 닫힌 열원
- initial/boundary/interface/overlap-grid 명세
- potential/conductivity 및 temperature/Species/front raw histories
- 후퇴율 postprocessing script 또는 동일한 명세

## 최종 판정

`NOT FULLY VERIFIED`

논문 식 구조의 reference implementation과 명시적 가정·fail-closed preflight는
제공하지만, 공개 정보와 현재 미결합 구조만으로 paper-faithful quantitative
reproduction을 검증할 수 없다. 이 상태에서 `3.6094 mm/s` 또는 `1804.8 K`에
맞춘 계수 조정은 재현 검증이 아니라 calibration이며 수행하지 않았다.
