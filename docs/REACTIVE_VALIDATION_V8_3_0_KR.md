# ECSP v8.3.0 reactive reference 검증 보고서

## 판정

**`NOT FULLY VERIFIED`**

이 문서는 논문 식 골격을 구현한 단일-process CPU FP64 reference의 소프트웨어
검증 범위와 검증 불가능 범위를 구분한다. 최종 release ZIP과 결합한 evidence가
동결되기 전이므로 테스트 개수·통과 개수·실행 시간은 선기록하지 않는다. 최종
산출물의 JUnit/JSON/manifest와 일치하는 값만 패키징 단계에서 채워야 한다.

## 검증 대상

| 대상 | 확인하려는 계약 | paper/experimental validation 의미 |
|---|---|---|
| Tait Eq.(3) | `p(rho)`, analytic `c^2=dp/drho`, energy/temperature roundtrip | 식·구현 일관성만 확인; 실제 ECSP Tait 계수 검증 아님 |
| Tait `N≈1` | `N=1` 극한과 인접 FP64 값에서 `expm1` cold-energy 연속성 | cancellation 방지 확인; caloric closure는 여전히 가정 |
| conservative/primitive 변환 | rho, velocity, energy, lambda roundtrip과 invalid-state 거부 | 상태 변환 구현 확인 |
| WENO5-JS | 상수 보존, smooth stencil, trace별 정확한 5-cell 의존성 | 논문 WENO variant 재현이 아니라 선택한 scheme 검증 |
| WENO tiny epsilon | 극소 양의 epsilon에서 weight/face/flux 유한성 | overflow 방어 확인 |
| energy-gauge covariance | Tait reference energy offset 전후 density/momentum/progress flux와 energy flux 변환 일관성 | arbitrary reference-energy dependence 방지 확인 |
| HLL/FV | 방향별 flux, periodic telescoping, face-local fallback 계측 | characteristic/interface scheme 검증 아님 |
| SSPRK(3,3) | stage time, 3차 시간 적분 reference, transactional retry | 논문 RK tableau는 미공개 |
| reaction source | `rho*omega`, `rho*q*omega` 공통 extent 및 terminal tolerance | kinetics/Q의 실험 적합성 검증 아님 |
| source timestep | progress headroom, cooling thermal headroom, subnormal-rate unbounded 처리 | explicit-stage 안전성 확인 |
| electrical field | `sigma*|grad(V)|^2`, nonnegative Joule, 음의 total source 거부 | potential PDE/전도도/BV closure 검증 아님 |
| Eq.(2) solid | periodic conduction integral, alpha/heat bookkeeping, thermal headroom | Eq.(1)–Eq.(2) interface coupling 검증 아님 |
| material level set | advection, sign convention, optional reinitialization drift | phase/EOS/jump coupling 검증 아님 |
| Species front | threshold/ray interpolation, birth/death/ambiguity bookkeeping | fixed `+x` ray 진단; true local-normal speed 검증 아님 |
| configuration/provenance | duplicate/unknown key 거부, strict fail-closed, runtime snapshot audit | paper missing input을 숨기지 않는 계약 확인 |
| source callback | read-only bytes-backed state/geometry, strict JSON snapshot, retry accounting | callback 물리 정확성은 self-attested이며 인증하지 않음 |
| result serialization | immutable arrays/metadata, deterministic JSON/NPZ hash, atomic per-file replace | 두 파일 전체가 하나의 filesystem transaction은 아님 |
| resource guard | history/metadata/working-set preflight와 runtime 누적 gate | OS RSS/allocator peak 보증 아님 |
| legacy regression | v8.2.1 parent 기능과 허용된 v8.3 추가분 비교 | reactive physics validation이 아니라 보존 회귀 |

## 수치 안전성의 구현 상태

### Tait와 WENO

- Tait cold energy는 `N=1`에서 analytic logarithmic limit를 쓰고, 그 주변에서는
  `expm1((N-1)*log(rho/rho0))/(N-1)`을 써 cancellation을 줄인다.
- WENO 입력은 conservative cell average에서 계산한 primitive
  `rho,u,v,T,lambda`다. 이 값은 엄밀한 primitive cell average가 아니며
  characteristic reconstruction도 아니다.
- left/right trace 각각의 exact five-cell stencil로 독립 affine normalization한다.
- WENO-JS nonlinear weight는 denominator 최솟값에 대한 비율로 구성해 tiny
  positive epsilon에서 inverse-square overflow를 피한다.
- face를 conservative state로 되돌린 다음 HLL을 계산하므로 Tait reference-energy
  gauge offset 때문에 density/momentum/progress flux가 바뀌는 것을 방지한다.
- fallback의 `maximum_abs_state_correction`은 raw conservative components의
  최대값이다. 성분마다 물리 단위가 달라 무차원 수렴오차로 해석할 수 없다.

### source와 stage

- flow와 solid reaction은 진행 increment/remaining headroom 외에 흡열 부호에서
  temperature reject floor까지 남은 thermal headroom도 timestep 제한에 넣는다.
- 아주 작은 양의 rate 때문에 비율이 `+Inf`가 되면 “제한 없음”으로 해석하고,
  NaN 또는 nonpositive limit만 오류로 처리한다.
- `joule_heat`는 비음수여야 하며 `joule+electrochemical` total도 비음수여야 한다.
  이 API는 signed electrical cooling을 지원하지 않는다.
- SSPRK stage에서 안정한계가 감소하면 기존 state/budget을 commit하지 않고 더
  작은 `dt`로 retry한다. proposed limiter와 accepted retry limiter를 구분한다.
- callback은 pure/deterministic해야 한다. solver state는 rollback되지만 callback이
  외부 시스템에 남긴 side effect는 rollback할 수 없다.

## 보존 수지 판정

NaN 또는 Inf state, source, reduction, accumulator와 residual은 fail-closed한다.
반면 유한 Euler/solid closure residual에는 현재 calibrated absolute tolerance와
scale-aware relative tolerance가 없다. 따라서 결과 metadata는 유한 residual을
**`DIAGNOSTIC_ONLY_NOT_ENFORCED`**로 기록한다. synthetic periodic telescoping
테스트의 tolerance를 실제 ECSP 형상·시간적분 전체의 production acceptance
criterion으로 전용하지 않는다.

이 제한만으로도 `geometry_ranking_eligible=false`이며, conservation field가
존재하거나 유한하다는 이유로 형상 순위를 신뢰할 수 없다.

## resource preflight 검증 범위

- history estimator는 `maximum_steps`, front-history stride, threshold별 ray 자료,
  step diagnostics와 세 SSPRK stage의 electrical metadata를 계산한다.
- prescribed source는 canonical record의 recursive Python-object size와 compact
  UTF-8 JSON bytes를 측정해 per-record charge를 만든다.
- callback metadata는 호출 전 정확한 크기를 알 수 없어 baseline planning을 하고,
  매 반환 record를 재측정해 accepted-history 누적 한도를 넘으면 실패한다.
- working-set estimator는 `nx*ny`, 격자당 versioned scratch allowance, 시간 배열,
  configured history budget와 NPZ staging allowance를 주요 grid allocation 전에
  검사한다.

이는 OOM 위험을 줄이는 planning guard이지 Python/NumPy allocator fragmentation,
제3자 library workspace, OS RSS, memory mapping 또는 순간 peak를 보증하는 측정이
아니다. 최종 deployment 환경에서는 별도의 peak-RSS/VRAM 측정이 필요하다.

## 구조적으로 검증되지 않은 부분

| 항목 | 상태 | 결과에 미치는 영향 |
|---|---|---|
| phi→phase/EOS/property/source/jump 연결 | 미구현 | material interface를 가진 multiphase solution이 아님 |
| Eq.(1)↔Eq.(2) 열·질량·운동량 coupling | 미구현 | 두 상태가 전체 격자에서 독립적으로 전진 |
| electrical heat partition | Eq.(1) only | Eq.(2) 중복 가열은 피하지만 논문 기반 partition law가 아님 |
| true normal front tracking | 미구현 | fixed positive-`x` ray 결과는 진단용이며 다중 교점 제외 |
| concrete corrected BC adapter | 미구현 | convergence/current balance/BV residual을 source metadata가 인증하지 못함 |
| integrated MPI solver | 미구현 | row/halo API만 있고 rank 1/2/4/8 time-loop parity 없음 |
| reactive CUDA/sm_80/A100 | 미구현 | batch 1/32/64 CPU–GPU parity와 성능 측정 불가 |
| paper quantitative case | 입력 부족 | 1804.8 K와 3.6094 mm/s 독립 재현 불가 |
| experimental validation | 미수행 | 실제 ECSP predictive accuracy를 주장할 수 없음 |

## 최종 evidence 기록 규칙

최종 검증 후에만 다음을 release evidence에서 복사한다.

1. reactive 전용 pytest/JUnit의 pass/fail/skip 수
2. 전체 legacy+reactive pytest/JUnit의 pass/fail/skip 수
3. `tools/validate_reactive_v8_3.py` check 수와 50-repeat determinism 결과
4. grid/time convergence JSON의 schema, verdict와 개별 check
5. assumed smoke JSON/NPZ hash 및 parent-preservation audit
6. 최종 source-tree digest, release ZIP SHA-256와 fresh-extraction manifest 검증

미실행 CUDA/MPI/A100 항목은 skip을 pass로 세지 않고
`NOT_IMPLEMENTED_NOT_TESTED` 또는 `NOT_TESTED_NO_RUNTIME/HARDWARE`로 유지한다.

