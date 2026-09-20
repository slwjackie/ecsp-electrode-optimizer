# ECSP v8.3.0 논문–코드 추적성 및 재현 한계

## 판정 범위

대상은 「전기 제어 고체 추진제 연소 메커니즘의 수치적 해석」의 PDF 10쪽
전체(학술지 쪽수 31–40)와 식 (1)–(4), 그림 설명 및 참고문헌이다. 공개 논문은
지배식의 골격은 제시하지만, 계산을 유일하게 닫을 물성·초기/경계조건·반응계수와
수치 알고리즘 세부를 제공하지 않는다. 그러므로 v8.3.0은 다음을 분리한다.

- `paper_faithful`: 논문에 없는 BV 반응을 사용하지 않는다. strict 설정에서는
  필수 미기재 값이나 알고리즘을 만나면 실행 전에 실패한다.
- `ecsp_extended`: 논문 기반 코어에 v8.2.1 surface-footprint BV 열원 callback을
  명시적으로 연결하는 비논문 확장이다. callback이 없으면 실행하지 않는다.
- 기존 v8.2.1 condensed propagation은 `legacy` 기능으로 보존하지만 논문
  level-set 또는 논문 후퇴율이라고 부르지 않는다.

## 논문–현재 코드 차이와 구현 상태

| 구현 항목 | 논문 페이지/식 | 논문의 정의 | v8.2.1 현재 코드 | v8.3.0 변경 | 재현 가능 여부 |
|---|---:|---|---|---|---|
| 지배방정식 | p.33, Eq.(1) | 2-D 압축성 무점성 반응 Euler 보존식 | 온도·농도·반응 진행 중심 condensed FV; 유동 보존식 없음 | `ecsp_reactive/core.py`에 5개 보존변수와 x/y flux | 식 골격 구현 가능, case 재현 불가 |
| 보존변수 | p.33, Eq.(1) | `[rho,rho*u_x,rho*u_y,rho*E,rho*lambda]^T` | 해당 상태 없음 | 마지막 축 5개 FP64 배열로 구현 | 가능 |
| 원시변수 | Eq.(1)에서 암시 | rho,u,v,p,T,lambda가 필요 | T·species 위주 | EOS를 통한 양방향 변환 | caloric EOS가 없어 가정 필요 |
| 질량·운동량·에너지 | p.33, Eq.(1) | conservative flux | 미해석 | FV flux divergence | 가능 |
| Species/lambda | p.33, p.35–38 | 반응 전 0, 후 1; `rho*lambda` 수송·생성 | global progress/alpha와 별도 inventory | Euler lambda를 별도 변수로 구현 | rate closure 미기재 |
| Euler 반응률 | p.33, Eq.(1) | source `rho*dot(omega)` | v8.2 kinetics와 의미가 다름 | caller-supplied one-step Arrhenius interface | 계수·함수 미기재; 가정 필요 |
| 반응열 Q | p.33, Eq.(1) | energy source `rho*Q*dot(omega)` | nominal condensed channel heat | 진행 source와 정확히 동일 extent를 쓰는 J/kg source | Q 미기재 |
| 전기화학 열 H | p.33, Eq.(1) | 논문은 H를 J/mol로 설명하면서 `rho*H` 기재 | BV reaction별 열원 존재 | W/m3의 명시적 field/callback만 허용 | 인쇄 단위로는 차원 불일치; 그대로 계산 불가 |
| Joule heating | p.33, Eq.(1) | `sigma[(V_x)^2+(V_y)^2]` | corrected FV/J·E modes | `joule_heating_sigma_e2`로 식 그대로 계산 | V,sigma field 미공개 |
| 전위 방정식 | p.32–33 설명 | 실험 전도도/전위 분포를 사용했다고 서술 | potential PCG + nonlinear surface Robin/BV | paper 모드는 prescribed field만, extended는 기존 callback | PDE·BC 미기재; paper solver로 해석 불가 |
| 전기전도도 | p.32–33 설명 | 측정값 사용 언급 | nominal composition/T dependent | 모든 field/값에 provenance 요구 | 원자료 미공개 |
| 고체 열전도 | p.34, Eq.(2) | `rho*C*T_t=k laplacian(T)-r_alpha*Q_r` | nonlinear condensed heat equation | 전체 격자의 독립 `solid.py` Eq.(2) reference | k,C,rho, BC 및 Eq.(1) 결합 미기재 |
| 열분해 alpha | p.34, Eq.(2) | `r_alpha=dalpha/dt=A_alpha*f(alpha)*exp(-E_alpha/RT)` | v8.2 two-channel alpha | 전체 격자에서 Euler lambda와 별도 상태로 전진 | A,E,f 및 phase/interface closure 미기재 |
| Eq.(2) 단위 | p.34, Eq.(2) | r_alpha를 `dalpha/dt`라 쓰면서 surface molar-rate 단위 표기 | 별도 nominal closure | 차원 보정 closure를 `ASSUMED_NOT_FROM_PAPER`로만 제공 | 원문만으로 결정 불가 |
| Tait EOS | p.34, Eq.(3) | `p=B[(rho/rho0)^N-1]+A` | 미사용 | 식 그대로 구현 | 상수 미기재 |
| 음속 | p.34, Eq.(3) | `c^2=(BN/rho0)(rho/rho0)^(N-1)` | 미사용 | 식과 `dp/drho` 검증 구현 | 상수 미기재 |
| caloric EOS | 미기재 | 에너지에서 T를 얻는 관계 필요 | v8.2 cp(T) table | Tait cold energy + constant-cv를 명시적 비논문 closure로 격리 | 논문만으로 불가 |
| 주변 매질 EOS | 미기재 | 다중재료이면 별도 EOS 필요 | 해당 유동 매질 없음 | ideal gas는 shock-tube 검증 전용으로만 제공 | 논문 case에는 사용 불가 |
| material level set | p.34, Eq.(4) | `phi_t+u*phi_x+v*phi_y=0` | alpha=0.5 진단 distance 재구성 | 반응 변수와 별도인 transported scalar | 수송은 구현; phase/EOS/interface 조건과는 미결합 |
| level-set 부호 | p.34 | 내부 음수, 외부 양수, 경계 0 | 진단 distance에 별도 의미 | `phi<0` material interior로 고정 | 가능 |
| 수송속도 | p.34, Eq.(4) | Euler의 u_x,u_y | gas/material velocity 미해석 | 매 RK step의 Euler 원시속도 사용 | 가능하나 interface closure 필요 |
| 재초기화 | p.34 설명 | 주기적으로 수행한다고만 서술 | 매 시점 alpha contour 재구성 | optional Sussman PDE; 사용 시 면적 drift 기록 | 방법·주기 미기재, 가정 필요 |
| 계면 경계조건 | p.34–35 | 적절한 material-interface 처리/ghost cell 언급 | 해당 유동 계면 없음 | runtime은 `PASSIVE_SINGLE_EOS`; phi가 phase/EOS/물성/source mask/jump를 선택하지 않음 | jump condition 미기재; paper case 미폐쇄 |
| WENO | p.34 | WENO 공간 차분 | 해당 Euler WENO 없음 | 보존 셀 평균에서 계산한 primitive `rho,u,v,T,lambda`를 component-wise FV WENO5-JS로 재구성 | primitive 값은 엄밀한 primitive cell average가 아니며 차수·variant·characteristic 처리 미기재; 가정 |
| Riemann/flux split | 미기재 | 보존 flux에 필요 | 해당 없음 | HLL + invalid reconstructed face의 국부 1차 HLL fallback | 전부 가정, fallback 계측 |
| Runge–Kutta | p.34 | 3차 RK | 기존 explicit time stepping | SSPRK(3,3) | tableau 미기재; 가정 |
| CFL | 미기재 | 안정 시간간격 필요 | thermal CFL 별도 | x/y wave speed 합 + solid/level-set 제한의 최솟값 | 수치값 미기재; 설정·가정 |
| source coupling | 미기재 | flux와 heat/reaction 결합 필요 | condensed step contract | 각 SSPRK stage에서 source 재평가 | 가정 |
| 외부/초기 BC | 미기재 | 계산을 닫는 데 필수 | 기존 condensed BC와 의미 상이 | outflow/periodic 명시 선택 | paper case 재현 불가 |
| positivity 처리 | 미기재 | rho,p,T와 lambda 유효성 필요 | cap/floor 및 fail-closed 혼재 | cell state는 무클리핑 reject, face fallback은 횟수와 mixed-unit raw conservative 최대보정 기록 | 알고리즘 가정; 최대보정은 무차원 norm 아님 |
| MPI decomposition | p.35 | subdomain/virtual cell과 매 유동 step 통신 | candidate process pool, MPI 없음 | mpi4py row partition + 3-row WENO halo 보조 API; solver는 single-process만 허용 | time loop 미통합, halo·topology 미기재, 현 host runtime 없음 |
| ghost/virtual cell | p.35 | 두 1-D primitive arrays 통신 언급 | 유동 ghost 없음 | serial/MPI outflow halo와 Sendrecv 구현 | exact paper ordering 불명 |
| cell-vertex/overlap grid | p.35 | cell-vertex를 쓰다가 overlapping grid로 변경 | cell-centred condensed grid | 이번 reference는 cell-centred FV; 논문 방식이라고 주장하지 않음 | 세부 미기재로 미구현 |
| 전극 contact 의미 | 그림/설명 + ECSP 요구 | 상·하부 전극으로 전기 가열 | v8.2.1에서 top-surface overlay로 수정됨 | full propellant domain + separate anode/cathode masks 유지 | 코드 계약 검증 가능 |
| BV footprint | 논문에는 BV 없음 | 해당 없음 | v8.2.1에서 전체 footprint 적용 | extended callback에서만 보존; paper 모드 금지 | 논문 검증 대상 아님 |
| onset 이후 전기 | 논문은 onset cutoff를 정의하지 않음 | 열원이 Eq.(1)에 포함 | legacy propagation은 전기/BV 중단 | 새 reactive solver는 모든 stage에 source provider 호출 | 논문 field가 없어 정량 재현 불가 |
| 후퇴 전선 | p.35–38 | Species 0→1 전선의 위치 변위 | `delta area/(front length*dt)` legacy metric | 고정 `+x` ray의 subcell contour displacement | true local-normal speed가 아니며 다중 교점 ray 제외; contour/대응 규칙 미기재 |
| threshold | 미기재 | Species 전선만 설명 | 0.5 진단 | 0.3/0.5/0.7 모두 비논문 민감도로 기록 | paper 값 재현 불가 |
| 격자 연구 | p.34 | 50²/100²/150²/200², 100² 선택 | 다른 격자 사용 | 같은 격자군 실행 도구 + synthetic convergence | paper error norm/time step 없음 |
| 온도 비교 | p.37 | 실험 평균 1788.93 K, 계산 1804.8 K | 다른 모델/조성 결과 | 보고서에 숫자만 reference로 보존 | raw 위치·시간·평균법 없음 |
| 후퇴율 비교 | p.38 | 수치 3.6094 mm/s, 실험 3.3696 mm/s | legacy proxy 출력 | normal displacement metric으로 분리 | raw front series/fit 구간 없음; exact reproduction 불가 |
| 조성 | p.32 | LP 27.37%, PVA 10%, W 5%, glycerol 5%, H3BO3 2%, water 50.63% | v8.2 default 조성과 다름 | paper metadata로 기록, 물성으로 임의 변환하지 않음 | 조성만 재현 가능 |

## 식의 차원 해석

Eq.(1)의 flux 차원은 `E`가 J/kg인 specific total energy일 때만 일관된다.
논문 설명의 모호한 에너지 단위를 코드가 문자 그대로 사용하지 않는다.
`rho*Q*dot(omega)`는 Q[J/kg], dot(omega)[1/s]일 때 W/m3이다. 반면 논문이
H를 J/mol로 정의한다면 `rho*H`는 W/m3가 아니므로, 다음처럼 molar rate가
추가된 측정/외부 closure 없이는 실행할 수 없다.

`q_echem''' = H_echem [J/mol] * n_dot_echem''' [mol/(m3*s)]`

Eq.(2) 역시 alpha가 무차원이고 r_alpha가 1/s라면 차원적으로 닫히는 source는
`-rho*Q_r*r_alpha`이다. 공개 식의 surface molar-rate 단위와 `Q_r[J/kg]`를
동시에 유지하면 차원이 맞지 않는다. 이 보정은 코드에서 비논문 가정으로
표시되며 strict paper 설정에서는 거부된다.

## 코드가 추가한 수치 안전장치와 그 한계

| 조치 | 구현 목적 | 검증·해석 한계 |
|---|---|---|
| Tait cold energy의 `N=1` 극한 및 `expm1` | `N≈1`에서 `pow-1` 소거오차 방지 | 논문이 Tait 상수와 caloric closure를 주지 않아 case validation은 별도 |
| primitive-from-conservative-cell-average WENO | reference-energy gauge offset에 따른 conservative energy weight 변화를 제거 | 입력 primitive는 엄밀한 primitive 셀 평균이 아니고 characteristic WENO가 아님 |
| left/right trace별 5-cell affine normalization | 한 trace 밖 cell 및 물리 단위 scale에 대한 불필요한 weight 의존 제거 | 논문이 normalization/epsilon을 명시하지 않아 비논문 알고리즘 |
| minimum-denominator-scaled WENO weight | 극소 양의 epsilon에서 inverse-square overflow 방지 | epsilon 값 자체는 caller-supplied 가정 |
| progress + thermal headroom timestep | signed reaction/decomposition heat가 temperature reject floor를 통과하는 stage 방지 | explicit sufficient guard이며 물리 kinetics 검증을 의미하지 않음 |
| 음의 `joule+electrochemical` total 거부 | “electrical heating” API를 무제한 냉각 callback으로 쓰는 것을 방지 | signed electrical cooling physics는 현재 지원하지 않음 |
| stage transactional retry | 중간 SSPRK stage의 더 작은 안정한계에 맞추되 부분 state/budget commit 방지 | callback의 외부 side effect는 되돌릴 수 없어 pure/deterministic callback 요구 |
| measured metadata accounting | stage JSON뿐 아니라 recursive Python object 크기도 누적 charge | interpreter 측정값이며 OS RSS/allocator fragmentation 보증 아님 |
| history/working-set preallocation guard | 큰 step/history/grid를 주요 배열 할당 전에 거부 | versioned planning bound일 뿐 제3자 workspace·순간 peak·실제 RSS를 보증하지 않음 |

비유한 보존 residual은 fail-closed한다. 유한 residual은 모두 출력하지만 공개
논문 또는 교정 데이터에서 정한 absolute/scale-aware tolerance가 없으므로 현재
pass/fail criterion으로 쓰지 않는다. 이 정책 때문에 conservation metadata가
존재한다는 사실만으로 물리 검증 또는 형상 ranking 적합성을 주장할 수 없다.

## 구조적 결합 상태

현재 `material_level_set`은 Eq.(4)에 따라 수송되는 독립 상태지만 phase/EOS/물성
선택, 반응·전기 source mask, ghost-fluid 또는 interface jump를 구동하지 않는다.
Eq.(1) Euler 상태와 Eq.(2) condensed 상태는 같은 전체 격자에서 독립적으로
전진하며 열·질량·운동량 exchange가 없다. electrical heat는 double counting을
피하기 위해 Eq.(1)에만 투입된다. 이 결정은 논문 미기재 항목을 임의로 채우지
않기 위한 제한이며 완전한 multiphase closure가 아니다.

전선 결과는 fixed positive-`x` ray의 등치선 교점 변위이다. true local contour
normal speed가 아니고 다중 교점 ray는 제외되므로 전극 형상 순위 지표로 쓰지
않는다. 코드 결과의 `geometry_ranking_eligible`은 항상 `false`다.

## 참고문헌으로 확인된 범위

- 대상 논문: <https://doi.org/10.6108/KSPE.2024.28.3.031>
- Yoh & Kim, J. Appl. Phys. 103, 113507:
  <https://doi.org/10.1063/1.2937936>. 관련 reactive hydrocode를 설명하지만
  대상 논문의 정확한 WENO/flux/CFL을 확정하지 않는다.
- Kim & Yoh, J. Math. Phys. 49, 043511:
  <https://doi.org/10.1063/1.2905152>. ghost-fluid 관련 외부 근거이지만 대상
  논문의 직접적인 jump-condition 명세는 아니다.
- Kim et al., Combustion and Flame:
  <https://doi.org/10.1016/j.combustflame.2019.08.017>. 고차 Eulerian
  hydrocode 맥락은 제공하지만 이 ECSP case의 closure를 복원하지 못한다.

## strict 재현에 아직 필요한 정보

1. 초기 rho, p/T, u, v, lambda, alpha의 공간장
2. 모든 외부 경계조건과 전극–추진제 접촉조건
3. Tait A, B, N, rho0 및 온도–내부에너지 caloric EOS
4. Q, dot(omega), A_alpha, E_alpha, f(alpha), Q_r와 일관된 단위/부호
5. sigma(T,alpha,composition), V field 또는 전위 PDE와 BC
6. H를 W/m3로 바꾸는 전기화학 반응률·전자수·화학양론
7. material interface jump condition와 각 재료 EOS/물성
8. WENO variant/order, Riemann solver/flux split, RK3 tableau, CFL
9. 재초기화식·주기·pseudo-time 및 overlap-grid 구성
10. MPI halo 폭·분할·corner/message ordering
11. Species contour 값, subcell 보간, front correspondence, fit interval
12. 1804.8 K와 3.6094 mm/s를 산출한 raw field/time series

## 구현 순서와 완료 상태

| 단계 | 내용 | 상태 |
|---:|---|---|
| 1 | 논문 10쪽/식/그림/참고문헌 분석, dimensional audit | 완료 |
| 2 | provenance + strict fail-closed configuration | 완료 |
| 3 | CPU FP64 Tait/reactive Euler reference | 완료 |
| 4 | primitive WENO5-JS/HLL/SSPRK3/CFL 및 fallback 계측 | 구현 완료(선택은 가정, 최종 evidence 동결 전) |
| 5 | Eq.(2) alpha/thermal reference 및 Joule source | 완료(차원 closure는 가정) |
| 6 | 실제 material level-set 수송/재초기화 drift | 완료(선택은 가정) |
| 7 | Species subcell fixed-`+x`-ray/threshold sensitivity | 구현 완료; true normal speed 아님 |
| 8 | surface overlay와 extended BV callback 경계 | 경계 구현; concrete BC adapter/형상 ranking 검증 미완 |
| 9 | serial/MPI halo 구현 및 실제 rank 1/2/4/8 | 보조 API 구현, solver 미통합, 실기 미검증 |
| 10 | synthetic 보존·해석해·격자/시간 수렴 | 최종 검증 evidence 동결 전; paper validation과 구분 |
| 11 | CUDA/A100 reactive solver 및 batch parity | 미구현/미검증 |
| 12 | 논문 ECSP 수치값 독립 재현 | 필수 입력 부족으로 불가 |

## 모델의 정확한 성격

논문은 압축성 반응 Euler와 material level set을 사용하므로 단순 열전도 모델은
아니다. 그러나 별도 기상 종·기상반응기구·고체–기체 질량전달·상변화 및 완전한
계면 jump condition을 공개하지 않는다. 또한 현재 reference의 phi와 두 방정식
상태도 그런 결합을 수행하지 않는다. 따라서 이 코드를 “검증된 다상 기상 연소
CFD”, “논문과 동일” 또는 “전극 형상 순위에 사용 가능한 production solver”라고
부르지 않는다. 정확한 표현은 **논문 식 골격을 구현한 단일-process CPU FP64
reactive-Euler/독립 condensed-thermal/passive-material-level-set reference이며,
공개 정보가 빠진 closure는 격리된 실험적 구현**이다. 최종 판정은
`NOT FULLY VERIFIED`다.
