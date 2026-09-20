# ECSP v8.3.0 Paper-Reactive Euler Experimental 변경 기록

## 릴리스 성격

v8.3.0은 v8.2.1의 전극 형상, full-domain surface overlay, contact-footprint BV,
inventory/heat bookkeeping, C++/CUDA FP64, batched `V_min`, hybrid scheduling,
NSGA-II 및 결과 경로를 보존하고 `python/ecsp_reactive`에 opt-in 단일-process
CPU FP64 reference를 추가한 실험 릴리스다. 기존 condensed propagation을 논문의
reactive Euler 또는 level-set solver로 재명명하지 않았으며 legacy 비교 경로로
유지한다.

최종 판정은 **`NOT FULLY VERIFIED`** 이다. 이 릴리스는 production 다상 연소
CFD나 production 전극 형상 ranking backend가 아니다.

## 추가된 논문 식 골격

- Eq.(1)의 2-D 반응 Euler 보존변수
  `[rho,rho*u,rho*v,rho*E,rho*lambda]`, 보존 flux, 반응 진행/열 및
  `sigma*|grad(V)|^2` 체적 열원
- Eq.(2)의 별도 condensed temperature/decomposition-`alpha` 상태와
  dimensionally repaired `-rho*Q_r*dalpha/dt` source
- Eq.(3)의 Tait pressure, analytic sound speed 및 constant-`cv` caloric closure
- Eq.(4)의 reaction progress와 별도인 transported material-level-set scalar
- Species 등치선의 subcell 교점 이동 진단
- CPU FP64 HLL/SSPRK(3,3)/adaptive timestep reference
- 별도 row partition/3-row halo API
- 보존 수지, fallback, retry, 재초기화 drift, source-stage provenance와 resource
  accounting 출력

논문에 없는 caloric closure, Arrhenius 계수/함수, interface method, 외부 경계,
WENO variant/reconstruction variable, HLL, RK tableau, CFL, source coupling,
reinitialization 및 front threshold는 모두 `ASSUMED_NOT_FROM_PAPER`다.

## 수치 구현의 현재 정확한 선택

- 보존 셀 평균에서 primitive `rho,u,v,T,lambda`를 계산한 뒤 component-wise
  WENO5-JS로 face를 재구성한다. 입력 primitive 값은 엄밀한 primitive cell
  average가 아니며 characteristic reconstruction도 아니다.
- left와 right trace는 각각 자신의 5-cell stencil로 독립 affine normalization한다.
  한 trace 밖의 여섯 번째 cell이 nonlinear weight를 바꾸지 않도록 했다.
- WENO-JS weight는 최소 denominator에 대한 비율로 scale해 아주 작은 양의
  epsilon에서도 `1/(epsilon+beta)^2` overflow를 피한다.
- primitive face를 conservative state로 복원하고 HLL flux를 계산한다. 이로써
  Tait reference energy에 상수를 더한 gauge 변환에서 component별 conservative
  WENO weight가 달라지는 문제를 제거했다.
- Shu–Osher SSPRK(3,3)의 각 stage에서 flux, reaction, electrical callback,
  condensed thermal RHS와 level set을 다시 평가한다.
- cell state는 clip하지 않는다. 비물리 reconstructed face에만 1차 HLL fallback을
  쓰며 횟수와 mixed-unit raw conservative correction을 기록한다.

## 검토 중 발견해 수정한 안전성·계약 문제

| 영역 | 이전 위험 | 현재 조치와 남는 의미 |
|---|---|---|
| Tait `N≈1` | `pow(x,N-1)-1` 소거오차 | `expm1((N-1)log(x))/(N-1)`와 `N=1` 극한식 사용 |
| WENO trace scaling | 공유 6-cell scale이 반대 trace 밖 cell에 의존 | trace별 정확한 5-cell normalization |
| WENO tiny epsilon | inverse-square weight overflow | pointwise minimum-denominator scaling으로 유한 weight 정규화 |
| energy gauge | conservative component weight가 reference-energy offset에 의존 | primitive 재구성 후 conservative face 복원 |
| reaction timestep | progress increment만 제한하면 흡열 source가 temperature floor 통과 가능 | flow/solid reaction의 진행 headroom과 thermal headroom을 함께 적용 |
| 아주 작은 rate | `increment/rate` overflow를 invalid limit로 오판 | 양의 무한 limit는 unbounded로 취급하고 NaN/nonpositive만 거부 |
| 전기 source | electrochemical 항이 Joule보다 더 음수이면 무제한 냉각 | `joule+electrochemical < 0`을 native stage 진입 전 fail-closed |
| RK stage | stage별 제한 감소 후 부분 commit 또는 stale limiter label 가능 | transactional retry와 proposed/accepted limiter 분리 기록 |
| completion | order<1/0에서 terminal에 무한 기하 step 가능 | 명시 tolerance 안 remainder의 rate만 0; 저장 state는 clip하지 않고 영향량 기록 |
| boundary provenance | flow BC를 thermal/level-set/electrical gradient에 재사용 | 네 경계 의미를 분리해 선언·runtime 검증 |
| level-set reinit | method/interval/parameter 조합 불일치 | parser/runtime에서 양방향 계약 검증, area drift와 sign change 기록 |
| front metric | fixed-axis 이동을 normal regression으로 표시 가능 | `+x` fixed-ray 진단으로 명명하고 다중 교점 제외·ranking 금지 |
| MPI 표시 | helper API를 integrated MPI solver로 오해 가능 | 실행 solver는 `SERIAL_SINGLE_PROCESS`만 허용 |
| source provenance | provider 문자열을 corrected BC 인증으로 오해 가능 | geometry hash/bytes snapshot을 solver가 계산, callback은 self-attested로 표시 |
| callback aliasing | 외부 mutation과 retry side effect가 결과를 바꿀 수 있음 | bytes-backed read-only state/geometry 및 strict JSON snapshot; callback purity 요구 |
| metadata memory | JSON 길이만으로 Python object memory를 과소계상 | recursive measured object size + canonical JSON bytes를 stage마다 charge |
| history allocation | step/ray/stage 기록으로 OOM 가능 | stride와 versioned history planning estimate, runtime callback record 누적 gate |
| grid allocation | 큰 `nx*ny`가 solver 생성 즉시 OOM 가능 | versioned per-cell scratch/NPZ allowance 기반 preallocation working-set guard |
| result mutability | `setflags(write=True)` 또는 nested metadata 변경 가능 | bytes-backed array/bool snapshot과 recursively immutable JSON metadata |
| serialization | NaN/Inf, object `str()`, mixed key로 비결정 JSON 가능 | finite JSON scalar/string-key contract와 `allow_nan=false` |
| config | duplicate/unknown key, float integer 손실, 상속 provenance 불일치 | exact schema, duplicate-key loader, integer validation, effective-runtime audit |
| legacy bool mask | Pillow mode-1의 raw `0xff`가 C++ bool UB 유발 | v8.2.1 canonical 0/1 경계 방어 보존 |

## 전기 source 경계

- `paper_faithful`은 Butler–Volmer를 금지한다.
- `CONFIG_UNIFORM_FIELD`는 설정의 sigma/Ex/Ey/electrochemical scalar만 소비한다.
- `PRESCRIBED_ARRAY_FIELDS`는 Python API의 배열과 provenance만 소비한다.
- `ecsp_extended`는 full-domain propellant 위의 양극·음극 surface mask 및 매
  SSPRK-stage `EXISTING_BC_GLOBAL_CALLBACK`을 요구한다.
- callback metadata는 `CALLBACK_SELF_ATTESTED_UNVERIFIED`이고 총 전기 열은
  비음수여야 한다. source는 Eq.(1) Euler energy에만 더하며 Eq.(2)에 중복하지 않는다.
- concrete BC adapter가 potential convergence, current balance와 BV residual을
  강제하지 않으므로 `geometry_ranking_eligible=false`다.

## resource gate의 정확한 범위

history 및 working-set 검사는 계산 전 예상 OOM을 줄이는 **versioned planning
guard**다. prescribed stage record는 실제 canonical metadata를 측정하며 callback은
baseline으로 preflight한 후 반환 record마다 재측정해 한도를 강제한다. working-set
추정에는 격자당 scratch allowance, 시간 배열, history budget와 NPZ staging
allowance가 포함된다. 그러나 Python/NumPy allocator fragmentation, 제3자
workspace, OS RSS나 순간 peak를 보증하지 않는다. 실제 배포 환경의 peak-memory
측정은 별도로 필요하다.

## 의도적으로 닫지 않은 구조

- `material_level_set`은 수송되지만 phase/EOS/물성 선택, source mask, ghost-fluid
  또는 interface jump condition에 연결되지 않는다.
- Eq.(1)과 Eq.(2)는 전체 격자에서 독립적으로 전진하며 열·질량·운동량을 교환하지
  않는다. 전기 열은 Eq.(1)에만 들어간다.
- Species front는 fixed `+x` ray diagnostic이며 true local-normal propagation
  metric이 아니다.
- finite conservation residual은 기록하지만 calibrated absolute/scale-aware
  tolerance가 없어 유한값에 pass/fail을 부여하지 않는다. nonfinite만 fail-closed한다.
- MPI row/halo API는 solver loop에 통합되지 않았고 실제 rank 1/2/4/8을 실행하지
  않았다.
- reactive CUDA kernel/sm_80/A100 batch backend는 구현하거나 시험하지 않았다.
- 공개 논문에 Tait/caloric/kinetic/electrical/BC/IC/interface/raw history가 없어
  1804.8 K와 3.6094 mm/s를 독립 재현할 수 없다.

따라서 이 변경은 논문 식 골격과 실험적 CPU 수치 reference이며, 형상 순위와
정량 예측은 별도 물리 closure·교정·실험·MPI/CUDA 실기 검증을 마친 뒤에만 논할
수 있다.

