# ECSP v8.3.0 Paper-Reactive Euler Experimental

## 결론과 사용 범위

이 패키지는 v8.2.1의 전극 형상·surface-contact overlay·full-domain 추진제
상태·contact footprint BV·C++/CUDA/하이브리드/NSGA-II 경로를 보존하고,
`python/ecsp_reactive`에 opt-in **단일 프로세스 CPU FP64 연구용 reference**를
추가한다. 논문 식 (1)–(4)의 계산 골격과 수치 안전장치를 구현했지만, 공개 논문에
필수 물성·반응계수·전위장·초기/경계/계면조건 및 원시 결과가 없다. 따라서 논문
정량 재현본, 검증된 다상 연소 CFD, reactive CUDA/A100 구현 또는 production
전극 최적화기로 해석하면 안 된다.

최종 판정은 **`NOT FULLY VERIFIED`** 이며 결과 metadata의
`geometry_ranking_eligible`은 항상 `false`다.

## 보존된 v8.2.1 기능

- 전극은 추진제를 제거하는 체적 재료가 아니라 별도 top-surface contact mask다.
- 전극 아래 셀을 포함한 전체 격자에 추진제 상태가 존재한다.
- BV는 전극 perimeter가 아닌 실제 contact footprint 전체에 적용된다.
- LP/PVA/수분/생성수/열 bookkeeping과 fail-closed inventory 계약을 유지한다.
- C++/CUDA FP64, batch 32/64, batched `V_min`, CPU+A100 hybrid 및 NSGA-II를
  보존한다. 이 항목들은 **legacy v8.2.1 경로**이며 새 reactive solver가 CUDA나
  A100을 지원한다는 뜻은 아니다.
- v8.2.1의 Pillow `0xff` bool canonicalization, host/CUDA finite helper,
  overflow/PCG/error-boundary 수정도 보존한다.

## 새 reference의 실제 계산

| 상태/기능 | 현재 구현 | 중요한 한계 |
|---|---|---|
| Eq.(1) | `U=[rho,rho*u,rho*v,rho*E,rho*lambda]` 2-D 보존형 reactive Euler | 논문의 kinetics와 caloric closure가 없어 caller-supplied/assumed 값 사용 |
| Eq.(2) | 별도 condensed temperature와 decomposition `alpha` | Eq.(1)과 열·질량·운동량을 교환하지 않고 전체 격자에서 독립적으로 전진 |
| Eq.(3) | Tait pressure, sound speed, Tait cold energy + constant-`cv` closure | Tait 계수와 caloric EOS가 논문에 없음 |
| Eq.(4) | 반응진행도와 독립된 `material_level_set` 수송 및 optional reinitialization | `phi`는 현재 phase/EOS/물성 선택이나 interface jump/ghost-fluid 조건을 구동하지 않는 passive scalar |
| 전기 열 | `sigma*|grad(V)|^2` 및 명시적 electrochemical volumetric field/callback | 합계가 음수인 source는 거부하며, 열은 Eq.(1)에만 들어가고 Eq.(2)에는 넣지 않음 |
| Species 전선 | 0.3/0.5/0.7 등치선의 subcell 위치 이동 | 고정 `+x` ray 진단이며 true local-normal speed가 아니고 다중 교점 ray는 제외 |
| 병렬화 | row partition/3-row halo 보조 API | solver time loop에 MPI가 통합되지 않았고 실제 rank 실행 미검증 |
| 가속 | 기존 v8.2.1 CUDA 코드는 보존 | 새 reactive solver용 CUDA/sm_80/A100/batch backend는 없음 |

이 분리는 Eq.(1)과 Eq.(2)의 공개된 interface closure가 없는 상태에서 임의
overwrite나 전기 열 double counting을 피하기 위한 것이다. 물리적으로 닫힌
다중재료 결합을 대신하지 않는다.

## 가정한 수치법과 안전장치

논문에는 WENO와 3차 Runge–Kutta 계열만 적혀 있다. 이 reference는 다음을
`ASSUMED_NOT_FROM_PAPER`로 기록한다.

- 보존 셀 평균에서 계산한 primitive 값 `rho,u,v,T,lambda`를 component-wise
  WENO5-JS로 재구성한다. 이 값은 엄밀한 primitive 셀 평균이 아니다.
- left/right trace 각각 **자신의 5-cell stencil**로 affine normalization하고,
  작은 epsilon에서도 역제곱 overflow가 없도록 WENO weight를 scale한다.
- 재구성된 primitive face를 conservative state로 되돌린 뒤 HLL flux를 쓴다.
  이 선택은 Tait reference-energy gauge 이동에 대한 flux covariance를 보존하도록
  설계됐지만 characteristic reconstruction이나 논문 고유 알고리즘은 아니다.
- 시간 적분은 Shu–Osher SSPRK(3,3), source는 각 stage에서 다시 평가한다.
- 비물리 reconstructed face만 1차 HLL로 fallback하며 횟수와 raw conservative
  component별 최대 절대 보정을 기록한다. 이 최대값은 성분 단위가 서로 달라
  무차원 error norm으로 사용할 수 없다.
- Tait cold energy는 `N`이 1 또는 그 근처일 때 `expm1` 기반 극한식을 사용한다.
- 반응 진행 headroom뿐 아니라 signed reaction/decomposition heat가 온도 reject
  floor를 넘지 않도록 thermal headroom timestep을 적용한다. 음의 전기 총열원은
  지원하지 않고 fail-closed한다.
- cell-state clipping, density/pressure/temperature floor 적용 및 반응률 magnitude
  cap은 하지 않는다. 설정된 completion tolerance는 저장값을 clip하지 않고
  terminal remainder의 rate만 0으로 만드는 명시적 비논문 수치 의미다.
- stage에서 안정 한계가 줄면 부분 상태를 commit하지 않고 제한 횟수만큼 더 작은
  `dt`로 transactional retry하며 proposed/accepted limiter를 따로 기록한다.

## 모드

| 모드 | 전기/전기화학 입력 | 실행 의미 |
|---|---|---|
| `paper_faithful` strict | 논문/측정 provenance의 prescribed field만 허용 | 미기재 입력 또는 assumed algorithm이 있으면 preflight 실패 |
| `paper_faithful` assumed | 명시적 assumed prescribed field | synthetic 수치 smoke·민감도 연구; 논문 재현 아님 |
| `ecsp_extended` | anode/cathode surface mask와 매 SSPRK-stage callback 필수 | 비논문 확장; callback은 self-attested이며 ranking 금지 |
| 기존 evaluator | v8.2.1 경로 | legacy 비교와 기존 최적화 기능 |

`paper_faithful`은 검증 등급이 아니라 논문 밖 BV를 섞지 않는 모드명이다.
strict template은 공개되지 않은 입력을 사용자가 공급하기 전까지 의도적으로
fail-closed한다.

## 실행

~~~bash
cd ECSP_v8_3_0_PaperReactiveEuler_Experimental
export PYTHONPATH="$PWD/python"

# strict template: 미공개 필수 입력 때문에 의도적으로 preflight 실패
python -m ecsp_reactive.cli \
  config/paper_faithful_required_inputs_v8_3_0.yaml \
  --output-prefix runs/strict_should_fail

# assumed synthetic CPU reference smoke; 논문값 재현이 아님
bash tools/run_reactive_reference.sh \
  config/paper_faithful_assumed_v8_3_0.yaml \
  runs/reactive_assumed_smoke

# reactive 수치 검증과 전체 legacy+reactive 회귀시험
python tools/validate_reactive_v8_3.py \
  --output runs/reactive_validation.json --deterministic-repeats 50
python -m pytest -q
~~~

`ecsp_extended_reactive_v8_3_0.yaml`은 CLI가 임의 BV source를 만들지 않는다.
Python API에서 geometry와 corrected BC-global electrical callback을 명시적으로
제공해야 한다. 현재 callback의 문자열/metadata는 물리 적합성 인증이 아니며,
구체 adapter가 potential convergence, current balance와 BV residual을 검증하기
전까지 형상 순위 산출은 hard-gated다.

## 출력과 보존 수지의 의미

`--output-prefix runs/example`은 `runs/example.npz`와 `runs/example.json`을 만든다.
각 파일은 원자적으로 교체되지만 두 파일 전체가 한 filesystem transaction은
아니므로 JSON의 `npz_sha256`으로 결과 쌍을 확인해야 한다.

NPZ의 배열 key는 다음과 같다.

- `conservative_state`
- `solid_temperature_K`
- `solid_decomposition_alpha`
- `material_level_set_m`
- `step_time_s`
- `step_dt_s`

JSON schema는 `ecsp.paper-reactive-result/v1`이다. stage source provenance,
fallback, retry/reinitialization, 전선 history, 보존 수지 및 resource accounting을
기록한다. 비유한 residual은 즉시 실패하지만, **유한 conservation residual의
calibrated absolute/scale-aware 허용오차가 아직 없으므로 pass/fail로 분류하지
않고 diagnostic으로만 저장한다.** 따라서 수지 숫자가 유한하다는 사실만으로
정량 검증을 주장할 수 없다.

## 메모리 preflight의 범위

- history preflight는 최대 step, history stride, threshold/ray history와 전기
  stage metadata를 대상으로 한 versioned planning estimate다.
- prescribed metadata는 canonical record의 measured recursive Python-object size와
  compact UTF-8 JSON byte 수를 사용한다. callback metadata는 사전 크기를 알 수
  없어 baseline으로 계획하고 실제 반환마다 같은 방식으로 재측정하여 누적 한계를
  넘으면 실패한다.
- working-set preflight는 격자당 고정 scratch allowance, 시간 배열, history budget,
  artifact staging allowance를 포함해 **격자 배열을 만들기 전에** 검사한다.
- 두 값 모두 NumPy/Python allocator fragmentation, 제3자 라이브러리 workspace,
  운영체제 RSS 또는 순간 peak를 보증하지 않는 versioned planning guard다. 실제 배포
  메모리 측정을 대신하지 않는다.

## 검증 상태와 문서

최종 evidence가 동결되기 전에는 시험 수를 `passed`로 선기록하지 않는다.
현재 패키징 host에는 mpi4py/mpiexec, CUDA-enabled PyTorch, nvcc와 A100이 없으므로
실제 MPI rank 1/2/4/8, reactive CUDA compile, A100 batch parity 및 A100 성능은
미검증 상태다. 논문의 1804.8 K와 3.6094 mm/s는 reference 값일 뿐이며 raw
field/time series와 closure 없이 assumed 상수를 보정해 맞추지 않는다.

- `docs/PAPER_CODE_TRACEABILITY_V8_3_0_KR.md`
- `docs/PAPER_ASSUMPTIONS_V8_3_0.json`
- `docs/PAPER_REPRODUCTION_V8_3_0_KR.md`
- `docs/REACTIVE_VALIDATION_V8_3_0_KR.md`
- `docs/REACTIVE_PARITY_PERFORMANCE_V8_3_0_KR.md`
- `docs/REACTIVE_CONVERGENCE_V8_3_0.json`
