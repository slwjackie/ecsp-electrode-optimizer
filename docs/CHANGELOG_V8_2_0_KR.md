# ECSP v8.2.0 변경 기록

## 목적

v8.1.0은 v8.0.0 수치 결과 보존을 우선하여 기존 B/C 물리 오류도 그대로
유지했다. v8.2.0은 그 parity 기준을 종료하고, 알려진 P0/P1 오류를 수정한
물리를 새 production 기준으로 삼는다. v8.1에서 추가한 native C++17-compatible FP64,
CUDA FP64, batch 32/64, batched V_min, early-stop 및 CPU8/A100 hybrid
스케줄링은 유지한다.

설치된 LibTorch가 C++20 header를 요구하는 torch 2.14 이상이면 build helper가
C++20 translation unit을 선택한다. 그 미만에서는 C++17을 사용하며 선택한
표준은 cache key와 runtime metadata에 기록된다.

## P0 수정

### 1. surface-contact geometry

- propellantMask는 전 영역에서 true다.
- anode/cathode는 별도 top-surface footprint mask다.
- 전극 mask가 bulk fixed-potential 셀이나 제거된 물질영역으로 쓰이지 않는다.
- 초기 이온, mobile water, 온도와 반응상태가 접촉면 아래에도 존재한다.
- 물리 raster spacing은 cell-centred `dx = L/N`으로 통일한다.
- physics-grid resize 뒤 edge-to-edge gap, polarity별 component topology/cap,
  접촉면적·양극/음극 balance와 유효 최소폭을 다시 검사하고 위반 후보를
  명시적으로 거부한다.

### 2. footprint Butler-Volmer

- B/C profile의 canonical coupling은 surface_overlay_bv다.
- BV는 접촉 footprint 모든 픽셀에 적용된다.
- 총 전류는 각 접촉 셀의 전류밀도에 dx²를 곱해 적분한다.
- 표면 반응을 열·species 체적 source로 바꿀 때 surface-layer 두께로 나눈다.
- surface overlay 선형화는 전체 추진제 셀을 미지수로 유지하고 외곽에는
  자연 zero-normal-current 조건을 적용한다.

### 3. 반응 inventory와 열 보존

- remaining reactive mass는 conversion만으로 LP를 복원하지 않고 실제
  mobile LP를 사용한다.
- cumulative electrochemical LP consumption을 별도 추적한다.
- 수송 후 요청된 Faradaic LP/water sink가 가용 inventory를 초과하면 sink만
  clip해서 계속하지 않는다. self-consistent BV/potential 재해석이 없는 현재
  계약에서는 상태·전류·energy commit 전에 명시적으로 실패하고 더 작은
  timestep 또는 inventory-constrained electrochemical solver를 요구한다.
- chemical channel increment는 alpha clip 뒤 가용 LP/PVA inventory에 맞춰
  두 channel을 공동 scaling한다. 이 chemical extent limiter는 수락된 extent를
  만들기 위한 것이며 위 Faradaic fail-closed 검사와 다른 의미다.
- accepted channel increment를 alpha, global progress, stoichiometric
  species/product와 chemical heat의 단일 근거로 사용한다.
- channel heat-release 계수는 initial bulk propellant 기준 각 channel 기여량으로
  명시하며, global-progress conversion weight를 열에 다시 곱하지 않는다.
- alpha가 1에서 clip되거나 inventory limiter가 작동해도 미수락 raw rate로
  과도한 chemical heat를 발생시키지 않는다.
- 마지막 fractional timestep은 실제 step_dt로 species, heat와 energy를
  함께 적분한다.

### 4. 물 상태 분리

- condensed mobile water와 global-reaction generated-water product를 서로
  다른 상태변수로 저장한다.
- generated water는 transport/BV closure의 mobile water로 자동 투입되지
  않는다.

### 5. 보존적인 electrical power

- surface-overlay의 in-plane conductive, total J·E 및 diffusion cross-term은
  harmonic face transport/flux로 계산하고 각 내부 face의 power를 인접 두
  control volume에 절반씩 배분한다. domain integral에서는 물리적 face가
  정확히 한 번만 집계된다.
- surface contact의 unresolved normal resistor power `j * normalOhmicDrop`을
  surface-layer 두께로 나눠 별도 체적열로 계산하고 active electrical power에
  포함한다. electrochemical reaction enthalpy와 중복하지 않는다.
- 기본 `conductive_sigma_E2`는 비음수 비가역 전도열이다.
  `total_j_dot_e`는 diffusion cross-term을 포함한 signed electrical-energy
  transfer이므로 음의 국소값을 clip하지 않는다. `legacy_magnitude`는 호환용
  비음수 closure로 유지한다.

## P1 수정

### 1. onset과 propagation 경계

- 최초 accepted onset 상태의 temperature, channel alpha, global progress,
  mobile species, cumulative consumption/product, potential과 열 source를
  snapshot한다.
- propagation은 이 snapshot의 inventory에서 시작한다.
- B/C production profile은 onset 뒤 electrical heating을 기본적으로 끈다.
- pre-flame qJ/qEchem history를 post-onset 시간축에서 반복 재생하지 않는다.
- 과거 stale-history replay는 명시적 legacy 모드를 요청해도 안전검사 없이
  실행되지 않는다.
- corrected propagation은 LP, reactive PVA, mobile water, generated-water
  product, cumulative electrochemical LP consumption 및 xiMax의 완전한
  inventory handoff를 요구한다. partial handoff와 inventory가 전부 없는
  heat-only handoff를 모두 거부한다.
- onset generated-water product는 `2 * xiMax * globalProgress`와 일치해야
  하며, 두 channel weight·progress·species/product inventory의 shape,
  finite/non-negative 및 stoichiometric invariant를 매 step 검사한다.
- provided post-onset electrical history 모드는 닫힌 전기화학-열-species
  재해석기가 없으므로 현재 릴리스에서 크기와 무관하게 거부한다.

### 2. 설정의 단일 의미

- condensed ignition과 B/C onset temperature/progress가 충돌하면 evaluator
  생성 시 명시적 오류를 낸다.
- initial temperature는 thermal/electrical 경로에서 일치해야 한다.
- debug forced-onset profile도 서로 다른 임계값을 숨기지 않고 동일한 값을
  사용한다.
- corrected B/C profile 여섯 개는 temperature/species/chemical-rate cap의
  numerical-validity 허용 fraction을 0으로 강화한다.
- debug profile의 evaluator physics grid를 17에서 design grid와 같은 32로
  변경한다. 따라서 이는 역사적 debug 수치 결과를 보존하는 설정이 아니다.

### 3. V_min 의미

- Python reference와 native evaluator 모두 동일한 `batched_voltage_search`
  state machine을 사용한다.
- tested voltage response에서 lower igniting / higher non-igniting 모순을
  검사한다.
- 같은 전압의 반복 계산이 서로 다른 onset 판정을 내리면 trial 순서와
  무관하게 search를 invalid 처리한다.
- invalid 수치응답을 물리적 non-ignition threshold로 보고하지 않는다.
- 탐색 중 성공 trial은 onset에서 early-stop할 수 있지만, 보고할 최종 upper
  igniting bracket은 full horizon으로 다시 실행해 수치 유효성과 ignition을
  검증한다.

### 4. 평가시각 metric

- 정식 objective 이름을 area_undecomposed_fraction_at_evaluation_time으로
  변경했다.
- remainingReactiveMassFractionAtEvaluationTime 등 AtEvaluationTime 계열을
  정식 output으로 사용한다.
- 과거 At2s key는 파일/API 호환을 위해 남기되 항상 같은 configured
  evaluationTime_s 값을 복제하고 deprecated alias metadata를 기록한다.
- current-congestion objective는 `peakCurrentCongestionToEvaluationTime`으로
  `evaluationTime_s`까지의 `J99/mean(J)` 최대값만 사용한다. 기존
  `peakCurrentCongestion`은 전체 simulation horizon 진단으로 유지하며 objective에
  섞지 않는다.
- onset이 evaluationTime_s보다 빠른 후보는 최초 onset snapshot에서 전기열을
  0으로 둔 응축상 전도·대류/복사 손실·동일한 inventory-limited chemistry를
  공통 평가시각까지 계속한다. 따라서 서로 다른 onset 시각의 frozen state를
  U_rem으로 직접 비교하지 않는다.
- 고정 길이 pre-flame history tail은 상태·누적량 hold와 순간 source 0을
  명시한다. 공통 평가시각 scalar는 canonical AtEvaluationTime metric으로
  제공한다. full-field 저수준 API 호출이면서 evaluationTime_s와 endTime_s가
  같은 경우에만 별도 evaluationFields도 제공한다.

### 5. 시간격자, 안정성 및 수렴의 fail-closed 계약

- step 수는 floating-point roundoff가 0-duration 마지막 step을 만들지 않는
  ceil-like 규칙으로 정하고, 마지막 `step_dt`를 `endTime_s - step_start`로
  계산해 end time에 정확히 도달한다.
- 내부 `evaluationTime_s`는 timestep grid에 정렬되어야 하며,
  `electricalUpdateInterval_s`는 timestep 이상인 정수배여야 한다. evaluation,
  상태 update 및 metric index가 서로 다른 시각을 가리키는 설정은 거부한다.
- B/C Python/native 경로는 `maximumTimeSteps=100000`과
  `maximumHistoryAllocationBytes=17179869184`를, propagation은 동등한
  `maximum_time_steps`와 `maximum_history_allocation_bytes`를 기본 계약으로
  사용한다. 예상 step 수와 retained/transient history byte 수를 allocation 전에
  계산하며, 비유한 값·정수 계약 위반·상한 초과 요청은 메모리 확보를 시도하기
  전에 거부한다. byte 상한은 사전 예약량이 아니라 fail-closed 허용 한도다.
- B/C와 propagation 모두 harmonic-face diffusion과 convection/radiation loss
  Jacobian을 합친 total explicit thermal stability CFL을 매 step 계산하고 설정
  한도를 넘으면 상태를 수락하지 않는다. diffusive CFL은 별도 진단값으로
  기록한다.
- main pre-flame, 공통 평가시각 continuation, propagation 및 native time loop는
  각 candidate temperature update에서 cp/k를 다시 평가한다. 새 온도에서 물성이
  non-finite/non-physical이면 마지막 step이라도 state를 commit/반환하지 않는다.
- propagation은 설정된 허용 fraction보다 많은 temperature clipping이 필요한
  step을 즉시 거부한다. B/C의 temperature-cap fraction은 candidate/V_min의
  numerical-validity threshold에서 거부된다.
- B/C candidate 수렴은 최종 linear solve 하나가 아니라 전체 simulation의
  모든 electrical linear solve와 nonlinear Robin solve를 누적해 판정한다.
  앞선 update의 실패를 마지막 성공이 가리지 않는다.

## 호환성과 의도적 보존

- v8.1 native protocol의 batch와 standalone/extension 실행 구조를 보존한다.
- legacy non-B/C profile, 과거 README와 검증보고서는 재현·감사를 위해
  삭제하지 않는다.
- old At2s consumer는 값 호환성을 유지하지만 새 분석 코드는 canonical
  AtEvaluationTime 이름으로 옮겨야 한다.
- 기상 CFD/OpenFOAM은 추가하지 않았다. propagation은 계속 condensed
  reaction-progress/level-set refinement다.

## 검증 경계

이 릴리스는 코드 회귀·보존식·수치 parity 대상이다. 물성 및 kinetics의
experimental calibration은 수행하지 않았다. 패키징 host에는 NVIDIA
GPU/CUDA runtime이 없어 실제 A100 compile/runtime 및 처리율 측정은 대상
장치 preflight 항목으로 남는다.
