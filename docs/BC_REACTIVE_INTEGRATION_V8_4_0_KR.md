# v8.4.0 — BC-global → 2-channel condensed Reactive Euler 통합

## 1. 실행되는 것과 실행되지 않는 것

첨부 `ECSP_v8_3_0_PaperReactiveEuler_Experimental.zip`에서 출발했다. 기존
`ecsp_nsga2/propagation.py` baseline과 독립 `ecsp_reactive/solver.py` reference를 보존했다.
새 후속 모델은 **`ecsp_reactive/condensed/`**에 있으며, workflow가 실제로 호출한다.
별도 reference의 `solid.py`를 동시에 적분하지 않는다. 새 해석기는 **CPU FP64**이다.
기존 pre-onset native CPU/CUDA 선택과 독립이며, 새 후속 CUDA 구현을 주장하지 않는다.

```
기존 grammar/NSGA-II → BC pre-onset → 승인된 onset snapshot
                                     ├─ condensed_propagation (기존 baseline)
                                     └─ reactive_euler (새 2-channel continuum)
                                         ↓ 동일 조건 비교/최종 8목적 Pareto
```

`post_onset`이 없는 기존 설정의 기본 경로는 바뀌지 않는다. 루트의 기존
`RUN_NSGA2_ONLY.sh`도 그대로이며, 그것을 실행했다고 BC/Reactive가 자동 선택되지는 않는다.
새 진입점은 `tools/run_bc_reactive.sh`이다. 기존 NSGA-II는 모든 세대의 pre-onset 평가를
먼저 수행하고 **후속 refinement에서** 이 해석기를 호출한다. 새 Reactive 결과를 매 세대의
부모 선택으로 되먹임하는 별도 최적화 구조를 새로 넣은 것은 아니다.

## 2. 최초 실행

패키지 루트에서 기존 requirements를 설치한 뒤 실행한다.

```bash
python -m pip install -r python/requirements.txt
bash tools/run_bc_reactive.sh "$PWD/runs/bc_reactive_debug"
```

기본은 `config/nsga2_bc_reactive_debug.yaml`이다. 4후보+별도 staggered를 BC→두 후속
모델로 계산한다. **debug는 0.01초 pre-onset/0.006초 continuation, 인위적인 낮은 onset
기준과 단축 반응계수를 사용하는 소프트웨어 시험**이다. 실제 점화조건/순위 검증용이 아니다.

기존 baseline을 추천 기준으로 유지하면서 두 모델을 비교:

```bash
CONFIG="$PWD/config/nsga2_bc_dual_backend_debug.yaml" \
  bash tools/run_bc_reactive.sh "$PWD/runs/bc_dual_debug"
```

C++ BC pre-onset를 거쳐 새 후속으로 이어지는 짧은 시험:

```bash
CONFIG="$PWD/config/nsga2_bc_reactive_native_debug.yaml" \
  bash tools/run_bc_reactive.sh "$PWD/runs/bc_native_reactive_debug"
```

native 경로는 C++17 컴파일러가 필요하다. 기존 native loader/build 경로를 사용한다.
실험 크기 계산의 출발 설정은 `nsga2_bc_dual_backend_native_cpu8.yaml`(baseline 추천)과
`nsga2_bc_reactive_native_cpu8.yaml`(Reactive 추천)이다. 전체 생산 규모/시간에 대한 실행
검증과 시간 예측은 제공하지 않는다. `nsga2_bc_reactive_native_a100_batch64.yaml`은
pre-onset A100+post-onset CPU 옵션이며 이 릴리스 환경에서는 CUDA 하드웨어 시험을 하지 못했다.

모든 실행은 **새 결과 디렉터리**를 지정해야 한다. 출력 디렉터리의 무조건 덮어쓰기는 금지한다.

## 3. 선택 스위치와 전원 정책

```yaml
post_onset:
  backend: condensed_propagation  # 또는 reactive_euler
  compare_backends: true          # 같은 onset을 두 해석기에 독립 복사
  allow_experimental_ranking: false
  reactive_euler:
    eos:
      A_Pa: 101325.0
      B_Pa: 2000000000.0
      N: 7.0
      rho0_kg_per_m3: null          # 실제 resolved BC 초기 밀도
      provenance: ASSUMED_NOT_FROM_PAPER_NOT_MEASURED
    caloric_closure: bc_cp_integral_plus_tait_cold_energy
    boundary: reflective
    riemann_solver: hllc
    stationary_mechanics_fast_path: true
```

위 EOS 수치는 v8.3 예제에서 가져온 **미보정 예시**이며 실제 ECSP 물성으로 확정한 값이 아니다.
반응 Q/E/L/w는 여기에 재입력하지 않고 resolved BC 설정을 그대로 쓴다.
Reactive를 최종 추천 기준으로 쓰는 경우에만 `allow_experimental_ranking: true`를 명시한다.
이 플래그는 실험 검증 완료 표시가 아니라 미검증 모델의 잠정 수치 순위를 사용한다는 선택이다.

전원 차단 비교(기본):

```yaml
propagation_refinement:
  continued_electrical_heating: false
  electrical_heating_mode: off
```

전압 유지/재계산(새 Reactive 단독):

```yaml
post_onset:
  backend: reactive_euler
  compare_backends: false
  allow_experimental_ranking: true
  # reactive_euler EOS/수치 설정도 위 또는 제공 프로필처럼 필요
propagation_refinement:
  continued_electrical_heating: true
  electrical_heating_mode: recomputed
```

이 경우 각 SSPRK stage에서 **현재** T/α/농도에 대해 기존 BC NP/BV solver를 호출한다.
전위 수렴·BV 수렴·전류수지를 검사하고 현재 qJ/qEchem을 얻는다. 이전 발열장 재생이나
두께로 중복 나누기는 하지 않는다. 전압은 `resolved_bc_config.coupled.voltage_V`를 사용한다.
전압이력/전극 운동/접촉 박리 모델은 추가하지 않았으며 표면 접촉 footprint는 고정이다.
기존 baseline에는 같은 recomputed electrical 경로가 없으므로 **전압 유지 paired 비교는
허용하지 않는다**. 전원 정책을 몰래 다르게 해서 비교하는 것을 막는다.

## 4. 보존 상태와 동일 chemistry

핵심 보존 상태:

\[
U=[\rho,\rho u,\rho v,\rho E,\rho\alpha_1,\rho\alpha_2]^T.
\]

여기에 cation, anion, mobile water, PVA repeat, generated product water,
electrochemically consumed LP의 **6개 체적 재고**를 보존 수송해 총 12변수로 계산한다.
이 재고 항목들이 모두 독립 질량분율이라는 뜻은 아니다. BC와 같은 반응물/생성물
bookkeeping을 갖는 축약 모델이며, 모든 생성물 화학종의 상세 보존식을 새로 푼 것은 아니다.

\[
\partial_t(\rho\alpha_i)+\nabla\cdot(\rho\mathbf u\alpha_i)=\rho r_i,
\quad r_i=\exp[L_i(\alpha_i)-E_i(\alpha_i)/(RT)].
\]

기존 BC와 같은 보간식·Q1/Q2·w1/w2·LP/PVA 화학양론을 사용한다. 반응물 고갈이나
α=1에 의해 실제 받아들인 증가량이 줄면 **발열도 같은 증가량으로 줄인다**.

\[
q_{chem}=\rho(Q_1r_{1,accepted}+Q_2r_{2,accepted}).
\]

Q1/Q2는 초기 벌크 질량 기준 채널별 기여량이다. 여기에 w1/w2를 다시 곱하지 않는다.
물 생성물은 mobile water에 자동으로 합치지 않아, 생성수가 즉시 다시 전기분해되는 새 가정을
추가하지 않는다. 단일 X만 저장했다가 α1/α2를 추정하는 변환도 하지 않는다.

`BCReactiveHandoffAdapter`는 T, α1/α2, X, 재고, 전위, q snapshot, contact/full-propellant mask,
밀도·길이·두께·BC hash를 검사한다. 다른 격자에 임의 보간하지 않고 같은 cell-centred 격자를 쓴다.
승인되지 않은 onset, 중첩 접촉 mask, 비물리 상태, 수지 불일치는 거부한다.
기존 NPZ에 contact mask가 없으면 heat map에서 추측하지 않는다. 이 버전에서 handoff를
다시 추출하거나 검증된 원래 contact mask를 명시적으로 제공해야 한다.

## 5. 단 하나의 응축상 에너지

\[
\partial_t(\rho E)+\nabla\cdot[(\rho E+p)\mathbf u]
=\nabla\cdot[k(T)\nabla T]+q_J+q_{echem}+q_{chem}-q_{loss}.
\]

BC의 k(T), cp(T), 대류·복사 표면손실을 유지한다. 기체로 이동하는 질량 source는 0이고
별도 `solid.py` 온도식도 호출하지 않는다. 동일 물질을 solid/flow 두 겹으로 저장하지 않으므로
solid 질량과 flow 질량을 따로 만들었다가 주고받는 구조가 아니다.

- qJ: BC 전도장으로 정해지는 벌크 Joule 열 + 기존 contact-normal Ohmic 손실.
- qEchem: 표면 전극 contact footprint에만 존재하는 기존 계면열의 체적화 결과.
- qChem: 두 채널의 실제 반응이 진행되는 셀에만 발생.

Surface overlay의 두께평균 모델이므로 접촉면 열유속을 유효두께로 나눈 **기존 BC 체적열원**을
그대로 전달한다. 이것은 전극 아래 얇은 z-방향 반응층을 3D로 직접 해상한 것이 아니다.

열역학 닫힘은 명시적인 추가 선택이다:

\[
e(\rho,T)=\underbrace{\int_{\rho_0}^{\rho}p(r)/r^2\,dr}_{e_{cold}(\rho)}
+\int_{T_{ref}}^T c_{p,BC}(\vartheta)d\vartheta.
\]

Tait가 온도와 무관한 barotropic 모델이므로 BC의 cp를 caloric 용량으로 사용했다.
이 식은 첨부 논문에서 그대로 제공된 닫힘이 아니며, 동일 BC 온도/열용량을 총에너지와
일관되게 연결하기 위해 추가한 모델 선택이다. 압축 일은 cold energy에 반영한다.
표 물성의 선형보간을 해석적으로 적분하고 역변환한다. T를 cap으로 잘라 에너지를 없애지 않는다.

질량, 총에너지, 미방출 화학에너지를 포함한 수지, 반응 재고 수지를 기록한다.
SSPRK의 **동일 quadrature 가중치**로 경계 플럭스와 모든 열원/손실을 적분한다.
양성/수지 검사에 실패하면 dt를 줄여 재시도하며, 완료시간 전에 자원 한도를 넘으면 실패한다.
실패한 Reactive를 성공한 baseline으로 몰래 바꾸지 않는다.

## 6. Level set: 내부 반응경계와 외부 소모면을 구분

\[
X=w_1\alpha_1+w_2\alpha_2,\qquad \phi_r=X_{front}-X.
\]

`phi_r=0`이 반응 진행에서 직접 정해지는 내부 미반응/반응 경계이다. `phi_r>0`은 미반응측,
`phi_r<=0`은 반응측이다. 이 경계의 signed-distance를 별도 저장한다.
**X=0.5는 진단/전선 정의이지 새로운 반응속도 계수가 아니다.** 제공 debug 프로필만은
짧은 소프트웨어 시험 때문에 1e-8 문턱을 쓰며, 생산 프로필은 기존 front 문턱을 보존한다.

경계는 α1/α2의 보존 수송·반응 결과를 따라 이동한다. 원래 reference의 독립 passive
material-level-set PDE와 별개이며, 추정한 burning speed를 덧붙이지 않는다.
`X>=Xfront` 셀도 질량·에너지·미완료 α2를 그대로 유지한다. 같은 Tait/물성으로 이어지는
**내부 반응 상태 구분**이며, 미반응물/반응생성물별 서로 다른 EOS, 자유표면,
기화, 외곽 ablation을 구현한 것은 아니다. 그런 구성 관계는 원자료에 없어 임의로 만들지 않았다.

후퇴속도 출력은 반응 면적 변화/평균 전선 길이로 구한 **실험실 좌표계의 유효 지표**이다.
움직이는 continuum에서 이 값은 양/음일 수 있으며 실제 외곽 표면의 법선 ablation 속도와
동일하지 않다. 경계가 격자에 해상되지 않으면 0 또는 문턱 민감도가 나타날 수 있다.

## 7. Tait와 수치해법의 중요한 한계

\[
p=A+B[(\rho/\rho_0)^N-1],\qquad (\partial p/\partial T)_\rho=0.
\]

BC handoff는 압력파를 제공하지 않으므로 **초기 rho=BC 밀도, u=v=0**이다.
균일 밀도/정지/고정 경계에서는 열이 발생해도 압력이 직접 상승하지 않는다. 따라서 이 조건의
계산은 Fourier 열전도·반응이 중심이며, Reactive라는 이름만으로 압력구동 연소파가 생기지 않는다.
밀도/속도 비균일 초기조건을 쓰는 별도 검증에서는 Euler 파동을 계산한다. 그런 교란을 실제
handoff에 몰래 넣지는 않는다. 열팽창/반응별 기준밀도 변화는 추가하지 않았다.

이 정확한 정지 부분공간에서만 hydro RHS=0을 이용해 acoustic CFL을 생략할 수 있다.
`stationary_mechanics_fast_path`는 새로운 감쇠식이나 임의 음속 저하가 아니다. 조건이
깨지면 원래 acoustic CFL로 돌아간다. 전체 유동 경로와의 일치 시험도 포함한다.

WENO5-JS/SSPRK33은 유지하되 신규 경로의 기본 Riemann flux는 **HLLC**다. 기존 독립
reference는 HLL 그대로이다. HLL은 정지 접촉면의 T/반응물도 음속 척도로 확산시킬 수 있어
HLLC의 접촉파 보존이 필요했다. 비교용 `riemann_solver: hll`도 가능하지만 fast path를 false로
두어야 한다. 유효하지 않은 고차 face state는 보존 플럭스를 유지한 first-order로, 유효하지 않은
HLLC star state는 HLL로 fallback하고 횟수를 기록한다.

기본 mechanical 경계는 reflective; periodic/transmissive도 수치시험용으로 지원한다.
**열전도/NP의 외곽 경계는 기존 BC의 no-flux**이며 mechanical periodic 설정과 별개이다.
지속전압 상태에서도 표면 전극은 고정 footprint이고 접촉역학을 풀지 않는다.
전원 OFF이면 이 후속 모델은 이온의 별도 NP 확산을 계속 풀지 않고 재고를 continuum과 운반한다.
이 역시 기존 power-off baseline과 비교하기 위한 범위이며 모든 응축상 수송의 완전 모델은 아니다.

## 8. 비교 출력과 개별 handoff 재계산

후보별:

```
final/propagation_candidates/<ID>/
  bc_handoff/{bc_handoff_fields.npz,bc_handoff_metadata.json}
  condensed_propagation/{propagation_metrics.json,propagation_history.csv,propagation_fields.npz}
  reactive_euler/{handoff_audit.json,propagation_metrics.json,propagation_history.csv,propagation_fields.npz}
  backend_comparison.{csv,json}
```

설정에 따라 사용한 backend만 생성한다. Paired 비교는 같은 handoff의 SHA256을 기록하고
established time/검열 여부, 미반응 면적·재고량, 유효 속도, 전선 불균일도, 최대온도,
질량·에너지 잔차 및 열항 적분을 비교한다. 최종 `backend_rank_comparison.{csv,json}`은
같은 후보/같은 8목적을 공동 정규화하여 Pareto front와 목적별/종합 Spearman 순위를 비교한다.
모두 동률이거나 1후보뿐이면 상관계수를 임의로 1로 만들지 않고 null로 둔다.
Staggered는 최적화군에 섞지 않고 별도 동일조건 비교를 유지한다.

기존 승인 snapshot만 다시 계산:

```bash
python python/run_bc_post_onset.py \
  --config config/nsga2_bc_reactive_debug.yaml \
  --handoff-dir RUN/final/propagation_candidates/ID/bc_handoff \
  --resolved-bc-config RUN/adapter/resolved_physics_config.json \
  --output "$PWD/runs/one_handoff_dual"
```

실제 RUN/ID로 치환한다. `backend_comparison`의 두 모델 일치는 **실험 정확도 증명**이 아니다.
비교 모델이 같은 미보정 kinetics/열물성을 공유하기 때문이다.

## 9. 코드 위치

| 기능 | 파일 |
|---|---|
| 실제 dispatch/8목적 비교 | `python/ecsp_nsga2/post_onset.py`, `workflow.py` |
| BC snapshot 확장 | `python/ecsp_nsga2/evaluator.py` |
| 엄격한 state handoff | `python/ecsp_reactive/condensed/handoff.py` |
| 동일 2채널/재고 수지 | `python/ecsp_reactive/condensed/chemistry.py` |
| T ↔ 보존 에너지/Tait | `python/ecsp_reactive/condensed/thermo.py` |
| WENO/HLLC/FV 경계 플럭스 | `python/ecsp_reactive/condensed/finite_volume.py` |
| 기존 NP/BV 재계산 연결 | `python/ecsp_reactive/condensed/electrical.py` |
| 단일 에너지/SSPRK/진단 | `python/ecsp_reactive/condensed/solver.py` |
| 보존된 baseline + 추가 진단 | `python/ecsp_nsga2/propagation.py` |
| 회귀/수치/전체 연결 시험 | `python/tests/test_bc_reactive_integration.py` |

첨부 논문에 있는 Euler/Tait/Arrhenius 구조, 기존 BC의 두 채널 구성·계수,
이 통합에서 추가한 caloric/수치/경계 선택은 서로 구분했다. 원문에 없는 물성이나
실험적 burning-speed law를 논문에서 검증된 것처럼 표시하지 않는다.
