# ECSP v8.4.0 — BC / 2-channel condensed Reactive 통합

**이번 버전 사용법:** [통합 실행·물리 범위](docs/BC_REACTIVE_INTEGRATION_V8_4_0_KR.md)
 · [시험 결과](docs/VALIDATION_V8_4_0_KR.md)

```bash
bash tools/run_bc_reactive.sh "$PWD/runs/bc_reactive_debug"
```

기존 propagation과 독립 v8.3 Reactive reference는 보존했다. 새 후속 해석기는
`python/ecsp_reactive/condensed/`이며 기본 debug는 BC→두 모델의 동일 상태 비교다.
새 Reactive는 CPU FP64, 2-channel/단일 응축상 에너지이며 기체 CFD가 아니다.
실험 검증 완료가 아니고 Tait의 온도 비의존성도 유지된다.

아래 v8.3 이하 문서/시험 수치는 **해당 과거 릴리스의 기록**이다. 이번 변경의 검증 범위는
위 v8.4 문서를 따른다. 기존 root run script의 기본 backend를 몰래 바꾸지 않았다.

---

# ECSP v8.3.0 — 논문 반응성 Euler 실험적 reference

> **현재 안내:** [README_V8_3_0_KR.md](README_V8_3_0_KR.md)를 먼저
> 확인하십시오. v8.3.0은 검증된 v8.2.1 기능을 보존한 채 논문 Eq.(1)–(4)의
> CPU FP64 reference를 opt-in으로 추가한 연구용 버전입니다. 공개 논문에
> 필수 입력이 빠져 있으므로 최종 판정은 `NOT FULLY VERIFIED`이며 production
> 또는 paper-faithful 재현 완료를 주장하지 않습니다.

## 역사적 v8.2.1 — P0/P1 수정 계열 maintenance patch 안내

> **v8.2.1 작업본:** [README_V8_2_1_KR.md](README_V8_2_1_KR.md)를 먼저
> 확인하십시오. v8.2.1은 SHA-256이
> `799cfd001a8e28143e16e33f32424a9545584f2c95691370943c804f210832a6`인
> immutable v8.2.0 ZIP을 기준으로 변경 범위를 감사하는 patch release입니다.
> 아래 v8.2.0 및 이전 설명은 역사적 동작과 계보 기록입니다.

## 역사적 v8.2.0 설명 — P0/P1 수정 production 기준

> **v8.2.0 P0/P1 수정본:** production B/C 실행 전
> [README_V8_2_KR.md](README_V8_2_KR.md)를 먼저 확인하십시오. v8.2는 full-domain
> propellant + 별도 surface contact mask, footprint BV, 보존적 반응·열·species
> bookkeeping과 strict onset handoff를 사용합니다. 아래 v8.1/v8.0 설명은 과거
> 동작과 legacy 경로의 기록입니다.

## 역사적 v8.0.0 설명 — B/C global pre-flame + post-onset condensed propagation

이 릴리스는 **v7.9.5의 모든 기존 backend와 실행 profile을 보존**하면서, `ECSP 최적 전극 형상 시뮬` 명세를 opt-in 경로로 추가한다.

```text
1000 CAD/Mask geometries
  → B/C pre-flame (NP + local BV + global chemistry + thermal)
  → 4-objective Pareto + near-Pareto/diverse 15%
  → condensed reaction-progress/level-set propagation
  → final Pareto
  → identical B/C+propagation evaluation of area-matched staggered
```

주요 실행:

```bash
# CPU FP64 전체 배관 smoke
bash tools/run_bc_global_debug.sh "$PWD/runs/bc_global_debug"

# 명세 정적/실행 감사
bash tools/validate_bc_global_pipeline.sh

# A100/CUDA FP64 code-path preflight (A100 node에서 먼저 실행)
bash tools/run_bc_global_a100_preflight.sh "$PWD/runs/bc_global_a100_preflight"

# A100 production profile: population 1000, generations 5
bash tools/run_bc_global_a100.sh "$PWD/runs/bc_global_a100" 1000 5
```

A100 profile은 실제 장치에서 아직 실행 검증되지 않았다. 수송·BV·열물성 및 Fig. 8 digitisation은 `literature_nominal_uncalibrated`이고, 결과는 실험보정 전에는 정량예측으로 해석하지 않는다. 후단 propagation은 **기상 CFD가 아니라 응축상 reaction-progress/level-set refinement**이다.

상세 문서:

- `docs/BC_GLOBAL_PREFLAME_AND_PROPAGATION_V8_KR.md`
- `docs/PARAMETER_PROVENANCE_BC_GLOBAL_KR.md`
- `docs/ECSP_OPTIMAL_ELECTRODE_SIM_IMPLEMENTATION_AUDIT_KR.md`
- `docs/VALIDATION_REPORT_V8_0_0_KR.md`
- `docs/V795_FEATURE_PRESERVATION_AUDIT_KR.md`
- `docs/V795_BASELINE_FILE_HASH_AUDIT.json`

---

## 역사적 v7.9.5 설명 — A100 CUDA FP64 + 48 vCPU hybrid, minimum ignition voltage objective, surface-contact 전극

이 패키지는 전극을 추진제에서 제거되는 물질영역이 아니라, **설정된 정사각형 추진제 표면 위의 전기적 접촉영역**으로 모델링한다. 기본값은 20×20 mm이며 외부 YAML로 25×25 mm 등도 사용한다. 따라서 전극 접촉면적이 변해도 계산되는 추진제 면적·질량은 항상 동일하다.

## v7.9.5 핵심 변경: A100과 48 vCPU 동시 사용

v7.9.5는 기존 CPU 물리해석기를 제거하지 않고, 다음 두 경로를 함께 제공한다.

```text
reference CPU backend
└─ cpp_fp64_cpu
   └─ cpp/ecsp_cpp_solver.cpp        # v7.9.4 원본, 변경 없음

hybrid production backend
└─ hybrid_cuda_cpu
   ├─ A100: Torch/CUDA FP64 batched condensed-phase solver
   └─ CPU : cpp/ecsp_cpp_solver_hybrid.cpp 독립 프로세스 pool
```

Hybrid 모드에서는 한 세대의 reference-voltage trial과 각 `V_min` 이분탐색 wave를 하나의 shared queue로 만든다. A100은 기본 64개 후보를 한 batch로 계산하고, 48 vCPU 중 기본 40개는 서로 다른 후보를 C++ FP64 프로세스로 동시에 계산한다. 나머지 8 vCPU는 Python, CUDA kernel launch, 결과 직렬화, 파일 I/O 및 OS에 남긴다.

```yaml
evaluator:
  backend: hybrid_cuda_cpu
  hybrid_cpu_worker_cases: 40
  hybrid_reserved_host_vcpus: 8
  hybrid_cuda_batch_size: 64
  hybrid_cuda_min_batch_size: 8
  hybrid_cpu_reserve_per_wave: 16
  hybrid_cuda_fallback_to_cpu: true
```

작은 `V_min` wave에서는 일부 trial을 CPU용으로 예약해 두 장치가 함께 진행한다. GPU batch OOM·런타임 오류 또는 GPU에서 수치적으로 거부된 후보는 batch 축소 재시도 후 C++ CPU queue로 이동한다. 다만 이분탐색 마지막 wave가 8개 미만이거나 queue tail만 남은 순간에는 의존성 때문에 한 장치가 잠시 유휴일 수 있으며, 코드가 모든 시점의 100% utilization을 보장하는 것은 아니다.

### A100 + 48 vCPU 필수 preflight

```bash
bash tools/run_a100_cpu48_hybrid_preflight.sh \
  "$PWD/runs/a100_cpu48_preflight"
```

이 preflight는 같은 one-step 물리를 CPU와 A100에서 모두 계산하고 다음을 확인한다.

- GPU와 CPU가 같은 stage에서 실제로 모두 사용됨
- 두 backend 모두 FP64
- physics rejection 0
- peak current 및 입력에너지의 CPU/GPU 상대차 ≤ `5e-4`

현재 배포 환경에는 A100이 없어 실제 CUDA 실행은 검증하지 못했으므로, 이 preflight를 통과하기 전에는 production run을 시작하면 안 된다.

### Hybrid production 실행

20×20 mm, 최대 4 component 예시:

```bash
ECSP_SKIP_HYBRID_PREFLIGHT=1 \
CONFIG="$PWD/config/experiments/d20_multi_up_to4_a100_cpu48_hybrid.yaml" \
bash tools/run_a100_cpu48_hybrid.sh \
  "$PWD/runs/d20_multi_500x4_hybrid" 500 4
```

`V_min`을 15 V bracket으로 빠르게 탐색하려면:

```bash
ECSP_SKIP_HYBRID_PREFLIGHT=1 \
CONFIG="$PWD/config/nsga2_condensed_phase_no_f_a100_cpu48_hybrid_fast_vmin.yaml" \
bash tools/run_a100_cpu48_hybrid.sh \
  "$PWD/runs/hybrid_fast_vmin_200x3" 200 3
```

실행 중 모니터링:

```bash
bash tools/monitor_a100_cpu48_hybrid.sh 10
```

환경변수로 병렬도를 임시 조정할 수도 있다.

```bash
ECSP_HYBRID_CPU_WORKERS=42 \
ECSP_HYBRID_HOST_RESERVE=6 \
ECSP_HYBRID_CUDA_BATCH_SIZE=32 \
ECSP_SKIP_HYBRID_PREFLIGHT=1 \
CONFIG="$PWD/config/experiments/d20_multi_up_to4_a100_cpu48_hybrid.yaml" \
bash tools/run_a100_cpu48_hybrid.sh \
  "$PWD/runs/profile_test" 200 1
```

48-vCPU 환경의 권장 시작점은 `40 CPU workers + 8 host reserve + CUDA batch 64`이다. 최적값은 node의 실제 CPU 종류, A100 모델, cgroup throttling 및 193/241 grid에 따라 달라지므로 제공된 benchmark로 `CPU 36/40/42`, `CUDA batch 32/64`를 비교해야 한다.

### 기존 CPU backend는 그대로 유지

A100을 사용하지 않거나 reference 결과가 필요하면 기존 YAML의:

```yaml
evaluator:
  backend: cpp_fp64_cpu
```

를 그대로 사용한다. `cpp/ecsp_cpp_solver.cpp`는 v7.9.4 원본과 byte-for-byte 동일하며, `cpp/reference/ecsp_cpp_solver_v7_9_4_reference.cpp`에도 복제 보존돼 있다.

## v7.9.4 핵심 변경

NSGA-II 제3 목적함수를 `E_ign`에서 **최소 점화전압 `V_min`**으로 교체했다. `V_min`은 설정된 전압 탐색 구간에서 동일한 응축상 점화 기준을 2 s 이내에 처음 만족하는 최소 인가전압으로 정의한다. 각 형상에 대해 260 V reference 계산을 먼저 수행하고, 저전압 trial + bracketed bisection으로 threshold를 찾는다.

기본 설정은 `20–260 V`, 목표 bracket 폭 `5 V`, 최대 bisection 8회다. 보고값은 마지막 bracket의 **낮은 쪽 non-igniting 전압이 아니라 높은 쪽 igniting 전압**이므로 보수적 upper bound이다. 20 V에서도 점화하면 `V_min ≤ 20 V`인 left-censored 결과로, 260 V에서도 점화하지 않으면 `V_min > 260 V`인 right-censored 결과로 기록한다.

`E_ign = ∫_0^{t_ign} V I dt`와 2 s 누적 입력에너지는 삭제하지 않고 diagnostic으로 보존하지만 NSGA-II ranking에는 사용하지 않는다. `t_ign`, `U_alpha(2s)`, `C_J`는 계속 고정 reference voltage(기본 260 V)에서 계산한다. 따라서 V_min을 찾기 위해 낮은 전압을 쓰더라도 다른 세 목적함수의 비교 조건은 바뀌지 않는다.

V_min trial은 threshold 판정에만 쓰므로 점화가 일어난 trial은 C++ solver가 그 순간 조기 종료할 수 있다. 점화하지 않은 trial은 2 s 전체를 계산한다. 이 inner voltage search 때문에 후보당 physics 비용은 v7.9.3보다 증가하며, 기본 20–260 V / 5 V 설정에서는 reference 1회 + lower-bound 1회 + 대략 6회의 bisection으로 **최대 약 8회의 물리계산**이 일반적이다.

## 핵심 모델

```text
configured full propellant domain (항상 100%)
          +
anodeContactMask / cathodeContactMask
(각각 목표 17.5%, 합계 목표 35%)
          ↓
A100 Torch/CUDA FP64 + buffered C++ CPU FP64 condensed-phase physics
          ↓
Python NSGA-II / Pareto selection / geometry evolution
```

- 추진제 계산영역: 기본 **20×20 mm 전체**; 실행 YAML로 변경 가능
- 양극 접촉면적 목표: 기본 **17.5%**; 실행 YAML 값을 baseline도 그대로 사용
- 음극 접촉면적 목표: 기본 **17.5%**; 실행 YAML 값을 baseline도 그대로 사용
- 총 전극 접촉면적 목표: 기본 **35%**
- 양극 component: 1–3개
- 음극 component: 1–3개
- 전체 component: 최대 4개
- 최소 양·음극 간격: 0.5 mm
- 물리 backend: hybrid A100 Torch/CUDA FP64 + C++17 CPU FP64; 기존 `cpp_fp64_cpu`도 보존
- 기상 CFD·OpenFOAM·flame-progress `F`: 포함하지 않음

분리된 같은 극성 component는 2D 계산영역 밖의 **backside/out-of-plane hidden bus**로 동일 terminal에 연결되어 있다고 가정한다. 실제 제작에서도 이 연결을 구현하거나, 해당 component 조합을 사용하지 않아야 한다.

## 형상 생성과 진화

Generation 0의 topology template는 다음 component 조합을 순환해 포함한다.

```text
(Na, Nc) = (1,1), (1,2), (2,1), (2,2), (3,1), (1,3)
Na ≤ 3, Nc ≤ 3, Na + Nc ≤ 4
```

각 component는 독립적인 시작점과 heading을 가진다. 이후 NSGA-II mutation은 기존 segment·branch 연산 외에 다음을 수행한다.

```text
add_component
remove_component
split_component
merge_component
```

crossover는 polarity 전체를 통째로 교환하지 않고 component 단위로 교환·대체한다. 96×96 생성 mask와 193×193 physics grid에서 component 수, 극성 겹침, 직접 접촉, 최소간격, 면적 목표를 다시 검사한다.


## v7.9.2: hidden-bus 2+2 area-matched staggered reference

NSGA-II가 종료되면 현재 실행 설정과 동일한 조건으로 다음 고정 staggered 기준형상 한 개를 자동 생성·평가한다.

```text
상단 hidden anode bus (접촉 mask 밖)
        │        │
        A   C    A   C      ← 좌우 contact order
            │        │
하단 hidden cathode bus (접촉 mask 밖)
```

- 양극 contact finger 정확히 2개, top boundary에서 아래로 연장
- 음극 contact finger 정확히 2개, bottom boundary에서 위로 연장
- 같은 극성 팔은 동일 길이, 네 팔은 동일 폭
- 양·음극이 y 방향으로 깊게 겹치는 staggered/interdigitated 구조
- 같은 극성 접합부는 out-of-plane/backside hidden bus
- hidden bus는 추진제 contact mask에 포함하지 않아 접촉면적·BV 발열 기여가 0

기준형상은 추천 AI 형상의 실제 physics-grid 극당 contact area 평균을 우선적으로 맞추며, 동일 domain, 폭 범위, 최소간격, 전압, 2 s horizon, physics grid와 C++ FP64 solver를 사용한다. 정수 raster search는 design/physics grid에서 면적·간격·두 팔의 동일 길이·동일 폭 및 최소 45% 수직 중첩을 모두 검증한다.

visible contact component는 극당 2개이지만 hidden bus를 포함한 electrical terminal은 극당 1개다. 그러므로 AI search가 1+1 component로 제한된 run에서도 이 post-optimization baseline만 명시적·제한적 override로 평가할 수 있다. 이 override는 일반 NSGA-II candidate에는 적용되지 않는다.

baseline은 initial population, crossover/mutation parent, Pareto front 또는 최종 AI 추천 선택에 들어가지 않는다. AI 추천형상을 먼저 동결한 뒤 독립 reference로만 계산한다.

기존 완료 run에는 다음처럼 적용한다.

```bash
bash tools/evaluate_area_matched_staggered_existing_run.sh \
  "$PWD/runs/기존_run_폴더" \
  "/실행에/사용한/YAML/절대경로"
```

20%/극·폭 1–5 mm·간격 2 mm 실험에서는 실제 YAML을 두 번째 인수로 명시하는 것이 안전하다.

## surface-contact 물리 의미

기존의 잘못된 해석:

```text
전극 mask cell = 추진제를 제거한 hole / fixed-potential material cell
```

v7.9.0 물리모델:

```text
propellantDomainMask = 모든 셀에서 true
anode/cathode mask   = top-surface contact label
```

따라서 전극 아래에서도 다음 상태를 계속 적분한다.

\[
T,\quad c_{\mathrm{Li^+}},\quad c_{\mathrm{ClO_4^-}},\quad
c_{\mathrm{H_2O}},\quad \alpha,\quad \text{liquid fraction}
\]

전극 접촉은 thin-layer 전위방정식의 면적 Robin/Butler–Volmer source로 들어가며, 전극 footprint의 실제 면적 \(h^2\)로 계면전류와 발열을 적분한다.

## 목적함수

NSGA-II는 다음 네 값을 최소화한다.

\[
\min\left[
 t_{\mathrm{ign,cond}}(V_{\mathrm{ref}}),\;
 U_\alpha(2\,\mathrm{s};V_{\mathrm{ref}}),\;
 V_{\min,\mathrm{ign}},\;
 C_J(V_{\mathrm{ref}})
\right]
\]

- `t_ign,cond`: reference voltage(기본 260 V)에서 622.15 K 응축상 분해 개시 기준 최초 도달시간
- `U_alpha(2s)`: reference voltage에서 전체 추진제 domain의 2초 미분해 분율
- `V_min,ign`: 동일 점화 기준을 설정 horizon(기본 2 s) 안에 만족시키는 최소 인가전압의 bracketed conservative upper bound
- `C_J`: reference voltage에서 시간 전체의 최대 `J99 / mean(J)`

V_min search의 기본 설정:

```yaml
minimum_ignition_voltage_search:
  enabled: true
  lower_bound_V: 20.0
  upper_bound_V: 260.0
  tolerance_V: 5.0
  maximum_bisection_iterations: 8
  stop_successful_trials_at_ignition: true
```

각 voltage trial은 reference run과 동일한 C++ FP64 electrochemical/thermal/decomposition physics를 사용한다. threshold 판정에는 ignition success뿐 아니라 numerical cap 기준도 적용한다. sampled voltage에서 저전압은 점화하지만 더 높은 전압은 점화하지 않는 비단조 응답이 관측되면 V_min search를 invalid로 표시한다.

`inputElectricalEnergyToIgnition_J`와 `inputElectricalEnergyAt2s_J`는 계속 결과 JSON/CSV에 저장되지만 **diagnostic only**이며 Pareto ranking에는 들어가지 않는다.

현재 물성·반응계수는 `nominal_unvalidated`이다. 따라서 `V_min` 역시 실험 보정 전에는 절대적인 실제 점화전압 예측치라기보다 **동일 모델·동일 조건에서 형상 간 저전압 점화성을 비교하는 계산지표**로 해석해야 한다.

## 설치

### A100/Linux 또는 Kubernetes Pod

CUDA 지원 PyTorch가 설치된 환경이어야 한다. 확인:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY
```

그다음:

```bash
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
sudo apt-get update && sudo apt-get install -y build-essential
```

### Mac/CPU reference backend

```bash
xcode-select --install

conda create -n ecsp-cpp python=3.11 -y
conda activate ecsp-cpp
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
```

## 1. 필수 preflight

```bash
bash tools/run_m2_cpp_fp64_preflight.sh \
  "$PWD/runs/v794_surface_preflight"
```

성공 기준:

```text
C++ FP64 preflight: successful=4 rejected=0
```

## 2. surface-contact·다중 component 전용 검증

```bash
bash tools/validate_surface_contact_cpp.sh \
  "$PWD/runs/v794_surface_validation"
```

이 검증은 geometry mutation/crossover, full-propellant mask, C++ surface-contact 해석 및 4형상 preflight를 함께 확인한다.

## 3. 100개 × 2세대 pilot

```bash
bash tools/run_m2_cpp_fp64_100x2.sh \
  "$PWD/runs/v794_100x2"
```

개체수와 세대수는 두 번째·세 번째 인수로 바꿀 수 있다.

```bash
bash tools/run_m2_cpp_fp64_100x2.sh \
  "$PWD/runs/v792_200x3" 200 3
```

동일 workdir은 재사용하지 않는다. 이전 rejection·fitness 파일과 새 결과가 섞이는 것을 막기 위해 실행기가 기존 경로를 거부한다.

## 최종 출력

feasible 점화형상이 존재하면:

```text
<workdir>/final/
├── final_pareto_designs.csv
├── recommended_design_metrics.csv
├── recommended_design.json/.npz/.png
├── area_matched_staggered.json/.npz/.png
├── area_matched_staggered_metrics.csv
├── recommended_vs_area_matched_staggered.csv
├── recommended_vs_area_matched_staggered.json
└── recommended_vs_area_matched_staggered.png
```

2초 내 feasible 점화형상이 없으면 임의 추천을 만들지 않고:

```text
NO_FEASIBLE_IGNITING_DESIGN.json
```

을 저장한다.

## 중요한 해석 제한

1. 여러 분리 component는 hidden bus 연결을 전제로 한다.
2. 전극은 zero-thickness surface contact로 모델링하며, 실제 전극 두께·열용량·면내 금속전도는 별도 solid region으로 풀지 않는다.
3. 기상 화염·유동·표면 열피드백은 포함하지 않는다.
4. 경화 후 잔류수분, 전도도, 확산계수, BV kinetics, 반응열은 실험 보정이 필요하다.
5. 전극면적 17.5%/극은 이번 설계조건이며, 최적 면적이라는 실험적 결론은 아니다.

자세한 변경사항과 검증범위는 다음 문서를 참조한다.

- `docs/HYBRID_A100_CPU48_V7_9_5_KR.md`
- `docs/VALIDATION_REPORT_V7_9_5_KR.md`
- `docs/CHANGELOG_V7_9_5_KR.md`
- `docs/AREA_MATCHED_STAGGERED_BASELINE_V7_9_2_KR.md`
- `docs/SURFACE_CONTACT_MULTICOMPONENT_V7_9_0_KR.md`
- `docs/CHANGELOG_V7_9_0_KR.md`
- `docs/CHANGELOG_V7_9_2_KR.md`
- `docs/VALIDATION_REPORT_V7_9_2_KR.md`
