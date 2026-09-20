> **보관 문서:** 현재 구현은 `../MPS_ROBUST_SPD_PCG_V7_7_5_KR.md`를 기준으로 합니다.

# MPS 최종 수정: BV 전류균형·검증·예상 실행시간

## 1. 이번 오류의 실제 원인

기존 nonlinear Robin 반복은 각 전극면의 Butler–Volmer 식과 벌크 전위장을 반복해서 계산했지만, 벌크 전해질 전위의 **절대 기준값(상수 gauge offset)**을 실제로 조절하지 않았다. 따라서 다음과 같은 정지점이 가능했다.

```text
dphi = 0
dI   = 0
anode/cathode current balance = 1
```

이는 반복값은 더 이상 변하지 않지만 양극 총 Faradaic current와 음극 총 Faradaic current 중 한쪽이 사실상 0인 비물리적 상태다. `currentBalanceTolerance`를 완화해서 통과시킬 문제가 아니며, 전류보존 조건 자체를 nonlinear solve에 넣어야 한다.

## 2. 최종 해법

전해질의 공간적 전위형상을 \(\tilde\phi(\mathbf{x})\), 조정할 상수 offset을 \(c\)라고 두면

\[
\phi(\mathbf{x})=\tilde\phi(\mathbf{x})+c
\]

이고, 금속 전극 전위는 각각 260 V와 0 V로 고정한다. 각 offset에서 모든 BV Robin face를 암시적으로 푼 뒤 다음 전류보존식을 만족시키는 \(c\)를 찾는다.

\[
f(c)=I_a(c)-I_c(c)=0
\]

각 face에서 BV 미분기울기를 \(k=\partial j_{BV}/\partial\eta\), 벌크 face conductance를 \(G=\sigma/h\)라고 하면, interface face를 제거한 정확한 국소 gauge 민감도는

\[
\frac{kG}{k+G}
\]

이다. 새 코드는 이 민감도를 이용한 Newton step을 사용하되, step이 물리적 bracket을 벗어나거나 민감도가 작으면 midpoint bisection으로 전환한다. 따라서 빠른 수렴성과 안전한 bracket 수렴을 함께 확보한다.

## 3. 포함된 수치 안전장치

- MPS 비정수배 adaptive pooling 제거: 3×3 full-weighting restriction 사용
- BiCGStab 수렴은 recurrence residual이 아니라 최종 true residual \(b-Ax\)로 판정
- MPS FP32 potential tolerance: relative `1e-5`, absolute `1e-6`
- MPS local Robin pilot tolerance: absolute `0.1 A/m²`, relative `5e-4`
- FP32에서 interface voltage의 ULP보다 작은 residual을 강제하지 않도록 계산된 roundoff floor를 적용하고, 해당 face 수를 별도 진단값으로 기록
- outer potential under-relaxation: `0.35`
- global gauge balance: safeguarded Newton–bisection, 최대 24회
- 전류균형 허용치: 상대 0.5%; 완화하지 않음
- 각 batch 완료 후 성공/거부/경과시간 출력
- 미점화 후보는 최종 feasible로 인정하지 않되, 최고온도의 점화온도 부족량을 연속 constraint로 사용하여 NSGA-II가 점화에 가까운 형상을 우선 진화

## 4. 검증 결과

### 자동시험

```text
20 passed
```

검증 범위에는 MPS full-weighting, true-residual BiCGStab, FP32 profile, no-F contract, NSGA-II core 및 새 global gauge regression이 포함된다.

### 의도적으로 잘못 편향한 gauge 회귀시험

전해질 전위를 약 220–235 V로 편향한 뒤 balance solver를 적용했다.

```text
initial mean gauge ≈ 227.5 V
resolved gauge      ≈ 129.9971 V
Ia                  ≈ 0.07298078 A
Ic                  ≈ 0.07298038 A
relative mismatch   ≈ 5.50e-6
gauge iterations    = 3
local unresolved faces = 0
```

### 사용자의 실제 실패 형상 대표본

`G000_T00_b5964f_V000`을 33×33 CPU FP64 one-step으로 계산했다.

```text
Ia                  = 0.0051438898 A
Ic                  = 0.0051439120 A
relative mismatch   = 4.31e-6
gauge iterations    = 2
gauge offset        = 105.2301 V
linear true residual= 5.36e-16
```

즉 기존 `balance=1.0` 정지점이 사라지고 동일 전류보존 기준을 충분히 만족했다. 상세 JSON은 `docs/validation_data/representative_geometry_fp64_one_step.json`에 있다.

현재 제작 환경에는 Apple MPS 장치가 없으므로 실제 M2 GPU end-to-end는 사용자의 Mac에서 포함된 preflight로 최종 확인해야 한다.

## 5. 권장 실행 순서

### 전용 환경

```bash
conda create -n ecsp-mps python=3.11 -y
conda activate ecsp-mps
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
```

### 4개 형상·1 timestep preflight

```bash
bash tools/run_m2_mps_preflight.sh "$PWD/runs/m2_final_preflight"
```

성공 기준:

```text
MPS preflight: successful=4 rejected=0
```

### 4개 형상·2초 실제 benchmark

```bash
bash tools/benchmark_m2_mps_batch4.sh "$PWD/runs/m2_batch4_benchmark"
```

완료되면 `m2_batch4_benchmark_summary.json`에 1세대·3세대·5세대 환산시간이 기록된다.

### 전체 1,000개 × 최대 5세대

```bash
bash tools/run_nsga2_m2_mps.sh "$PWD/runs/nsga2_m2_final"
```

이 명령은 먼저 별도 preflight를 자동 수행한다. 이미 검증한 환경에서 preflight를 생략하려면 다음처럼 실행한다.

```bash
ECSP_SKIP_MPS_PREFLIGHT=1 \
bash tools/run_nsga2_m2_mps.sh "$PWD/runs/nsga2_m2_final"
```

항상 새 workdir을 사용한다. 이전 `physics_rejection.txt`와 metric 파일이 새 결과에 섞이는 것을 막기 위해 실행 스크립트가 기존 workdir을 거부한다.

## 6. M2 Pro 예상시간

코드는 193×193 grid, \(dt=2.5\times10^{-4}\,s\), 2초 horizon이므로 형상당 8,000 timestep이며 전기장은 약 801회 갱신된다. 사용자의 실제 실패 로그에서는 full-grid nonlinear outer iteration 120회가 후보당 약 2분 20초 걸렸다. 새 gauge solver는 초기 대표형상에서 약 23 outer iteration, gauge는 보통 2–3회에 수렴했지만, 2초 동안의 801개 warm-start electrical update 비용은 여전히 크다.

현재 근거로 보수적으로 추정하면:

| 범위 | M2 Pro 예상 |
|---|---:|
| 4개 full-horizon batch | 약 30–90분 |
| 1,000개 1세대 | 약 5–16일 |
| 조기종료 3세대 | 약 2–7주 |
| 최대 5세대 | 약 4–11주 |

이 범위는 Mac의 메모리 대역폭, 열 스로틀링, 형상별 nonlinear iteration 수에 따라 크게 달라진다. 이전의 3–7일 추정은 801회의 전기장 재해석 비용을 과소평가한 값이다. 따라서 논문용 전체 실행 전에 `benchmark_m2_mps_batch4.sh`의 실측값으로 반드시 갱신해야 한다.

M2에서 전체 5세대를 수행하는 것은 가능하더라도 실용성이 낮을 수 있다. 현실적인 운용은 M2에서 100–200개·1세대 pilot과 코드 검증을 수행하고, 1,000×5 production은 A100 FP64 또는 CUDA 환경에서 돌린 뒤 MPS는 탐색 보조 backend로 사용하는 방식이다.
