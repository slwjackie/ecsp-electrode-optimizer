# v7.7.5 MPS robust SPD-PCG 구현

## 목적

v7.7.3~v7.7.4의 실제 M2 Pro preflight에서는 MPS FP32 Krylov solver가 일부 193×193 자유형 전극에서 잔차를 낮추지 못하거나 `1e11~1e12` 수준으로 폭주했다. v7.7.5는 tolerance를 느슨하게 하거나 CPU로 우회하지 않고, `_build_linear_system`이 만드는 대칭 양의 정부호 계를 그 구조에 맞게 다시 푼다.

MPS 탐색 프로필의 목표는 다음과 같다.

- 실제 true residual 기준 `relativeToleranceStatic/Coupled = 1e-6`
- FP32에서 발산이 구조적으로 억제될 것
- 실제 실행 method가 PCG인지 명시적으로 기록할 것
- 모든 연산을 MPS에 둔 채 실패 후보별로 복구할 것
- 최종 논문 후보는 CPU/A100 FP64로 재검증할 것

## 1. 대칭 Jacobi equilibration

원래 선형계를

\[
A x=b
\]

라고 하면, 양의 대각행렬 \(D=\operatorname{diag}(A)\)로

\[
S=D^{-1/2},\qquad \widehat A=SAS,\qquad \widehat b=Sb,
\qquad x=Sy
\]

를 구성한다. 변환된 계는 대칭성을 유지하며 활성 셀의 대각이 정확히 1이다. 전도도와 Robin 항이 만드는 대각 스케일 차이가 Krylov 내적에 직접 들어가지 않는다.

구현: `python/ecsp_v6/physics/potential.py::_equilibrate_linear_system`

## 2. correction-form PCG와 iterative refinement

절대 전위 \(x\sim260\,\mathrm{V}\) 자체를 계속 갱신하지 않고 warm start \(x_0\)에 대한 보정량을 푼다.

\[
r_0=b-Ax_0,
\qquad \widehat A\,\delta y=S r_0,
\qquad x_1=x_0+S\delta y
\]

각 보정 라운드 뒤 true residual \(b-Ax\)를 다시 계산하고, 필요하면 최대 3회 반복한다. recurrence residual만 tolerance를 통과한 경우는 수렴으로 인정하지 않는다.

구현:

- `_pcg_correction_core`
- `_solve_pcg_system`
- `solve_pcg`

## 3. FP32 breakdown 보호

PCG의 곡률은 dtype의 `tiny`가 아니라 벡터 크기에 상대적인 기준으로 판정한다.

\[
p^TAp > C\epsilon\,\lVert p\rVert_2\lVert Ap\rVert_2,
\qquad r^Tz>0
\]

또한 다음 상황에서는 해당 batch row만 재시작한다.

- NaN/Inf
- 양의 곡률 상실
- recurrence residual과 true residual의 과도한 불일치
- true residual이 지금까지의 최량값보다 설정 배수 이상 증가

재시작할 때 폭주한 현재 iterate가 아니라 저장해 둔 최량 iterate로 되돌린다. 후보 하나가 다른 batch 후보의 계산을 오염시키지 않는다.

## 4. MPS-resident fallback

PCG와 refinement 후에도 수렴하지 않은 행은 CPU로 보내지 않는다. 대칭 평형화된 계에서 residual 방향의 1차원 최소잔차 step

\[
\alpha=\frac{r^TAr}{(Ar)^T(Ar)}
\]

을 적용한다. 정확연산에서는 각 step의 residual norm이 증가하지 않는다. 최량 iterate만 채택하므로 fallback 자체가 해를 악화시키지 않는다.

## 5. solver routing 고정

`method=pcg`와 비대칭 legacy multigrid V-cycle을 함께 요청하면 더 이상 조용히 BiCGStab으로 바꾸지 않고 즉시 오류를 낸다. M2 프로필은 다음으로 고정한다.

```yaml
potentialSolver:
  methodStatic: pcg
  methodCoupled: pcg
  preconditioner: jacobi
  symmetricEquilibration: true
  correctionForm: true
  relativeToleranceStatic: 1.0e-6
  relativeToleranceCoupled: 1.0e-6
  absoluteTolerance: 1.0e-8
```

실제 결과에는 `finalElectricalSolverMethod`, restart 수, refinement 라운드, fallback 사용 여부가 저장된다.

## 6. MPS 취약 연산 제거

`python/ecsp_v6/physics/electrochem.py`에서 다음을 고정형 연산으로 교체했다.

- int64 `scatter_add_`
- 버전 의존적인 `scatter_reduce_(amax)`
- `index_add_` 및 `index_put_(accumulate=True)`

후보 batch가 작다는 점을 이용해 `[face,batch]` membership reduction을 사용한다. 따라서 이전 MPS deterministic warning을 발생시키던 atomic accumulate 경로를 사용하지 않는다.

`batch_quantile_masked`도 boolean advanced indexing + `torch.quantile` 대신, 고정형 masked sort와 두 order statistic의 선형보간으로 바꿨다. 현재밀도 p99는 전기장을 다시 계산하는 step에서만 갱신한다.

## 7. dtype-independent physical floors

전류·전류밀도·농도·전도도처럼 물리식과 보고값에 쓰이는 하한은 `torch.finfo(dtype).eps`가 아니라 `numerics.physicalFloors`를 사용한다. CPU FP64와 MPS FP32의 차이가 dtype epsilon 9자리 차이에서 인위적으로 만들어지지 않는다.

`species._limited_update`의 활성 판정도 농도 크기에 대한 상대 허용치와 명시적 농도 하한으로 변경했다.

## 8. local BV를 작은 전위강하 변수로 풂

기존에는 \(\phi_{face}\sim260\,\mathrm{V}\) 자체를 이분하여 FP32 ULP의 영향을 받았다. v7.7.5는

\[
g=|V_{electrode}-\phi_{face}|
\]

를 직접 미지수로 사용한다. BV overpotential도 `260 - phi_face`를 다시 계산하지 않고 이 작은 \(g\)에서 직접 산출한다. mass-transfer saturation 미분은

\[
\frac{dj}{d\eta}
=\frac{dj_{kin}}{d\eta}
\left(1+\frac{j_{kin}}{j_{lim}}\right)^{-2}
\]

형태로 계산해 Metal의 denormal flush 위험을 줄였다.

## 9. 강화된 preflight

`tools/run_m2_mps_preflight.sh`는 전체 physics 전에 다음 193×193 수치 gate를 먼저 실행한다.

1. 약 8% 면적의 맞물린 빗형 전극 구성
2. 설정된 전도도 최소~최대 범위의 이질장 적용
3. 행렬 대칭성 검사
4. 무작위 SPD Rayleigh quotient 검사
5. 평형화 후 unit diagonal 검사
6. 알려진 정확해 복원 및 true residual 검사
7. 실제 diagnostics method가 PCG인지 확인
8. int64 batch sum, batch max, fixed-shape quantile MPS 연산 검사
9. 이후 실제 4개 후보의 193×193 nonlinear one-step physics

명령:

```bash
bash tools/run_m2_mps_preflight.sh "$PWD/runs/m2_v775_preflight"
```

성공 기준:

```text
MPS preflight: successful=4 rejected=0
```

## 10. 빌드 환경 검증

Apple MPS 장치가 없는 빌드 환경에서 확인한 결과:

- pytest: 28개 통과
- 193×193 대표 빗형 알려진 해, CPU FP32:
  - true relative residual: 약 `5.1e-7`
  - method: `pcg_symmetric_equilibrated_correction_fp32_true_residual`
- 193×193 실제 경계전위 문제 CPU FP32 vs FP64:
  - 최대 전위차: 약 `4.8e-4 V`
  - RMS 전위차: 약 `7.6e-5 V`
  - RMS 차이에 따른 단순 BV 증폭 추정: 약 `1.0015×`
- 193×193, 4개 자유형 후보, FP32, nonlinear one-step: `4/4 successful`, `0 rejected`

실제 M2/MPS 커널 실행 여부는 사용자의 Apple Silicon에서 위 preflight가 `4/4`를 통과해야 최종 확인된다. 이 문서는 M2 실측을 대신하지 않는다.
