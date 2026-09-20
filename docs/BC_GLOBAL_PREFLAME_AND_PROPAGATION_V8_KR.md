# v8.2.0 corrected B/C global pre-flame + 응축상 propagation 모델

## 1. 모델 범위

corrected 경로는 legacy v7.9.5를 삭제하지 않고
`bc_global_preflame`, `bc_global_native` 또는 `bc_global_native_hybrid`
evaluator로 활성화되는 B/C 모델이다. v8.2 production profile은 full-domain
propellant와 서로 분리된 anode/cathode top-surface footprint mask를 사용한다.

```text
전극 CAD/Mask
  → B/C pre-flame: 전위·이온수송·국부 BV·열전달·global 분해
  → 4-objective pre-flame Pareto + near-Pareto/diverse 10–20%
  → post-onset condensed reaction-progress/level-set refinement
  → 8-objective final Pareto
  → 동일 면적 staggered의 동일 B/C + propagation 비교
```

이 경로는 **외부 화염, plume, 기상 Navier–Stokes/OpenFOAM CFD를 풀지 않는다.** 후단은 점화 이후의 응축상 반응진행과 reaction front를 평가하는 저비용 모델이다.

## 2. B/C pre-flame 지배식

사용하는 논문식 번호는 다음과 같다.

```text
사용:     (2), (4)–(9), (11), (13), (15)–(17), (22)–(24)
미사용:   (3), (18), (25)–(31)
식 (32): qJ + qechem 검산용 진단값. 식 (23)에 두 번째 열원으로 더하지 않음.
```

### 전기장·전기적 중성

\[
\mathbf E=-\nabla\phi,
\qquad
\sum_i z_i c_i=0.
\]

### 무대류 Nernst–Planck 수송

\[
\mathbf N_i
=-D_i(T)\nabla c_i
-\frac{z_iF D_i(T)c_i}{RT}\nabla\phi.
\]

\[
\frac{\partial c_i}{\partial t}+\nabla\cdot\mathbf N_i
=R_i^{\rm chem}.
\]

### 이온전류와 전류보존

\[
\mathbf J=F\sum_i z_i\mathbf N_i,
\qquad
\nabla\cdot\mathbf J=0.
\]

Strict B/C profile에서는 별도의 `sigma_ion E`를 더하지 않는다. 이온 migration은 이미 Nernst–Planck flux에 포함되기 때문이다. 전자전도는 `bcGlobal.augmentedElectronicConduction.enabled=false`로 명시적으로 비활성화되어 있다.

### 국부 Butler–Volmer와 Faraday 계면 flux

\[
\eta_e(\mathbf x,t)=\phi_{s,e}-\phi(\mathbf x,t)-E_{{\rm eq},e},
\]

\[
j_e=j_{0,e}
\left[
\frac{c_{\rm Ox}}{c_{\rm Ox}^*}e^{\alpha_a nF\eta_e/(RT)}
-
\frac{c_{\rm Red}}{c_{\rm Red}^*}e^{-\alpha_c nF\eta_e/(RT)}
\right],
\]

\[
-\mathbf n\cdot\mathbf N_i
=\sum_e\frac{\nu_{i,e}j_e}{n_eF}.
\]

자유형 전극의 각 접촉 cell에서 국부 전해질 전위를 사용한다. v8.2
nonlinear-Robin 해석은 접촉 perimeter가 아니라 footprint 전체에 BV를 적용한다.
총전류는 각 footprint cell에서 `j * dx²`를 합산하고, 표면 반응 source는
surface-layer 두께로 나눠 체적항으로 바꾼다. 접촉 mask는 bulk potential을
고정하거나 추진제 셀을 제거하지 않는다.

### 에너지보존

\[
\rho c_p(T)\frac{\partial T}{\partial t}
=
\nabla\cdot[k(T)\nabla T]
+q_J'''+q_{\rm echem}'''+q_{\rm chem}'''-q_{\rm loss}'''.
\]

기본 `electrical.jouleHeatModel=conductive_sigma_E2`는 harmonic internal face에서
계산한 비음수 비가역 전도열과 unresolved contact-normal resistor 열을 사용한다.
선택적 `total_j_dot_e`는 diffusion cross-term을 포함하는 signed electrical-energy
transfer이며 국소 음수값을 clip하지 않는다. 따라서 일반적으로 기본 열원을
단순한 cell-centred `J·E` 하나로 동일시하지 않는다. `legacy_magnitude`는 과거
호환용 비음수 closure다.

계면 전기화학 열은 기존 surface-contact solver가 실제 cell 면적과 reaction-layer 두께를 일관되게 환산한 `qTotal_W_per_m3`를 사용한다. Strict B/C profile에서는 논문 식 (23)에 맞추어 별도 activation heat `jη` 추가를 끄고, Faraday molar rate × 지정 reaction enthalpy만 사용한다. 열전도는 유한체적 harmonic face conductivity로 계산하고, 대류·복사는 thin-layer 체적손실로 적용한다.

## 3. 3번 논문 global chemistry

Wet recipe와 같은 non-metallized global reaction을 사용한다.

\[
1.45\,\mathrm{LiClO_4}+[-\mathrm{C_2H_4O}-]_n
\rightarrow
2\,\mathrm{CO_2}+2\,\mathrm{H_2O}+0.4\,\mathrm{O_2}+1.45\,\mathrm{LiCl}.
\]

상세 12-step을 풀지 않고 두 주요 exotherm의 conversion-dependent kinetics를 사용한다.

\[
\dot\alpha_r
=
\exp\left[g_r(\alpha_r)-\frac{E_r(\alpha_r)}{RT}\right],
\qquad r=1,2.
\]

\[
X_g=\omega_1\alpha_1+\omega_2\alpha_2,
\qquad \omega_1+\omega_2=1.
\]

LP/PVA inventory는 global reaction extent로 한 번만 차감한다. 두 kinetic channel 각각에 global stoichiometry를 독립 적용하여 반응물을 이중 소비하지 않는다.

## 4. 실제 cured composition

입력 wet recipe:

| 성분 | 초기 wt.% |
|---|---:|
| LiClO4 | 31.58 |
| H2O | 58.42 |
| PVA | 9.00 |
| H3BO3 | 1.00 |
| glycerol | 0 |
| W | 0 |

실제 solver 초기조건은 wet recipe가 아니라 cured composition이다. `cured_water_mass_fraction=0.20`은 3번 논문의 저온 TGA 질량감소를 근거로 한 **literature nominal, uncalibrated** 값이다. 건조고형분 비율은 LP:PVA:H3BO3 = 31.58:9:1로 유지된다. 향후 측정값이 있으면 이 한 값만 교체하면 조성·농도·밀도가 다시 계산된다.

## 5. pre-flame 네 목적함수

모두 minimisation vector로 저장한다.

1. **Condensed decomposition onset time**

\[
t_{\rm onset}=\min\left\{t:
\frac{A[T\ge T_{\rm onset}\ \land\ X_g\ge X_{\rm crit}]}{A_p}
\ge f_{\rm crit}\right\}.
\]

기본값은 `T_onset=523.15 K`, `Xcrit=0.01`, `fcrit=0.01`. visible flame ignition이 아니라 응축상 분해 onset이다.

2. **Remaining reactive LP+PVA mass fraction**

\[
U_{\rm rem}(t_f)=
\frac{\int_\Omega [M_{LP}c_{LP,react}+M_{PVA}c_{PVA,react}]_{t_f}\,dV}
{\int_\Omega [M_{LP}c_{LP,react}+M_{PVA}c_{PVA,react}]_{0}\,dV}.
\]

기존 `1-<alpha>` 대신 실제 mobile LP와 reactive PVA inventory의 질량보존으로
계산한다. 정식 objective 이름은
`area_undecomposed_fraction_at_evaluation_time`이다. 기존
`area_undecomposed_fraction_at_2s` key는 같은 값을 갖는 deprecated 호환 alias로만
남는다.

3. **Minimum onset voltage**

동일한 T+Xg+면적 criterion을 설정된 evaluation horizon 안에 만족하는 최소
수치유효 전압을 bracketed bisection으로 찾고, 마지막 igniting upper bracket을
full-horizon 재검증한 뒤 보수적으로 보고한다.

4. **Current congestion**

\[
C_J=\max_{0<t\le t_{\mathrm{eval}}}\frac{J_{99}(t)}{\bar J(t)}.
\]

지배법칙이 아니라 전류분포에서 직접 계산되는 engineering descriptor이다.
`peakCurrentCongestionToEvaluationTime`이 objective 값이며, 호환용
`peakCurrentCongestion`은 전체 계산 horizon의 별도 진단값이다.

## 6. post-onset condensed propagation

B/C onset 시점의 전체 2-D field를 넘긴다.

```text
T(x,y,t_onset)
Xg(x,y,t_onset)
alpha1(x,y,t_onset), alpha2(x,y,t_onset)
mobile LP/water, reactive PVA, generated-water product
cumulative electrochemical LP consumption
potential, xiMax, initial reactive-inventory scalars
qJ(x,y,t_onset), qechem(x,y,t_onset) diagnostics
propellant mask
```

점화 좌표 하나를 강제로 지정하지 않는다. `Xg` front로부터 signed-distance level
set을 재초기화하고, 동일 두-channel chemistry와 열전도를 계속 적분한다. 현재
corrected propagation은 닫힌 post-onset 전기화학/species solver가 없으므로 전기
열원은 0이며, 저장된 pre-flame `qJ/qechem` 시간장을 재생하지 않는다. 완전하고
보존식에 맞는 onset inventory가 없거나 provided electrical-history 모드를 요청하면
fail-closed한다.

출력:

- `finalUnreactedAreaFraction`
- `meanEffectiveRegressionVelocity_m_per_s`
- `maximumEffectiveRegressionVelocity_m_per_s`
- `establishedTimeAfterOnset_s`
- `reactionFrontNonuniformity`
- T/Xg/level-set snapshots 및 front arrival time

`reactionFrontNonuniformity`는 도착시간 변동계수와 최종 미반응 면적을 합한 refinement descriptor이며 fundamental law로 주장하지 않는다.

## 7. 선택·비교 정책

- pre-flame Pareto는 전부 유지하되 설정 상한을 넘으면 crowding-distance로 대표 후보를 보존한다.
- 여기에 pre-flame rank 1–3에서 max-min geometry diversity로 nominal 15%를 추가한다.
- 최종 추천은 4 pre-flame + 4 propagation objective의 final Pareto에서 normalised utopia distance로 선정한다.
- area-matched 2-anode/2-cathode hidden-bus staggered는 optimization에 넣지 않고 AI 추천 동결 후 동일 B/C와 동일 propagation으로 평가한다.

## 8. 모델 주장 범위

가능한 주장:

> B/C-based pre-flame electrochemical–thermal and post-onset condensed reaction-propagation ranking of electrode geometries.

불가능한 주장:

- detailed LP/PVA elementary mechanism 규명
- HClO4·acetaldehyde·polyene 등 중간종의 정확한 정량예측
- 외부 gas flame/plume CFD
- visible flame spread, gas-phase unburned fraction, thrust의 직접예측

현재 A100 profile의 수송·BV·열물성 및 Fig. 8 digitisation은 literature nominal이므로 실험보정 전 결과는 pipeline development와 sensitivity/ranking 연구용이다.
