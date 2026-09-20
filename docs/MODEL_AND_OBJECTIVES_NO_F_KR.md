# F 제거형 응축상 모델과 네 목적함수

## 1. 활성 물리모델

활성 해석 경로는 다음으로 제한된다.

```text
전위·이온수송·계면반응
        ↓
Joule 및 전기화학 발열
        ↓
고체 열전도·대류/복사 손실·상변화
        ↓
global 응축상 열분해 진행도 α
```

OpenFOAM, gas Navier–Stokes, species reacting-flow 및 경험적 flame-progress 변수 `F`는 호출되지 않는다.

## 2. 열분해 진행도 \(\alpha\)

각 추진제 셀의 global chemical progress는

\[
0\le\alpha(\mathbf{x},t)\le1
\]

이며 코드에서는 다음 reduced global rate를 적분한다.

\[
\dot\alpha=
A\exp\!\left(-\frac{E_a}{RT}\right)
(1-\alpha)^n
\left(\frac{c_{\mathrm{ox}}}{c_{\mathrm{ox},0}}\right)^m
G_T(T).
\]

여기서

\[
G_T(T)=\left[1+\exp\left(-\frac{T-T_a}{20\,\mathrm K}\right)\right]^{-1}
\]

은 수치적으로 부드러운 temperature gate이다. Explicit update는

\[
\alpha^{k+1}=\operatorname{clip}
\left(\alpha^k+\Delta t\,\dot\alpha^k,0,1\right)
\]

이다.

현재 \(\alpha\)는 **검증된 12-step mass balance가 아니라 기존 v7.7.2의 global reduced conversion variable**이다. 따라서 \(1-\alpha\)를 실제 잔류질량과 동일시하지 않는다.

## 3. 목적함수 1 — 응축상 분해 개시시간

\[
t_{\mathrm{ign,cond}}
=
\min\left\{t:\max_{\Omega_p}T(\mathbf{x},t)\ge622.15\,\mathrm K\right\}.
\]

622.15 K는 non-metallized M0의 보고 분해 개시온도 349 °C에 대응한다. 이 값은 **gas-flame ignition이 아니라 condensed-phase decomposition onset의 operational definition**이다.

설정 파일에는 602.15/622.15/642.15 K 민감도 확인 범위도 기록되어 있다. 최종 논문에서는 임계온도에 대한 순위 안정성을 확인해야 한다.

점화하지 않은 후보의 목적값은

\[
t_{\mathrm{penalty}}=t_{\mathrm{end}}+2\,\mathrm{s}
\]

로 유한하게 변환되어 NaN이 NSGA-II 정렬을 깨뜨리지 않도록 한다.

연구 설정에서는 non-ignition 후보에 constraint violation도 부여하므로, 점화한 feasible 후보가 존재하는 한 최종 권장형상으로 선택되지 않는다. 모든 후보가 점화하지 않으면 가짜 권장형상을 출력하지 않고 실행을 실패로 종료한다.

## 4. 목적함수 2 — 2초 평균 미분해 진행분율

\[
U_{\alpha}(2s)
=
\frac{1}{A_p}
\int_{\Omega_p}
\left[1-\alpha(\mathbf{x},2s)\right]dA.
\]

격자에서는

\[
U_{\alpha}(2s)
=
\frac{1}{N_p}
\sum_{j\in\Omega_p}(1-\alpha_j)
\]

로 계산한다. 임의의 \(\alpha_{crit}\)를 두지 않는 threshold-free spatial average이다.

정확한 표현은 `area-averaged undecomposed progress fraction`이며 다음 표현은 사용하지 않는다.

- 실제 burned area
- gas unburned fraction
- stable combustion area

## 5. 목적함수 3 — 점화까지 필요한 입력 전기에너지

\[
E_{\mathrm{ign}}=
\int_0^{t_{\mathrm{ign,cond}}}
\Delta V\,I(t)\,dt.
\]

코드는 매 timestep의 \(P=\Delta V I\)를 trapezoidal integration하고, 응축상 점화 기준이 최초 충족되는 timestep의 누적에너지를 `inputElectricalEnergyToIgnition_J`로 저장한다. 따라서 점화 후 2초까지 소비한 에너지는 목적함수에 포함하지 않는다.

`inputElectricalEnergyAt2s_J`는 후처리 진단값으로 계속 저장한다. 2초 안에 점화하지 못한 후보는 진짜 \(E_{\mathrm{ign}}\)가 관찰되지 않은 right-censored case이므로, evaluation-horizon energy를 유한 placeholder로만 사용하고 ignition constraint가 해당 후보를 infeasible로 처리한다.

## 6. 목적함수 4 — 전류 혼잡도

각 시간에서

\[
C_J(t)=\frac{J_{99}(t)}{\overline{J}(t)}
\]

를 계산하고 목적함수는

\[
C_J=\max_{0\le t\le2s}C_J(t)
\]

이다. 단일 셀 \(J_{max}\)보다 격자 특이성에 덜 민감하면서 hotspot 위험을 반영한다.

## 7. 제거된 항목

활성 solver와 목적함수에서 다음을 제거했다.

- `flameProgress`
- \(D_F\nabla^2F\)
- seed/spread logistic source
- `qFlame`
- `burnedAreaFraction`
- `establishedIgnitionDelay`
- OpenFOAM 및 reduced CFD runtime

## 8. 논문상의 명칭

권장 명칭:

> NSGA-II optimization based on an electrochemical–solid-thermal–decomposition model

비권장 명칭:

> fully coupled combustion CFD optimization

> stable-flame optimization
