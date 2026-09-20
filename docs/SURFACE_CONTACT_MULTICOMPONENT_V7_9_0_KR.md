# v7.9.0 모델 구조 — full propellant surface contact + multi-component geometry

## 1. 변경 목적

v7.8.1은 전극 mask 셀을 고정전위 물질영역으로 만들고 그 위치를 추진제 상태변수 계산에서 제외했다. 표면에 전극이 접촉하는 실제 구조에서는 전극 면적이 증가해도 추진제 질량이 줄지 않으므로, v7.9.0은 물질영역과 접촉영역을 분리한다.

\[
\Omega_p = [0,20\,\mathrm{mm}]^2
\]

은 항상 전체 추진제이며,

\[
\Gamma_a,\;\Gamma_c\subset\Omega_p
\]

는 상부 표면에 투영된 양극·음극 접촉 footprint이다.

## 2. 설정값

| 항목 | v7.9.0 |
|---|---:|
| 추진제 domain | 20×20 mm |
| 양극 접촉면적 목표 | 17.5% |
| 음극 접촉면적 목표 | 17.5% |
| 총 접촉면적 목표 | 35% |
| 생성 mask | 96×96 |
| physics grid | 기본 193×193 |
| 최소 양·음극 간격 | 0.5 mm |
| 극당 component | 1–3 |
| 전체 component | 최대 4 |

면적 tolerance는 `0.035`이며, 생성 mask뿐 아니라 193×193 resize 후에도 다시 검사한다.

## 3. 형상 component 표현

각 polarity는 다음과 같이 여러 root component를 가진다.

```text
anode.components   = [root_0, root_1, ...]
cathode.components = [root_0, root_1, ...]
```

각 root는 독립적으로 다음 값을 가진다.

```text
start = [x_mm, y_mm]
initial_heading_deg
LINE / ARC tree
width_mm
length_mm
branch data
```

Generation 0은 `(1,1),(1,2),(2,1),(2,2),(3,1),(1,3)`을 포함한다. mutation은 component 생성·삭제·분할·병합을 수행하며, crossover는 개별 component를 전달한다.

## 4. hidden bus 가정

2D surface mask에서 서로 떨어진 같은 극성 island에 동일 terminal 전위를 적용하려면 실제 구조에 별도의 연결도체가 있어야 한다. v7.9.0은 다음을 가정한다.

> 모든 분리된 같은 극성 surface-contact component는 계산영역 밖의 backside 또는 out-of-plane conductor로 동일 terminal에 연결된다.

이 가정은 계산 설정과 출력 metadata에 기록된다. 실제 제작에서 hidden bus를 만들지 않는다면 multi-component 후보는 물리적으로 실현되지 않으므로 component 수를 1로 제한해야 한다.

## 5. 전위·계면 결합

전체 추진제 박막에서 bulk 전도식을 풀고, contact footprint에는 Butler–Volmer 계면전류를 면적 source로 결합한다.

\[
-\nabla\cdot(\sigma\nabla\phi)
=
\frac{j_a-j_c}{\delta_s}
\]

여기서 source는 해당 polarity contact에서만 활성화된다. Newton/Picard 선형화에서 비음이 아닌 BV slope가 대각에 추가되어 전위 선형계의 SPD 성질을 보존한다.

계면 반응의 전위강하는 `contactNormalConductionLength_m`를 사용하고, 총 계면전류는 contact footprint 면적 \(h^2\)로 적분한다. 전극 mask는 추진제 상태변수를 0으로 만들지 않는다.

## 6. 추진제 질량 보존 의미

모든 grid cell에서:

```text
propellant = true
```

이므로 접촉면적을 바꿔도 다음은 변하지 않는다.

- 추진제 cell 수
- 면적평균 분해율의 분모
- 초기 Li⁺/ClO₄⁻/water inventory
- 열용량과 추진제 질량

변하는 것은 contact footprint, BV source 분포, 계면발열, 전류장과 그에 따른 열·분해 성능이다.

## 7. 남는 모델 한계

이 버전은 표면 접촉 모델의 일관성을 고친 것이지 3D 전극을 완전하게 해석한 것은 아니다.

- 전극 금속의 면내 전위강하와 열전도는 계산하지 않음
- 전극 두께·열용량을 별도 solid region으로 두지 않음
- hidden bus의 저항·발열을 계산하지 않음
- gas-phase reacting CFD와 flame feedback을 계산하지 않음
- 물성·kinetics는 실험 보정 전 nominal 값

따라서 현재 결과는 동일 접촉면적·동일 모델조건에서 topology를 비교하는 설계 연구용이다.
