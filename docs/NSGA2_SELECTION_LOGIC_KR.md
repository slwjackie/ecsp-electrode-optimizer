# 1,000개 후보에서 NSGA-II와 70+20+10이 작동하는 방식

## 1. 모든 후보를 먼저 평가

제작성 제약을 통과한 1,000개 후보 각각에 대해

\[
\mathbf f_i=
[t_{\mathrm{ign,cond}},U_\alpha(2s),E_{\mathrm{in}}(2s),C_J]
\]

를 계산한다. 따라서 70+20+10은 해석 대상을 줄이는 screening 단계가 아니다.

## 2. Constraint-domination

Deb의 constraint-domination 원칙을 적용한다.

1. feasible 후보는 infeasible 후보를 지배한다.
2. 둘 다 infeasible이면 총 violation이 작은 후보가 우선한다.
3. 둘 다 feasible이면 ordinary Pareto dominance를 적용한다.

제작성, solver 미수렴 및 과도한 limiter/cap 사용이 constraint violation에 반영된다.

## 3. Non-dominated sorting

후보 \(a\)가 \(b\)보다 모든 목적에서 나쁘지 않고 하나 이상에서 더 좋으면

\[
a\prec b
\]

이다. 지배받지 않는 후보가 Front 1, Front 1을 제거한 뒤 지배받지 않는 후보가 Front 2가 된다.

## 4. Crowding distance

같은 front 안에서는 objective-space에서 주변 후보와 멀리 떨어진 후보에 큰 crowding distance를 부여한다. 이는 빠른 점화형, 낮은 에너지형, 낮은 혼잡도형 등 서로 다른 trade-off 해를 보존한다.

## 5. 70+20+10 mating pool

### 70 performance

`(Pareto rank 오름차순, crowding distance 내림차순)`으로 정렬한 상위 70개를 선택한다.

### 20 topology coverage

각 topology의 최상위 후보를 하나씩 확보한다. 이미 performance 70에 포함된 topology는 중복 추가하지 않으며, 부족한 수는 다음 NSGA-II 상위 후보로 보충한다.

### 10 max-diversity

LINE/ARC/BRANCH 개수, branch depth, component 수, 길이·곡률 통계, 방향 entropy, 최소간격, perimeter, spatial dispersion으로 descriptor를 구성하고 robust scaling한다.

후보 \(i\)의 현재 선택집합 \(S\)에 대한 거리는

\[
D_i=\min_{j\in S}\|\tilde{\mathbf z}_i-\tilde{\mathbf z}_j\|_2
\]

이며 \(D_i\)가 가장 큰 후보를 하나씩 10번 추가한다.

## 6. Offspring 생성

1,000개 자식은 기본적으로 다음 비율이다.

- crossover: 600
- mutation: 250
- new grammar immigrant: 150

모든 자식은 다시 제작성 검사를 받고 같은 condensed-phase physics로 평가된다.

## 7. 진짜 NSGA-II 생존선택

\[
P_t(1000)+Q_t(1000)=R_t(2000)
\]

을 만든 뒤 non-dominated sorting과 crowding distance로 다음 세대 1,000개를 선택한다.

따라서 100개 mating pool은 번식 부모집합이고, **2,000→1,000이 elitist environmental selection**이다.

## 8. 최종 단일 추천형상

NSGA-II는 Pareto set을 반환하므로 단일 권장안은 각 목적을 final Pareto 범위에서 0–1 정규화한 뒤 utopia point까지의 Euclidean distance가 가장 작은 후보로 정한다.

\[
x^*=\arg\min_{x\in\mathcal P}
\|\tilde{\mathbf f}(x)-\mathbf 0\|_2.
\]

이는 물리단위가 다른 목적을 임의 가중합하는 것보다 투명하지만, 최종 의사결정 기준 자체는 별도로 보고해야 한다.
