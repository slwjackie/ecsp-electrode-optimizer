# v7.9.4 최소 점화전압 목적함수 정의

## 연구 목적과의 대응

제3 목적함수는

\[
V_{\min,\mathrm{ign}}
\]

으로 정의한다. 이는 **동일한 condensed-phase ignition criterion을 유한한 평가시간 안에 처음 만족시킬 수 있는 최소 인가전압**을 직접 최소화하므로, "저전압 점화가 가능한 전극 형상"이라는 설계 목표와 직접 대응한다.

현재 ignition criterion은 local condensed-phase decomposition-onset temperature가 설정값(기본 622.15 K)에 처음 도달하는 것이다. 따라서 이 값은 gas-phase flame의 최소 점화전압이 아니라 **현재 condensed-phase model에서의 최소 점화전압**이다.

## 왜 나머지 세 목적함수는 260 V에서 계산하는가

`t_ign`, `U_alpha(2s)`, `C_J`까지 각 형상의 V_min에서 계산하면 형상마다 평가 전압이 달라져 직접 비교가 어려워진다. 특히 threshold 바로 위에서 계산한 점화지연은 대부분 evaluation horizon에 가까워지는 경향이 있어 형상 성능을 구분하는 지표로서 약해진다.

따라서 v7.9.4는:

- `t_ign(V_ref)` — 고정 reference voltage에서 점화속도 비교
- `U_alpha(2s; V_ref)` — 고정 reference voltage에서 condensed decomposition coverage 비교
- `V_min,ign` — 저전압 점화 능력 자체 비교
- `C_J(V_ref)` — 고정 reference voltage에서 전류집중 비교

로 역할을 분리한다.

## 수치적 정의

연속적인 정확한 threshold를 직접 구하는 대신 bracketed search를 사용한다. 최종 bracket이

\[
V_{\mathrm{no\ ign}} < V_{\min,\mathrm{ign}} \le V_{\mathrm{ign}}
\]

을 만족하면 목적함수에는 보수적으로 `V_ign`, 즉 실제 점화가 확인된 upper bound를 넣는다. 기본 허용 bracket 폭은 5 V이다.

V_min trial에는 reference run과 동일한 numerical-validity 정책을 적용한다. 따라서 낮은 전압에서 "점화"가 발생했더라도 계산이 cap/limiter에 과도하게 의존하면 유효한 threshold로 인정하지 않는다.

## 에너지 지표

다음 값은 계속 저장하지만 NSGA-II objective에는 포함하지 않는다.

\[
E_{\mathrm{ign}}=\int_0^{t_{\mathrm{ign}}}V(t)I(t)\,dt
\]

\[
E_{2s}=\int_0^{2s}V(t)I(t)\,dt
\]

따라서 후처리에서 저전압 점화와 에너지 소비 사이의 trade-off를 별도로 분석할 수 있다.
