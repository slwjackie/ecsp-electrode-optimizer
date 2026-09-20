# 정량 논문결과 전에 필요한 보정·검증

## 필수 우선순위

1. **경화 전후 질량**: retained-water fraction 결정
2. **multi-rate TGA**: global/단계별 apparent kinetics와 mass-loss curve
3. **DSC**: heat release 및 stage별 열효과
4. **온도별 전도도 또는 I–V/I(t)**: electrical/ionic model 검증
5. **고속영상·IR**: 최초 반응위치와 condensed-phase onset proxy 검증

## 최소 검증 절차

### A. 문헌값 고정 validation

먼저 문헌 reference와 기존 nominal parameter를 수정하지 않고 계산한다.

### B. 민감도

- onset threshold: 602.15/622.15/642.15 K
- retained water
- global \(A,E_a,n\)
- heat release
- 계면 kinetics

### C. 필요한 항목만 calibration

모든 계수를 동시에 맞추면 identifiability가 무너질 수 있으므로 순차 보정한다.

1. retained water 및 thermophysical properties
2. \(\sigma(T)\), I(t)
3. TGA/DSC global kinetics
4. interface/electrochemical parameters

### D. 독립 validation

보정에 쓰지 않은 전압·형상·반복시료에서 다음을 비교한다.

- onset delay
- onset location
- temperature history
- current history
- \(U_\alpha(t)\)

## 중요한 표현 제한

실험 검증 전 결과는 `nominal-model Pareto recommendation`이다. 이를 `experimentally validated optimum`으로 표현하면 안 된다.
