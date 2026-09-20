# B/C global 모델 파라미터 출처와 교체 지점

## 1. 논문에서 직접 가져온 항목

| 항목 | 코드 위치 | 현재값/출처 | 상태 |
|---|---|---|---|
| wet recipe | `physics.composition` | LP/H2O/PVA/H3BO3 = 31.58/58.42/9/1 wt.% | 3번 논문과 동일 |
| non-metallized global reaction | `bc_global.py::BC_GLOBAL_REACTION` | 1.45 LP + 1 PVA repeat → 2 CO2 + 2 H2O + 0.4 O2 + 1.45 LiCl | 3번 논문 reduced global chemistry |
| channel 1 heat | `bc_global.kinetics.channels[0]` | 881 kJ/kg | 3번 논문 second exotherm |
| channel 2 heat | `bc_global.kinetics.channels[1]` | 1162 kJ/kg | 3번 논문 third exotherm |
| conversion-dependent E and ln(Af) | same | Fig. 8 approximate digitisation | nominal; supplementary numerical data로 교체 권장 |
| mass-conversion weights | `mass_conversion_weights` | 2/3, 1/3 | 2번 논문 main stage 24%:12%를 정규화한 nominal 값 |
| onset-temperature prior | `onsetCriterion.temperature_K` | 523.15 K | 2번 250°C; 3번 약 260°C는 sensitivity |
| cured water prior | `physics.cured_water_mass_fraction` | 0.20 | 3번 논문의 <150°C mass loss 기반 nominal |

2번과 3번의 총 질량감소/반응열을 평균해 섞지 않는다. 3번을 nominal chemistry로 사용하고 2번은 stage interpretation, weight prior 및 sensitivity 범위에만 사용한다.

## 2. 코드 실행용 literature-nominal 입력

아래 값은 코드 실행에는 충분하지만 해당 formulation에서 직접 측정된 값이 아니다.

| 계수 | 코드 위치 | nominal 값 | 향후 교체 실험 |
|---|---|---:|---|
| Li+ diffusivity | `bc_global.transport.cation_diffusivity` | 2.0e-13 m2/s @ 298 K, Ea=22 kJ/mol | EIS+전달수 또는 7Li PFG-NMR |
| ClO4- diffusivity | `anion_diffusivity` | 1.5e-13 m2/s @ 298 K, Ea=25 kJ/mol | EIS+전달수/PFG-NMR/농도완화 |
| water diffusivity | `water_diffusivity` | 2.0e-10 m2/s @ 298 K, Ea=12 kJ/mol | DVS 또는 1H PFG-NMR |
| cp(T) | `bc_global.thermal.heat_capacity` | 2200–2700 J/kg/K table | DSC specific heat |
| k(T) | `bc_global.thermal.thermal_conductivity` | 0.40→0.32 W/m/K table | LFA+rho cp 또는 TPS/Hot Disk |
| convection h | thermal config | 12 W/m2/K | dummy cooling inverse fit |
| emissivity | thermal config | 0.85 | IR/emissometer calibration |
| density | `density_kg_per_m3: null` | component additive-volume estimate | cured coupon density |
| BV Eeq, j0, alpha, n, reaction enthalpy | inherited `config/default_lp_pva.yaml::interface` | explicit nominal water/LP anode/cathode channels | OCV, polarization/EIS, coulometry, calorimetry |

## 3. 코드 실행과 논문 정량주장의 구분

```text
코드 실행/배관 검증:
  논문 kinetics + literature nominal D/cp/k/BV → 가능, 새 실험 불필요

실제 formulation 정량 calibration:
  cured water/rho + EIS/transport + BV + cp/k → 필요

독립 validation:
  staggered와 calibration에 쓰지 않은 AI 형상의 V(t), I(t), IR T(x,y,t), onset/잔류량 비교
```

모든 nominal 값은 `calibration_status: literature_nominal_uncalibrated` 및 각 `provenance` 필드로 결과에 남는다.
