# Non-metallized LP/PVA/GLY/BA 문헌 데이터와 코드 적용 범위

## 1. 사용자 조성

| 성분 | wet recipe |
|---|---:|
| LP | 29.1 wt% |
| H₂O | 53.9 wt% |
| PVA | 10.0 wt% |
| glycerol | 5.0 wt% |
| boric acid | 2.0 wt% |
| W | 0 wt% |

## 2. 가장 가까운 non-metallized M0

보고된 M0 wet formulation은 LP/H₂O/PVA/glycerol/boric acid = 29.47/54.53/10/4/2 wt%이며 W=0이다. 사용자 조성은 glycerol이 1%p 높고 LP와 water가 소폭 낮다.

## 3. 직접 코드에 쓴 문헌값

### 3.1 응축상 onset reference

- 값: 349 °C = 622.15 K
- 용도: `t_ignition`의 operational condensed-phase decomposition-onset threshold
- 코드: `config/nsga2_condensed_phase_no_f.yaml` → `condensed_ignition.onset_temperature_K`
- 주의: gas-flame ignition temperature가 아님

### 3.2 전체 열방출 reference

- 값: 약 2043 J/g
- 용도: TGA/DSC/열모델 validation target
- 현재 열원에 직접 강제하지 않음
- 이유: 기존 global heat term과 중복될 수 있고 reaction별 배분이 보고되지 않음

### 3.3 반응구조

- non-metallized sample의 multi-stage decomposition
- molten oxidizer와 PVA pyrolysis product의 primary reaction
- secondary gas-phase reaction과 잔여 LP decomposition의 중첩
- 별도 연구의 non-metallized multi-stage/12-reaction mechanistic pathway

이 정보는 모델구조의 근거지만, 12개 반응별 CFD-ready \(A_i,E_{a,i},n_i,\Delta H_i\) 세트로 전환하지 않았다.

## 4. 코드에 유지된 nominal 값

기존 v7.7.2 global solid kinetics:

- \(A=5.0\times10^9\,\mathrm{s^{-1}}\)
- \(E_a=110\,\mathrm{kJ/mol}\)
- \(n=1\)
- effective heat release = 200 kJ/kg

이 값들은 아래 문헌에서 완전한 user-formulation parameter로 보고된 값이라고 주장하지 않으며, 코드에서 `nominal_unvalidated`로 취급한다.

## 5. 임의 생성하지 않은 값

- reaction별 pre-exponential factor
- reaction별 activation energy/order
- reaction별 enthalpy
- gas species yield
- 경화 후 잔류수분
- \(\sigma(T)\)
- Li⁺/ClO₄⁻ diffusivity
- Butler–Volmer \(j_0,\alpha_a,\alpha_c\)

## 6. 주요 출처

1. K. Gnanaprakash, M. Yang, J. J. Yoh, **Thermal decomposition behaviour and chemical kinetics of tungsten based electrically controlled solid propellants**, *Combustion and Flame* 238 (2022).  
   https://www.sciencedirect.com/science/article/pii/S0010218021004958

2. M. Yang et al., **Effect of tungsten size on thermal analysis and mechanism of lithium perchlorate-based electrically controlled solid propellant**, *Case Studies in Thermal Engineering* 47 (2023), 103135.  
   https://www.sciencedirect.com/science/article/pii/S2214157X23004410

3. K. Gnanaprakash and J. J. Yoh, **Understanding the pyroelectric combustion behaviour of metallized electrically controlled solid propellants**, *Proceedings of the Combustion Institute* 39 (2023).  
   https://www.sciencedirect.com/science/article/pii/S1540748922000669

## 7. 해석 원칙

문헌은 reaction pathway와 validation reference를 제공한다. 사용자 조성에 대한 정량 kinetics는 다음 순서로 확정한다.

```text
문헌 prior
→ 사용자 시료 multi-rate TGA/DSC validation
→ 필요 시 global kinetics calibration
→ 보정에 쓰지 않은 조건/형상에서 validation
```
