# `ECSP 최적 전극 형상 시뮬` 구현 검토표

이 문서는 요구사항을 코드·설정·출력에 대조한 release-level 감사표다. 자동 검증은 `python/validate_bc_global_pipeline.py`와 `tools/validate_bc_global_pipeline.sh`에서 수행한다.

| 명세 요구사항 | 구현 위치 | 판정 |
|---|---|---:|
| v7.9.5 기존 기능 전부 유지 | 기존 파일 154개 중 삭제 0, 핵심 C++ byte-identical, 새 backend는 opt-in; `V795_FEATURE_PRESERVATION_AUDIT_KR.md` | 구현·회귀검증 |
| pre-flame 식 (22)–(24) | `python/ecsp_v6/physics/bc_global.py` | 구현 |
| 보조식 (2),(4)–(9),(11),(13),(15)–(17) | `bc_global.py`, 기존 `electrochem.py` local BV/Faraday | 구현 |
| Poisson (3) 미사용 | bulk 1:1 electroneutrality로 cation/anion projection | 구현 |
| density 식 (18) 강제 미사용 | measured/specified 또는 composition estimate density 입력 | 구현 |
| post-flame 식 (25)–(28) pre-flame에서 제외 | B/C module에 velocity/momentum 없음 | 구현 |
| 식 (29)–(31) 중복 제한전류 제외 | `usePaperMassTransferSaturation=false` 강제 | 구현 |
| 식 (32)는 검산만 | 별도 heat-rate/energy diagnostic; thermal RHS에 재추가 없음 | 구현 |
| 3번 논문 non-metallized global reaction | `BC_GLOBAL_REACTION`, output metadata | 구현 |
| 두 conversion-dependent exotherm | `BCKineticChannel`, config lookup tables | 구현 |
| 두 채널에서 global species 이중소비 방지 | weighted `Xg`, one global reaction extent | 구현 |
| 국부 BV overpotential | existing nonlinear-Robin local electrolyte potential solver | 구현 |
| Faraday species flux | existing interface water/salt sink + B/C species equation | 구현 |
| wet recipe 31.58/58.42/9/1, glycerol/W 0 | A100/debug configs | 구현 |
| 실제 cured composition 사용 | `cured_water_mass_fraction` final-specimen basis | 구현 |
| unknown cured water = 0.20 literature nominal | A100 profile and output provenance | 구현 |
| Usp를 `1-alpha`가 아닌 LP+PVA remaining mass로 | `_reaction_inventory`, `remainingReactiveMassFraction*` | 구현 |
| onset = T AND Xg over minimum area | `onsetCriterion`, `qualified` cell fraction | 구현 |
| Vmin 동일 criterion | B/C evaluator bracket/bisection | 구현 |
| J99/Jbar 유지 | pre-flame current field quantile/mean | 구현 |
| 1,000 형상 B/C 전수평가 | A100 profile population 1000 | 구현 |
| Pareto + near-Pareto/diverse 10–20% | `_select_propagation_candidates`, 15% default | 구현 |
| 후단은 condensed propagation, gas CFD 아님 | `python/ecsp_nsga2/propagation.py` | 구현 |
| T/Xg/channel progress/species inventory/potential/qJ/qechem full-field handoff | evaluator batch handoff NPZ and in-memory fields | 구현 |
| 점화 좌표 강제 금지 | entire Xg field initializes reaction front/level set | 구현 |
| post-onset 전기 열원 정책 | 닫힌 species/charge accounting이 없는 electrical-on 모드는 fail-closed; 현재 condensed continuation은 전기 열원 0 | 구현 |
| A_unreacted(t) | `propagation_history.csv` | 구현 |
| regression velocity | area-loss/front-perimeter effective velocity | 구현 |
| t_established | configured reacted-area threshold time | 구현 |
| front uniformity | front arrival-time CV + residual unreacted descriptor | 구현·공학지표 명시 |
| 최종 Pareto | 4 pre-flame + 4 propagation minimisation objectives | 구현 |
| staggered 동일 B/C+propagation | post-optimization exact persisted baseline masks | 구현 |
| reduced chemistry reviewer caveat | README 및 v8 모델 문서 | 구현 |
| TGA/DSC 비교 필요성 | parameter provenance/validation docs | 구현 |

## 실행 검증 게이트

```bash
PYTHONPATH=python pytest -q
bash tools/validate_bc_global_pipeline.sh
```

자동 감사가 확인하는 항목:

- exact composition와 cured-water basis
- 사용·미사용 equation contract
- global reaction/two-channel kinetics
- proxy closure 비활성화
- 네 pre-flame objective와 onset criterion
- 1,000 → Pareto+15% diversity → propagation → final Pareto
- staggered 동일 경로
- CPU FP64 end-to-end smoke에서 필수 파일과 전체 field 저장

## 남아 있는 의도적 한계

- Fig. 8 kinetic table은 supplementary numerical data가 아니라 approximate digitisation이다.
- D_i, BV, cp, k, cured water는 literature nominal이며 아직 실험보정되지 않았다.
- 후단 level set은 Xg front의 signed-distance representation이며 독립적인 flame-speed law가 아니다.
- external flame/plume/gas-phase chemistry는 포함하지 않는다.
- 실제 A100 장치의 end-to-end 성능과 장기 수치오차는 이 패키징 환경에서 검증하지 못했다.

따라서 구현 완전성은 검증했지만, 실제 추진제의 정량 예측 정확도는 향후 calibration/validation의 대상이다.
