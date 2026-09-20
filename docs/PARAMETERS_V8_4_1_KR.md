# v8.4.1 계수·설정 전체 정리

기준: `config/nsga2_bc_reactive_a100_cpu8.yaml`의 production 물리. evaluator의 최종 병합 결과는 `docs/audit/resolved_bc_production_physics.json`입니다. CUDA가 없는 검토 환경에서 설정을 CPU로만 해석했으며 물리값은 바꾸지 않았습니다. production 실행 장치/배치 값은 원 YAML에 기록되어 있습니다.

**수치상수·운영입력·문헌 prior·실측 물성을 구분해야 합니다.** 현재 시료의 실험보정 완료 계수는 첨부자료에서 확인되지 않았습니다. 원시 8,520행 YAML은 반복 프로필과 비활성 값도 모두 포함합니다. 코드 기본값과 수치리터럴 목록은 추출 인덱스이지 모든 숫자가 nominal 물성이라는 주장이 아닙니다.

## 네 계면 채널

| 채널 | 극 | Eeq_V | j0_A_m2 | beta | n | reverse | DeltaH_J_mol | km_m_s | 시약소모_mol_per_mol_e | gas_yield | BC활성 | 근거 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| water | anode | 1.23 | 8.0 | 0.5 | 2.0 | 1.0 | 25000.0 | 1e-05 | 0.5 | 0.75 | Eeq,j0,beta,n,reverse,ΔH,소모계수 활성; km/gasyield 미사용 | config/default_lp_pva.yaml interface; 실제 시료 보정 없음 |
| water | cathode | 1.23 | 10.0 | 0.5 | 2.0 | 1.0 | 20000.0 | 1.2e-05 | 0.5 | 0.75 | Eeq,j0,beta,n,reverse,ΔH,소모계수 활성; km/gasyield 미사용 | config/default_lp_pva.yaml interface; 실제 시료 보정 없음 |
| lp | anode | 3.0 | 1.0 | 0.5 | 1.0 | 1.0 | 80000.0 | 4e-06 | 1.0 | 0.4 | Eeq,j0,beta,n,reverse,ΔH,소모계수 활성; km/gasyield 미사용 | config/default_lp_pva.yaml interface; 실제 시료 보정 없음 |
| lp | cathode | 3.0 | 0.8 | 0.5 | 1.0 | 1.0 | 50000.0 | 3e-06 | 1.0 | 0.5 | Eeq,j0,beta,n,reverse,ΔH,소모계수 활성; km/gasyield 미사용 | config/default_lp_pva.yaml interface; 실제 시료 보정 없음 |

## 두 채널 반응속도 LUT — 모든 좌표

| 채널 | alpha | E_J_per_mol | ln_Af_per_s | Q_J_per_kg | w | 출처표시 | 검증상태 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0 | 85000 | 13.0 | 881000.0 | 0.6666666666666666 | approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 1 | 0.1 | 85000 | 13.0 | 881000.0 | 0.6666666666666666 | approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 1 | 0.2 | 91000 | 14.5 | 881000.0 | 0.6666666666666666 | approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 1 | 0.3 | 98000 | 15.5 | 881000.0 | 0.6666666666666666 | approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 1 | 0.4 | 105000 | 16.5 | 881000.0 | 0.6666666666666666 | approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 1 | 0.5 | 112000 | 18.0 | 881000.0 | 0.6666666666666666 | approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 1 | 0.6 | 119000 | 19.0 | 881000.0 | 0.6666666666666666 | approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 1 | 0.7 | 127000 | 20.0 | 881000.0 | 0.6666666666666666 | approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 1 | 0.8 | 136000 | 21.0 | 881000.0 | 0.6666666666666666 | approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 1 | 0.9 | 150000 | 23.8 | 881000.0 | 0.6666666666666666 | approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 1 | 1.0 | 150000 | 23.8 | 881000.0 | 0.6666666666666666 | approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 2 | 0.0 | 150000 | 19.0 | 1162000.0 | 0.3333333333333333 | approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 2 | 0.1 | 150000 | 19.0 | 1162000.0 | 0.3333333333333333 | approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 2 | 0.2 | 145000 | 18.5 | 1162000.0 | 0.3333333333333333 | approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 2 | 0.3 | 140000 | 18.0 | 1162000.0 | 0.3333333333333333 | approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 2 | 0.4 | 140000 | 18.0 | 1162000.0 | 0.3333333333333333 | approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 2 | 0.5 | 150000 | 20.0 | 1162000.0 | 0.3333333333333333 | approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 2 | 0.6 | 190000 | 26.0 | 1162000.0 | 0.3333333333333333 | approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 2 | 0.7 | 220000 | 29.9 | 1162000.0 | 0.3333333333333333 | approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 2 | 0.8 | 190000 | 26.0 | 1162000.0 | 0.3333333333333333 | approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 2 | 0.9 | 150000 | 18.0 | 1162000.0 | 0.3333333333333333 | approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |
| 2 | 1.0 | 150000 | 18.0 | 1162000.0 | 0.3333333333333333 | approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | 원시 좌표 독립검증 미완료; 시료보정 없음 |

## 해석된 물리/수치/운영 값 — 전체 578행

`classification`과 단위 추정은 보조 분류입니다. `source`와 full key가 최종 정의입니다. `values[]`의 단위는 부모 열물성의 단위를 따릅니다. `null`은 누락을 숨긴 상수가 아니라 조성에서 유도하거나 기본값으로 채우는 설정이며 실제 해석값은 resolved/hand-off에 별도 저장됩니다.


### 초기조건/기하 입력

| key | value | units | classification | declared_source | source |
| --- | --- | --- | --- | --- | --- |
| composition.masses_g.LP | 11.053 | g | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: config/default_lp_pva.yaml:13 |
| composition.masses_g.water | 20.447000000000003 | g | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: config/default_lp_pva.yaml:14 |
| composition.masses_g.PVA | 3.15 | g | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: config/default_lp_pva.yaml:15 |
| composition.masses_g.glycerol | 0.0 | g | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: config/default_lp_pva.yaml:16 |
| composition.masses_g.boric_acid | 0.35000000000000003 | g | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: config/default_lp_pva.yaml:17 |
| composition.retained_water_fraction | 1.0 | 키 정의 확인(주로 무차원) | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:18 |
| composition.density_mode | "additive_volume_estimate" | kg/m³ | 모델 선택/가정 |  | config/default_lp_pva.yaml:19 |
| composition.mixture_density_kg_per_m3 | 1261.36 | kg/m³ | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:20 |
| composition.calibration_required | true | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:21 |
| composition.molar_masses_g_per_mol.LP | 106.39 | g/mol | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:23 |
| composition.molar_masses_g_per_mol.water | 18.01528 | g/mol | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:24 |
| composition.molar_masses_g_per_mol.PVA_repeat | 44.0526 | g/mol | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:25 |
| composition.molar_masses_g_per_mol.glycerol | 92.09382 | g/mol | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:26 |
| composition.molar_masses_g_per_mol.boric_acid | 61.83 | g/mol | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:27 |
| composition.component_densities_kg_per_m3.LP | 2430.0 | kg/m³ | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:29 |
| composition.component_densities_kg_per_m3.water | 997.0 | kg/m³ | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:30 |
| composition.component_densities_kg_per_m3.PVA | 1269.0 | kg/m³ | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:31 |
| composition.component_densities_kg_per_m3.glycerol | 1260.0 | kg/m³ | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:32 |
| composition.component_densities_kg_per_m3.boric_acid | 1435.0 | kg/m³ | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:33 |
| composition.retained_water_fraction_basis | "literature_nominal_uncalibrated_3rd_paper_low_temperature_TGA_loss" | 키 정의 확인(주로 무차원) | 출처/의미/상태 메타데이터 |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: config/default_lp_pva.yaml:34 |
| composition.retained_water_measurement_id | null | 키/정의 참조 | 출처/의미/상태 메타데이터 |  | config/default_lp_pva.yaml:35 |
| composition.requireMeasuredRetainedWater | false | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:36 |
| composition.cured_water_mass_fraction | 0.2 | 키 정의 확인(주로 무차원) | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: python/ecsp_nsga2/evaluator.py: 파생/override; resolved_bc_production_physics.json |
| composition.cured_water_mass_fraction_basis | "literature_nominal_uncalibrated_3rd_paper_low_temperature_TGA_loss" | 키 정의 확인(주로 무차원) | 출처/의미/상태 메타데이터 |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: python/ecsp_nsga2/evaluator.py: 파생/override; resolved_bc_production_physics.json |
| geometry.domainSize_m | 0.02 | m | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:39 |
| geometry.surfaceLayerThickness_m | 0.001 | m | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:40 |
| geometry.minimumElectrodeGap_m | 0.0005 | m | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:49 |

### 공통 저장 설정; 새 NSGA 경로 활성 미주장

| key | value | units | classification | declared_source | source |
| --- | --- | --- | --- | --- | --- |
| geometry.maskSize | 64 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:38 |
| geometry.minimumWidthPixels | 2 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:41 |
| geometry.minimumGapPixels | 3 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:42 |
| geometry.minimumObjectPixels | 4 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:43 |
| geometry.minimumAreaFractionPerPolarity | 0.17 | 키 정의 확인(주로 무차원) | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:44 |
| geometry.maximumTotalElectrodeAreaFraction | 0.36 | 키 정의 확인(주로 무차원) | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:45 |
| geometry.maximumAreaImbalanceFraction | 0.0 | 키 정의 확인(주로 무차원) | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:46 |
| geometry.maximumComponentsPerPolarity | 3 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:47 |
| geometry.minimumBorderMarginPixels | 1 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:48 |
| geometry.rejectResizeShorts | true | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:50 |
| geometry.maximumTotalComponents | 4 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:51 |
| geometry.targetTotalElectrodeAreaFraction | 0.35 | 키 정의 확인(주로 무차원) | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:52 |
| geometry.targetAnodeShareOfElectrodeArea | 0.5 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:53 |
| geometry.enforceEqualPolarityArea | true | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:54 |
| geometry.enforceFixedTotalElectrodeArea | true | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:55 |
| geometry.fixedAreaTolerancePixels | 0 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:56 |
| geometry.minimumCornerRadiusPixels | 1 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:57 |
| geometry.minimumCornerRadius_m | 0.00025 | m | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:58 |
| geometry.maximumCornerRoundingChangeFraction | 0.01 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:59 |
| geometry.generatorMode | "cad_quantized_ar_latent_diffusion" | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:60 |
| geometry.electrodeMaskSemantics | "surface_contact_overlay_not_material_removal" | 키/정의 참조 | 기타 입력 |  | config/default_lp_pva.yaml:61 |
| geometry.propellantDomainFraction | 1.0 | 키 정의 확인(주로 무차원) | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:62 |
| geometry.targetAreaFractionPerPolarity | 0.175 | 키 정의 확인(주로 무차원) | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:63 |

### BC 전도/수치 입력

| key | value | units | classification | declared_source | source |
| --- | --- | --- | --- | --- | --- |
| electrical.gridSize | 257 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:274 |
| electrical.appliedVoltage_V | 260.0 | V | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:275 |
| electrical.cathodeVoltage_V | 0.0 | V | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:276 |
| electrical.initialTemperature_K | 298.15 | K | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:277 |
| electrical.omega | 1.72 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:278 |
| electrical.maximumIterations | 3000 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:279 |
| electrical.tolerance_V | 1e-08 | V | 수치 설정/제한 |  | config/default_lp_pva.yaml:280 |
| electrical.diffusionPotentialEnabled | true | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:281 |
| electrical.jouleHeatModel | "conductive_sigma_E2" | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:282 |
| electrical.conductivityMinimum_S_per_m | 0.0001 | S/m | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:286 |
| electrical.conductivityMaximum_S_per_m | 0.5 | S/m | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:287 |
| electrical.lowCurrentThresholdFraction | 0.5 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:288 |

### BC 비활성(전자전도 OFF)

| key | value | units | classification | declared_source | source |
| --- | --- | --- | --- | --- | --- |
| electrical.electronicConductivity0_S_per_m | 0.0 | S/m | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: config/default_lp_pva.yaml:283 |
| electrical.electronicConductivityTemperatureCoefficient_per_K | 0.0 | S/m 또는 W/(m K): 키 문맥 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: config/default_lp_pva.yaml:284 |
| electrical.electronicLiquidSuppression | 0.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: config/default_lp_pva.yaml:285 |
| bcGlobal.augmentedElectronicConduction.conductivity.mode | "constant" | S/m 또는 W/(m K): 키 문맥 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:140 |
| bcGlobal.augmentedElectronicConduction.conductivity.value | 0.0 | S/m 또는 W/(m K): 키 문맥 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:141 |

### BC 수송/수치 입력

| key | value | units | classification | declared_source | source |
| --- | --- | --- | --- | --- | --- |
| transport.gasConstant_J_per_molK | 8.314462618 | J/(mol K) | 상수/화학종·반응 정의(반응 정의는 별도 검증 필요) |  | config/default_lp_pva.yaml:290 |
| transport.faradayConstant_C_per_mol | 96485.33212 | C/mol | 상수/화학종·반응 정의(반응 정의는 별도 검증 필요) |  | config/default_lp_pva.yaml:291 |
| transport.chargeNumberCation | 1.0 | 키/정의 참조 | 상수/화학종·반응 정의(반응 정의는 별도 검증 필요) |  | config/default_lp_pva.yaml:292 |
| transport.chargeNumberAnion | -1.0 | 키/정의 참조 | 상수/화학종·반응 정의(반응 정의는 별도 검증 필요) |  | config/default_lp_pva.yaml:293 |
| transport.concentrationMinimumFraction | 1e-06 | 키 정의 확인(주로 무차원) | 수치 설정/제한 |  | config/default_lp_pva.yaml:306 |
| transport.concentrationMaximumMultiple | 2.0 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:307 |
| transport.electroneutralRelaxation | 1.0 | 1 | 수치 설정/제한 |  | config/default_lp_pva.yaml:308 |
| transport.maximumRelativeConcentrationChangePerStep | 0.05 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:309 |

### BC에서 bcGlobal.transport로 대체/비활성

| key | value | units | classification | declared_source | source |
| --- | --- | --- | --- | --- | --- |
| transport.cationDiffusivityDry_m2_per_s | 5e-15 | m²/s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:294 |
| transport.anionDiffusivityDry_m2_per_s | 3e-15 | m²/s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:295 |
| transport.cationDiffusivityWet_m2_per_s | 2e-13 | m²/s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:296 |
| transport.anionDiffusivityWet_m2_per_s | 1.5e-13 | m²/s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:297 |
| transport.waterDiffusivity_m2_per_s | 2e-10 | m²/s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:298 |
| transport.cationTransportActivationEnergy_J_per_mol | 22000.0 | J/mol | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:299 |
| transport.anionTransportActivationEnergy_J_per_mol | 25000.0 | J/mol | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:300 |
| transport.referenceTemperature_K | 298.15 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:301 |
| transport.liquidDiffusivityGain | 120.0 | m²/s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:302 |
| transport.waterActivityExponent | 1.2 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:303 |
| transport.glycerolPlasticizationGain | 1.5 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:304 |
| transport.crosslinkTransportPenalty | 0.8 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:305 |

### 기존 경로 전용; BC에서 미사용

| key | value | units | classification | declared_source | source |
| --- | --- | --- | --- | --- | --- |
| phase.softeningTemperature_K | 355.0 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:311 |
| phase.glycerolSofteningShift_K_perMassRatio | 30.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:312 |
| phase.boricAcidSofteningShift_K_perMolarRatio | 30.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:313 |
| phase.transitionWidth_K | 15.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:314 |
| phase.relaxationTime_s | 0.08 | s | 수치 설정/제한 |  | config/default_lp_pva.yaml:315 |
| phase.latentHeat_J_per_kg | 0.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:316 |
| phase.minimumLiquidFractionForFastIonTransport | 0.1 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:317 |
| phase.liquidDiffusivityGain | 0.0 | m²/s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: python/ecsp_nsga2/evaluator.py: 파생/override; resolved_bc_production_physics.json |
| chemical.enabled | false | 키/정의 참조 | 모델 선택/가정 |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: config/default_lp_pva.yaml:439 |
| chemical.preExponentialFactor_per_s | 5000000000.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:440 |
| chemical.activationEnergy_J_per_mol | 110000.0 | J/mol | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:441 |
| chemical.reactionOrder | 1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:442 |
| chemical.oxidizerOrder | 0.5 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:443 |
| chemical.oxidizerConsumptionFractionPerUnitProgress | 0.25 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:444 |
| chemical.maximumRate_per_s | 50.0 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:445 |
| chemical.heatRelease_J_per_kg | 200000.0 | J/kg | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:446 |
| chemical.effectiveGasMolarMass_kg_per_mol | 0.03 | 키/정의 참조 | 상수/화학종·반응 정의(반응 정의는 별도 검증 필요) |  | config/default_lp_pva.yaml:447 |
| chemical.gasYieldMassFraction | 0.35 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:448 |
| chemical.activationTemperature_K | 430.0 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:449 |
| gas.diffusivity_m2_per_s | 3e-06 | m²/s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:462 |
| gas.lossRate_per_s | 1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:463 |
| gas.maximumConcentration_mol_per_m3 | 5000.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:464 |

### BC 계면/수치 입력

| key | value | units | classification | declared_source | source |
| --- | --- | --- | --- | --- | --- |
| interface.contactModel | "surface_overlay_full_propellant" | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:319 |
| interface.contactNormalConductionLength_m | 3e-05 | m | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:320 |
| interface.electrodeMasksRemovePropellant | false | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:321 |
| interface.hiddenBusConnectionAssumed | true | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:322 |
| interface.currentDirectionModel | "signed_normal" | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:327 |
| interface.useFullButlerVolmer | true | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:328 |
| interface.numericalReactionLayerMinimum_m | 3e-05 | m | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:327 |
| interface.exponentialArgumentLimit | 45.0 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:330 |
| interface.water.anode.equilibriumPotential_V | 1.23 | V | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:334 |
| interface.water.anode.exchangeCurrentDensity_A_per_m2 | 8.0 | kg/m³ | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:335 |
| interface.water.anode.chargeTransferCoefficient | 0.5 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:336 |
| interface.water.anode.reverseAvailabilityFraction | 1.0 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:337 |
| interface.water.anode.electronNumber | 2.0 | 키/정의 참조 | 상수/화학종·반응 정의(반응 정의는 별도 검증 필요) |  | config/default_lp_pva.yaml:338 |
| interface.water.anode.massTransferCoefficient_m_per_s | 1e-05 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:339 |
| interface.water.anode.reactionEnthalpy_J_per_mol | 25000.0 | J/mol | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:340 |
| interface.water.anode.waterStoichiometry_mol_per_molElectron | 0.5 | 키/정의 참조 | 상수/화학종·반응 정의(반응 정의는 별도 검증 필요) |  | config/default_lp_pva.yaml:341 |
| interface.water.cathode.equilibriumPotential_V | 1.23 | V | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:346 |
| interface.water.cathode.exchangeCurrentDensity_A_per_m2 | 10.0 | kg/m³ | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:347 |
| interface.water.cathode.chargeTransferCoefficient | 0.5 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:348 |
| interface.water.cathode.reverseAvailabilityFraction | 1.0 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:349 |
| interface.water.cathode.electronNumber | 2.0 | 키/정의 참조 | 상수/화학종·반응 정의(반응 정의는 별도 검증 필요) |  | config/default_lp_pva.yaml:350 |
| interface.water.cathode.massTransferCoefficient_m_per_s | 1.2e-05 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:351 |
| interface.water.cathode.reactionEnthalpy_J_per_mol | 20000.0 | J/mol | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:352 |
| interface.water.cathode.waterStoichiometry_mol_per_molElectron | 0.5 | 키/정의 참조 | 상수/화학종·반응 정의(반응 정의는 별도 검증 필요) |  | config/default_lp_pva.yaml:353 |
| interface.lp.anode.equilibriumPotential_V | 3.0 | V | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:359 |
| interface.lp.anode.exchangeCurrentDensity_A_per_m2 | 1.0 | kg/m³ | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:360 |
| interface.lp.anode.chargeTransferCoefficient | 0.5 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:361 |
| interface.lp.anode.reverseAvailabilityFraction | 1.0 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:362 |
| interface.lp.anode.electronNumber | 1.0 | 키/정의 참조 | 상수/화학종·반응 정의(반응 정의는 별도 검증 필요) |  | config/default_lp_pva.yaml:363 |
| interface.lp.anode.massTransferCoefficient_m_per_s | 4e-06 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:364 |
| interface.lp.anode.reactionEnthalpy_J_per_mol | 80000.0 | J/mol | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:365 |
| interface.lp.anode.saltStoichiometry_mol_per_molElectron | 1.0 | 키/정의 참조 | 상수/화학종·반응 정의(반응 정의는 별도 검증 필요) |  | config/default_lp_pva.yaml:366 |
| interface.lp.cathode.equilibriumPotential_V | 3.0 | V | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:371 |
| interface.lp.cathode.exchangeCurrentDensity_A_per_m2 | 0.8 | kg/m³ | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:372 |
| interface.lp.cathode.chargeTransferCoefficient | 0.5 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:373 |
| interface.lp.cathode.reverseAvailabilityFraction | 1.0 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:374 |
| interface.lp.cathode.electronNumber | 1.0 | 키/정의 참조 | 상수/화학종·반응 정의(반응 정의는 별도 검증 필요) |  | config/default_lp_pva.yaml:375 |
| interface.lp.cathode.massTransferCoefficient_m_per_s | 3e-06 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:376 |
| interface.lp.cathode.reactionEnthalpy_J_per_mol | 50000.0 | J/mol | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:377 |
| interface.lp.cathode.saltStoichiometry_mol_per_molElectron | 1.0 | 키/정의 참조 | 상수/화학종·반응 정의(반응 정의는 별도 검증 필요) |  | config/default_lp_pva.yaml:378 |
| interface.blocking.minimumActiveAreaFraction | 1.0 | 키 정의 확인(주로 무차원) | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/nsga2_bc_reactive_a100_cpu8.yaml:333 |
| interface.boundaryCouplingModel | "surface_overlay_bv" | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:326 |
| interface.nonlinearRobin.minimumIterations | 2 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:412 |
| interface.nonlinearRobin.maximumIterations | 120 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:413 |
| interface.nonlinearRobin.potentialTolerance_V | 0.01 | V | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:414 |
| interface.nonlinearRobin.relativeReactionCurrentTolerance | 0.005 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:415 |
| interface.nonlinearRobin.currentBalanceTolerance | 0.005 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:416 |
| interface.nonlinearRobin.underRelaxation | 0.35 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:417 |
| interface.nonlinearRobin.failOnNonConvergence | true | 키/정의 참조 | 모델 선택/가정 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:418 |
| interface.nonlinearRobin.status | "implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5" | 키/정의 참조 | 출처/의미/상태 메타데이터 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:419 |
| interface.nonlinearRobin.potentialUnderRelaxation | 0.35 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:420 |
| interface.nonlinearRobin.currentBalanceAbsoluteTolerance_A | 1e-12 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:421 |
| interface.nonlinearRobin.gaugeBracketMargin_V | 25.0 | V | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:422 |
| interface.nonlinearRobin.gaugeMinimumIterations | 1 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:423 |
| interface.nonlinearRobin.gaugeMaximumIterations | 24 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:424 |
| interface.nonlinearRobin.gaugeUnderRelaxation | 1.0 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:425 |
| interface.nonlinearRobin.gaugeMaximumStep_V | 65.0 | V | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:426 |
| interface.nonlinearRobin.gaugeDerivativeFloor_A_per_V | 1e-14 | V | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:427 |
| interface.nonlinearRobin.gaugeBisectionIterations | 60 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:428 |
| interface.nonlinearRobin.localInterfaceIterations | 60 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:429 |
| interface.nonlinearRobin.localRobinResidualTolerance_A_per_m2 | 0.01 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:430 |
| interface.nonlinearRobin.localRobinRelativeResidualTolerance | 0.0001 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:431 |
| interface.nonlinearRobin.localRoundoffSafetyFactor | 2.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:432 |
| interface.nonlinearRobin.localInterfaceMinimumIterations | 8 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:433 |
| interface.nonlinearRobin.localInterfaceMaximumIterations | 60 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:434 |
| interface.nonlinearRobin.localNewtonPolishIterations | 2 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:435 |
| interface.nonlinearRobin.minimumIterationsStatic | 2 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:436 |
| interface.nonlinearRobin.minimumIterationsCoupled | 2 | 키/정의 참조 | 수치 설정/제한 | status: implicit_face_bv_drop_variable_exact_global_gauge_balance_v7_7_5 | config/default_lp_pva.yaml:437 |

### BC 비활성 또는 OFF 설정

| key | value | units | classification | declared_source | source |
| --- | --- | --- | --- | --- | --- |
| interface.deriveMassTransferCoefficientFromDiffusivity | true | m²/s | 모델 선택/가정 |  | config/default_lp_pva.yaml:325 |
| interface.physicalDiffusionLayerThickness_m | 3e-05 | m | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:326 |
| interface.includeActivationHeat | false | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:324 |
| interface.activationHeatFraction | 0.0 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:325 |
| interface.liquidKineticsGain | 0.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:329 |
| interface.water.anode.gasYield_mol_per_molElectron | 0.75 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:342 |
| interface.water.anode.referenceActivity | 1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:343 |
| interface.water.anode.nernstReactionQuotientExponent | -1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:344 |
| interface.water.cathode.gasYield_mol_per_molElectron | 0.75 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:354 |
| interface.water.cathode.referenceActivity | 1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:355 |
| interface.water.cathode.nernstReactionQuotientExponent | -1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:356 |
| interface.lp.anode.gasYield_mol_per_molElectron | 0.4 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:367 |
| interface.lp.anode.referenceActivity | 1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:368 |
| interface.lp.anode.nernstReactionQuotientExponent | -1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:369 |
| interface.lp.cathode.gasYield_mol_per_molElectron | 0.5 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:379 |
| interface.lp.cathode.referenceActivity | 1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:380 |
| interface.lp.cathode.nernstReactionQuotientExponent | -1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:381 |
| interface.nernst.enabled | false | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:331 |
| interface.nernst.activityCoefficientModel | "extended_debye_huckel_proxy" | 키 정의 확인(주로 무차원) | 모델 선택/가정 | status: nominal_unvalidated_concentration_activity_closure | config/default_lp_pva.yaml:384 |
| interface.nernst.activityFloor | 1e-06 | 키/정의 참조 | 수치 설정/제한 | status: nominal_unvalidated_concentration_activity_closure | config/default_lp_pva.yaml:385 |
| interface.nernst.activityA | 0.2 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: nominal_unvalidated_concentration_activity_closure | config/default_lp_pva.yaml:386 |
| interface.nernst.activityB | 1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: nominal_unvalidated_concentration_activity_closure | config/default_lp_pva.yaml:387 |
| interface.nernst.activityLinearPerMolL | 0.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: nominal_unvalidated_concentration_activity_closure | config/default_lp_pva.yaml:388 |
| interface.nernst.maximumLog10ActivityCoefficientMagnitude | 0.5 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: nominal_unvalidated_concentration_activity_closure | config/default_lp_pva.yaml:389 |
| interface.nernst.maximumAbsoluteShift_V | 0.25 | V | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: nominal_unvalidated_concentration_activity_closure | config/default_lp_pva.yaml:390 |
| interface.nernst.status | "nominal_unvalidated_concentration_activity_closure" | 키/정의 참조 | 출처/의미/상태 메타데이터 | status: nominal_unvalidated_concentration_activity_closure | config/default_lp_pva.yaml:391 |
| interface.blocking.passivation.enabled | false | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:335 |
| interface.blocking.passivation.formationRate_per_s | 0.02 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: reduced_nominal_unvalidated | config/default_lp_pva.yaml:396 |
| interface.blocking.passivation.removalRate_per_s | 0.005 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: reduced_nominal_unvalidated | config/default_lp_pva.yaml:397 |
| interface.blocking.passivation.referenceCurrentDensity_A_per_m2 | 1000.0 | kg/m³ | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: reduced_nominal_unvalidated | config/default_lp_pva.yaml:398 |
| interface.blocking.passivation.currentExponent | 1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: reduced_nominal_unvalidated | config/default_lp_pva.yaml:399 |
| interface.blocking.passivation.formationActivationTemperature_K | 360.0 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: reduced_nominal_unvalidated | config/default_lp_pva.yaml:400 |
| interface.blocking.passivation.formationActivationWidth_K | 25.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: reduced_nominal_unvalidated | config/default_lp_pva.yaml:401 |
| interface.blocking.passivation.status | "reduced_nominal_unvalidated" | 키/정의 참조 | 출처/의미/상태 메타데이터 | status: reduced_nominal_unvalidated | config/default_lp_pva.yaml:402 |
| interface.blocking.gasCoverage.enabled | false | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:337 |
| interface.blocking.gasCoverage.formationRate_per_s | 0.2 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: reduced_nominal_unvalidated | config/default_lp_pva.yaml:405 |
| interface.blocking.gasCoverage.detachmentRate_per_s | 1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: reduced_nominal_unvalidated | config/default_lp_pva.yaml:406 |
| interface.blocking.gasCoverage.referenceGasSource_mol_per_m3_s | 100.0 | 키/정의 참조 | 출처/의미/상태 메타데이터 | status: reduced_nominal_unvalidated | config/default_lp_pva.yaml:407 |
| interface.blocking.gasCoverage.thermalInsulationFraction | 0.5 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | status: reduced_nominal_unvalidated | config/default_lp_pva.yaml:408 |
| interface.blocking.gasCoverage.status | "reduced_nominal_unvalidated" | 키/정의 참조 | 출처/의미/상태 메타데이터 | status: reduced_nominal_unvalidated | config/default_lp_pva.yaml:409 |

### BC 열물성은 bcGlobal.thermal 사용; 일부 공통 기준

| key | value | units | classification | declared_source | source |
| --- | --- | --- | --- | --- | --- |
| thermal.density_kg_per_m3 | null | kg/m³ | 모델 선택/가정 |  | config/default_lp_pva.yaml:451 |
| thermal.heatCapacity_J_per_kgK | 2200.0 | J/(kg K) | 수치 설정/제한 |  | config/default_lp_pva.yaml:452 |
| thermal.thermalConductivity_W_per_mK | 0.4 | W/(m K) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:453 |
| thermal.initialTemperature_K | 298.15 | K | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:454 |
| thermal.ambientTemperature_K | 298.15 | K | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:455 |
| thermal.convectionCoefficient_W_per_m2K | 12.0 | W/(m² K) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:456 |
| thermal.emissivity | 0.85 | 1 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:457 |
| thermal.stefanBoltzmann_W_per_m2K4 | 5.670374419e-08 | 키/정의 참조 | 상수/화학종·반응 정의(반응 정의는 별도 검증 필요) |  | config/default_lp_pva.yaml:458 |
| thermal.minimumTemperature_K | 250.0 | K | 수치 설정/제한 |  | config/default_lp_pva.yaml:459 |
| thermal.maximumTemperature_K | 2500.0 | K | 수치 설정/제한 |  | config/default_lp_pva.yaml:460 |

### BC 일부 override/기존 공통 운영 입력

| key | value | units | classification | declared_source | source |
| --- | --- | --- | --- | --- | --- |
| coupled.gridSize | 193 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:466 |
| coupled.timeStep_s | 0.00025 | s | 수치 설정/제한 |  | config/default_lp_pva.yaml:467 |
| coupled.endTime_s | 2.0 | s | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: config/default_lp_pva.yaml:468 |
| coupled.electricalUpdateInterval_s | 0.0025 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:469 |
| coupled.voltage_V | 260.0 | V | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:470 |
| coupled.voltageSweep_V[0] | 50 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:472 |
| coupled.voltageSweep_V[1] | 75 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:473 |
| coupled.voltageSweep_V[2] | 100 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:474 |
| coupled.voltageSweep_V[3] | 125 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:475 |
| coupled.voltageSweep_V[4] | 150 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:476 |
| coupled.voltageSweep_V[5] | 175 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:477 |
| coupled.voltageSweep_V[6] | 200 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:478 |
| coupled.voltageSweep_V[7] | 225 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:479 |
| coupled.voltageSweep_V[8] | 250 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:480 |
| coupled.voltageSweep_V[9] | 260 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:481 |
| coupled.electricalOmega | 1.7 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:482 |
| coupled.electricalMaximumIterations | 1500 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:483 |
| coupled.electricalTolerance_V | 1e-08 | V | 수치 설정/제한 |  | config/default_lp_pva.yaml:484 |
| coupled.saveLevel | "light" | 키/정의 참조 | 기타 입력 |  | config/default_lp_pva.yaml:485 |
| coupled.snapshotTimes_s[0] | 0.25 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:487 |
| coupled.snapshotTimes_s[1] | 0.5 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:488 |
| coupled.snapshotTimes_s[2] | 1.0 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:489 |
| coupled.snapshotTimes_s[3] | 2.0 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:490 |
| coupled.snapshotTimes_s[4] | 5.0 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:491 |
| coupled.snapshotTimes_s[5] | 10.0 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:492 |
| coupled.earlyStopAfterEstablished_s | 0.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:493 |
| coupled.usePaperMassTransferSaturation | false | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:322 |
| coupled.useFullNernstPlanckTransport | true | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:321 |
| condensedIgnition.criterion | "first_local_decomposition_onset_temperature" | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:727 |
| condensedIgnition.decompositionOnsetTemperature_K | 523.15 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: config/default_lp_pva.yaml:728 |
| condensedIgnition.source | "Gnanaprakash_Yang_Yoh_non_metallized_M0_reported_decomposition_onset_349_degC" | 키/정의 참조 | 출처/의미/상태 메타데이터 |  | config/default_lp_pva.yaml:729 |
| condensedIgnition.interpretation | "condensed_phase_onset_not_gas_flame" | 키/정의 참조 | 출처/의미/상태 메타데이터 |  | config/default_lp_pva.yaml:730 |
| condensedIgnition.minimumChemicalProgressNumericalGuard | 0.01 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: config/default_lp_pva.yaml:731 |
| condensedIgnition.minimumChemicalRateNumericalGuard_per_s | 0.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:732 |
| condensedIgnition.sensitivityStudyRequiredForPublication | true | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:733 |
| condensedPhaseMetrics.evaluationTime_s | 2.0 | s | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:735 |
| condensedPhaseMetrics.undecomposedMetric | "area_average_of_one_minus_chemicalProgress" | 키/정의 참조 | 기타 입력 |  | config/default_lp_pva.yaml:736 |
| condensedPhaseMetrics.undecomposedMetricFormula | "A_p^-1 integral_Omega_p (1-alpha) dA" | 키/정의 참조 | 기타 입력 |  | config/default_lp_pva.yaml:737 |
| condensedPhaseMetrics.thresholdFree | true | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:738 |

### 공통 수치 제어

| key | value | units | classification | declared_source | source |
| --- | --- | --- | --- | --- | --- |
| numerics.physicsDevice | "cpu" | 키/정의 참조 | 기타 입력 |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: config/default_lp_pva.yaml:571 |
| numerics.physicsDtype | "float64" | 키/정의 참조 | 기타 입력 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:314 |
| numerics.deterministic | true | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:316 |
| numerics.electricalBatchSize | 16 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:574 |
| numerics.coupledBatchSize | 64 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:315 |
| numerics.potentialSolver.method | "auto" | 키/정의 참조 | 기타 입력 |  | config/default_lp_pva.yaml:577 |
| numerics.potentialSolver.methodStatic | "pcg" | 키/정의 참조 | 기타 입력 |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: config/default_lp_pva.yaml:578 |
| numerics.potentialSolver.methodCoupled | "pcg" | 키/정의 참조 | 기타 입력 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:318 |
| numerics.potentialSolver.directMaximumUnknowns | 500000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:580 |
| numerics.potentialSolver.maximumIterationsStatic | 3000 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:581 |
| numerics.potentialSolver.maximumIterationsCoupled | 1500 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:582 |
| numerics.potentialSolver.relativeToleranceStatic | 1e-10 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:583 |
| numerics.potentialSolver.relativeToleranceCoupled | 1e-09 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:584 |
| numerics.potentialSolver.absoluteTolerance | 1e-12 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:585 |
| numerics.potentialSolver.failOnNonConvergence | true | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:586 |
| numerics.potentialSolver.convergenceCheckInterval | 5 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:587 |
| numerics.potentialSolver.preconditioner | "jacobi" | 키/정의 참조 | 기타 입력 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:319 |
| numerics.potentialSolver.breakdownGuardFactor | 16.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:589 |
| numerics.potentialSolver.residualGrowthRestartFactor | 4.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:590 |
| numerics.potentialSolver.residualReplacementRelativeDrift | 0.25 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:591 |
| numerics.potentialSolver.maximumKrylovRestarts | 8 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:592 |
| numerics.potentialSolver.solverRoutingPolicy | "explicit_no_silent_substitution" | 키/정의 참조 | 기타 입력 |  | config/default_lp_pva.yaml:593 |
| numerics.potentialSolver.multigrid.maximumLevels | 6 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:595 |
| numerics.potentialSolver.multigrid.minimumGridSize | 7 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:596 |
| numerics.potentialSolver.multigrid.preSmooth | 2 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:597 |
| numerics.potentialSolver.multigrid.postSmooth | 2 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:598 |
| numerics.potentialSolver.multigrid.coarseSmooth | 24 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:599 |
| numerics.potentialSolver.multigrid.jacobiOmega | 0.7 | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/default_lp_pva.yaml:600 |
| numerics.potentialSolver.multigrid.scaleRestrictedResidualByGridRatioSquared | true | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:601 |
| numerics.potentialSolver.allowNonSPDPreconditionedPCG | false | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:602 |
| numerics.potentialSolver.symmetricEquilibration | true | 키/정의 참조 | 모델 선택/가정 |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: python/ecsp_nsga2/evaluator.py: 파생/override; resolved_bc_production_physics.json |
| numerics.potentialSolver.correctionForm | true | 키/정의 참조 | 모델 선택/가정 |  | python/ecsp_nsga2/evaluator.py (override/derived); resolved_bc_production_physics.json; upstream: python/ecsp_nsga2/evaluator.py: 파생/override; resolved_bc_production_physics.json |
| numerics.physicalFloors.current_A | 1e-15 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:604 |
| numerics.physicalFloors.currentDensity_A_per_m2 | 1e-12 | kg/m³ | 수치 설정/제한 |  | config/default_lp_pva.yaml:605 |
| numerics.physicalFloors.currentDensitySlope_A_per_m2_V | 1e-12 | kg/m³ | 수치 설정/제한 |  | config/default_lp_pva.yaml:606 |
| numerics.physicalFloors.conductivity_S_per_m | 1e-12 | S/m | 수치 설정/제한 |  | config/default_lp_pva.yaml:607 |
| numerics.physicalFloors.concentration_mol_per_m3 | 1e-12 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:608 |
| numerics.physicalFloors.dimensionless | 1e-12 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:609 |
| numerics.limiterComparisonRelativeTolerance | 1e-07 | 키/정의 참조 | 수치 설정/제한 |  | config/default_lp_pva.yaml:610 |
| numerics.speciesSubcycling | true | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:611 |
| numerics.speciesSubcyclingMode | "fixed" | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:612 |
| numerics.speciesFixedSubsteps | 2 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:613 |
| numerics.speciesMaximumSubsteps | 32 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:614 |
| numerics.speciesRelativeChangeTarget | 0.01 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:615 |
| numerics.recomputeTransportEachSpeciesSubstep | true | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:616 |
| numerics.failureIsolation.enabled | true | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:618 |
| numerics.failureIsolation.retryFailedOnResume | false | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:619 |
| numerics.failureIsolation.singleCandidateRetryEligible | false | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:620 |
| numerics.failureIsolation.includeTraceback | false | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:621 |
| numerics.failureIsolation.maximumFailureFraction | 0.2 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:622 |
| numerics.failureIsolation.minimumAttemptsBeforeFailureFractionGuard | 20 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:623 |
| numerics.failureIsolation.failureFractionAction | "raise" | 키 정의 확인(주로 무차원) | 기타 입력 |  | config/default_lp_pva.yaml:624 |
| numerics.progressEnabled | false | 키/정의 참조 | 모델 선택/가정 |  | config/default_lp_pva.yaml:625 |
| numerics.progressIntervalSteps | 400 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/default_lp_pva.yaml:626 |

### BC 활성

| key | value | units | classification | declared_source | source |
| --- | --- | --- | --- | --- | --- |
| bcGlobal.timeStep_s | 0.00025 | s | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:111 |
| bcGlobal.endTime_s | 2.0 | s | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/nsga2_bc_reactive_a100_cpu8.yaml:112 |
| bcGlobal.evaluationTime_s | 2.0 | s | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/nsga2_bc_reactive_a100_cpu8.yaml:113 |
| bcGlobal.electricalUpdateInterval_s | 0.0025 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:114 |
| bcGlobal.handoffSnapshotInterval_s | 0.02 | 키/정의 참조 | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:115 |
| bcGlobal.continuedElectricalHeatingAfterOnset | false | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:116 |
| bcGlobal.transport.cation_diffusivity.mode | "reference_arrhenius" | m²/s | 모델 선택/가정 | provenance: literature_nominal_pending_EIS_transference_or_PFG_NMR | config/nsga2_bc_reactive_a100_cpu8.yaml:119 |
| bcGlobal.transport.cation_diffusivity.reference_value | 2e-13 | m²/s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_EIS_transference_or_PFG_NMR | config/nsga2_bc_reactive_a100_cpu8.yaml:120 |
| bcGlobal.transport.cation_diffusivity.reference_temperature_K | 298.15 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_EIS_transference_or_PFG_NMR | config/nsga2_bc_reactive_a100_cpu8.yaml:121 |
| bcGlobal.transport.cation_diffusivity.activation_energy_J_per_mol | 22000.0 | m²/s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_EIS_transference_or_PFG_NMR | config/nsga2_bc_reactive_a100_cpu8.yaml:122 |
| bcGlobal.transport.cation_diffusivity.provenance | "literature_nominal_pending_EIS_transference_or_PFG_NMR" | m²/s | 출처/의미/상태 메타데이터 | provenance: literature_nominal_pending_EIS_transference_or_PFG_NMR | config/nsga2_bc_reactive_a100_cpu8.yaml:123 |
| bcGlobal.transport.anion_diffusivity.mode | "reference_arrhenius" | m²/s | 모델 선택/가정 | provenance: literature_nominal_pending_EIS_transference_or_PFG_NMR | config/nsga2_bc_reactive_a100_cpu8.yaml:125 |
| bcGlobal.transport.anion_diffusivity.reference_value | 1.5e-13 | m²/s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_EIS_transference_or_PFG_NMR | config/nsga2_bc_reactive_a100_cpu8.yaml:126 |
| bcGlobal.transport.anion_diffusivity.reference_temperature_K | 298.15 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_EIS_transference_or_PFG_NMR | config/nsga2_bc_reactive_a100_cpu8.yaml:127 |
| bcGlobal.transport.anion_diffusivity.activation_energy_J_per_mol | 25000.0 | m²/s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_EIS_transference_or_PFG_NMR | config/nsga2_bc_reactive_a100_cpu8.yaml:128 |
| bcGlobal.transport.anion_diffusivity.provenance | "literature_nominal_pending_EIS_transference_or_PFG_NMR" | m²/s | 출처/의미/상태 메타데이터 | provenance: literature_nominal_pending_EIS_transference_or_PFG_NMR | config/nsga2_bc_reactive_a100_cpu8.yaml:129 |
| bcGlobal.transport.water_diffusivity.mode | "reference_arrhenius" | m²/s | 모델 선택/가정 | provenance: literature_nominal_pending_DVS_or_PFG_NMR | config/nsga2_bc_reactive_a100_cpu8.yaml:131 |
| bcGlobal.transport.water_diffusivity.reference_value | 2e-10 | m²/s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_DVS_or_PFG_NMR | config/nsga2_bc_reactive_a100_cpu8.yaml:132 |
| bcGlobal.transport.water_diffusivity.reference_temperature_K | 298.15 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_DVS_or_PFG_NMR | config/nsga2_bc_reactive_a100_cpu8.yaml:133 |
| bcGlobal.transport.water_diffusivity.activation_energy_J_per_mol | 12000.0 | m²/s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_DVS_or_PFG_NMR | config/nsga2_bc_reactive_a100_cpu8.yaml:134 |
| bcGlobal.transport.water_diffusivity.provenance | "literature_nominal_pending_DVS_or_PFG_NMR" | m²/s | 출처/의미/상태 메타데이터 | provenance: literature_nominal_pending_DVS_or_PFG_NMR | config/nsga2_bc_reactive_a100_cpu8.yaml:135 |
| bcGlobal.augmentedElectronicConduction.enabled | false | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:137 |
| bcGlobal.augmentedElectronicConduction.interpretation | "strict_BC_ionic_current_only" | 키/정의 참조 | 출처/의미/상태 메타데이터 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:138 |
| bcGlobal.kinetics.model | "two_conversion_dependent_exothermic_channels_one_global_stoichiometry" | 키/정의 참조 | 모델 선택/가정 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses | config/nsga2_bc_reactive_a100_cpu8.yaml:143 |
| bcGlobal.kinetics.mass_conversion_weights[0] | 0.6666666666666666 | 1 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses | config/nsga2_bc_reactive_a100_cpu8.yaml:145 |
| bcGlobal.kinetics.mass_conversion_weights[1] | 0.3333333333333333 | 1 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses | config/nsga2_bc_reactive_a100_cpu8.yaml:146 |
| bcGlobal.kinetics.weight_provenance | "literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses" | 키/정의 참조 | 출처/의미/상태 메타데이터 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses | config/nsga2_bc_reactive_a100_cpu8.yaml:147 |
| bcGlobal.kinetics.heat_release_basis | "per_initial_bulk_propellant_channel_contribution" | J/kg | 출처/의미/상태 메타데이터 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses | config/nsga2_bc_reactive_a100_cpu8.yaml:148 |
| bcGlobal.kinetics.maximum_rate_per_s | 1000.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses | config/nsga2_bc_reactive_a100_cpu8.yaml:149 |
| bcGlobal.kinetics.channels[0].label | "second_exotherm_275_to_420C" | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:151 |
| bcGlobal.kinetics.channels[0].provenance | "approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data" | 키/정의 참조 | 출처/의미/상태 메타데이터 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:152 |
| bcGlobal.kinetics.channels[0].alpha_grid[0] | 0.0 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:154 |
| bcGlobal.kinetics.channels[0].alpha_grid[1] | 0.1 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:155 |
| bcGlobal.kinetics.channels[0].alpha_grid[2] | 0.2 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:156 |
| bcGlobal.kinetics.channels[0].alpha_grid[3] | 0.3 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:157 |
| bcGlobal.kinetics.channels[0].alpha_grid[4] | 0.4 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:158 |
| bcGlobal.kinetics.channels[0].alpha_grid[5] | 0.5 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:159 |
| bcGlobal.kinetics.channels[0].alpha_grid[6] | 0.6 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:160 |
| bcGlobal.kinetics.channels[0].alpha_grid[7] | 0.7 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:161 |
| bcGlobal.kinetics.channels[0].alpha_grid[8] | 0.8 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:162 |
| bcGlobal.kinetics.channels[0].alpha_grid[9] | 0.9 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:163 |
| bcGlobal.kinetics.channels[0].alpha_grid[10] | 1.0 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:164 |
| bcGlobal.kinetics.channels[0].activation_energy_J_per_mol[0] | 85000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:166 |
| bcGlobal.kinetics.channels[0].activation_energy_J_per_mol[1] | 85000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:167 |
| bcGlobal.kinetics.channels[0].activation_energy_J_per_mol[2] | 91000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:168 |
| bcGlobal.kinetics.channels[0].activation_energy_J_per_mol[3] | 98000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:169 |
| bcGlobal.kinetics.channels[0].activation_energy_J_per_mol[4] | 105000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:170 |
| bcGlobal.kinetics.channels[0].activation_energy_J_per_mol[5] | 112000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:171 |
| bcGlobal.kinetics.channels[0].activation_energy_J_per_mol[6] | 119000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:172 |
| bcGlobal.kinetics.channels[0].activation_energy_J_per_mol[7] | 127000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:173 |
| bcGlobal.kinetics.channels[0].activation_energy_J_per_mol[8] | 136000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:174 |
| bcGlobal.kinetics.channels[0].activation_energy_J_per_mol[9] | 150000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:175 |
| bcGlobal.kinetics.channels[0].activation_energy_J_per_mol[10] | 150000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:176 |
| bcGlobal.kinetics.channels[0].ln_Af_per_s[0] | 13.0 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:178 |
| bcGlobal.kinetics.channels[0].ln_Af_per_s[1] | 13.0 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:179 |
| bcGlobal.kinetics.channels[0].ln_Af_per_s[2] | 14.5 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:180 |
| bcGlobal.kinetics.channels[0].ln_Af_per_s[3] | 15.5 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:181 |
| bcGlobal.kinetics.channels[0].ln_Af_per_s[4] | 16.5 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:182 |
| bcGlobal.kinetics.channels[0].ln_Af_per_s[5] | 18.0 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:183 |
| bcGlobal.kinetics.channels[0].ln_Af_per_s[6] | 19.0 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:184 |
| bcGlobal.kinetics.channels[0].ln_Af_per_s[7] | 20.0 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:185 |
| bcGlobal.kinetics.channels[0].ln_Af_per_s[8] | 21.0 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:186 |
| bcGlobal.kinetics.channels[0].ln_Af_per_s[9] | 23.8 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:187 |
| bcGlobal.kinetics.channels[0].ln_Af_per_s[10] | 23.8 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:188 |
| bcGlobal.kinetics.channels[0].heat_release_J_per_kg | 881000.0 | J/kg | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8a_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:189 |
| bcGlobal.kinetics.channels[1].label | "third_exotherm_380_to_525C" | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:190 |
| bcGlobal.kinetics.channels[1].provenance | "approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data" | 키/정의 참조 | 출처/의미/상태 메타데이터 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:191 |
| bcGlobal.kinetics.channels[1].alpha_grid[0] | 0.0 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:193 |
| bcGlobal.kinetics.channels[1].alpha_grid[1] | 0.1 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:194 |
| bcGlobal.kinetics.channels[1].alpha_grid[2] | 0.2 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:195 |
| bcGlobal.kinetics.channels[1].alpha_grid[3] | 0.3 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:196 |
| bcGlobal.kinetics.channels[1].alpha_grid[4] | 0.4 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:197 |
| bcGlobal.kinetics.channels[1].alpha_grid[5] | 0.5 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:198 |
| bcGlobal.kinetics.channels[1].alpha_grid[6] | 0.6 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:199 |
| bcGlobal.kinetics.channels[1].alpha_grid[7] | 0.7 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:200 |
| bcGlobal.kinetics.channels[1].alpha_grid[8] | 0.8 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:201 |
| bcGlobal.kinetics.channels[1].alpha_grid[9] | 0.9 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:202 |
| bcGlobal.kinetics.channels[1].alpha_grid[10] | 1.0 | 1 | 운영/초기/기하 조건(수분·두께 등 측정 필요) | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:203 |
| bcGlobal.kinetics.channels[1].activation_energy_J_per_mol[0] | 150000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:205 |
| bcGlobal.kinetics.channels[1].activation_energy_J_per_mol[1] | 150000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:206 |
| bcGlobal.kinetics.channels[1].activation_energy_J_per_mol[2] | 145000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:207 |
| bcGlobal.kinetics.channels[1].activation_energy_J_per_mol[3] | 140000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:208 |
| bcGlobal.kinetics.channels[1].activation_energy_J_per_mol[4] | 140000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:209 |
| bcGlobal.kinetics.channels[1].activation_energy_J_per_mol[5] | 150000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:210 |
| bcGlobal.kinetics.channels[1].activation_energy_J_per_mol[6] | 190000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:211 |
| bcGlobal.kinetics.channels[1].activation_energy_J_per_mol[7] | 220000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:212 |
| bcGlobal.kinetics.channels[1].activation_energy_J_per_mol[8] | 190000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:213 |
| bcGlobal.kinetics.channels[1].activation_energy_J_per_mol[9] | 150000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:214 |
| bcGlobal.kinetics.channels[1].activation_energy_J_per_mol[10] | 150000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:215 |
| bcGlobal.kinetics.channels[1].ln_Af_per_s[0] | 19.0 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:217 |
| bcGlobal.kinetics.channels[1].ln_Af_per_s[1] | 19.0 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:218 |
| bcGlobal.kinetics.channels[1].ln_Af_per_s[2] | 18.5 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:219 |
| bcGlobal.kinetics.channels[1].ln_Af_per_s[3] | 18.0 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:220 |
| bcGlobal.kinetics.channels[1].ln_Af_per_s[4] | 18.0 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:221 |
| bcGlobal.kinetics.channels[1].ln_Af_per_s[5] | 20.0 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:222 |
| bcGlobal.kinetics.channels[1].ln_Af_per_s[6] | 26.0 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:223 |
| bcGlobal.kinetics.channels[1].ln_Af_per_s[7] | 29.9 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:224 |
| bcGlobal.kinetics.channels[1].ln_Af_per_s[8] | 26.0 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:225 |
| bcGlobal.kinetics.channels[1].ln_Af_per_s[9] | 18.0 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:226 |
| bcGlobal.kinetics.channels[1].ln_Af_per_s[10] | 18.0 | ln(Af / s⁻¹) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:227 |
| bcGlobal.kinetics.channels[1].heat_release_J_per_kg | 1162000.0 | J/kg | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | weight_provenance: literature_nominal_from_2nd_paper_24_to_12_percent_main_stage_mass_losses; provenance: approximate_digitisation_of_3rd_paper_Fig8b_pending_supplementary_numeric_data | config/nsga2_bc_reactive_a100_cpu8.yaml:228 |
| bcGlobal.onsetCriterion.temperature_K | 523.15 | K | 모델 선택/가정 | status: operational_threshold_pending_experimental_calibration | config/nsga2_bc_reactive_a100_cpu8.yaml:230 |
| bcGlobal.onsetCriterion.minimum_progress | 0.01 | 키/정의 참조 | 모델 선택/가정 | status: operational_threshold_pending_experimental_calibration | config/nsga2_bc_reactive_a100_cpu8.yaml:231 |
| bcGlobal.onsetCriterion.minimum_area_fraction | 0.01 | 키 정의 확인(주로 무차원) | 모델 선택/가정 | status: operational_threshold_pending_experimental_calibration | config/nsga2_bc_reactive_a100_cpu8.yaml:232 |
| bcGlobal.onsetCriterion.status | "operational_threshold_pending_experimental_calibration" | 키/정의 참조 | 출처/의미/상태 메타데이터 | status: operational_threshold_pending_experimental_calibration | config/nsga2_bc_reactive_a100_cpu8.yaml:233 |
| bcGlobal.thermal.density_kg_per_m3 | null | kg/m³ | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:235 |
| bcGlobal.thermal.heat_capacity.mode | "table" | 키/정의 참조 | 모델 선택/가정 | provenance: literature_nominal_pending_DSC_specific_heat | config/nsga2_bc_reactive_a100_cpu8.yaml:237 |
| bcGlobal.thermal.heat_capacity.temperature_K[0] | 298.15 | K | 수치 설정/제한 | provenance: literature_nominal_pending_DSC_specific_heat | config/nsga2_bc_reactive_a100_cpu8.yaml:239 |
| bcGlobal.thermal.heat_capacity.temperature_K[1] | 373.15 | K | 수치 설정/제한 | provenance: literature_nominal_pending_DSC_specific_heat | config/nsga2_bc_reactive_a100_cpu8.yaml:240 |
| bcGlobal.thermal.heat_capacity.temperature_K[2] | 473.15 | K | 수치 설정/제한 | provenance: literature_nominal_pending_DSC_specific_heat | config/nsga2_bc_reactive_a100_cpu8.yaml:241 |
| bcGlobal.thermal.heat_capacity.temperature_K[3] | 573.15 | K | 수치 설정/제한 | provenance: literature_nominal_pending_DSC_specific_heat | config/nsga2_bc_reactive_a100_cpu8.yaml:242 |
| bcGlobal.thermal.heat_capacity.temperature_K[4] | 773.15 | K | 수치 설정/제한 | provenance: literature_nominal_pending_DSC_specific_heat | config/nsga2_bc_reactive_a100_cpu8.yaml:243 |
| bcGlobal.thermal.heat_capacity.values[0] | 2200.0 | J/(kg K) | 수치 설정/제한 | provenance: literature_nominal_pending_DSC_specific_heat | config/nsga2_bc_reactive_a100_cpu8.yaml:245 |
| bcGlobal.thermal.heat_capacity.values[1] | 2250.0 | J/(kg K) | 수치 설정/제한 | provenance: literature_nominal_pending_DSC_specific_heat | config/nsga2_bc_reactive_a100_cpu8.yaml:246 |
| bcGlobal.thermal.heat_capacity.values[2] | 2350.0 | J/(kg K) | 수치 설정/제한 | provenance: literature_nominal_pending_DSC_specific_heat | config/nsga2_bc_reactive_a100_cpu8.yaml:247 |
| bcGlobal.thermal.heat_capacity.values[3] | 2500.0 | J/(kg K) | 수치 설정/제한 | provenance: literature_nominal_pending_DSC_specific_heat | config/nsga2_bc_reactive_a100_cpu8.yaml:248 |
| bcGlobal.thermal.heat_capacity.values[4] | 2700.0 | J/(kg K) | 수치 설정/제한 | provenance: literature_nominal_pending_DSC_specific_heat | config/nsga2_bc_reactive_a100_cpu8.yaml:249 |
| bcGlobal.thermal.heat_capacity.provenance | "literature_nominal_pending_DSC_specific_heat" | 키/정의 참조 | 출처/의미/상태 메타데이터 | provenance: literature_nominal_pending_DSC_specific_heat | config/nsga2_bc_reactive_a100_cpu8.yaml:250 |
| bcGlobal.thermal.thermal_conductivity.mode | "table" | S/m 또는 W/(m K): 키 문맥 | 모델 선택/가정 | provenance: literature_nominal_pending_LFA_or_TPS | config/nsga2_bc_reactive_a100_cpu8.yaml:252 |
| bcGlobal.thermal.thermal_conductivity.temperature_K[0] | 298.15 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_LFA_or_TPS | config/nsga2_bc_reactive_a100_cpu8.yaml:254 |
| bcGlobal.thermal.thermal_conductivity.temperature_K[1] | 373.15 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_LFA_or_TPS | config/nsga2_bc_reactive_a100_cpu8.yaml:255 |
| bcGlobal.thermal.thermal_conductivity.temperature_K[2] | 473.15 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_LFA_or_TPS | config/nsga2_bc_reactive_a100_cpu8.yaml:256 |
| bcGlobal.thermal.thermal_conductivity.temperature_K[3] | 573.15 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_LFA_or_TPS | config/nsga2_bc_reactive_a100_cpu8.yaml:257 |
| bcGlobal.thermal.thermal_conductivity.temperature_K[4] | 773.15 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_LFA_or_TPS | config/nsga2_bc_reactive_a100_cpu8.yaml:258 |
| bcGlobal.thermal.thermal_conductivity.values[0] | 0.4 | W/(m K) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_LFA_or_TPS | config/nsga2_bc_reactive_a100_cpu8.yaml:260 |
| bcGlobal.thermal.thermal_conductivity.values[1] | 0.39 | W/(m K) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_LFA_or_TPS | config/nsga2_bc_reactive_a100_cpu8.yaml:261 |
| bcGlobal.thermal.thermal_conductivity.values[2] | 0.37 | W/(m K) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_LFA_or_TPS | config/nsga2_bc_reactive_a100_cpu8.yaml:262 |
| bcGlobal.thermal.thermal_conductivity.values[3] | 0.35 | W/(m K) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_LFA_or_TPS | config/nsga2_bc_reactive_a100_cpu8.yaml:263 |
| bcGlobal.thermal.thermal_conductivity.values[4] | 0.32 | W/(m K) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: literature_nominal_pending_LFA_or_TPS | config/nsga2_bc_reactive_a100_cpu8.yaml:264 |
| bcGlobal.thermal.thermal_conductivity.provenance | "literature_nominal_pending_LFA_or_TPS" | S/m 또는 W/(m K): 키 문맥 | 출처/의미/상태 메타데이터 | provenance: literature_nominal_pending_LFA_or_TPS | config/nsga2_bc_reactive_a100_cpu8.yaml:265 |
| bcGlobal.thermal.initialTemperature_K | 298.15 | K | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/nsga2_bc_reactive_a100_cpu8.yaml:266 |
| bcGlobal.thermal.ambientTemperature_K | 298.15 | K | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/nsga2_bc_reactive_a100_cpu8.yaml:267 |
| bcGlobal.thermal.convectionCoefficient_W_per_m2K | 12.0 | W/(m² K) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:268 |
| bcGlobal.thermal.emissivity | 0.85 | 1 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:269 |
| bcGlobal.thermal.stefanBoltzmann_W_per_m2K4 | 5.670374419e-08 | 키/정의 참조 | 상수/화학종·반응 정의(반응 정의는 별도 검증 필요) |  | config/nsga2_bc_reactive_a100_cpu8.yaml:270 |
| bcGlobal.thermal.minimumTemperature_K | 250.0 | K | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:271 |
| bcGlobal.thermal.maximumTemperature_K | 2500.0 | K | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:272 |

### 후속 모델/운영/최적화

| key | value | units | classification | declared_source | source |
| --- | --- | --- | --- | --- | --- |
| post_onset.backend | "reactive_euler" | 키/정의 참조 | 기타 입력 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:351 |
| post_onset.compare_backends | true | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:352 |
| post_onset.allow_experimental_ranking | true | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:353 |
| post_onset.reactive_euler.eos.A_Pa | 101325.0 | Pa | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: ASSUMED_NOT_FROM_PAPER; inherited v8.3 example, not measured ECSP Tait coefficients; rho0 defaults to resolved BC density | config/nsga2_bc_reactive_a100_cpu8.yaml:356 |
| post_onset.reactive_euler.eos.B_Pa | 2000000000.0 | Pa | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: ASSUMED_NOT_FROM_PAPER; inherited v8.3 example, not measured ECSP Tait coefficients; rho0 defaults to resolved BC density | config/nsga2_bc_reactive_a100_cpu8.yaml:357 |
| post_onset.reactive_euler.eos.N | 7.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | provenance: ASSUMED_NOT_FROM_PAPER; inherited v8.3 example, not measured ECSP Tait coefficients; rho0 defaults to resolved BC density | config/nsga2_bc_reactive_a100_cpu8.yaml:358 |
| post_onset.reactive_euler.eos.rho0_kg_per_m3 | null | 키/정의 참조 | 모델 선택/가정 | provenance: ASSUMED_NOT_FROM_PAPER; inherited v8.3 example, not measured ECSP Tait coefficients; rho0 defaults to resolved BC density | config/nsga2_bc_reactive_a100_cpu8.yaml:359 |
| post_onset.reactive_euler.eos.provenance | "ASSUMED_NOT_FROM_PAPER; inherited v8.3 example, not measured ECSP Tait coefficients; rho0 defaults to resolved BC density" | 키/정의 참조 | 출처/의미/상태 메타데이터 | provenance: ASSUMED_NOT_FROM_PAPER; inherited v8.3 example, not measured ECSP Tait coefficients; rho0 defaults to resolved BC density | config/nsga2_bc_reactive_a100_cpu8.yaml:360 |
| post_onset.reactive_euler.caloric_closure | "bc_cp_integral_plus_tait_cold_energy" | 키/정의 참조 | 기타 입력 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:362 |
| post_onset.reactive_euler.boundary | "reflective" | 키/정의 참조 | 기타 입력 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:363 |
| post_onset.reactive_euler.riemann_solver | "hllc" | 키/정의 참조 | 기타 입력 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:364 |
| post_onset.reactive_euler.weno_epsilon | 1e-06 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:365 |
| post_onset.reactive_euler.cfl | 0.35 | 키/정의 참조 | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:366 |
| post_onset.reactive_euler.thermal_cfl | 0.7 | 키/정의 참조 | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:367 |
| post_onset.reactive_euler.maximum_channel_increment | 0.02 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:368 |
| post_onset.reactive_euler.stationary_mechanics_fast_path | true | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:369 |
| post_onset.reactive_euler.maximum_time_steps | 100000 | 키/정의 참조 | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:370 |
| post_onset.reactive_euler.maximum_step_retries | 12 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:371 |
| post_onset.reactive_euler.minimum_time_step_s | 1e-14 | s | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:372 |
| post_onset.reactive_euler.mass_budget_relative_tolerance | 1e-10 | 키/정의 참조 | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:373 |
| post_onset.reactive_euler.energy_budget_relative_tolerance | 1e-09 | 키/정의 참조 | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:374 |
| post_onset.reactive_euler.interpretation | "single condensed continuum; no thermal pressure source in barotropic Tait; retain BC Fourier conduction; no gas mass transfer" | 키/정의 참조 | 출처/의미/상태 메타데이터 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:375 |
| post_onset.note | "Experimental FP64 tensor continuation on A100; CPU workers run paired baseline; no gas CFD or hardware speedup claim. All physical coefficients unchanged from v8.4.0." | 키/정의 참조 | 출처/의미/상태 메타데이터 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:377 |
| post_onset.execution.backend | "torch_batch" | 키/정의 참조 | 기타 입력 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:381 |
| post_onset.execution.device | "cuda" | 키/정의 참조 | 기타 입력 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:382 |
| post_onset.execution.batch_size | 8 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:383 |
| post_onset.execution.cuda_graph | false | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:384 |
| post_onset.execution.cpu_budget | 8 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:385 |
| post_onset.execution.host_reserve | 2 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:386 |
| post_onset.execution.cpu_workers | 6 | 키/정의 참조 | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:387 |
| post_onset.execution.torch_threads | 1 | 키/정의 참조 | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:388 |
| post_onset.execution.maximum_batch_working_bytes | 8589934592 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:389 |
| post_onset.execution.maximum_batch_history_bytes | 8589934592 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:390 |
| post_onset.execution.oom_split_retry | true | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:391 |
| propagation_refinement.enabled | true | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:274 |
| propagation_refinement.selection.maximum_pareto_candidates | 50 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:276 |
| propagation_refinement.selection.additional_fraction | 0.15 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:277 |
| propagation_refinement.selection.minimum_additional_candidates | 4 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:278 |
| propagation_refinement.selection.maximum_total_candidates | 60 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:279 |
| propagation_refinement.selection.source | "preflame_pareto_plus_near_pareto_max_diversity" | 키/정의 참조 | 출처/의미/상태 메타데이터 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:280 |
| propagation_refinement.selection.maximum_near_pareto_rank | 3 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:281 |
| propagation_refinement.duration_s | 1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:282 |
| propagation_refinement.time_step_s | 0.00025 | s | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:283 |
| propagation_refinement.snapshot_interval_s | 0.05 | 키/정의 참조 | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:284 |
| propagation_refinement.front_progress_threshold | 0.5 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:285 |
| propagation_refinement.established_reacted_area_fraction | 0.5 | 키 정의 확인(주로 무차원) | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/nsga2_bc_reactive_a100_cpu8.yaml:286 |
| propagation_refinement.continued_electrical_heating | false | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:287 |
| propagation_refinement.combined_final_objectives[0] | "ignition_delay_s" | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:289 |
| propagation_refinement.combined_final_objectives[1] | "area_undecomposed_fraction_at_evaluation_time" | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:290 |
| propagation_refinement.combined_final_objectives[2] | "minimum_ignition_voltage_V" | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:291 |
| propagation_refinement.combined_final_objectives[3] | "current_congestion" | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:292 |
| propagation_refinement.combined_final_objectives[4] | "final_unreacted_area_fraction" | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:293 |
| propagation_refinement.combined_final_objectives[5] | "established_time_after_onset_s" | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:294 |
| propagation_refinement.combined_final_objectives[6] | "negative_mean_regression_velocity_m_per_s" | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:295 |
| propagation_refinement.combined_final_objectives[7] | "reaction_front_nonuniformity" | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:296 |
| propagation_refinement.interpretation | "post_onset_condensed_phase_reaction_progress_and_level_set_not_gas_phase_cfd" | 키/정의 참조 | 출처/의미/상태 메타데이터 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:297 |
| propagation_refinement.execution.backend | "numpy_cpu" | 키/정의 참조 | 기타 입력 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:299 |
| propagation_refinement.execution.parallel_cases | 6 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:300 |
| propagation_refinement.execution.note | "Historical fallback-only setting; post_onset.execution controls this profile." | 키/정의 참조 | 출처/의미/상태 메타데이터 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:301 |
| physics.voltage_V | 260.0 | V | 운영/초기/기하 조건(수분·두께 등 측정 필요) | calibration_status: literature_nominal_uncalibrated | config/nsga2_bc_reactive_a100_cpu8.yaml:62 |
| physics.end_time_s | 2.0 | s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | calibration_status: literature_nominal_uncalibrated | config/nsga2_bc_reactive_a100_cpu8.yaml:63 |
| physics.metric_evaluation_time_s | 2.0 | s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | calibration_status: literature_nominal_uncalibrated | config/nsga2_bc_reactive_a100_cpu8.yaml:64 |
| physics.interpretation | "bc_global_preflame_condensed_phase_not_visible_flame" | 키/정의 참조 | 출처/의미/상태 메타데이터 | calibration_status: literature_nominal_uncalibrated | config/nsga2_bc_reactive_a100_cpu8.yaml:65 |
| physics.composition.lithium_perchlorate_mass_fraction | 0.3158 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | calibration_status: literature_nominal_uncalibrated | config/nsga2_bc_reactive_a100_cpu8.yaml:67 |
| physics.composition.water_mass_fraction | 0.5842 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | calibration_status: literature_nominal_uncalibrated | config/nsga2_bc_reactive_a100_cpu8.yaml:68 |
| physics.composition.pva_mass_fraction | 0.09 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | calibration_status: literature_nominal_uncalibrated | config/nsga2_bc_reactive_a100_cpu8.yaml:69 |
| physics.composition.glycerol_mass_fraction | 0.0 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | calibration_status: literature_nominal_uncalibrated | config/nsga2_bc_reactive_a100_cpu8.yaml:70 |
| physics.composition.boric_acid_mass_fraction | 0.01 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | calibration_status: literature_nominal_uncalibrated | config/nsga2_bc_reactive_a100_cpu8.yaml:71 |
| physics.composition.tungsten_mass_fraction | 0.0 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 | calibration_status: literature_nominal_uncalibrated | config/nsga2_bc_reactive_a100_cpu8.yaml:72 |
| physics.cured_water_mass_fraction | 0.2 | 키 정의 확인(주로 무차원) | 운영/초기/기하 조건(수분·두께 등 측정 필요) | calibration_status: literature_nominal_uncalibrated | config/nsga2_bc_reactive_a100_cpu8.yaml:73 |
| physics.cured_water_mass_fraction_basis | "literature_nominal_uncalibrated_3rd_paper_low_temperature_TGA_loss" | 키 정의 확인(주로 무차원) | 출처/의미/상태 메타데이터 | calibration_status: literature_nominal_uncalibrated | config/nsga2_bc_reactive_a100_cpu8.yaml:74 |
| physics.calibration_status | "literature_nominal_uncalibrated" | 키/정의 참조 | 출처/의미/상태 메타데이터 | calibration_status: literature_nominal_uncalibrated | config/nsga2_bc_reactive_a100_cpu8.yaml:75 |
| condensed_ignition.criterion | "temperature_and_global_conversion_over_minimum_area" | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:77 |
| condensed_ignition.onset_temperature_K | 523.15 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:78 |
| condensed_ignition.onset_temperature_source | "2nd_paper_250C_onset_with_3rd_paper_260C_sensitivity" | K | 출처/의미/상태 메타데이터 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:79 |
| condensed_ignition.minimum_conversion_numerical_guard | 0.01 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:80 |
| condensed_ignition.minimum_rate_numerical_guard_per_s | 0.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:81 |
| condensed_ignition.reference_time_s | 2.0 | s | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:82 |
| condensed_ignition.interpretation | "condensed_phase_decomposition_onset_not_visible_gas_flame" | 키/정의 참조 | 출처/의미/상태 메타데이터 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:83 |
| condensed_ignition.sensitivity_onset_temperature_K[0] | 523.15 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:85 |
| condensed_ignition.sensitivity_onset_temperature_K[1] | 533.15 | K | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:86 |
| minimum_ignition_voltage_search.enabled | true | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:88 |
| minimum_ignition_voltage_search.lower_bound_V | 20.0 | V | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/nsga2_bc_reactive_a100_cpu8.yaml:89 |
| minimum_ignition_voltage_search.upper_bound_V | 260.0 | V | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/nsga2_bc_reactive_a100_cpu8.yaml:90 |
| minimum_ignition_voltage_search.tolerance_V | 5.0 | V | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:91 |
| minimum_ignition_voltage_search.maximum_bisection_iterations | 8 | 키/정의 참조 | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:92 |
| minimum_ignition_voltage_search.right_censor_objective_penalty_V | 50.0 | V | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/nsga2_bc_reactive_a100_cpu8.yaml:93 |
| minimum_ignition_voltage_search.invalid_search_objective_penalty_V | 100.0 | V | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/nsga2_bc_reactive_a100_cpu8.yaml:94 |
| minimum_ignition_voltage_search.stop_successful_trials_at_ignition | true | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:95 |
| minimum_ignition_voltage_search.verify_final_upper_full_horizon | true | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:96 |
| minimum_ignition_voltage_search.reported_value | "conservative_upper_igniting_bracket" | 키/정의 참조 | 운영/초기/기하 조건(수분·두께 등 측정 필요) |  | config/nsga2_bc_reactive_a100_cpu8.yaml:97 |
| minimum_ignition_voltage_search.interpretation | "minimum_voltage_for_identical_temperature_plus_conversion_area_onset_within_configured_evaluation_horizon" | 키/정의 참조 | 출처/의미/상태 메타데이터 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:98 |
| optimization.algorithm | "constrained_grammar_nsga2" | 키/정의 참조 | 기타 입력 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:12 |
| optimization.population_size | 1000 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:13 |
| optimization.generations | 5 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:14 |
| optimization.initial_topologies | 20 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:15 |
| optimization.variants_per_topology | 50 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:16 |
| optimization.physics_batch_size | 70 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:17 |
| optimization.mating_performance_count | 70 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:18 |
| optimization.mating_topology_count | 20 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:19 |
| optimization.mating_diversity_count | 10 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:20 |
| optimization.offspring_fractions.crossover | 0.6 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:22 |
| optimization.offspring_fractions.mutation | 0.25 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:23 |
| optimization.offspring_fractions.new_grammar | 0.15 | 키 정의 확인(주로 무차원) | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:24 |
| optimization.offspring_generation_attempts | 40 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:25 |
| optimization.objectives_minimise[0] | "ignition_delay_s" | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:27 |
| optimization.objectives_minimise[1] | "area_undecomposed_fraction_at_evaluation_time" | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:28 |
| optimization.objectives_minimise[2] | "minimum_ignition_voltage_V" | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:29 |
| optimization.objectives_minimise[3] | "current_congestion" | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:30 |
| optimization.no_ignition_penalty_s | 2.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:31 |
| optimization.early_stop_patience | 2 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:32 |
| optimization.early_stop_min_improvement | 0.001 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:33 |
| optimization.numerical_cap_thresholds.temperature | 0.0 | K | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:35 |
| optimization.numerical_cap_thresholds.species | 0.0 | 키/정의 참조 | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:36 |
| optimization.numerical_cap_thresholds.gas | 0.0 | 키/정의 참조 | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:37 |
| optimization.numerical_cap_thresholds.chemical_rate | 0.0 | 키/정의 참조 | 수치 설정/제한 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:38 |
| optimization.no_ignition_constraint_violation | 1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:39 |
| optimization.continuous_ignition_constraint | true | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:40 |
| optimization.minimum_no_ignition_violation | 1e-06 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:41 |
| optimization.require_ignition_for_feasibility | true | 키/정의 참조 | 모델 선택/가정 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:42 |
| optimization.vmin_invalid_search_constraint_violation | 1.0 | 키/정의 참조 | 물성/경험/속도론 값: 실제 시료 미보정 또는 코드 유도값 |  | config/nsga2_bc_reactive_a100_cpu8.yaml:43 |
