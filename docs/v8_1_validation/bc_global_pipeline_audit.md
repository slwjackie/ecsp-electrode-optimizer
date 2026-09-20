# B/C global pre-flame + condensed propagation 구현 감사

- 판정: **통과**
- 통과: **24/24**

| ID | 판정 | 요구사항 | 근거 |
|---|---:|---|---|
| `eq.used` | PASS | B/C pre-flame 식 (22)–(24)와 필요한 보조식만 사용 | `{"scope": "bc_global_preflame_species_electrochemical_thermal", "global_reaction": "1.45 LiClO4 + PVA_repeat -> 2 CO2 + 2 H2O + 0.4 O2 + 1.45 LiCl", "paper_equations_used": [2, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 22, 23, 24], "paper_equations_not_used_in_preflame": [3, 18, 25, 26, 27, 28, 29, 30, 31], "equation_32_role": "postprocessing_identity_only_not_added_as_a_second_heat_source", "legacy_proxy_closures_used": false, "gas_phase_cfd_used": false, "post_onset_handoff_fields": ["temperatureAtOnset_K", "globalProgressAtOnset", "qJAtOnset_W_per_m3", "qEchemAtOnset_W_per_m3"]}` |
| `eq.excluded` | PASS | Poisson/밀도합/점화후 유동/중복 한계전류 식 제외 | `{"scope": "bc_global_preflame_species_electrochemical_thermal", "global_reaction": "1.45 LiClO4 + PVA_repeat -> 2 CO2 + 2 H2O + 0.4 O2 + 1.45 LiCl", "paper_equations_used": [2, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 22, 23, 24], "paper_equations_not_used_in_preflame": [3, 18, 25, 26, 27, 28, 29, 30, 31], "equation_32_role": "postprocessing_identity_only_not_added_as_a_second_heat_source", "legacy_proxy_closures_used": false, "gas_phase_cfd_used": false, "post_onset_handoff_fields": ["temperatureAtOnset_K", "globalProgressAtOnset", "qJAtOnset_W_per_m3", "qEchemAtOnset_W_per_m3"]}` |
| `eq32.diagnostic` | PASS | 식 (32)는 중복 열원이 아니라 검산용 | `"postprocessing_identity_only_not_added_as_a_second_heat_source"` |
| `gas.none` | PASS | 후단 모델은 외부 기상 CFD가 아닌 응축상 propagation | `false` |
| `composition.wet_recipe` | PASS | 3번 논문 non-metallized wet recipe와 동일 | `{"lithium_perchlorate_mass_fraction": 0.3158, "water_mass_fraction": 0.5842, "pva_mass_fraction": 0.09, "glycerol_mass_fraction": 0.0, "boric_acid_mass_fraction": 0.01, "tungsten_mass_fraction": 0.0}` |
| `composition.cured_water` | PASS | 경화 후 수분 0.20은 literature-nominal로 명시 | `{"value": 0.2, "basis": "literature_nominal_uncalibrated_3rd_paper_low_temperature_TGA_loss"}` |
| `chem.global` | PASS | 3번 논문 global LP/PVA 반응 사용 | `"1.45 LiClO4 + PVA_repeat -> 2 CO2 + 2 H2O + 0.4 O2 + 1.45 LiCl"` |
| `chem.two_channels` | PASS | 두 conversion-dependent exothermic channel | `"two_conversion_dependent_exothermic_channels_one_global_stoichiometry"` |
| `chem.weights` | PASS | global species inventory는 두 채널 가중합으로 한 번만 소비 | `[0.6666666666666666, 0.3333333333333333]` |
| `objectives.four` | PASS | Urem, t_onset, Vmin, J99/Jbar 네 목적함수 | `["ignition_delay_s", "area_undecomposed_fraction_at_2s", "minimum_ignition_voltage_V", "current_congestion"]` |
| `onset.and_area` | PASS | T AND Xg over minimum area onset criterion | `{"temperature_K": 523.15, "minimum_progress": 0.01, "minimum_area_fraction": 0.01, "status": "operational_threshold_pending_experimental_calibration"}` |
| `proxy.disabled` | PASS | 질량전달 직렬 제한·passivation·gas coverage·liquid multiplier 비활성 | `{"usePaperMassTransferSaturation": false, "passivation": false, "gasCoverage": false, "liquidKineticsGain": 0.0, "activationHeat": false, "electronicAugmentation": false}` |
| `pipeline.1000` | PASS | 1,000개 형상 B/C 전수평가 기본 profile | `1000` |
| `pipeline.diversity` | PASS | Pareto + near-Pareto/diverse 10–20% | `{"maximum_pareto_candidates": 50, "additional_fraction": 0.15, "minimum_additional_candidates": 4, "maximum_total_candidates": 60, "source": "preflame_pareto_plus_near_pareto_max_diversity", "maximum_near_pareto_rank": 3}` |
| `pipeline.propagation` | PASS | post-onset condensed reaction-progress/level-set refinement | `"post_onset_condensed_phase_reaction_progress_and_level_set_not_gas_phase_cfd"` |
| `pipeline.handoff` | PASS | T, Xg, qJ, qEchem full-field handoff | `["temperatureAtOnset_K", "globalProgressAtOnset", "qJAtOnset_W_per_m3", "qEchemAtOnset_W_per_m3"]` |
| `pipeline.staggered` | PASS | area-matched staggered는 동일 B/C+propagation 경로로 후평가 | `{"enabled": true, "layout": "hidden_bus_vertical_2anode_2cathode", "fingers_per_polarity": 2, "target_interdigitation_overlap_fraction": 0.5, "minimum_interdigitation_overlap_fraction": 0.45, "hidden_bus_in_contact_mask": false, "maximum_gap_safety_pixels": 8, "include_in_nsga2_population": false, "include_in_pareto_selection": false}` |
| `pipeline.staggered_constructible` | PASS | production width bound permits the required two-finger area-matched staggered reference | `{"maximum_width_mm": 5.0, "approximate_required_width_mm": 1.75}` |
| `legacy.configs` | PASS | v7.9.5 backend/config/source 보존 | `"legacy files present"` |
| `runtime.exit` | PASS | CPU FP64 debug pipeline completes | `0` |
| `runtime.outputs` | PASS | pre-flame, propagation, final Pareto, staggered outputs persisted | `[]` |
| `runtime.objectives` | PASS | final Pareto combines 4 pre-flame + 4 propagation outputs | `{"ignition_delay_s": 0.002, "area_undecomposed_fraction_at_2s": 0.9999349503803872, "minimum_ignition_voltage_V": 1.0, "current_congestion": 0.0, "final_unreacted_area_fraction": 0.0, "established_time_after_onset_s": 0.002, "negative_mean_regression_velocity_m_per_s": -0.0, "reaction_front_nonuniformity": 1.0}` |
| `runtime.scope` | PASS | runtime scope labels match implementation | `{"recommended_geometry_id": "G000_T00_1A1C_f430c6_V000", "elapsed_s": 0.8311038017272949, "openfoam_used": false, "model_scope": "bc_global_preflame_plus_post_onset_condensed_reaction_propagation", "bc_global_preflame_used": true, "post_onset_condensed_propagation_used": true, "gas_phase_cfd_used": false, "area_matched_staggered_evaluated": true, "area_matched_staggered_included_in_optimization": false}` |
| `runtime.full_fields` | PASS | full T/Xg/level-set propagation fields written | `["/mnt/data/v81_validation/legacy_bc_run/final/propagation_candidates/G000_T00_1A1C_f430c6_V001/condensed_propagation/propagation_fields.npz", "/mnt/data/v81_validation/legacy_bc_run/final/propagation_candidates/G000_T00_1A1C_f430c6_V000/condensed_propagation/propagation_fields.npz", "/mnt/data/v81_validation/legacy_bc_run/final/propagation_candidates/G000_T01_1A2C_64e3cb_V000/condensed_propagation/propagation_fields.npz"]` |

## 실행 정보

```json
{
  "executed": true,
  "command": [
    "/opt/pyvenv/bin/python",
    "/mnt/data/ECSP_v8_1_0_BCNative_Hybrid_A100_CPU8/python/run_nsga2_electrical_solid_loop.py",
    "--config",
    "/mnt/data/ECSP_v8_1_0_BCNative_Hybrid_A100_CPU8/config/nsga2_bc_global_preflame_propagation_debug.yaml",
    "--workdir",
    "/mnt/data/v81_validation/legacy_bc_run",
    "--package-root",
    "/mnt/data/ECSP_v8_1_0_BCNative_Hybrid_A100_CPU8",
    "--allow-no-feasible"
  ],
  "returncode": 0,
  "workdir": "/mnt/data/v81_validation/legacy_bc_run",
  "stdout_tail": "[physics] batch 1/1 start size=2 ids=G000_T00_1A1C_f430c6_V000..G000_T00_1A1C_f430c6_V001\n[physics] batch 1/1 complete successful=2 elapsed=0.1s\n[generation 000] physics 2/4 successful=2 rejected=0 elapsed=0.3s\n[physics] batch 1/1 start size=2 ids=G000_T01_1A2C_64e3cb_V000..G000_T01_1A2C_64e3cb_V001\n[physics] batch 1/1 complete successful=2 elapsed=0.1s\n[generation 000] physics 4/4 successful=4 rejected=0 elapsed=0.3s\n[propagation] preparing B/C handoff fields for 4 candidates in batched mode\n[propagation] candidate 1/4 G000_T00_1A1C_f430c6_V000\n[propagation] candidate 2/4 G000_T00_1A1C_f430c6_V001\n[propagation] candidate 3/4 G000_T01_1A2C_64e3cb_V001\n[propagation] candidate 4/4 G000_T01_1A2C_64e3cb_V000\n[physics] batch 1/1 start size=1 ids=AREA_MATCHED_STAGGERED..AREA_MATCHED_STAGGERED\n[physics] batch 1/1 complete successful=1 elapsed=0.0s\n{\n  \"recommended_geometry_id\": \"G000_T00_1A1C_f430c6_V000\",\n  \"topology_id\": \"T00_1A1C_f430c6\",\n  \"objectives\": [\n    0.002,\n    0.9999349503803872,\n    1.0,\n    0.0,\n    0.0,\n    0.002,\n    -0.0,\n    1.0\n  ],\n  \"workdir\": \"/mnt/data/v81_validation/legacy_bc_run\",\n  \"openfoam_used\": false,\n  \"gas_phase_cfd_used\": false,\n  \"model_scope\": \"bc_global_preflame_plus_post_onset_condensed_reaction_propagation\"\n}\n",
  "stderr_tail": ""
}
```
