# PHIDL DOE600 검증 결과

## 신규 구현

- 신규 테스트: **58 passed**. `new_tests.log`, `new_tests.xml`.
- 자동 graph/LHS/CAD 보존, 네 objective Pareto/crowding, 단계별 mock 호출 수,
  completed resume 0 calls, 실패 저장/재시도, 고정 selection, baseline 분리,
  handoff 네 가지 재사용 경우, 동일 model-validity rule을 검증했습니다.
- 실제 PHIDL 4 topology × 2 variant → 선택 2 topology × 새 2 variant의
  mini end-to-end workflow도 mock physics로 검증했습니다.
- production 설정으로 `NativeBCHybridEvaluator` 생성 및 grid_size=193 해석을
  확인했습니다. 이 확인의 physics evaluation call은 **0**입니다.
- 실제 600 Pre-flame 또는 21 Post-flame production 작업을 시작하지 않았습니다.

## 기존 suite

기존 테스트 전체: **668 passed, 6 skipped, 8 failed**.
`existing_regression.log`, `existing_regression.xml`에 원본 출력을 보존했습니다.
6 skipped는 M2에 CUDA/A100 하드웨어가 없어 실행할 수 없는 GPU 테스트입니다.

8개 실패는 아래 기존 코드/config 테스트입니다. 이 작업은 해당 코드와 YAML을
수정하지 않았습니다. `protected_source_verification.json`은 관련 source 파일이
Desktop 원본과 바이트 단위로 동일함을 확인한 기록입니다.

1. `test_geometry_safe_bootstrap.py::test_fitted_primitives_reproduce_masks_and_do_not_refit_on_reload`
2. `test_geometry_safe_bootstrap.py::test_all_20_topologies_have_unique_feasible_small_quotas`
3. `test_geometry_safe_bootstrap.py::test_bootstrap_never_pads_invalid_candidates`
4. `test_geometry_safe_bootstrap.py::test_duplicate_masks_do_not_fill_a_topology_quota`
5. `test_geometry_safe_bootstrap.py::test_bootstrap_cache_is_revalidated_and_bound_to_seed_and_grids`
6. `test_m2_post_onset_profile.py::test_production_preflame_sections_are_immutable[path0]`
7. `test_surface_contact_multicomponent_geometry.py::test_components_have_independent_poses_and_surface_contact_area`
8. `test_surface_contact_multicomponent_geometry.py::test_component_mutations_obey_caps_and_change_counts`

Desktop 원본에서도 **동일 8개 테스트가 모두 실패**했습니다.
첫 2건은 `original_eight_failures.log/xml`, 나머지는
`original_no_padding.*`, `original_duplicate_masks.*`,
`original_remaining_contracts.*`에 있습니다. 직렬 runner는 첫 2건 완료 후
중복 실행 방지를 위해 중단했으며, 나머지 6건은 별도 프로세스에서 완료했습니다.
`regression_comparison.json`에 테스트별 원본/작업본 비교를 기록했습니다.
기존 geometry/fitter/physics나 기준 hash를 변경해서 실패를 숨기지 않았습니다.

초기 환경 탐색에서 PATH에 Ninja가 없어 native setup이 실패했던 로그는
`initial_environment_probe.*`입니다. ecsp-m2/bin PATH를 적용한 최종 기존 suite
결과는 위의 **668/6/8**이며, 초기 환경 탐색 결과와 구분합니다.

## 설치된 프로젝트 및 실제 geometry 검증

- Desktop 설치 경로에서 신규 **58 passed** (`installed_tests.log/xml`).
- 검증된 300 geometry를 Desktop run directory에서 재로드하고 config/code/dependency fingerprint 일치를 확인했습니다.
- 유효 topology **100개**, 1A1C/2A2C **50/50**, unique physics masks **300개**.
- 대표 최대 IoU **0.79798545** (제한 0.80).
- 두 grid에서 최대 per-polarity 면적 상대 오차 **0.99063062%** (제한 1%).
- CAD 폭 **0.859116–1.485739 mm**, 초기 폭 대비 최대 조정 **7.573242%** (제한 15%).
- CAD와 두 raster의 모든 구조 검사 및 기존 solver-grid validator 통과.
- 개별 검증 데이터와 PNG/JSON/NPZ는 `runs/doe600_phidl_geometry_validation/`에 있습니다.
