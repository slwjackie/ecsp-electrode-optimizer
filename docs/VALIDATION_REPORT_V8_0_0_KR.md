# ECSP v8.0.0 B/C global pipeline 검증 보고서

## 1. 검증 요약

| 항목 | 결과 |
|---|---:|
| Python test suite | **56/56 통과** |
| Python byte-code compile | **55개 파일 통과** |
| 전체 shell script `bash -n` | 통과 |
| legacy C++ warning build | 통과 |
| hybrid C++ warning build | 통과 |
| B/C 명세 자동감사 | **24/24 통과** |
| B/C CPU FP64 end-to-end debug | 통과 |
| pre-flame Pareto 저장 | 통과 |
| near-Pareto/diverse propagation selection | 통과 |
| condensed propagation full fields | 통과 |
| final 8-objective Pareto/recommendation | 통과 |
| area-matched staggered 동일 B/C+propagation | 통과 |
| legacy surface-contact component pairs | **6/6 통과** |
| legacy C++ preflight | **4/4 성공, rejection 0** |
| legacy C++ source byte identity | 통과 |
| 실제 A100 B/C run | **미실행: 패키징 환경에 NVIDIA GPU 없음** |
| 실험보정/물리 validation | **미수행** |

## 2. 자동 명세 감사

실행:

```bash
bash tools/validate_bc_global_pipeline.sh
```

결과:

```text
status: passed
checks: 24/24
```

감사 범위:

- 식 (2),(4)–(9),(11),(13),(15)–(17),(22)–(24) 사용 contract
- 식 (3),(18),(25)–(31) pre-flame 제외
- 식 (32) diagnostic-only 및 heat-source 중복 방지
- exact non-metallized wet recipe와 cured-water basis
- global reaction, two-channel kinetics, weights
- onset T AND Xg AND area criterion
- Urem, Vmin, J99/Jbar 목적함수
- proxy closure 비활성화
- 1,000 → Pareto+15% diversity → propagation → final Pareto
- T/Xg/qJ/qechem full-field handoff
- staggered 동일 pipeline
- legacy v7.9.5 파일 존재와 backend 비파괴성

생성 보고서:

- `docs/generated_bc_audit/bc_global_pipeline_audit.json`
- `docs/generated_bc_audit/bc_global_pipeline_audit.md`

## 3. CPU FP64 end-to-end debug 재현

`config/nsga2_bc_global_preflame_propagation_debug.yaml`의 작은 deterministic case로 다음 전체 경로를 실행했다.

```text
geometry generation
→ B/C batch physics
→ pre-flame four-objective Pareto
→ Pareto + diversity selection
→ full-field handoff
→ condensed propagation/level set
→ final eight-objective Pareto
→ area-matched staggered B/C
→ staggered propagation
→ AI-vs-staggered comparison
```

필수 산출물 생성 확인:

```text
final/preflame_final_pareto_designs.csv
final/propagation_selection.csv
final/all_propagation_refined_designs.csv
final/final_pareto_designs.csv
final/recommended_design.json
final/area_matched_staggered.json
final/area_matched_staggered_propagation.json
final/recommended_vs_area_matched_staggered_propagation.json
RUN_COMPLETE.json
```

각 propagation candidate는 `bc_handoff_fields.npz`와 `propagation_fields.npz`에 전체 2-D fields를 저장한다.

## 4. v7.9.5 회귀검증

### Python

기존 52개 시험에 B/C 시험 4개를 추가했고 전체 56개가 통과했다. 새 backend는 별도 factory branch/config로만 활성화되며 legacy evaluator 선택은 변경하지 않는다.

### C++/surface-contact

- `cpp/ecsp_cpp_solver.cpp` SHA-256:
  `453168254960d20d93e0e52c4eac4f0638c391297baaa544691980acc0cb0e94`
- reference copy와 byte-for-byte 동일
- required component pairs `(1,1),(1,2),(2,1),(2,2),(3,1),(1,3)` 모두 physics rejection 0
- legacy C++ preflight 4/4 성공
- short-horizon Python FP64 vs C++ parity gate 통과

## 5. 검증하지 못한 항목

패키징 환경에 A100이 없어 다음은 실제 장치에서 검증하지 못했다.

- CUDA FP64 B/C kernel execution
- A100 batch size 4 이상의 sustained utilization
- 1,000×5 production wall time
- propagation CPU parallelism 8/40의 실제 node scaling
- full 2 s에서 literature-nominal parameter의 수치 민감도

A100에서 먼저 실행:

```bash
bash tools/run_bc_global_a100_preflight.sh \
  "$PWD/runs/bc_global_a100_preflight"
```

그 다음 production:

```bash
bash tools/run_bc_global_a100.sh \
  "$PWD/runs/bc_global_a100" 1000 5
```

## 6. 물리적 검증 상태

아래는 **코드 실행값은 존재하지만 아직 해당 LP/PVA/H3BO3 cured formulation에서 측정되지 않은 값**이다.

- cured water fraction
- Li+/ClO4-/water diffusivity
- water/LP interface BV coefficients와 reaction enthalpy
- density, cp(T), k(T), h, emissivity
- Fig. 8 digitised conversion-dependent kinetics의 정확한 원자료

따라서 본 보고서가 입증하는 것은 **명세 구현·수치 배관·회귀 무결성**이다. 실제 전류, 온도, onset 및 형상 순위의 정량 정확도는 향후 parameter calibration과 독립 geometry validation으로 입증해야 한다.
