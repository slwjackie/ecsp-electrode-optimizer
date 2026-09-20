# v7.9.5 기존 기능 보존 감사

이 릴리스는 원본 v7.9.5를 기준 commit으로 고정한 뒤 B/C 경로를 **opt-in**으로 추가했다. 전체 파일 hash 비교 결과와 회귀시험을 함께 사용하여 기존 기능의 삭제·대체 여부를 확인했다.

## 1. 파일 보존 결과

- v7.9.5 기준 tracked file: **154개**
- 삭제된 기존 파일: **0개**
- byte-identical 기존 파일: **147개**
- 하위호환 확장을 위해 수정한 기존 파일: **7개**
- 상세 SHA-256 감사: `docs/V795_BASELINE_FILE_HASH_AUDIT.json`

수정한 기존 파일은 다음뿐이다.

| 파일 | 수정 이유 | 하위호환 방식 |
|---|---|---|
| `README_KR.md` | v8 실행·범위 문서 추가 | 기존 v7.9.5 문서를 아래에 그대로 유지 |
| `VERSION.json` | 새 release metadata 추가 | 기존 schema·backend metadata 유지 |
| `python/ecsp_nsga2/evaluator.py` | `bc_global_preflame` evaluator 추가 | 기존 backend factory branch 유지, 새 backend만 opt-in |
| `python/ecsp_nsga2/workflow.py` | optional propagation/final Pareto/staggered 동일 경로 추가 | `propagation_refinement.enabled=false`이면 기존 경로 유지 |
| `python/ecsp_v6/composition.py` | cured-water final-specimen basis 추가 | 기존 `retained_water_fraction` 입력도 계속 지원 |
| `python/ecsp_v6/physics/composition_model.py` | LP/PVA repeat inventory 필드 추가 | dataclass 기본값으로 기존 constructor 호환 |
| `python/run_nsga2_electrical_solid_loop.py` | 새 scope·CLI 설명 추가 | 기존 CLI와 config 선택 방식 유지 |

## 2. byte-identical 핵심 backend

- `cpp/ecsp_cpp_solver.cpp`는 reference copy와 byte-for-byte 동일하다.
- SHA-256: `453168254960d20d93e0e52c4eac4f0638c391297baaa544691980acc0cb0e94`
- 기존 C++ CPU, Torch/CUDA, hybrid CUDA+CPU 설정·실행 스크립트는 삭제하지 않았다.
- B/C production profile은 별도 `config/nsga2_bc_global_preflame_propagation_a100.yaml`로만 활성화된다.

## 3. 회귀검증 결과

| 게이트 | 결과 |
|---|---:|
| 전체 Python test suite | **56/56 통과** |
| 기존 surface-contact component pairs | **6/6 통과** |
| legacy C++ preflight | **4/4 성공, rejection 0** |
| short-horizon Python FP64 ↔ C++ parity | 통과 |
| legacy/hybrid C++ warning build | 통과 |
| 전체 shell syntax | 통과 |
| B/C 명세·실행 감사 | **24/24 통과** |

## 4. 보존 판정

기존 v7.9.5 파일은 삭제되지 않았고, 핵심 기존 solver는 그대로 보존됐다. 새 기능은 새 config/backend flag로만 진입하므로 **기존 v7.9.5 profile은 계속 실행 가능**하다. 실제 A100 장치에서의 production 성능은 패키징 환경에 GPU가 없어 별도 preflight가 필요하지만, 기존 CPU FP64 경로와 B/C CPU debug 경로는 실행 검증했다.
