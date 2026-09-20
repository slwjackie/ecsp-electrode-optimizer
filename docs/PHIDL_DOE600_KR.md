# PHIDL DOE600 — ECSP v8.4.2

## 구현 범위

독립 실행 파일은 `tools/run_phidl_doe600.py`입니다. 기존 NSGA-II 실행 파일,
bootstrap, mutation, crossover, geometry fitter, electrical/B/C/Post-onset solver와
physics YAML은 수정하지 않습니다. 기존 파일 중 변경한 것은
`tools/resume_final_only.py`의 handoff 재사용 wrapper뿐입니다.

기본 호출은 **Stage 1 형상만 생성**합니다. 실제 Pre-flame 계산에는
`--execute-physics`, Post-flame 계산에는 추가로 `--postflame`이 필요합니다.

## 구조

| 파일 | 역할 |
|---|---|
| `python/ecsp_doe/topology.py` | lattice/comb graph grammar, polarity 구분 WL signature, representative IoU 검사 |
| `python/ecsp_doe/geometry.py` | separated/interdigitated PHIDL 배치, CAD·두 raster grid 검증 |
| `python/ecsp_doe/sampling.py` | deterministic Latin Hypercube Sampling, 중복 및 재시도 제한 |
| `python/ecsp_doe/selection.py` | 4-objective Pareto rank, crowding, utopia tie-break |
| `python/ecsp_doe/storage.py` | atomic JSON/CSV/NPZ/PNG, hash, contact sheet, run lock |
| `python/ecsp_doe/workflow.py` | Stage 1 → Stage 2 → 고정 final20, candidate 단위 resume |
| `python/ecsp_doe/physics.py` | 기존 production evaluator·baseline·refinement 연결 |
| `config/phidl_doe600.yaml` | DOE 개수, component 비율, IoU 및 시도 제한 |

### 100개 topology 자동 생성

기본 설정 `hybrid_interdigitated`는 100개 중 50개를 기존 lattice 계열,
50개를 interdigitated comb 계열로 자동 생성합니다. 각 계열 안에서도
1A1C/2A2C가 25개씩이므로 전체 비율은 1A1C 50개 / 2A2C 50개입니다.

lattice 계열은 seeded spanning-tree 확장/회전/분기와 추가 edge의 loop/parallel
branch를 사용합니다. comb 계열은 spine과 네 구간으로 나뉜 finger graph를 만들고,
양·음극을 같은 active window 안에서 A/C/A/C 순으로 배치합니다. 두 polarity의
finger가 서로 안쪽으로 들어오며, orientation, 시작 polarity, row spacing,
finger reach와 LINE/ARC/SPLINE primitive sequence를 seed에서 결정합니다.
T000~T099 전용 함수는 없습니다.

signature는 polarity와 primitive 유형을 보존하는 edge-subdivision graph의
NetworkX WL hash 및 component/endpoint/branch/loop 수로 구성합니다. 좌표,
parameter vector, topology 이름 및 node 번호는 topology identity에 포함하지 않습니다.
WL hash 중복은 거부하고, **실제로 검증된 대표 raster**끼리 polarity별 IoU를
계산해 기본 최대 0.80을 적용합니다. interdigitated 후보는 finger 투영 중첩률과
row polarity 교대율의 작은 값을 `opposed_active_edge_fraction`으로 기록하며,
기본값 0.45 이상이어야 합니다.

생성 가능한 topology 공간은 이 planar lattice grammar입니다. 임의의 모든
electrode graph를 포괄한다는 의미는 아닙니다. 불가능한 제약이나 diversity
quota는 진단을 남기고 종료하며, 중복 topology로 개수를 채우지 않습니다.

### 면적과 구조 보존

`GeometryLimits(**experiment['geometry'])`에서 domain, 면적 목표/오차,
width, gap, margin, component 제한을 읽습니다. physics grid는 기존
`bootstrap.resolved_physics_grid()`로 experiment/base config에서 해석합니다.
별도 same-polarity gap을 지정하지 않으면 기존 `minimum_gap_mm`를 사용합니다.

초기 중심선 길이와 `target area / length` 관계를 사용해 skeleton envelope를
먼저 정합니다. 이후 **중심선 좌표를 고정**하고 width만 기본 ±15% 이내에서
이분 탐색합니다. width 조정 허용률은 20%를 넘길 수 없으며 실제 width도
GeometryLimits 범위를 지켜야 합니다. global length fit, slot shrink,
legacy area fitter, pixel repair는 사용하지 않습니다.

- CAD: per-polarity area, component·hole 수, centerline segment 길이, width, margin, 양극 간 separation,
  비인접 edge/같은 polarity component의 self-contact·gap을 검사합니다.
- Raster: design grid와 physics grid를 CAD에서 각각 직접 생성하고,
  overlap, 면적 오차, margin, component count, 양극·동극 gap을 검사합니다.
- Topology: 두 grid에서 skeleton의 endpoint·branch cluster 수 및 hole/Euler
  loop 수를 원래 graph와 **정확히 비교**합니다. loop filling, branch 소실/접촉은 거부합니다.
- 기존 solver-grid validator도 마지막으로 호출합니다. geometry generator나
  fitter를 호출하지 않는 read-only 검증입니다.

비인접 CAD edge 간격에서 두 grid 중 큰 pixel 폭의 두 배를 뺀 보수적 하한도
동극 gap 조건에 적용합니다. 각 거부는 즉시 `generation_rejections.csv`에
reason/parameters/validation details로 기록됩니다.

`config/phidl_doe600.yaml`의 관련 기본값은 다음과 같습니다.

```yaml
topology_style: hybrid_interdigitated
interdigitated_fraction: 0.5
interdigitated_reach_fraction: 0.78
interdigitated_curve_scale: 0.20
require_opposed_active_edges: true
minimum_opposed_edge_fraction: 0.45
```

`topology_style`은 `separated`, `interdigitated`, `hybrid_interdigitated` 중 하나이며,
hybrid에서만 `interdigitated_fraction`이 quota를 정합니다.

### Sampling 및 선택

Stage 1은 100×3개입니다. 각 topology에서 3-point LHS proposal batch를
사용하며, DRC 거부는 새 deterministic LHS batch로 보충합니다. Stage 2도
같은 방식으로 10-point batch를 사용합니다. 거부 필터 후의 accepted subset은
완벽한 Latin stratification을 유지한다고 보장하지 않습니다. 모든 proposal
batch는 실제 `scipy.stats.qmc.LatinHypercube`를 사용합니다.

Stage 1의 실제 성공 결과를 `pareto_rank ↑`, `crowding_distance ↓`,
`normalized_utopia_distance ↑`, `geometry_id ↑`로 정렬하고 처음 등장하는
서로 다른 topology 30개를 택합니다. 이들에서 새로운 10개씩 생성합니다.
Stage 1 parameter vector와 동일한 vector 또는 동일 physics raster는 거부합니다.

합계 600개의 실제 결과를 다시 Pareto 정렬해 앞 front부터 final20을 채우고,
마지막 부분 front는 crowding → utopia → ID 순으로 결정합니다. 순위는 1부터
시작합니다. Utopia 정규화는 해당 selection pool 전체의 min/max 기준입니다.
Weighted sum, surrogate 또는 predicted score는 사용하지 않습니다.

수치적으로 유효한 non-ignition은 기존 production의 finite penalty를 그대로
사용합니다. 수치 실패/invalid voltage search는 결과를 저장하지만 selection을
진행하지 않습니다. 실패를 수정한 뒤 `--retry-failed`로 재개할 수 있습니다.

### Staggered 및 Post-flame

final20을 고정한 뒤 기존 `generate_area_matched_staggered()`로 같은 target
면적의 external reference를 만들고, 동일 production evaluator로 평가합니다.
baseline은 600개 pool과 Pareto selection에 포함되지 않습니다.
`final/preflame_top20_vs_staggered.csv`에는 네 objective의 절대 차이와
`100 × (staggered − candidate) / abs(staggered)` 개선율을 기록합니다.
기준이 0이면 개선율은 비워 둡니다.

`ProductionPhysicsAdapter`는 기존 `create_evaluator()`,
`_run_propagation_refinement()`, `_write_propagation_comparison()`,
`_load_persisted_baseline_raster()`를 재사용합니다. NSGA-II workflow를
생성하거나 실행하지 않습니다. 전체 600개 목록은 refinement 함수에 넘기지
않으며, 저장된 final20 ID와 정확히 일치하는 목록 + baseline만 전달합니다.

`configured_model_temperature_range_exceeded`는 기존 model-validity rejection으로
저장됩니다. candidate와 staggered 모두 같은 2500 K 계약을 적용합니다.
실패한 design도 `preflame_top20.csv`에는 그대로 남고, post-onset valid ranking에서만
빠집니다. 21번째 후보를 승격하지 않습니다.

기존 `resume_final_only.py --reuse-handoff`는 patch 전에 original 단일/batch
method를 저장합니다. candidate의 persisted handoff가 없으면 fail closed합니다.
명시적 baseline role 또는 정확한 baseline output path인 경우에만, 두 handoff
파일이 모두 없을 때 original evaluator로 한 번 새 계산합니다. 부분 파일이나
손상된 파일은 새 계산을 허용하는 근거가 되지 않습니다. 결과에는
`reusedPersistedHandoff=True/False`를 기록합니다. error 문자열로 role을 추론하지 않습니다.
새 DOE 핵심 workflow는 이 monkey patch에 의존하지 않습니다.

### 재개와 파일

geometry별 NPZ에는 design/physics mask가 모두 들어 있고, JSON에는 topology
spec/signature, parameter vector, seed, CAD polygons와 holes, electrode area,
검증 결과 및 geometry/physics-mask hash가 들어 있습니다.

각 실제 평가는 `stageN/physics/<geometry_id>/evaluation.json`에 atomic 저장됩니다.
성공 완료는 재계산하지 않고, 중단된 running record는 재시도합니다. 실패 완료는
`--retry-failed`가 있어야 재시도합니다. config, base-config 내용, code 및 주요
dependency version의 fingerprint가 달라지면 resume를 거부합니다.

Post-onset은 candidate별 `result_checkpoint.json`을 사용합니다. 성공과
model-validity rejection 모두 완료 상태로 재사용하며, JSON에 NaN을 쓰지 않고
별도 sidecar로 원래 nonfinite diagnostic을 복구합니다. selection CSV/최종 geometry
artifact는 immutable이며 내용이 다른 덮어쓰기를 거부합니다.

## Dependency

검증 환경: production `ecsp-m2`, Python 3.10.21, NumPy 2.2.6, SciPy 1.15.3.
[PHIDL 1.7.2](https://pypi.org/project/phidl/1.7.2/)는 Python ≥3.6을 지원하며,
NumPy 2 compatibility 수정이 포함된 버전입니다.

`python/requirements-phidl-doe.txt`는 PHIDL 1.7.2, gdspy 1.6.13,
Shapely 2.0.7, NetworkX 3.2.1, scikit-image 0.24.0, matplotlib 3.10.8을 pin합니다.
현재 프로젝트에는 검증한 CAD dependency를 `.doe-deps/`에 격리합니다.
새 DOE entry point만 이 폴더를 우선 로드하므로 기존 NSGA-II 환경은 바뀌지 않습니다.
다른 환경에서는 별도 venv에 위 requirements를 설치할 수 있습니다.

## 실행 명령

아래 명령은 문서일 뿐, 구현 작업 중 production physics를 시작하지 않았습니다.

최종 hybrid geometry-only 검증 결과와 정확한 재개 명령은
`docs/validation_phidl_doe600/HYBRID_INTERDIGITATED_VALIDATION_KR.md`에 있습니다.

```bash
cd /Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe
export PATH="/Users/kimjiin/anaconda3/envs/ecsp-m2/bin:$PATH"

# 형상 생성만: 기본 동작
python tools/run_phidl_doe600.py --run-dir runs/doe600_phidl_study

# 위 형상 생성부터 이어 실제600 Pre-flame + external baseline,
# final20+baseline만 Post-flame
python tools/run_phidl_doe600.py --run-dir runs/doe600_phidl_study \
  --resume --execute-physics --postflame

# 새 production run을 한 번에 시작할 때
python tools/run_phidl_doe600.py --run-dir runs/doe600_phidl_production \
  --execute-physics --postflame

# 중단된 production run 재개
python tools/run_phidl_doe600.py --run-dir runs/doe600_phidl_production \
  --resume --execute-physics --postflame

# 수치 실패가 완료 상태로 저장된 candidate를 명시적으로 재시도
python tools/run_phidl_doe600.py --run-dir runs/doe600_phidl_production \
  --resume --execute-physics --postflame --retry-failed
```

다른 experiment를 쓰려면 최초 실행과 resume에 동일한 `--config` 및
`--doe-config`를 전달하십시오. backend, voltage, time step, temperature cap 등을
바꾸어 기존 결과를 같은 run directory에 혼합하지 않습니다.

## 검증

결과와 기존 원본 비교는 `docs/validation_phidl_doe600/`에 저장합니다.
기존 suite에는 이 작업 전부터 존재한 실패가 있으므로, 신규 테스트 결과와
기존 테스트의 원본 재현 결과를 구분합니다. GPU 전용 테스트는 M2에서 skip됩니다.

```bash
PYTHONPATH="$PWD/.doe-deps:$PWD/python" python -m pytest -q \
  python/tests/test_phidl_doe_geometry.py \
  python/tests/test_phidl_doe_workflow.py \
  python/tests/test_phidl_doe_physics.py \
  python/tests/test_phidl_handoff_reuse.py
```
