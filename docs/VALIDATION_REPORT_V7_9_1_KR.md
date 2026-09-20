> **Historical v7.9.1 document.** v7.9.2 replaces the connected comb baseline with the hidden-bus vertical 2-anode/2-cathode reference. See `AREA_MATCHED_STAGGERED_BASELINE_V7_9_2_KR.md`.

# ECSP v7.9.1 area-matched staggered 검증 보고서

## 검증 대상

v7.9.0의 surface-contact 다중-component C++ FP64 물리엔진은 변경하지 않고, 다음 기능만 추가했다.

1. 현재 run의 domain·폭·간격·physics grid를 그대로 따르고, 최종 추천형상의 실제 극당 접촉면적 평균을 맞추는 staggered 기준형상 생성
2. NSGA-II 종료 후 기준형상 한 개를 동일 evaluator로 독립 평가
3. AI 추천형상과 기준형상의 네 목적함수 비교 출력
4. 기존 완료 run에 기준형상만 추가 계산하는 backfill 도구

## 자동시험

```text
python compileall: 통과
pytest: 43 passed
C++17 release build: 통과
C++ solver version: ecsp_cpp_solver 7.9.0 (물리엔진 변경 없음)
```

## 형상 제약 검증

### 20×20 mm / design 96 / physics 193

공통 실험조건:

```text
극당 접촉면적 목표 20%
전극 폭 1–5 mm
최소 양·음극 간격 2 mm
anode 1 component + cathode 1 component
```

생성 결과:

```text
finger pairs: 3
bus width: 2.5263 mm
finger width: 1.2632 mm
finger extension: 9.2632 mm
design-grid 극당 면적: 0.2005208
physics-grid 극당 면적: 0.2000591
design-grid 최소간격: 2.3158 mm
physics-grid 최소간격: 2.1875 mm
component count: 1 + 1
overlap/direct contact: 0
```

### 25×25 mm / design 120 / physics 241

동일 상대면적·폭·간격 조건에서:

```text
finger pairs: 3
bus width: 1.6807 mm
finger width: 1.6807 mm
finger extension: 17.4370 mm
design-grid 극당 면적: 0.2005556
physics-grid 극당 면적: 0.1999966
design-grid 최소간격: 2.3109 mm
physics-grid 최소간격: 2.1875 mm
component count: 1 + 1
overlap/direct contact: 0
```

두 경우 모두 design grid와 실제 physics-grid resize 이후에 면적 허용오차, 최소간격, 1+1 connected-component 조건을 통과했다.

## C++ physics 연결 검증

20 mm/193과 25 mm/241 baseline을 각각 실제 C++ CPU FP64 backend로 one-step 계산했다.

```text
20 mm: rejection 0, converged true, final relative residual 9.94e-10
25 mm: rejection 0, converged true, final relative residual 9.29e-10
```

one-step 시험은 배선·BV·전위·형상 resize 검증용이며, 2초 점화성능을 검증한 시험은 아니다.

## NSGA-II 비간섭 검증

analytic-debug 20개 × 1세대 전체 workflow를 실행해 다음을 확인했다.

```text
AI recommendation을 먼저 확정
baseline geometry_id = AREA_MATCHED_STAGGERED
baseline source role = post_optimization_reference_only
baseline included_in_nsga2_population = false
baseline included_in_final_pareto_front = false
비교 CSV/JSON/PNG 생성
```

생성 확인 파일:

```text
final/area_matched_staggered.json/.npz/.png
final/area_matched_staggered_metrics.csv
final/recommended_vs_area_matched_staggered.csv
final/recommended_vs_area_matched_staggered.json
final/recommended_vs_area_matched_staggered.png
```

## 기존 run backfill 검증

`python/evaluate_area_matched_staggered.py`를 모의 완료 run에 적용해 NSGA-II 세대를 재계산하지 않고 baseline 및 비교 산출물이 생성되는 것을 확인했다.

## 해석 제한

- area-matched baseline은 사용자의 실제 실험 staggered CAD를 복원한 것이 아니라, 좌우 bus와 교번 finger로 구성한 고정 reference topology다.
- 기본 finger pair는 3개다. 실제 비교 프로토콜에서 finger 수를 고정해야 하면 YAML의 `finger_pairs`를 명시적으로 설정해야 한다.
- baseline은 현재 run과 동일한 응축상 C++ FP64 모델을 사용한다. 따라서 gas flame, OpenFOAM, 기상 열피드백은 포함하지 않는다.
- 절대 성능값은 LP/PVA/GLY/BA 물성 및 kinetics의 실험 보정 전에는 `nominal_unvalidated`다.
- 이 배포환경에는 Apple M2 장치가 없어 v7.9.1 baseline의 M2 2초 full-horizon 실행은 수행하지 않았다. 다만 C++ 물리엔진은 v7.9.0에서 변경되지 않았고, 새 baseline은 동일 executable에 표준 contact mask를 전달한다.

## 실제 추천면적 override 검증

단위시험에서 설정 목표와 다른 추천형상 면적 override를 전달해 baseline이 override 값을 기준으로 재보정되는 경로를 검증한다. 자동 workflow와 기존 run backfill 모두 추천형상을 먼저 읽은 뒤 baseline을 생성한다.
