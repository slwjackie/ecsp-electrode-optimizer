> **Historical v7.9.1 document.** v7.9.2 replaces the connected comb baseline with the hidden-bus vertical 2-anode/2-cathode reference. See `AREA_MATCHED_STAGGERED_BASELINE_V7_9_2_KR.md`.

# v7.9.1 Area-matched staggered baseline

## 목적

AI 추천형상을 기존 staggered 계열과 공정하게 비교하기 위한 **후처리 reference**다. 실험 staggered의 원래 면적을 그대로 쓰는 것이 아니라, 현재 NSGA-II run의 조건에 맞춰 접촉면적을 다시 맞춘다.

## 공정성 조건

baseline은 실행 시점의 `geometry`와 `evaluator` 설정을 그대로 사용하며, 면적은 최종 추천 AI 형상의 실제 physics-grid 접촉면적을 우선적으로 맞춘다.

- 동일 추진제 domain: 20×20 mm 또는 25×25 mm
- 추천 AI 형상의 실제 physics-grid 양·음극 접촉면적 평균을 기준으로 면적 일치; 해당 값이 없을 때만 설정 목표 사용
- 동일 최소/최대 전극 폭
- 동일 최소 양·음극 간격
- 동일 design/physics grid
- 동일 전압, 시간간격, 2 s horizon
- 동일 C++ CPU FP64 condensed-phase solver
- 전극 mask는 추진제를 제거하지 않는 surface-contact overlay

## 형상 정의

- 왼쪽 anode bus 1개, 오른쪽 cathode bus 1개
- 기본적으로 polarity당 3개의 수평 finger
- anode/cathode finger가 y 방향으로 교대로 배치되는 staggered 구조
- polarity당 connected component는 정확히 1개
- bus 폭, finger 폭 및 finger 길이는 geometry constraint만으로 결정
- objective 값은 geometry 생성에 사용하지 않음

기본 3쌍이 현재 domain·면적·폭·간격 조건으로 불가능할 때만 `allow_finger_pair_fallback: true` 설정에 따라 가까운 finger 수를 사용한다.

## NSGA-II와의 분리

baseline은 다음 단계에 들어가지 않는다.

```text
initial population
mating pool
crossover / mutation
environmental selection
Pareto front
utopia-distance recommendation
```

최종 AI 추천형상이 정해진 뒤 동일 evaluator로 한 번 계산한다. 따라서 baseline이 추천형상 선정에 영향을 줄 수 없다.

## 설정

```yaml
baselines:
  area_matched_staggered:
    enabled: true
    finger_pairs: 3
    allow_finger_pair_fallback: true
    maximum_gap_safety_pixels: 8
    include_in_nsga2_population: false
    include_in_pareto_selection: false
```

## 기존 완료 run에 적용

```bash
bash tools/evaluate_area_matched_staggered_existing_run.sh \
  "$PWD/runs/d20_single_1a1c"
```

명시적으로 YAML을 지정하려면 두 번째 인수로 전달한다.

```bash
bash tools/evaluate_area_matched_staggered_existing_run.sh \
  "$PWD/runs/d20_single_1a1c" \
  "$PWD/experiments/contact20pct_width1_gap2/config/d20_single_1a1c.yaml"
```

기본값은 기존 run의 `effective_config.json`이다. 이 방식은 NSGA-II 세대를 다시 계산하지 않고 staggered 한 형상만 추가 계산한다.

## 출력

```text
<workdir>/final/
├── area_matched_staggered.png
├── area_matched_staggered.npz
├── area_matched_staggered.json
├── area_matched_staggered_metrics.csv
├── recommended_vs_area_matched_staggered.csv
├── recommended_vs_area_matched_staggered.json
└── recommended_vs_area_matched_staggered.png
```

비교 JSON의 `ai_improvement_percent_relative_to_staggered`는 모든 목적함수가 낮을수록 좋다는 기준으로 계산한다.

\[
\mathrm{improvement}(\%)=100\frac{f_{\mathrm{staggered}}-f_{\mathrm{AI}}}{|f_{\mathrm{staggered}}|}
\]

양수면 AI 추천형상이 더 좋고, 음수면 area-matched staggered가 더 좋다.

## 해석 제한

이 비교는 동일 condensed-phase 모델 내부의 수치 비교다. 기상 화염, OpenFOAM, 실제 전극 두께 및 실험 오차를 포함하지 않으므로 최종 실험 검증을 대체하지 않는다.

## 면적 일치 우선순위

```text
1. recommended metrics의 anodeContactAreaFraction/cathodeContactAreaFraction 평균
2. recommended geometry descriptor의 두 극 면적 평균
3. geometry.target_area_fraction_per_polarity
```

따라서 rasterization 허용오차 내에서 AI 형상이 목표 20%와 조금 다르게 생성됐더라도 baseline은 단순히 20% 목표가 아니라 **실제 추천형상의 평균 극당 접촉면적**에 맞춰진다.
