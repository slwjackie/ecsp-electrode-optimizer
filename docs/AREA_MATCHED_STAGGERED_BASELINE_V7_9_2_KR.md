# v7.9.2 hidden-bus 2+2 area-matched staggered baseline

## 형상 정의

후처리 기준형상은 다음 top-view 접촉배치를 고정해서 사용한다.

```text
hidden anode bus: propellant contact 없음
           │                 │
           A        C        A        C
           │        │        │        │
           │        │        │        │
           │        │        │        │
                    │                 │
hidden cathode bus: propellant contact 없음
```

- 양극 접촉 팔: 정확히 2개, 위쪽 경계에서 아래로 연장
- 음극 접촉 팔: 정확히 2개, 아래쪽 경계에서 위로 연장
- 좌우 순서: `anode – cathode – anode – cathode`
- 네 팔은 동일한 폭을 사용
- 두 양극 팔은 서로 같은 길이
- 두 음극 팔은 서로 같은 길이
- 같은 극성의 팔은 surface-contact mask에서는 서로 끊어져 있음
- 같은 극성 팔의 전기적 연결은 추진제 접촉면 밖의 이상적인 hidden bus로 처리
- hidden bus는 mask에 포함하지 않으므로 접촉면적과 BV 발열에 기여하지 않음

동일한 극당 목표면적을 사용하므로 현재 구현에서는 네 팔의 폭과 길이가 모두 동일하다.

## area matching과 깊은 stagger

추천 AI 형상의 실제 극당 접촉면적 평균을 우선적으로 맞춘다. 해당 값이 없을 때만 YAML의 `target_area_fraction_per_polarity`를 사용한다.

정수 raster search가 다음을 동시에 만족하는 폭, 길이, 수평 간격을 선택한다.

- design grid 및 physics grid에서 극당 접촉면적 허용오차 이내
- 최소 양·음극 간격 이상
- 동일 극성 두 팔의 길이·폭 동일
- 네 팔의 resize 후 길이·폭 동일
- design/physics grid의 수직 중첩비가 최소 0.45 이상
- 목표 수직 중첩비 약 0.50
- domain 좌우 margin 유지

수직 중첩비는 위에서 내려오는 양극 팔과 아래에서 올라오는 음극 팔의 y-방향 공통 구간을 domain 높이로 나눈 값이다.

## component 의미

visible surface-contact component는 다음과 같다.

```text
Na_visible = 2
Nc_visible = 2
Ntotal_visible = 4
```

그러나 hidden bus를 포함한 전기 terminal은 극당 하나다.

```text
Na_terminal = 1
Nc_terminal = 1
```

따라서 AI search가 `1+1 component`로 제한된 run이라도 이 post-optimization reference는 명시적 baseline override를 통해 계산할 수 있다. 이 override는 `baseline_type=area_matched_staggered` 및 전용 내부 플래그가 모두 존재할 때만 허용되며, NSGA-II 후보에는 적용되지 않는다.

## 설정

```yaml
baselines:
  area_matched_staggered:
    enabled: true
    layout: hidden_bus_vertical_2anode_2cathode
    fingers_per_polarity: 2
    target_interdigitation_overlap_fraction: 0.50
    minimum_interdigitation_overlap_fraction: 0.45
    hidden_bus_in_contact_mask: false
    maximum_gap_safety_pixels: 8
    include_in_nsga2_population: false
    include_in_pareto_selection: false
```

`fingers_per_polarity`는 v7.9.2에서 반드시 2여야 한다. 현재 폭·면적 조건으로 이 정확한 2+2 형상을 만들 수 없을 때, 연결형 comb나 다른 finger 수로 조용히 대체하지 않고 `BaselineGeometryError`를 기록한다.

## NSGA-II와의 분리

baseline은 initial population, mating, mutation, crossover, Pareto front 및 AI 추천 선택에 포함되지 않는다. AI 추천을 먼저 동결한 다음 동일 C++ FP64 solver로 한 번 평가한다.

기존 완료 run에는 다음 명령으로 추가할 수 있다.

```bash
bash tools/evaluate_area_matched_staggered_existing_run.sh \
  "/기존/run/절대경로" \
  "/실행에/사용한/YAML/절대경로"
```

20%/극, 폭 1–5 mm, 간격 2 mm run에서는 YAML을 명시하는 편이 가장 안전하다.
