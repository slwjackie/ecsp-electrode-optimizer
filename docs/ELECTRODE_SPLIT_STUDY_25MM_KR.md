# 25×25 mm 분할 교대 전극 비교

첨부 표의 **좌우 간격 2 mm, 중앙 분할 간격 1 mm, 조각 길이 12 mm,
총 접촉면적 192 mm²** 조건을 독립 study로 실행한다. 폭은 양극·음극 모두 4 mm다.
앞서 논의한 좌우 3 mm·중앙 2 mm 조건은 이 profile에 포함하지 않는다.

기존 `run_electrode_parameter_study.py`에 새 `--study-config`를 지정한다.
기본 config의 직선 전극 8개 조건, 148개 catalogue, 공통 물리식과 계수는 변경하지 않는다.
Staggered 비교는 실행하지 않는다.

## 형상과 전원 연결

| ID | 좌우 간격 | 중앙 간격 | 조각 길이 | 극성별 조각 수 | 양극/음극 면적 | 총 접촉면적 |
|---|---:|---:|---:|---:|---:|---:|
| `WD_W4` | 2 mm | 없음 | 25 mm | 1 / 1 | 100 / 100 mm² | 200 mm² |
| `SPLIT_W4_G2_S1` | 2 mm | 1 mm | 12 mm | 2 / 2 | 96 / 96 mm² | 192 mm² |

SVG와 배열의 첫 행을 위쪽으로 보았을 때의 극성은 다음과 같다.

| 위치 | 왼쪽 | 오른쪽 |
|---|---|---|
| 위 | 양극 | 음극 |
| 아래 | 음극 | 양극 |

전극 x 구간은 `[7.5, 11.5]`, `[13.5, 17.5]` mm이고, 분할형의 y 구간은
`[0, 12]`, `[13, 25]` mm다. 최소 양·음극 edge 간격은 **1 mm**다.
동일 극성 두 조각은 각각 같은 외부 전원 단자에 연결된다고 가정한다
(`implicit_3d_hidden_bus`). 외부 배선은 추진제 접촉 mask에 넣지 않는다.
배선 저항이나 floating electrode를 계산하는 모델은 아니다.

전극 아래에도 추진제가 남는 surface overlay이며, 전체 추진제 면적은 625 mm²다.
분할로 전극 접촉면적이 8 mm²(4%) 줄어든 상태 그대로 비교한다.
면적을 보상하거나 형상을 자동 확대하지 않는다. 따라서 이 비교에는 분할/극성 배치와
접촉면적 감소 효과가 함께 포함된다.

기본 격자는 N=200, dx=25/200=0.125 mm이다. 분할형은 극성별 6144셀이다.
격자 변경은 새 study config와 새 output으로 실행한다. N은 100 이상인 50의 배수만
허용하며 모든 설계 edge가 cell face에 놓이는지 검사한다. 이 조건은 격자수렴 검증을
대신하지 않는다.

## 실행

기존 PyTorch/C++/CUDA 환경에서 저장소 루트를 기준으로 실행한다.

```bash
RUN=runs/split25_w4_g2_s1
python python/run_electrode_parameter_study.py generate --study-config config/electrode_split_study_25mm.yaml --out "$RUN"
python python/run_electrode_parameter_study.py audit --out "$RUN"
python python/run_electrode_parameter_study.py evaluate --out "$RUN" --device cuda --batch-size 2
```

`generate`와 `audit`는 PDE를 풀지 않는다. `evaluate`는 저장된 `study.json`에서 두 조건을
읽으므로 `--study-config`를 다시 지정할 필요가 없다. CPU는 `--device cpu`다.
기본 physics는 기존 `config/preflame_electrochemical_thermal_decomposition_a100_cpu8.yaml`이며,
실제 시료의 보정 config를 쓰려면 `--physics-config <파일>`을 지정한다.

동일 소스·설정·형상·장치·batch 크기의 중단된 실행은 아래처럼 재개한다.

```bash
python python/run_electrode_parameter_study.py evaluate --out "$RUN" --device cuda --batch-size 2 --resume
```

소스 변경 전의 실행은 새로운 output에서 다시 시작해야 한다. 기존 결과를 덮어쓰지 않는다.

## 출력과 해석 범위

- `library/<ID>/mask.npz`, `metadata.json`, `geometry.svg`: 정확한 mask·치수·극성.
- `cases/<ID>/result.json`: 원본 solver 결과와 수치 유효성.
- `parameter_study_summary.csv`, `split_comparison_summary.csv`: 비분할/분할 두 행.
  중앙 간격, 조각 길이, 접촉면적 감소량·감소율, 극성별 조각 수를 함께 기록한다.
- `requested_runtime_config.json`, `resolved_physics_config.json`, `run_identity.json`:
  실행 입력과 재개 검증 정보.
- 대표장 출력은 기존 evaluator의 field diagnostics를 그대로 사용한다.

시간적분, 전위/종 수송, Butler–Volmer 계면반응, 열전달, 분해반응, onset 판정,
Vmin 검색은 기존 C++/CUDA evaluator를 상속한다. 새 형상 검증은 지역 geometry builder에서
수행하며, 공통 equal-area 형상 검사를 완화하지 않는다.

`ignitionDelay_s`는 **응축상 분해 onset**이다. 가시 화염의 점화 지연, 화염 안정성,
합선이나 방전 발생 여부를 이 모델만으로 판정할 수 없다. 미보정 물성·반응계수가
이 기능 추가로 보정되는 것은 아니다. 수치적으로 무효인 결과는 유효한 무점화와
구분하고, 요약 CSV의 물리값을 비운다.

## 실험 기록과 검증

두 조건×5회인 빈 실험 기록 양식과 block별 무작위 순서표를 생성한다.
이는 실제 실험 기록용이며 동일 simulation을 5번 반복하지 않는다.
측정치를 복사한 CSV에 입력한 뒤, 이 profile을 명시하여 집계한다.

```bash
python python/run_electrode_parameter_study.py summarize-experiments --study-config config/electrode_split_study_25mm.yaml --input "$RUN/experiment_measurements.csv" --out "$RUN/experiment_summary.csv"
```

관련 자동 검증:

```bash
python -m pytest -q tests/test_electrode_parameter_study.py tests/test_electrode_split_study.py
```

기하학·실행 연결·재개 테스트의 성공은 실제 2초 생산 계산, A100 검증 또는
실험적 성능 우위를 뜻하지 않는다.

추가 시 검증 기록: 위 두 테스트 파일 **68개 통과**. 실제 C++ standalone CPU 경로에서
N=200, 기준 전압 260 V, 종료/평가시간 0.0005 s(기존 dt=0.00025 s의 2 step),
기존 Vmin 검색을 유지한 두 조건의 smoke run을 수행했다. 두 조건 모두 수치 유효성이
확인됐고 대표장의 수치값은 유한했다. smoke용 시간 변경은 임시 config에만 적용했으며,
저장소의 기본 2초 물리 설정은 그대로다. CUDA/A100 실행과 전체 2초 성능 비교는 미실시다.
