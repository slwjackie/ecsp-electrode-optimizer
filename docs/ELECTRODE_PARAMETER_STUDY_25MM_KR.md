# 25 mm full-height 직선 전극: 독립 parameter study

기준 main: `563ffbff7ee1cfbf46a81ff946aac69383b4a461`.
**기존 파일은 수정하지 않는다.** 기존 148개 형상, paired workflow, staggered,
NSGA-II, pre-flame/native solver, post-onset solver와 원본 config는 그대로다.
이번 추가 실행기는 staggered 형상을 생성하거나 비교하지 않는다.

## 고정 설계

추진제 평면 25×25 mm. 양극과 음극의 추진제 접촉 길이는 모두 25 mm이며,
아래/위 margin 없이 전체 높이를 덮는다. 각 극성은 하나의 직사각형이다.
`gap`은 edge-to-edge 거리다. `양극 + gap + 음극` 전체 폭을 domain 중앙에 놓는다.
접촉영역 밖 배선은 기존 solver처럼 별도 외부 전원 연결을 가정하며 접촉 mask에 추가하지 않는다.

| ID | 분석 그룹 | 양극 폭 mm | 음극 폭 mm | gap mm | 양극 면적 mm² | 음극 면적 mm² |
|---|---|---:|---:|---:|---:|---:|
| SP_G1 | spacing | 2 | 2 | 1 | 50 | 50 |
| BASE_G2_W2 | spacing, width | 2 | 2 | 2 | 50 | 50 |
| SP_G4 | spacing | 2 | 2 | 4 | 50 | 50 |
| WD_W1 | width | 1 | 1 | 2 | 25 | 25 |
| WD_W4 | width | 4 | 4 | 2 | 100 | 100 |
| AR_1TO2 | area_ratio | 2 | 4 | 2 | 50 | 100 |
| AR_1TO1 | area_ratio | 3 | 3 | 2 | 75 | 75 |
| AR_2TO1 | area_ratio | 4 | 2 | 2 | 100 | 50 |

고유 형상은 **8개**다. `BASE_G2_W2`는 staggered가 아닌 직선형 기준이다.
Area-ratio 그룹의 기준은 별개의 `AR_1TO1`이다. 해당 그룹의 총 전극면적은
모두 150 mm²다. Width 그룹에서는 폭과 총 접촉면적이 함께 변한다.
전극 밑의 추진제도 남아 있으므로 모든 경우 추진제 domain 면적은 625 mm²다.

## 추가 파일과 물리 보존

- `python/run_electrode_parameter_study.py`: geometry, audit, native 실행, 결과/실험 요약.
- `config/electrode_parameter_study_25mm.yaml`: 실험 설계만 저장.
- `tests/test_electrode_parameter_study.py`: 새 경로의 단위/실행 연결 테스트.
- 이 문서.

원래 evaluator의 equal-area 조건을 넓히거나 원본 파일에서 삭제하지 않는다.
새 실행기의 지역 subclass는 `NativeBCGlobalEvaluator._build_geometry_batch()`만
독립적으로 구현한다. 정확한 등록 직사각형과의 일치로 극성별 목표면적, 폭, gap,
연결성, 경계 접촉 및 위치를 검증하고 동일한 `GeometryBatch` 자료형을 반환한다.
1:2의 비대칭 면적을 허용하지만 임의 형상이나 잘못된 면적은 허용하지 않는다.
원본 class를 monkey-patch하지 않으며 mask 수정/resize/fitting도 하지 않는다.

시간적분, B/C 반응식, native CPU/CUDA 엔진, onset 판정, numerical validity와
Vmin search는 원래 메서드를 그대로 상속한다. `fixed=False`, `propellant=True`로
기존 surface-contact overlay 의미도 보존한다. 별도 NSGA-II/paired/post-onset
orchestrator는 호출하지 않는다. 상속된 reference run 자체에 있는 공통 평가시점까지의
응축상 continuation은 그대로이며, 이를 기체 화염 전파라고 해석하지 않는다.

## 격자: 201 대신 200

실제 B/C 코드는 cell-centred 격자, **dx=L/N**을 사용한다. 기본 N=200이면
25/200=0.125 mm이므로 1 mm 폭/gap은 정확히 8셀, 길이는 200셀이다.
앞서 설명한 `201에서 25/(201-1)` 방식은 이 코드의 격자 정의와 맞지 않는다.
이번 runner는 치수를 조용히 반올림하지 않고 201 같은 off-grid 입력을 거부한다.
100, 200, 400처럼 50의 배수인 격자에서 모든 설계 edge를 정확히 표현할 수 있다.
200은 초기 실행 격자이며, 격자 독립성이 검증되었다는 의미는 아니다.
격자 변경은 새 study config와 새 output으로 실행한다.

## 실행

저장소 루트의 기존 작업 환경에서 실행한다.

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests/test_electrode_parameter_study.py

RUN="$PWD/runs/straight25_parameter_study"
python python/run_electrode_parameter_study.py generate --out "$RUN"
python python/run_electrode_parameter_study.py audit --out "$RUN"

python python/run_electrode_parameter_study.py evaluate \
  --out "$RUN" --device cuda --batch-size 8
```

`generate`와 `audit`는 PDE를 실행하지 않는다. 기하학 단계는 NumPy/PyYAML만 필요하다.
수치 단계에는 저장소의 기존 의존성, PyTorch, native compiler와 CUDA 개발 환경이 필요하다.
CPU 경로는 `--device cpu`이며 MPS를 사용하지 않는다.
기존 native 초기화에서 지원하는 compiler/build 실패를 nonignition으로 숨기지 않는다.

기본 physics 입력은 기존
`config/nsga2_bc_reactive_a100_cpu8_poweroff_200x3.yaml`이다.
전압, 해석시간, 물성, 조성, 잔류수분, 두께 관련 설정, 열경계와 onset/Vmin 기준을
이 입력 및 그 `base_config`에서 상속한다. 원본 파일은 변경하지 않는다.
실제 실험과 일치하도록 이미 보정한 config가 있다면 다음처럼 명시한다.

```bash
python python/run_electrode_parameter_study.py evaluate \
  --out "$RUN" --device cuda --batch-size 8 \
  --physics-config config/YOUR_EXISTING_CALIBRATED_CONFIG.yaml
```

입력/장치/batch 크기/코드가 동일할 때만 재개한다.

```bash
python python/run_electrode_parameter_study.py evaluate \
  --out "$RUN" --device cuda --batch-size 8 --resume
```

이미 완료된 case 결과는 재사용한다. batch가 반환될 때마다 case별로 저장한다.
batch 도중 중단되면 그 batch의 미저장 case는 다시 계산한다.
case 단위로 자주 checkpoint하려면 처음부터 `--batch-size 1`로 실행한다.
설정이 바뀌면 새 output을 사용한다. 기존 library/result를 덮어쓰거나 삭제하지 않는다.

## 출력 및 실패 처리

`library/<ID>/`에 `mask.npz`, `metadata.json`, 치수 기반 `geometry.svg`를 저장한다.
`cases/<ID>/result.json`에는 변환 전 원본 solver 결과를 보존한다.
기존 representative-field 기능을 활성화하여 native 출력의 대표장을 저장한다.
정확한 생성 파일/field 의미는 기존 `ecsp_nsga2/field_diagnostics.py`를 따른다.
시간이력 전체나 새로운 ignition-location/화염전파 모델을 추가한 것은 아니다.

`parameter_study_summary.csv`와 `spacing_summary.csv`, `width_summary.csv`,
`area_ratio_summary.csv`에는 raw 값과 유효성/censoring flag를 기록한다.
Staggered 정규화, Pareto 선별, no-ignition penalty를 물리적 측정값으로 제시하지 않는다.
각 그룹의 CSV는 공유 직선 baseline을 포함해 3행이다.

Reference onset 실패는 `no_onset_within_horizon`, numerical rejection은
`numerically_invalid`로 구분한다. 계산/컴파일 예외는 `execution_failure.json`에
기록하고 실행을 중단한다. 수치 실패와 물리적 nonignition을 섞지 않는다.
Vmin의 left/right censoring과 search validity는 별도 열로 남긴다.
`ignitionDelay_s`는 **응축상 onset**이며, 고속영상의 visible-flame delay와 동일하다고
가정해서는 안 된다. 현재 모델의 미보정 계수도 이 기능 추가로 보정되는 것이 아니다.

`requested_runtime_config.json`, `resolved_physics_config.json`, `run_identity.json`,
`sources_before.json`, `sources_after.json`을 저장한다. 실행 전후 source/config hash가
달라지면 실패하며, code/config/mask가 달라진 상태의 resume도 거부한다.

## 실험 반복 데이터

8개 조건 × 5회 = 40행의 빈 `experiment_measurements_template.csv`를 제공한다.
이는 **실험 기록 양식**이지 가짜 실험결과나 40회 시뮬레이션이 아니다.
원본 template을 복사하여 `ignition_observed`에 true/false, 실제 지연시간과
`observation_window_s`를 기록한다. 무점화 시험의 delay는 빈칸으로 남긴다.

```bash
python python/run_electrode_parameter_study.py summarize-experiments \
  --input "$RUN/experiment_measurements.csv" \
  --out "$RUN/experiment_summary.csv"
```

성공 횟수, 무점화 횟수, 점화율과 **성공한 시험에 한정한** 평균/표본표준편차를
분리한다. 무점화를 0초/관측 종료시간으로 대체하지 않는다. 관측창이 다른 실험들의
점화율/지연시간을 직접 비교하면 안 된다. 공유 baseline 데이터를 독립적인 두 표본으로
중복 계산하지 않는다. 동일 deterministic simulation을 5회 반복하지 않는다.

## 검증 범위

배포 전 새 테스트 **34개 통과**: 8개 형상의 면적/폭/gap/full-height,
area-ratio mirror/150 mm², 격자 일치, 손상 입력 거부, full-propellant tensor,
원본 config 불변, subclass의 geometry-only override, mock evaluator를 이용한
8-case 실행/재개/caching/실패 분리 및 실험 통계 처리를 확인했다.

실행 연결 테스트에는 명시적인 fake evaluator를 사용했다. **생산 native PDE,
실제 A100/CPU 수치 실행, 기존 148개 전체 회귀시험, 격자수렴성 및 실험 정확도는
이 테스트로 검증되지 않는다.** 위 테스트를 저장소 환경에서 재실행한 뒤
실제 `evaluate`의 numerical validity 결과를 확인해야 한다.
