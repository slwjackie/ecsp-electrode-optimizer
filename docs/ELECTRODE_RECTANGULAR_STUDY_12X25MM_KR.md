# 가로 12 mm·세로 25 mm 전극 경사각·폭 비교

`config/electrode_inclination_study_12x25mm.yaml`은 **가로 12 mm·세로 25 mm**
추진제에 놓인 좌우 대칭 전극을 기존 preflame 해석기로 평가하는 별도 설정이다.
양극·음극의 수평 폭을 함께 **2 mm 또는 4 mm**로 선택한다. 기존 25×25 mm의
0~30° 비교 설정은 [별도 문서](ELECTRODE_INCLINATION_STUDY_25MM_KR.md)를 따른다.

## 형상과 비교 조건

중앙 높이 y=12.5 mm에서 양극은 왼쪽 경계, 음극은 오른쪽 경계에 닿는다.
각도 θ는 수직 방향 기준이며, 양극·음극의 위·아래 끝이 서로 가까워진다.
각 극의 수평 폭 w를 고정하므로 한 폭 그룹 안에서는 각도와 무관하게
접촉면적이 25w mm²로 일정하다. 중앙 간격과 끝 간격은 다음과 같다.

\[
g_{\mathrm{center}}=12-2w,\qquad
g_{\mathrm{end}}=12-2w-25\tan\theta.
\]

최소 끝 간격 1 mm와 경사 팔에 수직인 폭 \(w\cos\theta\ge1\) mm를 적용한다.

| 양극·음극 각각의 수평 폭 | 중앙 간격 | 극당 접촉면적 | 최대 연속기하 각도 | 기본 생성 각도 |
|---|---:|---:|---:|---|
| 2 mm | 8 mm | 50 mm² | 15.642° | 0°, 5°, 10°, 15° |
| 4 mm | 4 mm | 100 mm² | 6.843° | 0°, 5° |

기본 설정은 위 **6개 case**를 포함한다. 동일 각도에서 폭을 비교할 수 있는
공통 조건은 0°와 5°다. 폭 2 mm의 10°·15°는 해당 폭에서의 추가 각도 비교다.
**폭 2 mm와 4 mm 사이에서는 접촉면적과 중앙 간격도 함께 변한다.** 따라서
두 폭의 결과 차이를 면적과 간격이 고정된 순수 폭 효과로 해석하지 않는다.

| Case ID | 수평 폭 | 각도 | 연속 형상의 끝 간격 |
|---|---:|---:|---:|
| W2_ANGLE_00 | 2 mm | 0° | 8.000 mm |
| W2_ANGLE_05 | 2 mm | 5° | 5.813 mm |
| W2_ANGLE_10 | 2 mm | 10° | 3.592 mm |
| W2_ANGLE_15 | 2 mm | 15° | 1.301 mm |
| W4_ANGLE_00 | 4 mm | 0° | 4.000 mm |
| W4_ANGLE_05 | 4 mm | 5° | 1.813 mm |

## 직사각형 계산 격자

기본 `grid_size: 200`은 **세로 셀 수 Ny**다. 세로 25 mm를 200개로 나누면
dy=0.125 mm이며, 가로도 같은 dx=0.125 mm를 사용하므로 Nx=96이다.
실제 mask와 물리장은 **200행×96열**, 전체 추진제 면적은 **300 mm²**다.
25×25 mm 정사각형에 빈 여백을 넣거나 형상을 가로로 늘려 계산하지 않는다.

전극 아래에도 추진제가 존재하는 surface-contact overlay를 유지한다.
CPU/CUDA 계산 경로는 직사각형 배열의 실제 행·열 수와 공통 셀 간격을 사용한다.
서로 다른 dx와 dy를 사용하는 비등방 격자는 이 설정의 지원 범위가 아니다.
직사각형은 컴파일된 preflame 해석 경로를 사용한다. 별도 post-onset propagation으로
넘기는 handoff는 지원하지 않으며, 요청하면 오류로 중단한다. 기존 onset 이후
공통 평가 시각까지의 응축상 continuation과 성능 지표 계산은 지원한다.

각 행의 전극 셀 수를 고정하여 raster에서도 극당 면적을 유지한다. 경사 경계는
계단형으로 근사되며 연속 형상의 끝 간격과 실제 raster 최소 간격을 모두 검사한다.
`grid_size`를 바꿀 때에는 중앙 가장자리와 가로 domain을 같은 셀 간격으로
정확하게 나타낼 수 있어야 한다. N=200은 격자수렴을 입증한 값이 아니다.

## 실행과 폭 선택

저장소 루트에서 실행한다. 기본 설정의 양쪽 폭을 모두 생성하려면:

```bash
RUN="$PWD/runs/rect12x25_w2_w4"
python python/run_electrode_parameter_study.py generate \
  --study-config config/electrode_inclination_study_12x25mm.yaml \
  --out "$RUN"
python python/run_electrode_parameter_study.py audit --out "$RUN"
python python/run_electrode_parameter_study.py evaluate \
  --out "$RUN" --device cuda --batch-size 6
```

폭 2 mm만 선택하려면 생성 명령에 `--electrode-width-mm 2`를 지정한다.
폭 4 mm는 같은 옵션의 값을 `4`로 바꾼다. 각 실행에는 새 output 경로를 사용한다.

```bash
python python/run_electrode_parameter_study.py generate \
  --study-config config/electrode_inclination_study_12x25mm.yaml \
  --electrode-width-mm 2 \
  --out runs/rect12x25_w2

python python/run_electrode_parameter_study.py generate \
  --study-config config/electrode_inclination_study_12x25mm.yaml \
  --electrode-width-mm 4 \
  --out runs/rect12x25_w4
```

`evaluate`는 저장된 `study.json`을 읽으므로 폭 선택 옵션을 다시 지정하지 않는다.
CPU 실행은 `--device cpu`를 사용한다. 실제 시료에 맞춰 보정한 물리 설정이 있으면
`--physics-config <파일>`을 추가한다. 이전 source/config의 결과는 새 코드와
혼합하여 재개하지 않으며, 같은 입력으로 중단한 계산만 `--resume`으로 이어간다.

최소 점화 전압을 구하지 않고 고정 기준 전압에서 폭·각도만 비교하려면
`evaluate`에 `--skip-vmin`을 추가한다. 다른 성능 지표와 물리장은 계속 저장한다.

```bash
python python/run_electrode_parameter_study.py evaluate \
  --out "$RUN" --device cpu --batch-size 1 --skip-vmin
```

CSV의 최소 전압 값·유효성 칸은 공란, 상태는
`not_requested_reference_voltage_only`로 기록된다. 실제 기준 전압은
`referenceAppliedVoltage_V`에 저장한다. 탐색 포함 실행과 결과를 혼합하지
않도록 새 output에서 시작하고, 재개할 때도 `--skip-vmin --resume`을 지정한다.
시간 측정과 전압 탐색 횟수는 [25 mm 안내](ELECTRODE_INCLINATION_STUDY_25MM_KR.md#최소-점화-전압-탐색-생략)를 따른다.

## 각도 목록을 직접 지정하는 sweep

`generate-sweep`은 각 폭 그룹에 요청한 각도를 적용한다. 0° 대조군을 보존하고,
불가능한 각도는 해당 폭·각도·제외 이유를 `angle_feasibility.csv`와 `.json`에 남긴다.
원본 요청은 `sweep_request.json`에 저장된다. 예를 들어:

```bash
python python/run_electrode_parameter_study.py generate-sweep \
  --study-config config/electrode_inclination_study_12x25mm.yaml \
  --angles 0 5 10 15 \
  --out runs/rect12x25_custom_angles
```

이 목록에서는 폭 2 mm의 네 각도와 폭 4 mm의 0°·5°가 생성된다. 폭 4 mm의
10°·15°는 겹침/간격 조건을 위반하므로 제외된다. `--angles`를 생략하면
0~30°를 5° 간격으로 요청하며, 두 폭을 함께 선택한 경우 6개를 생성하고 8개를
제외 보고한다. `--electrode-width-mm`를 함께 사용하여 한 폭만 요청할 수 있다.
소수 각도도 지정할 수 있지만 격자에서 같은 mask가 되는 중복 형상은 제외한다.

## 출력과 해석 범위

- `library/<ID>/geometry.svg`, `mask.npz`, `metadata.json`: 실제 domain 치수,
  극성별 면적, 각도, 연속·raster 간격, 경사 팔 수직 폭과 경계 근사 오차.
- `cases/<ID>/result.json`: 원본 solver 결과와 수치 유효성.
- `parameter_study_summary.csv`, `inclination_summary.csv`: 폭·각도·domain 치수와
  점화 전 성능 지표. 수치적으로 무효인 물리값은 공란으로 둔다.
- `width_2mm_summary.csv`, `width_4mm_summary.csv`: 생성한 폭별 결과만 모은 표.
- 기본 6조건×5회인 실험 기록 양식과 순서표. 폭 하나만 선택하면 그 조건만 포함한다.

실험 집계는 실제 생성한 case 목록으로 수행한다.

```bash
python python/run_electrode_parameter_study.py summarize-experiments \
  --study-config "$RUN/study.json" \
  --input "$RUN/experiment_measurements.csv" \
  --out "$RUN/experiment_summary.csv"
```

`generate`와 `audit`는 PDE를 실행하지 않는다. 점화 지연 지표는 기존 해석기의
응축상 onset이며 가시 화염의 점화·안정성 판정을 대신하지 않는다. 외부 탭과
클램프 위치는 등전위 접촉 모델의 계산 domain에 포함하지 않는다.

## 구현 검증 범위

기본 6조건을 실제 C++ FP64 CPU 해석기로 200×96 격자, 260 V에서
0.00025 s 한 단계 실행했다. 여섯 조건 모두 선형·비선형 계면 풀이가 수렴했고
수치 유효성 검사를 통과했다. Vmin 실행 경로는 이 짧은 시간 안에 상한 전압에서
onset이 없음을 정상적으로 기록했으며, 저장 결과 재개와 폭별 CSV도 확인했다.
저장된 물리장 배열의 크기, 12×25 mm 좌표 및 극당 50/100 mm² 면적을 검사했다.
이 검증은 전체 2 s 성능 실험이나 실제 점화 전압 측정이 아니다.

형상·CLI·결과 검증, 공통 CPU/CUDA 셀 연산의 직사각형 인덱스·보존성,
기존 정사각형 해석과 Python continuation 경로를 회귀 검사한다.
관련 17개 테스트 파일에서 총 357개 테스트가 통과했다.
CUDA/A100 실행, 전체 시간창의 Vmin 이분 탐색, 격자수렴과 실험 성능은
별도 검증이 필요하다.
