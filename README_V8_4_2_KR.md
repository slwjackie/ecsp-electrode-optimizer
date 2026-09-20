# ECSP v8.4.2 — 폭 상한 5 mm 유지 / 면적 1% / strict bootstrap / Power-OFF

## 이번 수정 범위

직접 기반은 `ECSP_v8_4_1_A100CPU8.zip`입니다. 전기·열·분해·Reactive Euler 수치 엔진, 물성·반응계수, 물리시간과 시간간격은 바꾸지 않았습니다. 형상 생성·면적 보정·초기 개체군 검사·재사용 캐시와 실행기를 수정했습니다. 기존 condensed baseline과 A100 Torch Reactive는 보존합니다. 이 배포를 만든 환경은 CPU 5개, CUDA 없음이며 실제 A100 성능·생산 물리 실행은 검증하지 않았습니다.

## 검증된 제조 조건

생산 seed **20260904**, **20 topology × 각 50개**를 실제 생성했습니다. 설계 격자는 **96×96**, BC/Reactive 해석 격자는 **193×193**입니다. 각 격자에서 동일 극성 component 수 보존, 반대 극성 비접촉·최소 간격, 면적, 외곽 여유와 중복을 검사합니다. 실제 B/C `_build_geometry_batch()`에도 1,000개 전부를 통과시켰습니다. 이 검증은 제조·격자 적합성이며 점화 성공·물성 정확도를 보장하지 않습니다.

| 항목 | 결과 |
|---|---:|
| 서로 다른 topology | 20 |
| topology별 중복 없는 유효 후보 | 각각 50 |
| 설계 격자 / 해석 격자 고유 mask 쌍 | 각각 1,000 |
| 재생성을 포함한 원시 시도 | 2,613 |
| 허용 최대 선폭 | **5.0 mm 그대로** |
| 이번 bootstrap의 실제 최대 primitive 선폭 | 약 1.56213 mm |
| 양·음극 각각의 목표 접촉면적 | 전체 표면의 17.5% = 70 mm² |
| 설계 격자의 극성별 최대 상대 면적오차 | 약 0.29762% |
| 해석 격자의 극성별 최대 상대 면적오차 | 약 0.99063% |
| 설계 / 해석 격자 최소 반대 극성 간격 | 1.25 / 약 1.34715 mm |
| 요구 최소 간격 | 0.5 mm |
| 실제 B/C geometry builder 검사 | 1,000 / 1,000 통과 |

`maximum_width_mm=5.0`은 허용 상한이지 모든 component를 5 mm로 만든다는 뜻이 아닙니다. 5 mm 선폭이 가능한 별도의 여유공간 시험도 통과했습니다. 이번 bootstrap에서 가장 넓은 trace가 1.56 mm라는 사실은 상한을 숨겨서 1.56 mm로 바꿨다는 뜻이 아닙니다.

`area_tolerance_fraction: 0.01`은 도메인 면적의 ±1 percentage point가 아니라, **극성별 목표면적 70 mm²의 ±1% (69.3–70.7 mm²)** 입니다. 양·음극 오차를 평균하여 한 극성의 초과를 숨기지 않습니다.

## 면적 보정 로직

1. `_assign_component_poses()`는 더 이상 `maximum_width_mm=5`를 모든 trace의 pitch와 예약 폭으로 쓰지 않습니다. 각 component에 분리된 배치영역을 두고 목표면적과 centreline 길이로 예상 실제 선폭을 계산합니다.
2. 주경로뿐 아니라 실제 생성된 branch/arc envelope를 배치영역에 맞춥니다. LINE 길이와 ARC 반경을 같은 비율로 조정하므로 원호를 타원으로 바꾸지 않습니다.
3. 면적이 부족하면 선폭을 고정한 상태에서 먼저 centreline/arc/branch의 길이를 늘립니다.
4. 길이만으로 해결되지 않으면 이웃 경로·이미 맞춘 반대 극성·외곽까지의 여유거리로 선폭 배율을 제한합니다. 필요하면 길이를 되돌려 공간을 확보한 후 폭을 조정합니다. 최소 LINE 길이와 최대 5 mm 선폭을 유지합니다.
5. 서로 다른 component가 합쳐지는 scale은 penalty가 아니라 **무효 후보**로 제거합니다. 개별 component가 하나의 연결영역인지도 확인하므로, 한쪽의 분할과 다른 쪽의 합쳐짐으로 총 개수만 맞추는 것도 허용하지 않습니다.
6. 실제 최종 LINE/ARC/BRANCH 파라미터를 genome에 저장합니다. 저장·재로드·재평가 시 같은 mask를 재현하고 재보정으로 형상이 달라지지 않는 시험을 추가했습니다.

두 centreline 사이의 거리 d, 두 선폭 w_i, w_j, 공통 배율 s에 대한 조건은

`(s*w_i + s*w_j)/2 + required_gap <= d`

입니다. 상대 선폭도 들어가므로 이전 설명의 `g_available - g_min` 단독식 대신 두 반폭을 포함한 조건을 사용합니다. raster rounding과 round cap 여유를 추가하고, 최종 mask에서 다시 검사합니다.

## 초기 개체군은 실패 시 중단

`python/ecsp_nsga2/bootstrap.py`가 모든 topology의 quota를 채웁니다. 해석 격자로 변환한 뒤 면적·간격·component 수를 재검사하며, 같은 mask를 50개의 다른 후보로 세지 않습니다.

한 topology가 설정된 재시도 한도에서 50개를 확보하지 못하면 `BootstrapGeometryError`로 중단하고 실패 보고서를 남깁니다. 기존 `bootstrap_constrained` / `_INVALID`로 초기 개체군을 채우는 경로는 제거했습니다. **1,000개가 모두 유효한지 확인하기 전에 Generation 0 후보 PDE는 호출하지 않습니다.** 다른 seed·크기·topology 수로 바꾼 경우에도 이 검사를 새로 통과해야 합니다. 모든 임의 제조조건이 가능한 것은 아닙니다.

기존 NSGA-II의 후속 세대에서 발생하는 infeasible 자손에 대한 constrained ranking 자체를 삭제한 것은 아닙니다. 유효하지 않은 자손은 비싼 물리 평가에서 제외되는 기존 정책을 유지합니다.

## 생산 프로필 변경 범위

`config/nsga2_bc_*.yaml` 중 population 1,000인 11개 활성 BC 생산 프로필에 tolerance 0.01을 적용했습니다. 이들은 seed, design grid, solver grid, 기본 topology/variant 계약이 동일합니다. 기존 debug 프로필의 완화된 설정과 과거 non-BC/CPU48 실험 프로필은 보존했습니다. 새 power-off 실행기와 아래 명령은 명시적으로 수정된 BC 생산 프로필을 선택합니다.

## A100 + CPU8, 점화 이후 전원 OFF 실행

ZIP을 Pod에 올리고 압축을 푼 패키지 루트에서 실행합니다. CUDA-enabled PyTorch와 기존 C++/CUDA 컴파일 환경을 유지하십시오.

```bash
python -m pip install -r python/requirements.txt pytest

bash tools/run_bc_reactive_a100_cpu8_poweroff.sh \
  "$PWD/runs/v842_reactive_poweroff"
```

기본 파일은 `config/nsga2_bc_reactive_a100_cpu8_poweroff.yaml`입니다.

```yaml
bc_global:
  continuedElectricalHeatingAfterOnset: false

propagation_refinement:
  continued_electrical_heating: false
  electrical_heating_mode: off

post_onset:
  backend: reactive_euler
  compare_backends: true
  allow_experimental_ranking: true
  execution:
    backend: torch_batch
    device: cuda
    batch_size: 8
    cpu_budget: 8
    host_reserve: 2
    cpu_workers: 6
```

**Power-off는 onset 이후의 전기발열과 NP/BV 재계산을 끈다는 뜻입니다. onset 이전 BC-global의 전위·NP/BV 계산을 끄는 것은 아닙니다.** Reactive에서는 반응·열전도·열손실을 계속 계산합니다. 같은 onset의 CPU condensed baseline 비교도 유지됩니다. 잠정 최종 추천 기준은 신규 Reactive이며 실험 검증 완료를 뜻하지 않습니다.

전원 OFF 전용 실행기는 true/문자열 "false"/recomputed 모드 등을 발견하면 덮어쓰지 않고 거부합니다. 전압 유지가 필요한 다른 실험에는 일반 실행기를 사용하고 전원정책을 별도로 지정해야 합니다.

실행 순서:

```text
20×50 geometry-only 검사 (96 및 193 격자, 중복 검사)
→ geometry cache 저장
→ 실제 A100 장치·native BC CUDA·Reactive GPU 필수시험
→ 주 workflow에서 cache의 코드/seed/격자 계약과 모든 mask 재검사
→ BC pre-onset → Power-OFF GPU Reactive + CPU baseline
→ 최종 비교·순위
```

GPU 필수시험이 skip되거나 실패하면 본 계산을 시작하지 않습니다. 이 제작 환경에서 A100 시험을 대신 통과했다고 기록하지 않았습니다.

### 터미널을 닫아도 프로세스를 유지하는 실행

```bash
mkdir -p "$PWD/runs"
RUN="$PWD/runs/v842_poweroff_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/run_bc_reactive_a100_cpu8_poweroff.sh "$RUN" \
  > "${RUN}_launcher.log" 2>&1 < /dev/null &
echo "PID=$!"
echo "RUN=$RUN"
tail -f "${RUN}_launcher.log"
```

`tail`만 종료하려면 Ctrl+C를 사용합니다. 로그는 `${RUN}_launcher.log`를 사용하십시오. `${RUN}.log`는 내부 실행기가 생성하는 파일이므로 nohup 출력을 그 파일로 먼저 열면 기존 출력 보호장치가 거부합니다. `nohup`은 Pod 자체의 삭제·재시작·자원 회수까지 막지 않습니다.

### GPU 없이 형상 1,000개만 확인

```bash
python python/validate_nsga2_geometry.py \
  --config config/nsga2_bc_reactive_a100_cpu8_poweroff.yaml \
  --output "$PWD/runs/geometry_20x50_check" \
  --require-power-off
```

출력은 `geometry_bootstrap_report.json`, `geometry_bootstrap_candidates.csv`, `bootstrap_genomes.json`입니다. 런처가 만든 cache를 주 실행기가 재사용하므로 1,000개를 다시 area-fit하지 않습니다. 캐시가 다른 코드·seed·격자·한계에서 만들어졌으면 중단하며, 캐시를 읽더라도 mask 검사는 생략하지 않습니다.

## 회귀시험과 자료

전체 시험 결과는 `VALIDATION_V8_4_2_KR.md` 및 `docs/validation_v8_4_2/`를 참조하십시오. 새 제조 회귀시험은 `python/tests/test_geometry_safe_bootstrap.py`입니다.

```bash
PYTHONPATH=python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q python/tests
```

v8.4.1 물리·계수 표와 이전 manifest들은 역사적 자료로 보존했습니다. **현재 제조 tolerance는 위 v8.4.2 프로필과 새 보고서가 기준**입니다. 물리식·실험 미보정 계수·Tait의 열–압력 비결합 등 기존 모델 한계를 이번 형상 패치가 해소한 것은 아닙니다.
