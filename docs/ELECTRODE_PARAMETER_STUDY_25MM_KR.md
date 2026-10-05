# 25 mm full-height 평행 전극 parameter study

## 범위와 기존 연구 보존

기준 main: `563ffbff7ee1cfbf46a81ff946aac69383b4a461`.
**기존 파일 수정·삭제 없이 추가 파일 4개로 구현**했습니다.
148-topology library, 생성기, paired workflow, staggered 생성기, 공용 evaluator,
C++/CUDA 물리 코드, 기존 configuration은 그대로 둡니다.

새 설정 파일은 의도적으로 `config/`가 아니라 `studies/`에 있습니다.
기존 `paired_workflow.protected_hashes()`는 `config/` 전체를 fingerprint에
포함하므로, 그곳에 새 YAML을 추가하는 것만으로도 과거 148-case resume의
입력 signature가 달라질 수 있기 때문입니다. 이번 추가 파일 4개는 모두
그 보호 경로 밖에 있어 해당 fingerprint를 바꾸지 않습니다.

실행은 아래 새 runner로만 합니다. 기존 148-case launcher를 사용하지 않습니다.
NSGA-II, staggered 생성/비교/정규화, 별도 post-onset refinement는 호출하지 않습니다.

## 고정된 8개 조건

Domain은 25 x 25 mm, 전극 접촉 길이는 모두 25 mm입니다.
**25 mm는 두께가 아닙니다.** 두께/유효 깊이, 조성, 계면 물성 및 kinetics는
선택한 기존 physics config에서 상속하며, 새 값으로 임의 지정하지 않습니다.
실험값과 상속한 모델 값의 일치 여부는 실험 전에 확인해야 합니다.

| ID | 분석 그룹 | 양극 폭 mm | 음극 폭 mm | edge gap mm | 양극 면적 mm2 | 음극 면적 mm2 |
|---|---|---:|---:|---:|---:|---:|
| SP_G1 | spacing | 2 | 2 | 1 | 50 | 50 |
| BASE_G2_W2 | spacing, width | 2 | 2 | 2 | 50 | 50 |
| SP_G4 | spacing | 2 | 2 | 4 | 50 | 50 |
| WD_W1 | width | 1 | 1 | 2 | 25 | 25 |
| WD_W4 | width | 4 | 4 | 2 | 100 | 100 |
| AR_1TO2 | area_ratio | 2 | 4 | 2 | 50 | 100 |
| AR_1TO1 | area_ratio | 3 | 3 | 2 | 75 | 75 |
| AR_2TO1 | area_ratio | 4 | 2 | 2 | 100 | 50 |

`BASE_G2_W2`는 두 그룹에서 공유하는 **직선형 기준 조건**입니다.
Area-ratio 그룹의 1:1 기준은 별도의 `AR_1TO1`입니다.
Area-ratio 세 조건은 총 접촉면적 150 mm2로 같고, width 그룹은 폭과 coverage가 함께 변합니다.

전체 전극 assembly의 바깥 경계를 x방향 중앙에 맞춥니다.
`x0=(25-wa-gap-wc)/2`, 양극 `[x0,x0+wa]`, 음극 `[x0+wa+gap,x0+wa+gap+wc]`.
모든 strip은 y=0부터 y=25까지 이어집니다. 비대칭 면적비의 경우 gap의 중심은
항상 x=12.5가 아니며, AR_1TO2와 AR_2TO1은 극성 교환을 수반하는 좌우 반사 쌍입니다.
전극은 full-propellant 위의 surface-contact label입니다. 전극 밑 추진제를 제거하지 않습니다.

## 격자와 adapter

B/C는 **cell-centred**이므로 `dx=25/N` mm입니다.
기본값 `N=200`이면 dx=0.125 mm이고 폭·간격 및 모든 반 mm 좌표가 정확히 셀 경계에 놓입니다.
1 mm 폭/간격은 8개 셀입니다. 201은 이 고정 치수에 정확히 맞지 않아 허용하지 않습니다.
격자수는 100 이상인 50의 배수만 허용합니다. 정밀도 검토는 예를 들어 200, 400, 800에서
별도의 output directory와 study YAML 복사본을 사용하십시오.
**격자에 치수가 정확히 맞는다는 것은 해의 격자수렴성을 입증했다는 뜻이 아닙니다.**

`FullHeightStripGeometryMixin`은 이 runner에서 생성하는 subclass에만 적용됩니다.
유일하게 바꾸는 method는 `_build_geometry_batch()`입니다.
입력 mask를 승인된 여덟 직사각형으로 재구성해 완전히 동일한지 검사합니다.
위치, 길이, 폭, gap, 극성별 면적, 1A1C 연결성, grid 모두 강제합니다.
그 후 기존 `DirectCondensedV772NoFEvaluator._build_geometry_batch()`의
전극 overlap/contact/gap 검사 및 tensor 포장을 재사용합니다.
B/C와 같은 `fixed=False`, `propellant=True` overlay 의미로 반환하며
실제 비대칭 masks 자체를 그대로 numerical engine에 전달합니다.

기존 equal-area gate를 넓히거나 끄지 않으며, 다른 evaluator instance의 동작을 바꾸지 않습니다.
`NativeBCGlobalEvaluator`의 전기화학·열·분해 계산, Vmin, 수치 유효성 검사,
대표 field export는 그대로 상속합니다. Dummy mask나 global monkey-patch는 없습니다.

검토한 upstream `evaluator.py`, `bc_native.py`의 Git blob SHA를 실행 전에 확인합니다.
두 파일에 다른 수정이 있으면 호환성을 확인할 때까지 중단합니다.
**이 오류를 피하려고 사용자의 작업을 reset하거나 기존 코드를 덮어쓰지 마십시오.**

## 실행

저장소에 추가 파일이 들어 있는 branch/commit을 checkout한 후, 기존 `.venv`를 사용합니다.

```bash
cd /ECSP/ecsp-electrode-optimizer
source .venv/bin/activate
export PYTHONPATH="$PWD/python"

python -m pytest -q -o addopts='' tests/test_electrode_parameter_study.py

python python/run_electrode_parameter_study.py generate \
  --out runs/strip_doe_25mm

python python/run_electrode_parameter_study.py audit \
  --out runs/strip_doe_25mm
```

첫 실제 solver 확인은 양극/음극 면적이 다른 조건으로 할 수 있습니다.

```bash
python python/run_electrode_parameter_study.py run \
  --device cuda --only AR_1TO2 \
  --out runs/strip_doe_25mm

# 나머지 조건을 계산합니다. AR_1TO2의 같은 입력 결과는 재사용합니다.
python python/run_electrode_parameter_study.py run \
  --device cuda --resume \
  --out runs/strip_doe_25mm
```

CPU 실행은 `--device cpu`이며 같은 **native B/C** 경로를 사용합니다.
오래된 legacy standalone C++ solver로 전환하는 옵션은 아닙니다.
CUDA 선택 실패 시 CPU로 조용히 전환하지 않습니다. Native compiler 환경은 기존 저장소와 같습니다.

기본 physics 파일은 `config/nsga2_bc_reactive_a100_cpu8_poweroff_200x3.yaml`입니다.
기본 study 파일은 `studies/electrode_parameter_study_25mm.yaml`입니다.
`--config`, `--study`로 명시적으로 다른 파일을 지정할 수 있습니다.
study의 260 V와 2 s는 physics 값에 대한 **일치성 검사**이지 덮어쓰기 값이 아닙니다.
동일한 물리 설정 내에서 geometry와 계산 device/batching만 변경합니다.

중단 후에는 같은 명령에 `--resume`을 붙입니다. 입력이 달라지면 재사용을 거부합니다.
`execution_failed` 재시도만 허용하려면 `--resume --retry-failed`를 사용합니다.
수치적으로 invalid한 결과를 다른 tolerance로 몰래 재시도하지 않습니다.
한 batch가 완료될 때 case별 checkpoint가 저장됩니다. batch 도중 중단되면 아직
checkpoint가 없는 batch는 다시 계산합니다. 동시에 같은 out을 쓰는 실행은 lock으로 막습니다.

## 출력과 해석

`library/<ID>/mask.npz`, `metadata.json`, `manifest.json`, `geometry_audit.json`:
형상 및 무결성 정보. `cases/<ID>/result.json`: 원래 scalar 결과와 유효성/상태.
`parameter_study_summary.csv` 및 `spacing_summary.csv`, `width_summary.csv`,
`area_ratio_summary.csv`: 각 그룹의 raw 결과. 공통 기준은 두 그룹 CSV에 표시되지만 한 번만 계산합니다.
`resolved_physics_config.json`, `source_preservation.json`: 실제 설정 및 기존 source 변경 유무.
기존 native representative-field exporter의 출력은 각 case directory에 저장됩니다.

컴파일/실행 오류는 `execution_failed`, 수치 실패는 `numerically_invalid`,
유효한 비점화는 `no_condensed_onset_within_horizon`으로 분리합니다.
유효 onset은 `condensed_onset_reached`입니다.
비점화 delay는 비어 있으며 2 s로 대체하지 않습니다. Optimizer의 거대한 penalty 값을
측정치로 CSV에 내보내지 않습니다. Vmin의 validity 및 좌/우 검열 flags도 보존합니다.

**이 모델의 onset은 응축상 기준이며 영상에서 보이는 화염 점화와 자동으로 같지 않습니다.**
상속한 B/C 평가기는 최초 onset 후 전기 가열을 중단하고 고정 평가시간까지의 내부 응축상
continuation을 사용합니다. 이를 새로운 gas-flame propagation 해석으로 해석하면 안 됩니다.
이번 runner는 외부 post-onset refinement/handoff workflow를 호출하지 않습니다.

## 실제 실험 5회 반복

`experiment_template.csv`는 8조건 x 5회 = 40개 빈 관측 행입니다.
`experiment_schedule.csv`는 각 block에서 여덟 조건을 한 번씩 무작위 배치한 계획입니다.
5회는 실제 실험의 반복이지 deterministic simulation의 중복 실행 수가 아닙니다.
데이터는 템플릿의 복사본에 입력하십시오. `valid`, `ignited`는 true/false,
`ignition_delay_s`는 관측값, 비점화는 delay 공란과 실제 observation window를 기록합니다.

```bash
python python/run_electrode_parameter_study.py summarize-experiment \
  --input runs/strip_doe_25mm/experiment_observations.csv \
  --output runs/strip_doe_25mm/experiment_summary.csv
```

통계는 유효 횟수, 제외 횟수, 점화/우측 검열 횟수, 점화율, **점화된 시험에서만**
계산한 delay 평균 및 표본 표준편차(ddof=1)를 나눠 기록합니다.
검열된 비점화가 섞인 경우 이 평균은 모든 시험의 무조건 평균이 아닙니다.

## 이번 구현의 검증 범위

독립 add-on 환경: **51 passed, 1 skipped**.
포함 범위: 100/200/400 격자의 여덟 형상, 면적·간격·길이, 극성 반사,
mask/metadata 훼손 거부, 생성/audit CLI, 40행 템플릿, source 보호 경로,
mocked checkpoint/resume·오류 분리, 실험 통계.

Skip한 1개는 실제 upstream checkout의 원본 geometry builder와 연결해 검사하는 테스트입니다.
저장소 전체가 있는 사용자 환경에서는 해당 테스트도 수행됩니다.
이 환경에서는 GitHub connector로 원본 코드를 읽었지만 전체 checkout을 내려받을 수 없었고,
실제 native B/C integration 및 A100 실행은 수행하지 않았습니다.
**Mock 테스트의 수치는 physics 예측이나 점화성능 검증 결과가 아닙니다.**
