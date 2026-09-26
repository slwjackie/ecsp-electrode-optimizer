# 고정 형상 catalogue의 paired preflame screening

## 범위와 변경 금지 영역

이 경로는 전극 형상끼리 1~148등을 매기지 않는다. 각 candidate는 **자신과 같은 물리 domain, grid, 초기온도, 기준전압, 평가시간, 물성/반응/수치 설정을 사용하는 staggered**와만 비교한다. 실제 solver mask에서 측정한 양극/음극 접촉면적을 각각 1% 이내로 맞춘다. Coverage만 같은 다른 크기의 baseline은 재사용하지 않는다.

새 파일은 `paired_screening.py`, `paired_workflow.py`, `run_paired_preflame_screening.py`, catalogue manifest, 테스트와 이 문서다. 기존 `run_five_topology_preflame.py`의 **평가 호출만 새 screening 경로에 연결**했다. 그 파일의 generate/audit 함수와 `five_topologies.py`, 기존 143개 형상, `ecsp_nsga2`·native·CUDA·C++·preflame·postflame solver 및 원본 config 파일은 변경하지 않았다. 기존 `objectives.py`의 legacy 함수도 호환성을 위해 남아 있지만 새 경로에서는 Pareto/vector/rank를 생성하지 않는다.

## 지표와 분류

- `R_t = t_ign,c / t_ign,s`: 기준 전압에서 둘 다 점화하고 유한한 양의 시간이 있을 때만 계산.
- `R_V = V_min,c / V_min,s`: 두 탐색이 유효하고 **둘 다 uncensored bracketed**인 경우에만 계산. penalty 값은 절대 사용하지 않음.
- `R_J = C_J,c / C_J,s`, `C_J = peakCurrentCongestionToEvaluationTime`: 기존 solver 정의 그대로. 동일 이름의 필드를 양쪽에서 사용하며, 전류밀도나 전류를 새로 계산하지 않음.
- `R_T = (Tmax,c - T0) / (Tmax,s - T0)`: **두 기준전압 case가 모두 비점화**인 경우에만 계산. 온도는 `peakMaximumTemperature_K`, 초기온도는 실제 resolved B/C config에서 읽음. Kelvin 자체의 비율이 아님.

분모가 0·극소값이거나 누락·NaN·무한대이면 해당 ratio는 JSON `null`과 사유로 남긴다. 0/0, 임의의 큰 penalty, 무한대 점화시간으로 우수성을 만들지 않는다.

| 결과 | 분류 |
|---|---|
| candidate만 점화 | `ignition_superior_candidates`, 기준조건에서 강한 유망 후보 |
| 둘 다 점화, t가 짧고 Vmin 감소가 bracket으로 분리됨 | `ignition_superior_candidates` |
| baseline만 점화 | `ignition_reject_at_reference_condition` |
| 둘 다 비점화, RT > 1 | `thermal_promising_candidates`; 점화 우수성은 미확정 |
| RJ < 1 | `congestion_superior_candidates`; 다른 label과 동시 부여 가능 |
| 둘 다 점화하고 t/V/J 모두 baseline보다 나쁨 | `baseline_dominated` |
| 둘 다 비점화하고 RT < 1, RJ > 1 | `fixed_condition_baseline_dominated`; 실제 Vmin 열위라는 의미가 아님 |
| reference 수치 검증 실패 | `numerically_invalid`; 물리적 비점화와 구별 |
| 실행 예외/PCG/OOM 등 | `execution_failed`; failure.json에 traceback 보존 |
| context·면적·기록 불일치 | `invalid_pair` |

`t`와 `Vmin`에서 한 지표만 유리하거나 Vmin bracket이 겹치면 `tradeoff_or_unresolved`다. Vmin의 tested upper-bound ratio는 계속 기록하되, bracket이 겹치는데 임계전압의 차이가 확정됐다고 분류하지 않는다. 좌측/우측 censoring은 탐색정보이지 곧바로 수치실패가 아니다. Vmin 탐색이 invalid라도 reference run이 수치적으로 유효하면 그 reference의 thermal/current 지표는 별도로 보존한다.

`ScreeningPolicy`의 네 gain margin은 기본 0이다. 이는 요청한 방향성 비교를 구현한 **nominal screening 기준**이지 실험적 유의차 기준이 아니다. 수치/제조/물성 불확실성을 확인한 후 `--policy policy.json`으로 margin을 지정할 수 있다. 분모 보호용 1e-9 K, 1e-12 및 roundoff epsilon은 수치 보호값이며 물리 threshold가 아니다. 소수의 미세한 개선을 확정적 성능향상으로 보고하지 않는다.

## 온도·전류 시간창에 관한 제한

이 패치는 기존 onset 동결/종료 규칙을 바꾸지 않는다. 그래서 RT screening은 두 case가 모두 비점화인 동일 full-horizon 구간에 제한했다. RJ는 기존 peak-to-evaluation-time 정의를 유지하므로 점화한 case에는 기존 lane 동결/종료 규칙이 그대로 적용된다. 서로 다른 onset 이후 시간을 새로 비교하는 지표를 만들어 넣지 않는다.

RT 개선은 Vmin 감소를 보장하지 않는다. Domain이 다른 pair의 ratio끼리 비교하여 보편적인 topology 순위를 만들지 않는다. 후보와 baseline의 local gap/width 분포까지 같다는 뜻도 아니며, 그 차이는 비교되는 geometry 차이의 일부다.

## 기존 E114 결과는 재계산하지 않고 분류

Pod 안에서:

```bash
cd /ECSP/ecsp-electrode-optimizer
source ./a100_env.sh

python python/run_paired_preflame_screening.py report \
  --results runs/e114_pcg6000_smoke/preflame/E114/result.json \
  --out runs/e114_pcg6000_smoke/paired_screening
```

기존 파일은 덮어쓰지 않는다. Legacy result에 초기온도/시간/전압 또는 실제 candidate 면적이 없으면 **같이 저장된** `resolved_physics_config.json`, `library/E114/mask.npz`, `baseline_parameters.json`을 읽어 확인한다. 필요한 증거가 없으면 임의로 298.15 K/260 V/2 s를 넣지 않고 `invalid_pair`로 보류한다.

사용자가 제공한 E114 scalar 값은 RT 약 0.713, RJ 약 0.231이다. 이는 `neither_ignited`, `thermal_screen=inferior`, `congestion_superior_candidates`에 해당한다. 두 ignition ratio는 null이다. 테스트의 E114 예제는 이 scalar 값에 synthetic validity/context fixture를 결합한 회귀시험이며 새 PDE 실측 결과가 아니다.

## 143개 기존 형상 + 현재 5개 준비

현재 main에는 원래 143개 v5 library의 binary/CAD 데이터가 없었다. 그 데이터를 이름으로 재구성하지 않는다. 별도 배포 파일 **ECSP_frozen143_geometry.zip**에는 v5의 accepted 143개 `metadata.json`, `mask.npz`, `master.json`이 byte-identical하게 들어 있다. 저장소의 `paired_catalogue_148.json`에 정확한 ID 목록과 허용 archive SHA-256을 기록했다. 기존 `ECSP_ImageMatched_v5.zip` 원본도 허용한다.

Mac에서 다운로드한 bundle을 한 번 전송:

```bash
kubectl cp ~/Downloads/ECSP_frozen143_geometry.zip \
  ecsp-pod:/ECSP/ECSP_frozen143_geometry.zip
```

Pod에서 기존 5개 library를 사용:

```bash
python python/run_paired_preflame_screening.py prepare \
  --base-archive /ECSP/ECSP_frozen143_geometry.zip \
  --five-library runs/five_topologies_a100_validation/library \
  --out runs/paired_catalogue_148

python python/run_paired_preflame_screening.py audit \
  --library runs/paired_catalogue_148/library \
  --out runs/paired_catalogue_148
```

`--five-library`를 생략하면 현재 main의 기존 다섯 fitter를 호출한다. 143개는 어느 경우에도 refit/resize하지 않는다. 기존 output 폴더는 덮어쓰지 않으므로 새 경로를 선택해야 한다. Prepare는 hash/ID와 다섯 추가 형상을 검사하며, 전체 CAD/raster 감사는 `audit`에서 수행한다. 실제 evaluate에서도 각 case를 다시 검사한다.

## 5개 또는 148개 preflame 실행

새 평가기는 NSGA-II/workflow.run을 호출하지 않고 기존 native evaluator와 독립 staggered generator만 호출한다. 기본은 명시적 CUDA 단일 lane이며 **서로 다른 context의 pair는 순차 실행**한다. CPU-first hybrid scheduler에 두 case를 넘겨 GPU가 비는 문제를 피한 것이다. 이 패치가 CPU 6 worker와 GPU를 동시에 최적 스케줄링한다고 주장하지 않는다.

먼저 5개:

```bash
python -u python/run_paired_preflame_screening.py evaluate \
  --library runs/five_topologies_a100_validation/library \
  --ids E058,E114,R038,R050,R091 \
  --config config/nsga2_bc_reactive_a100_cpu8_poweroff_200x3.yaml \
  --device cuda --pcg-max-iterations 6000 \
  --out runs/paired_five_validation
```

그다음 148개:

```bash
python -u python/run_paired_preflame_screening.py evaluate \
  --library runs/paired_catalogue_148/library \
  --config config/nsga2_bc_reactive_a100_cpu8_poweroff_200x3.yaml \
  --device cuda --pcg-max-iterations 6000 \
  --out runs/paired_148
```

`--ids`를 생략하면 기본적으로 정확히 148개가 있어야 하며, 5개만 있는데 전체 실행이 된 것처럼 조용히 넘어가지 않는다. 명시적 subset 또는 다른 catalogue는 `--ids` / `--expected-count`로 구분한다.

`--pcg-max-iterations 6000`은 이미 E114에서 사용한 반복 예산을 **runtime config 복사본**에 적용하는 명시적 옵션이다. 원본 config, PCG 코드, residual tolerance, time step, 전압범위, onset criterion, 반응계수는 변경하지 않는다. 옵션을 생략하면 제공한 config의 budget을 그대로 사용한다.

Resume는 동일 명령에 `--resume`을 추가한다. 이전 실행 예외 case까지 다시 시도하려면 `--resume --retry-failed`를 사용한다. source/config/geometry hash가 달라진 완료 case는 재사용하지 않는다. Screening policy만 바꾼 경우 raw PDE 결과로 재분류한다. CUDA/PCG 예외가 나면 그 case를 `execution_failed`로 보존하고 다른 pair를 계속 처리한다. 비점화로 위장하지 않으며 자동 tolerance 완화도 없다. `RUN_FINISHED.json`에 실행실패 개수가 기록되고, CLI 종료코드는 실행 예외가 있으면 1이다.

전체 job 재시작 전에 기존 프로세스가 없는지 확인한다. 새 CLI는 같은 output에 대한 동시 실행을 lock으로 거부한다. nohup으로 실행하려면 `python -u ... > runs/<new_log>.log 2>&1 &`를 사용하고, 로그 디렉터리는 미리 만든다.

## 출력

- `preflame/<ID>/raw_pair.json`: 원시 candidate/baseline 결과와 input signature.
- `preflame/<ID>/result.json`, `screening.json`: paired raw 결과 및 분류.
- `preflame/<ID>/staggered_mask.npz`, `baseline_parameters.json`: 실제 비교 기준선.
- `runtime_config.json`, `resolved_physics_config.json`: 실제 적용한 context 근거.
- `screening_summary.csv/json`, `screening_groups.json`, `screening_report.html`: ID순 표와 겹침 가능한 그룹. **성능순 정렬 아님**.
- `fixed_25mm_revalidation_manifest.json`: 후보 ID·선정 목적·25 mm 재검증 요구사항. 실제 CAD를 25 mm로 자동 축소하지 않는다.
- `protected_before.json`, `protected_after.json`: physics/solver/config 소스가 실행 중 바뀌지 않았는지 검증.

2단계 shortlist는 `ignition`, `thermal`, `congestion` 목적을 구분한다. baseline만 점화했지만 RJ가 개선된 형상을 ignition 우수 후보로 되살리지 않는다. 해당 형상을 전류균일성 후속연구로 검토할 수는 있다. 25×25 mm에서 2 mm 폭/3 mm gap을 보존할 수 있는지는 별도 설계·QC·계산이 필요하며 성공을 보장하지 않는다.

## 테스트

```bash
python -m pytest -q tests/image_design/test_paired_screening.py \
  tests/image_design/test_paired_workflow.py
```

순수 classification과 mock evaluator 계약, resume/중복방지, 148 count guard, frozen copy, legacy 재분류를 검사한다. 실제 A100 full PDE/148 pair 성능은 이 코드 작성 환경에서 실행하지 않았다. 기존 geometry나 physics 테스트의 통과를 새 paired E2E의 통과로 대신 주장하지 않는다.


## Representative field output (reference 260 V run only)

Paired screening now stores representative full fields for the candidate and its
same-context staggered reference. This is diagnostics/output only: no PDE,
material property, onset criterion, voltage-search semantics, C++/CUDA time
integration, or baseline geometry rule is changed.

- Non-igniting reference: evaluation-time snapshot (normally 2 s).
- Igniting reference: first-onset snapshot plus evaluation-time preflame state.
  The strict B/C lane freezes at first onset, so the evaluation snapshot is
  explicitly labelled as a frozen onset state rather than a new post-onset
  electrical solution.
- Vmin lower/bisection/final-verification trials remain scalar-only; field
  archives are not duplicated for those trials.

Each `preflame/<ID>/candidate/` and `staggered/` directory contains:

```text
fields_eval.npz
geometry_eval.png
temperature_eval.png
current_eval.png
potential_eval.png
progress_eval.png
overview_panel.png
field_manifest.json
```

If that lane ignites, it also contains `fields_onset.npz`,
`temperature_onset.png`, `current_onset.png`, `potential_onset.png`,
`progress_onset.png`, `geometry_onset.png`, and
`overview_onset_panel.png`.

The NPZ stores full-grid `temperature_K`, `potential_V`,
`global_progress`, `current_density_x_A_per_m2`,
`current_density_y_A_per_m2`, `current_density_magnitude_A_per_m2`,
`joule_heat_W_per_m3`, electrode masks, coordinates and snapshot time.
The PDE remains FP64; archives are compressed float32 copies for diagnostics
and visualization. Current density is not a proxy: it is evaluated by the
production `compute_current()` closure from the native solver's saved
state/potential and therefore retains the configured conductive and diffusion
current terms.

`representativeFieldMetrics` in the candidate/baseline raw result also stores
temperature standard deviation, P95-P05, temperature-rise CV, current-density
CV and current-density P99. These are diagnostic quantities and do not alter
the paired screening labels or objectives.
