# 현재 배포: ECSP v8.4.2 — GeometrySafe / Power-OFF

**현재 실행은 `README_V8_4_2_KR.md`를 참조하십시오.** 5 mm 폭 상한을 유지하고, BC 생산 면적 tolerance를 1%로 강화했으며, 20×50 유효·고유 형상 확인 전 후보 해석을 시작하지 않습니다. 현재 전원 OFF 명령은 `bash tools/run_bc_reactive_a100_cpu8_poweroff.sh "$PWD/runs/v842_poweroff"`입니다.

아래 v8.4.1 설명은 상속한 GPU/물리 모델의 역사적 설명입니다. 현재 시험 결과는 `VALIDATION_V8_4_2_KR.md`, 현재 manifest는 `PACKAGE_SHA256_MANIFEST_V8_4_2.txt`를 따릅니다.

---

# ECSP v8.4.1 — A100 + CPU 8-core / BC → 2-channel Reactive

## 현재 구현과 검증 상태

v8.4.0 직접 기반. **신규 응축상 12변수 Reactive를 FP64 Torch 배치 연산으로 이식**했습니다. NumPy 기준 해석기, 기존 condensed propagation, 독립 논문 reference는 유지합니다. 생산 물성·반응계수·격자·물리시간·최적화 예산을 줄여 얻은 속도 개선이 아닙니다.

**빌드 환경에는 CUDA 장치가 없습니다.** CPU NumPy ↔ 동일 Torch FP64 연산, 전체 파이프라인, 후보별 독립 시간간격, 질량/에너지/재고 수지를 시험했습니다. **A100 성능·CUDA 실행·CUDA Graph capture 성공은 이 배포를 만든 환경에서 검증하지 못했습니다.** 제공 launcher가 실제 A100에서 필수시험을 통과해야 본 계산을 시작합니다. CUDA 요청 실패를 CPU 결과로 바꾸지 않습니다.

전체 시험: **554 passed / 5 skipped / 0 failed**. 5 skips는 기존 native CUDA 2개와 신규 CUDA 3개입니다. 실험적 물성 검증은 하지 않았습니다. 이론 모델의 정확성과 코드 시험 통과는 다릅니다.

## 실행

압축을 푼 디렉터리에서 실행합니다. 기존 CUDA-enabled PyTorch/컴파일 환경을 유지하고 부족한 의존성만 설치하십시오. CPU-only Torch를 새로 설치한 상태에서는 A100 preflight가 거부합니다.

```bash
python -m pip install -r python/requirements.txt pytest

# GPU 없이 전체 연결 확인용: 합성/단축 DEBUG 물리, 연구 순위로 사용 금지
CONFIG="$PWD/config/nsga2_bc_reactive_tensor_cpu_debug.yaml" \
bash tools/run_bc_reactive.sh "$PWD/runs/v841_cpu_debug"

# A100 + CPU8. 기본: 기존 baseline을 추천기준으로 유지하고 GPU Reactive와 비교
bash tools/run_bc_reactive_a100_cpu8.sh "$PWD/runs/v841_dual"

# 신규 Reactive 결과를 잠정 최종 추천기준으로 사용하는 명시적 선택
CONFIG="$PWD/config/nsga2_bc_reactive_a100_cpu8.yaml" \
bash tools/run_bc_reactive_a100_cpu8.sh "$PWD/runs/v841_reactive"
```

기존 `RUN_NSGA2_ONLY.sh` 기본 경로는 바꾸지 않았습니다. 원래 실행기를 쓴다고 새 GPU Reactive가 자동 선택되지 않습니다. 기존 output/log/preflight 폴더가 있으면 덮어쓰지 않고 거부합니다.

A100 launcher는 매번 새 경로에 **장치 identity → native BC CUDA compile/parity/hybrid → 신규 GPU 보존·동적파동·graph·실제 BC/NP/BV → native BC→GPU Reactive+CPU baseline 전체 연결**을 검사합니다. 필수 GPU test가 skip되면 실패입니다. `PREFLIGHT_COMPLETE.json`에는 소스 SHA-256, 장치, 시험명이 묶입니다. 이 검사는 초단기 기능 검사이며 생산 1,000×5 계산시간 벤치마크가 아닙니다.

## 새 실행 설정

```yaml
post_onset:
  backend: reactive_euler              # 물리모델 선택; baseline은 condensed_propagation
  compare_backends: true
  allow_experimental_ranking: true     # 실험검증 완료라는 뜻 아님
  execution:
    backend: torch_batch              # 연산 구현 선택; numpy_cpu도 보존
    device: cuda
    batch_size: 8                     # 시작값, A100 실측 최적값 아님
    cuda_graph: false                 # 선택적 graph 프로필은 별도
    cpu_budget: 8
    host_reserve: 2
    cpu_workers: 6
    torch_threads: 1
    maximum_batch_working_bytes: 8589934592
    maximum_batch_history_bytes: 8589934592
    oom_split_retry: true
```

`post_onset.backend`는 물리 모델, `post_onset.execution.backend`는 CPU/Torch 구현을 선택합니다. 실행설정이 없으면 기존 v8.4.0 경로입니다. 과거 `propagation_refinement.execution`은 새 execution block이 없는 fallback 경로용입니다.

CPU는 실제 affinity/cgroup/quota에 맞춰 요청 8개 이하로 줄입니다. 기본 최대 6개 single-thread spawned worker가 **독립 baseline 비교 또는 NumPy Reactive 후보**를 처리하고, 주 프로세스가 GPU 배치·입출력·스케줄을 담당합니다. CPU6+GPU가 같은 한 후보의 격자를 분할하는 구조가 아닙니다. pre-onset native hybrid의 CPU 풀은 단계 종료 후 닫히므로 post-onset CPU 풀과 중복시키지 않습니다.

## GPU 경로에 이식한 연산

- 12개 보존변수 `[ρ,ρu,ρv,ρE,ρα1,ρα2,c+,c−,water,PVA,product_water,ec_LP]`의 장과 LUT를 device에 유지.
- Fourier 열전도, cp 적분/역변환, 두 채널 Arrhenius/공유재고 제한, WENO5-JS/HLLC/SSPRK33, 수지 검사 모두 FP64 tensor 연산.
- 후보별 `dt`, 거부/재시도, 완료 mask를 독립 관리. 가장 stiff한 후보의 dt를 다른 후보에 강제하지 않음.
- 매 trial의 작은 후보별 진단표만 CPU로 복사. 격자장 다운로드는 snapshot/최종 출력 때. NP/BV ON에는 기존 반복 수렴 동기화가 남음.
- 실제 OOM 또는 계획 메모리 초과 시 동일 장치에서 batch를 분할. 격자, 시간, LUT, precision, model을 바꾸지 않음. 단일 후보도 감당할 수 없으면 실패.
- 선택적 CUDA Graph: `config/nsga2_bc_reactive_a100_cpu8_graph.yaml`. `U/dt/retry`를 static input buffer로 복사해 replay. **전원 ON의 iterative NP/BV는 graph=false만 허용**.

**작은 격자에서는 GPU가 빠르다는 보장이 없습니다.** 여기서 Torch CPU 경로는 GPU 연산식의 대조용이며, 실제 CPU 벤치마크에서 NumPy보다 느린 경우도 있었습니다. A100 kernel-only benchmark는 다음과 같습니다.

```bash
PYTHONPATH=python python python/benchmark_reactive_backends.py \
  --device cuda --grids 32,96,193 --batch-sizes 1,4,8 \
  --steps 10 --repeats 3 --output runs/reactive_a100_kernel.json

# graph 및 동적 역학은 각각 별도 출력으로 비교
PYTHONPATH=python python python/benchmark_reactive_backends.py \
  --device cuda --cuda-graph --grids 32,96,193 --batch-sizes 1,4,8 \
  --steps 10 --repeats 3 --output runs/reactive_a100_graph_kernel.json
```

이 benchmark는 합성장, RHS/SSPRK trial만 측정하고 NP/BV·파일 I/O·선별을 제외합니다. 결과를 전체 NSGA 속도향상 배수로 해석하지 마십시오. dynamic mode는 `--dynamic`을 추가합니다. 배치 wall time은 후보별 파일에 반복 기록되므로 후보 시간을 합산해 GPU 총시간으로 사용하면 안 됩니다.

## 보존한 물리와 한계

**Tait 압력은 p(ρ)입니다. 균일 밀도·정지 BC handoff에서 열원만으로 압력구동 유동을 만들지 못합니다.** 이때 HLLC stationary fast path는 이 모델의 정확한 불변부분공간이며 임의 감쇠가 아닙니다. 열전도/반응은 계속 계산됩니다. HLL 수치flux는 정지 접촉도 확산하므로 이 shortcut을 쓰지 않습니다.

`level-set`은 `X=w1α1+w2α2`의 내부 반응경계를 표현합니다. 외곽 추진제 질량 제거, 기체 질량주입, 반응물/생성물 EOS 분리, 점성유동, 고체 전단/탄성 해석을 추가하지 않았습니다. `Q1,Q2`는 BC와 같은 initial-bulk 기준이고 w를 다시 곱하지 않습니다. 분해량 제한과 화학발열 제한은 동일한 accepted increment를 씁니다.

## 자료

- `docs/PHYSICS_EQUATIONS_ASSUMPTIONS_V8_4_1_KR.md`: 정립된 식/변형/진단의 구분, 40개 가정.
- `docs/PARAMETERS_V8_4_1_KR.md`: 578개 resolved 물리·운영 입력, 22행 반응 LUT, 4개 계면 채널.
- `docs/VALIDATION_V8_4_1_KR.md`: 발견/수정, 시험, 미검증 범위.
- `docs/audit/`: 전체 YAML 8,520행, 코드 기본값 2,086행, Python 수치리터럴 7,845행 및 파일/함수 인덱스.

표의 `literature_nominal`은 **현재 시료로 보정된 값**을 뜻하지 않습니다. 기존 disabled proxy와 신규 활성 식을 혼동하지 않도록 별도 분류했습니다.
