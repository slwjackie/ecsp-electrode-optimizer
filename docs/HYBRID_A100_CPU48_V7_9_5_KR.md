# ECSP v7.9.5 — A100 FP64 + 48 vCPU 동시 활용 구조

## 1. 실행 구조

v7.9.5는 기존 CPU solver를 삭제하거나 대체하지 않는다. 다음 세 backend가 공존한다.

```text
cpp_fp64_cpu
└─ 기존 cpp/ecsp_cpp_solver.cpp 그대로 사용

hybrid_cuda_cpu
├─ A100: batched Torch/CUDA FP64 physics
└─ CPU: cpp/ecsp_cpp_solver_hybrid.cpp 독립 프로세스 pool
```

하나의 physics/Vmin trial queue를 A100과 CPU worker가 동시에 가져간다. 큰 wave에서는 A100이 최대 64개를 한 batch로 먼저 받고, 40개의 단일-thread C++ 프로세스가 다른 후보를 처리한다. GPU가 먼저 끝나면 남은 queue에서 다음 batch를 가져가고, CPU가 먼저 끝나면 CPU worker가 계속 work-steal한다.

```text
pending trials
     ├─ CUDA batch: 기본 64개
     └─ CPU pool: 기본 40개
```

48 vCPU 중 8개는 Python, CUDA host scheduling, 파일 I/O 및 OS에 남긴다. `sched_getaffinity()`로 실제 보이는 CPU가 더 적으면 CPU worker 수를 자동 감축한다.

## 2. 기존 CPU solver 보존

다음 두 파일은 byte-for-byte 동일하다.

```text
cpp/ecsp_cpp_solver.cpp
cpp/reference/ecsp_cpp_solver_v7_9_4_reference.cpp
SHA-256: 453168254960d20d93e0e52c4eac4f0638c391297baaa544691980acc0cb0e94
```

따라서 기존 `backend: cpp_fp64_cpu` 실행은 그대로 재현할 수 있다.

Hybrid CPU worker는 별도 파일 `cpp/ecsp_cpp_solver_hybrid.cpp`를 사용한다. GPU와 동일하게 explicit thermal/gas RHS를 한 timestep snapshot에서 계산하는 buffered update를 사용해, CPU와 GPU의 thread/order 차이를 제거한다. 원본 solver를 덮어쓰지 않는다.

## 3. A100에서 계산되는 항목

한 candidate의 전체 trial이 GPU memory에 상주한다.

- 전도도와 확산계수
- matrix-free surface-contact potential PCG
- Butler–Volmer / mass-transfer saturation
- Nernst–Planck species transport
- passivation 및 gas coverage
- Joule/계면/화학 발열
- 열전도와 분해 진행도
- 점화 판정과 네 목적함수용 reduction

각 timestep마다 CPU↔GPU 복사를 하지 않는다. 시작 시 mask/config를 한 번 올리고 마지막 scalar metrics만 회수한다.

## 4. Vmin 탐색

한 후보 내부의 이분탐색은 순차지만, 같은 wave에 있는 여러 후보는 동시에 처리한다.

```text
wave 0: 각 후보의 lower bound trial
wave 1: 각 후보의 첫 midpoint trial
wave 2: 후보별 서로 다른 midpoint trial
...
```

각 wave도 동일한 GPU/CPU shared queue로 처리되므로 A100과 CPU pool이 동시에 사용된다. Wave가 작아져도 기본 16개를 CPU용으로 남기고 나머지를 GPU에 보낸다. 후보 수가 8개 미만인 마지막 tail에서는 한 장치가 잠시 유휴일 수 있으며, 이는 의존성 있는 이분탐색 구조상 피할 수 없다.

## 5. 장애 격리

- GPU batch OOM/런타임 오류: batch를 절반으로 줄여 1회 재시도
- 최소 batch에서도 실패: 해당 trial을 C++ CPU worker로 재queue
- GPU candidate 수치 실패: 해당 candidate만 CPU로 재평가
- CPU candidate 실패: 기존처럼 `physics_rejection.txt` 기록
- 원본 CPU solver는 reference backend로 계속 유지

GPU 장애가 전체 generation을 중단시키지 않는다.

## 6. 기본 48-vCPU profile

```yaml
optimization:
  physics_batch_size: 192

evaluator:
  backend: hybrid_cuda_cpu
  hybrid_cpu_worker_cases: 40
  hybrid_reserved_host_vcpus: 8
  hybrid_cuda_batch_size: 64
  hybrid_cuda_min_batch_size: 8
  hybrid_cpu_reserve_per_wave: 16
  hybrid_cuda_fallback_to_cpu: true
```

최적값은 workload에 따라 달라진다. 반드시 preflight 후 benchmark를 실행해 `cuda_batch_size=32/64`, CPU worker `36/40/42`을 비교해야 한다.

## 7. 실행

```bash
bash tools/run_a100_cpu48_hybrid_preflight.sh \
  "$PWD/runs/a100_cpu48_preflight"
```

성공 후:

```bash
ECSP_SKIP_HYBRID_PREFLIGHT=1 \
CONFIG="$PWD/config/experiments/d20_multi_up_to4_a100_cpu48_hybrid.yaml" \
bash tools/run_a100_cpu48_hybrid.sh \
  "$PWD/runs/d20_multi_500x4_hybrid" 500 4
```

모니터링:

```bash
bash tools/monitor_a100_cpu48_hybrid.sh 10
```

## 8. 검증 한계

현재 제작 환경에는 A100이 없어 실제 CUDA end-to-end 실행시간과 utilization은 검증하지 못했다. 다음은 검증했다.

- 기존 CPU source 보존
- C++ 원본/Hybrid C++ warning build
- Torch CUDA equation path의 CPU FP64 test mode
- Hybrid C++와 Torch backend의 short-horizon 목적값 일치
- dynamic scheduler와 Vmin wave logic 단위시험
- GPU 실패의 CPU fallback 경로

A100에서는 preflight가 CPU와 GPU를 한 stage에서 모두 사용하고, 동일 mask의 전류·에너지 일치 gate를 통과해야 production을 시작한다.
