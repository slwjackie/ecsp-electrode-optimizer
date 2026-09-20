# ECSP v7.9.5 Hybrid A100 + 48 vCPU 검증 보고서

## 1. 패키징 환경에서 완료한 검증

- Python test suite: **52/52 통과**
- Python byte-code compile: **51개 파일 통과**
- 전체 shell script `bash -n`: 통과
- C++17 warning build:
  - `cpp/ecsp_cpp_solver.cpp`: 통과
  - `cpp/ecsp_cpp_solver_hybrid.cpp`: 통과
- Hybrid C++ AddressSanitizer/UndefinedBehaviorSanitizer short-horizon 실행: 통과
- 기존 CPU source 보존:
  - `cpp/ecsp_cpp_solver.cpp`
  - `cpp/reference/ecsp_cpp_solver_v7_9_4_reference.cpp`
  - SHA-256 둘 다 `453168254960d20d93e0e52c4eac4f0638c391297baaa544691980acc0cb0e94`

## 2. Torch/CUDA 방정식 경로의 CPU FP64 test mode 대 C++ hybrid 비교

실제 A100이 없는 환경에서도 동일 Torch tensor 방정식 경로를 CPU FP64로 실행하여 buffered C++ 해석기와 대조했다. 이는 CUDA kernel 성능 검증은 아니지만 방정식·배열 배선·출력 정의의 회귀검증이다.

### 193×193, one-step, 극당 약 20% contact

| 항목 | C++ hybrid | Torch tensor path | 상대차 |
|---|---:|---:|---:|
| peak current | 0.003827224106 A | 0.003827224622 A | 1.35e-7 |
| evaluation energy | 2.487695669e-4 J | 2.487696004e-4 J | 1.35e-7 |
| current congestion | 1.865695650 | 1.865695643 | 3.59e-9 |
| peak temperature | 298.150602645 K | 298.150602644 K | 3.83e-12 |
| undecomposed fraction | 1.0 | 1.0 | 0 |
| potential iterations | 156 | 156 | 동일 |

두 경로의 final potential relative residual은 각각 약 `8.19e-10`, `8.23e-10`이었다.

### 193×193, 10-step

- peak current 상대차: `2.85e-7`
- 누적 입력에너지 상대차: `2.85e-7`
- current congestion 상대차: `4.95e-9`
- peak temperature 상대차: `1.11e-10`
- species/temperature/gas/chemical cap fraction: 두 경로 모두 동일

## 3. Scheduler 및 Vmin 제어 검증

- 192 tasks: CUDA가 먼저 64개를 취하고 나머지는 CPU pool/queue에 유지
- 40 tasks, small-wave reserve 16: CUDA 24개 + CPU용 16개
- GPU row failure: 해당 trial만 CPU queue로 재삽입
- GPU batch failure/OOM: batch 절반 축소 후 재시도, 최종적으로 CPU fallback
- Vmin lower-bound/bisection wave: shared hybrid scheduler를 통해 실행
- `minimumIgnitionVoltage` bracket, censoring 및 결과 JSON 저장: 단위시험 통과
- CPU-only fallback smoke: 4/4 physics 성공, rejection 0

## 4. 실제 A100에서 반드시 통과해야 하는 gate

패키징 환경에는 NVIDIA GPU가 없으므로 다음 명령을 A100 node에서 먼저 실행해야 한다.

```bash
bash tools/run_a100_cpu48_hybrid_preflight.sh \
  "$PWD/runs/a100_cpu48_preflight"
```

성공 조건:

- `all_physics_successful = true`
- `gpu_used = true`
- `cpu_used = true`
- GPU/CPU 모두 `float64`
- 동일 분포의 one-step mask에서 peak current 상대차 ≤ `5e-4`
- 입력에너지 상대차 ≤ `5e-4`

## 5. 아직 미검증인 항목

- 실제 A100 kernel utilization과 power draw
- CUDA batch 32 대 64의 실측 throughput
- 2초 full-horizon에서 CPU/GPU 장기 누적 오차
- 48-vCPU cgroup 환경의 sustained CPU throughput
- 200×3 및 500×4 실제 wall time

따라서 v7.9.5는 **CPU·방정식·scheduler 회귀시험을 통과한 A100 실행 후보**이며, A100 preflight를 통과한 뒤에만 production 결과를 사용해야 한다.
