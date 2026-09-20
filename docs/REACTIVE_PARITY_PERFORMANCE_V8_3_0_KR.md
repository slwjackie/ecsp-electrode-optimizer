# ECSP v8.3.0 reactive parity 및 성능 범위

## 결론

새 reactive solver에 대해 검증 가능한 parity 대상은 현재 **같은 단일-process
CPU FP64 구현의 반복 결정성 및 수치 reference 내부 비교**뿐이다. integrated MPI,
reactive CUDA/sm_80/A100와 batch backend가 없으므로 CPU–MPI, CPU–GPU,
batch 1/32/64 parity 또는 A100 성능을 주장하지 않는다.

판정은 **`NOT FULLY VERIFIED`** 이며 `geometry_ranking_eligible=false`다.

## 기존 v8.2.1과 새 v8.3 경로의 분리

| 경로 | 구현 상태 | 이 문서에서의 의미 |
|---|---|---|
| v8.2.1 BC-native C++17 FP64 CPU | 보존 | parent regression 대상으로만 취급 |
| v8.2.1 CUDA FP64 batch 32/64 | 보존 | legacy evidence이며 reactive solver parity가 아님 |
| v8.2.1 CPU+A100 hybrid / batched `V_min` | 보존 | legacy 최적화 경로; v8.3 reactive time loop와 무관 |
| v8.3 reactive CPU FP64 | 구현 | 단일-process experimental reference |
| v8.3 MPI | row partition/3-row halo 보조 API만 구현 | solver time loop 미통합, rank 실기 parity 없음 |
| v8.3 reactive CUDA/A100 | 미구현 | compile/runtime/parity/throughput 모두 측정 대상 없음 |

기존 CUDA 코드가 release tree에 존재한다는 사실이나 v8.2.1 batch 시험 결과를 새
reactive backend의 구현·parity·성능 증거로 재사용하면 안 된다.

## 수행 가능한 parity/결정성 검사

| 비교 | 목적 | 제한 |
|---|---|---|
| 동일 입력 CPU FP64 반복 | JSON/NPZ 주요 배열·metric 결정성 | 한 host/NumPy/BLAS/runtime 범위 |
| grid/time refinement | 선택한 scheme의 synthetic observed behavior | 논문 case raw solution과 비교 아님 |
| analytic/manufactured component tests | EOS, flux, advection, conduction, source bookkeeping | fully coupled ECSP solution 검증 아님 |
| serial row-halo helper comparison | halo packing/fill API 계약 | distributed stage ordering·collectives·global budgets 검증 아님 |
| legacy parent audit | v8.3 추가가 허용된 파일 밖 v8.2.1 payload를 바꾸지 않는지 확인 | 물리 예측 parity가 아니라 파일/계약 보존 |

50-repeat determinism을 실행하더라도 그것은 periodic/source-controlled reference의
동일 환경 재실행 일치일 뿐, MPI decomposition invariance, GPU parity 또는 서로
다른 compiler/hardware의 bitwise reproducibility를 의미하지 않는다.

## 수행하지 못한 parity matrix

| 비교 | 상태 | 완료에 필요한 조건 |
|---|---|---|
| CPU rank 1 ↔ MPI rank 2/4/8 | `NOT_IMPLEMENTED_NOT_TESTED` | MPI-integrated solver, global reductions, boundary/corner ordering과 mpi4py/mpiexec host |
| CPU FP64 ↔ CUDA FP64 | `NOT_IMPLEMENTED_NOT_TESTED` | 동일 reactive equations/numerics/source contract의 CUDA backend |
| CPU ↔ sm_80 compile/runtime | `NOT_IMPLEMENTED_NOT_TESTED` | nvcc, CUDA-enabled runtime와 device gate |
| batch 1 ↔ 32 ↔ 64 | `NOT_IMPLEMENTED_NOT_TESTED` | reactive batch API와 state-isolation test |
| CPU ↔ A100 result parity | `NOT_IMPLEMENTED_NOT_TESTED` | NVIDIA A100 실기, tolerance 사전등록과 전체 field/budget comparison |
| A100 throughput/VRAM | `NOT_IMPLEMENTED_NOT_TESTED` | warm-up, repeat, synchronization, peak-VRAM 측정 protocol |
| Linux/GCC ↔ macOS/Clang | 최종 교차-host evidence 없음 | 고정 dependency/compiler matrix와 artifact hash 기록 |

## 성능 주장 범위

현재 구현은 정확성과 진단을 우선한 NumPy CPU reference이며 다음 비용을 포함한다.

- 각 accepted step마다 최대 세 SSPRK stage와 retry trial
- 각 stage의 x/y primitive WENO reconstruction, conservative face 복원과 HLL
- Eq.(1) reaction/electrical source, Eq.(2) conduction/decomposition 및 level-set RHS
- callback field와 metadata의 immutable snapshot/hash/JSON validation
- step/front/source history 및 conservation accumulator
- deterministic NPZ를 위한 in-memory artifact staging

따라서 v8.2.1 native/A100 backend보다 빠르다고 추론할 근거가 없고, 현 단계의
wall-clock 측정을 production throughput으로 외삽하면 안 된다. 최종 evidence 전에는
절대 시간, speedup, cells/s, steps/s, peak RSS/VRAM 숫자를 문서에 고정하지 않는다.

공정한 성능 측정에는 최소한 다음이 필요하다.

1. source tree와 config/effective provenance hash 고정
2. grid, end time, accepted/rejected step와 fallback 수 고정
3. CPU model, physical/logical core 수, thread/affinity와 BLAS 환경 기록
4. warm-up과 여러 번의 synchronized 반복, median/dispersion 보고
5. solver 생성, callback, serialization 포함/제외 시간을 구분
6. peak RSS와 history/working-set planning estimate를 별도 보고
7. CUDA 구현 후 compile arch, driver/runtime, A100 identity와 peak VRAM 기록
8. 성능 비교 전에 field/budget parity tolerance를 사전 정의하고 통과

## resource planning은 성능/메모리 측정이 아님

history estimator와 working-set estimator는 oversized request를 일찍 거부하는
versioned planning guard다. prescribed metadata는 recursive object size와 canonical
JSON bytes를 측정하고, callback record는 runtime에 재측정한다. working-set은
격자당 scratch allowance와 history/NPZ staging budget을 포함한다.

그러나 이 값은 운영체제 RSS, NumPy allocator fragmentation, Python interpreter,
제3자 library workspace, memory mapping 또는 순간 peak의 upper bound를 증명하지
않는다. “preflight 통과”를 “OOM 불가능” 또는 “메모리 최적화 완료”로 표현하면
안 되며, 실제 target host에서 peak-memory profiling이 필요하다.

## 형상 최적화와의 관계

성능이나 반복 결정성이 확보돼도 다음 물리 간극 때문에 상대 전극 형상 순위를
산출할 수 없다.

- transported phi가 phase/EOS/물성/interface jump를 구동하지 않는다.
- Eq.(1)과 Eq.(2)가 전체 격자에서 독립이며 전기 열은 Eq.(1)에만 들어간다.
- electrical callback은 self-attested이고 concrete potential convergence/current
  balance/BV residual gate가 없다.
- front metric은 fixed `+x` ray 진단이며 true local normal이 아니다.
- finite conservation residual tolerance가 교정되지 않았다.
- 논문의 case-closing 입력과 실험/raw numerical data가 공개되지 않았다.

따라서 parity/performance 최적화보다 먼저 physical closure와 검증 지표를 닫아야
하며, 현재 release metadata의 ranking hard gate를 우회해서는 안 된다.

