# v7.9.5 변경내역

- `python/ecsp_cuda/solver.py`: batched CUDA FP64 surface-contact physics backend 추가
- `python/ecsp_nsga2/hybrid.py`: A100/CPU shared work queue와 dynamic scheduler 추가
- `python/ecsp_nsga2/evaluator.py`: `hybrid_cuda_cpu` backend routing 추가
- `cpp/ecsp_cpp_solver.cpp`: **변경 없음**
- `cpp/reference/ecsp_cpp_solver_v7_9_4_reference.cpp`: 원본 CPU source 복제 보관
- `cpp/ecsp_cpp_solver_hybrid.cpp`: GPU와 동일한 buffered explicit update를 쓰는 hybrid CPU worker 추가
- `python/make_a100_cpu48_hybrid_config.py`: 기존 실험 YAML을 hybrid profile로 변환
- `tools/run_a100_cpu48_hybrid*.sh`: preflight, production, benchmark, monitor 추가
- A100 batch OOM/오류 시 batch 축소 및 C++ CPU fallback 추가
- 48 vCPU profile: CPU worker 40, host reserve 8, CUDA batch 64, workflow batch 192
- Vmin wave도 CPU/GPU가 함께 처리하도록 task queue 통합
- CUDA PCG/local BV/outer Robin/time loop의 불필요한 host synchronization 완화
- GPU와 CPU 결과에 실제 backend/device/revision 기록

- 환경변수 override 추가: `ECSP_HYBRID_CPU_WORKERS`, `ECSP_HYBRID_HOST_RESERVE`, `ECSP_HYBRID_CUDA_BATCH_SIZE`, `ECSP_HYBRID_CUDA_MIN_BATCH_SIZE`, `ECSP_HYBRID_CPU_RESERVE_PER_WAVE`
- 48-vCPU 기본 profile을 CPU worker 40 + host reserve 8 + small-wave CPU reserve 16으로 조정
