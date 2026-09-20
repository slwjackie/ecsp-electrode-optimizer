> **보관 문서:** 현재 구현은 `../MPS_ROBUST_SPD_PCG_V7_7_5_KR.md`를 기준으로 합니다.

# v7.7.4 MPS stability hotfix

실제 M2 Pro preflight에서 v7.7.3의 MPS 전용 geometric multigrid preconditioner가 193×193 자유형 mask에서 BiCGStab true residual을 안정적으로 낮추지 못하는 문제가 확인되었다.

v7.7.4에서는 M2/MPS 프로필만 다음처럼 변경한다.

- potential method: PCG
- preconditioner: Jacobi
- FP32 relative tolerance: 1e-5 유지
- PCG convergence: recursive residual이 아니라 true residual `b-Ax`를 기준으로 판정
- recursive/true residual drift가 확인되면 해당 batch row만 residual replacement 후 PCG direction을 restart

CPU/A100 production 프로필의 multigrid/BiCGStab 설정은 변경하지 않는다.

동일한 193×193, 4개 대표형상, 1 timestep, FP32 CPU reference에서 4/4 physics 성공 및 rejection 0을 확인했다. 실제 Apple MPS 장치는 이 빌드 환경에 없으므로 M2에서 `tools/run_m2_mps_preflight.sh`를 한 번 실행해 최종 확인한다.

MPS에서 `index_put_with_accumulate_mps does not have a deterministic implementation` 경고가 발생할 수 있다. 이는 PyTorch MPS backend의 결정론 지원 한계에 대한 warning이며 physics rejection 원인은 아니다. 완전 bitwise reproducibility가 필요한 최종 검증은 CPU/A100 FP64에서 수행한다.
