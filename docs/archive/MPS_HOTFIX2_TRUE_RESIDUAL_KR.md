# 보관 문서 — 최종본 아님

> 이 문서는 중간 hotfix 기록입니다. 현재 실행·검증 기준은 `../MPS_FINAL_GAUGE_BALANCE_AND_RUNTIME_KR.md`입니다.

# MPS Hotfix 2 — BiCGStab true-residual convergence

## 문제
MPS FP32에서 BiCGStab의 재귀 residual이 실제 행렬 residual `b-Ax`보다 먼저 tolerance 아래로 내려가면서 반복이 조기 종료되었다. 최종 true residual 검증에서만 실패해, tolerance를 1e-5에서 2e-5로 완화하면 최종 residual도 약 2배 커지는 tolerance-tracking 현상이 발생했다.

## 수정
- BiCGStab half-step 수렴 후보는 `b-Ax` true residual로 재확인한다.
- 정기 convergence check 역시 true residual을 기준으로 한다.
- 재귀 residual은 수렴을 주장하지만 true residual은 실패하는 경우 해당 batch row만 true residual로 residual replacement/restart 한다.
- CPU/CUDA/MPS 모두 같은 수렴 정의를 사용한다.
- M2 MPS profile은 pilot 기준 `relativeToleranceStatic/Coupled=1e-5`, `absoluteTolerance=1e-6`로 설정한다.

## 의미
Tolerance를 계속 완화해서 실패를 숨기는 패치가 아니다. 설정된 tolerance가 실제 `||b-Ax||/max(||b||,1)`에 대해 만족되었을 때만 solver가 성공한다.

## 검증
- Python test suite: 18/18 pass
- CPU FP32 physics-path evaluation: pass
- 실제 Apple MPS 하드웨어는 배포 환경에서 검증 필요
