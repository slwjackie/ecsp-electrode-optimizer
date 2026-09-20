# 보관 문서 — 최종본 아님

> 이 문서는 중간 hotfix 기록입니다. 현재 실행·검증 기준은 `../MPS_FINAL_GAUGE_BALANCE_AND_RUNTIME_KR.md`입니다.

# MPS multigrid restriction hotfix — 2026-08-30

## 재현된 오류

Apple MPS에서 응축상 physics potential multigrid가 다음 오류로 후보를 거부할 수 있었습니다.

```text
RuntimeError: Adaptive pool MPS: input sizes must be divisible by output sizes.
```

원인은 `python/ecsp_v6/physics/potential.py::_mg_restrict()`의
`F.interpolate(..., mode="area")`입니다. PyTorch MPS가 이를 adaptive average pooling으로
구현하며, 본 solver의 193 -> 97 -> 49 ... odd-grid hierarchy는 입력/출력 크기가 정수배가
아니어서 실패했습니다.

## 수정

- CPU/CUDA: 기존 `mode="area"` restriction 유지 (FP64 reference 경로 불변)
- MPS: GPU-resident 3x3 full-weighting restriction 사용
  - stencil = (1/16) [[1,2,1],[2,4,2],[1,2,1]]
  - stride 2, padding 1
  - 193 -> 97 -> 49 ... hierarchy를 CPU round-trip 없이 처리
- 미래의 비표준 hierarchy에 대해서만 CPU area-interpolation fallback 제공

## 검증

- MPS hierarchy shape regression: 통과
- CPU historical area restriction exact regression: 통과
- 전체 Python test suite: 18/18 통과

현재 CI/Linux 환경에는 Apple MPS device가 없으므로 실제 M2 GPU end-to-end 실행은
Mac에서 재검증해야 합니다. 기존 실패 workdir에는 rejected result가 남아 있으므로 새 workdir으로 실행하십시오.
