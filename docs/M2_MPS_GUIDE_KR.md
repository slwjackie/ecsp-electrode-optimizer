# v7.9.0 M2 실행 안내 — C++ CPU FP64 사용

v7.9.0의 corrected surface-contact physics는 **C++ CPU FP64** backend에 구현되어 있다. 기존 PyTorch/MPS 직접 물리경로는 전극 mask를 추진제 hole로 해석하는 legacy formulation이므로 production에서 비활성화했다.

사용할 설정:

```text
config/nsga2_condensed_phase_no_f_m2_cpp_fp64.yaml
```

필수 검증:

```bash
bash tools/run_m2_cpp_fp64_preflight.sh "$PWD/runs/v790_preflight"
bash tools/validate_surface_contact_cpp.sh "$PWD/runs/v790_surface_validation"
```

100개 × 2세대:

```bash
bash tools/run_m2_cpp_fp64_100x2.sh "$PWD/runs/v790_100x2"
```

응축상 물리계산에는 Metal/MPS가 사용되지 않는다. 생성 AI 등 별도 PyTorch 모듈에서 MPS를 사용하는 것은 가능하지만, Python↔C++ 간 timestep별 tensor 이동은 하지 않는다.
