# ECSP v8.2.1 maintenance patch

## 기준 릴리스

v8.2.1은 v8.2.0의 P0/P1 수정 물리와 native/CUDA/hybrid 성능 경로를 다시
구현하는 릴리스가 아니다. 다음 immutable archive를 직접 기준으로 변경 파일을
감사하는 patch release다.

- 기준 archive: `ECSP_v8_2_0_P0P1Fixed_BCNative_Hybrid_A100_CPU8.zip`
- SHA-256: `799cfd001a8e28143e16e33f32424a9545584f2c95691370943c804f210832a6`
- 기준 payload: 269 files
- 내장 기준 inventory: `docs/V820_BASELINE_MANIFEST.json`
- v8.2.1 변경 원장: `docs/V821_EXPECTED_CHANGES.json`

v8.2.0의 full-domain propellant, 별도 surface-contact footprint mask, footprint
전체 Butler–Volmer, 보존적 LP/PVA/species/heat bookkeeping, first-onset handoff,
zero-electrical post-onset propagation 계약은 v8.2.1에서도 기본 전제다.

## v8.2.1 변경 범위

이번 patch는 Pillow mode-1 mask가 NumPy에서 논리 dtype이면서 true storage byte는
`0xff`일 수 있다는 경계조건을 수정한다. Python에서는 generation과 resize에서
`uint8` 값 비교를 거쳐 소유권이 있는 0/1 bool 배열을 만들고, CUDA batch 입력
적재, native adapter와 직접 pybind entry point에서도 contact/full-domain mask를
다시 canonicalize한다. 따라서 Torch 전송과 C++ `bool*` 접근 어느 쪽에도 비표준
truth byte representation이 남지 않는다.

또한 CPU contact kernel은 state/property/contact/batch shape를 실행 전에 확인하고
ATen parallel task가 immutable parameter block을 값으로 소유한다. shared host/CUDA
FP64 finite check는 host에서 `std::isfinite`, device compilation에서 CUDA global
intrinsic을 사용해 libc++/nvcc 양쪽의 컴파일 계약을 분리한다. 물리식과 v8.2.0
P0/P1 의미는 바꾸지 않는다.

변경을 선언하지 않은 v8.2.0 파일의 byte hash가 달라지거나 파일이 추가·삭제되면
release validation은 실패해야 한다. 파일별 근거는
`docs/V821_EXPECTED_CHANGES.json`에 기록한다.

## 검증 및 A100 상태

v8.2.1의 CPU 전체 suite, native build, strict Python/native parity, pipeline
audit, v8.2.0 보존 감사 및 package manifest 검증은 아직 최종 release 결과로
기록하지 않는다. 최종 결과는 `docs/V8_2_1_VALIDATION_REPORT_KR.md`에 반영한 뒤
manifest-bearing tree에서 release validator가 다시 실행한다.

패키징 환경에는 NVIDIA GPU, CUDA-enabled PyTorch 및 nvcc가 없으므로 실제 CUDA
compile/runtime, A100 FP64, batch 32/64 불변성 및 CPU8/A100 hybrid 동작을 CPU
결과로 통과 처리하지 않는다. 대상 A100에서는 기존 fail-closed production
preflight를 별도로 통과해야 한다.

## 개발 중 pre-package 검증

출력 경로는 존재하지 않는 package 외부 경로여야 한다.

~~~bash
ECSP_V820_BASELINE_ZIP=/path/to/ECSP_v8_2_0_P0P1Fixed_BCNative_Hybrid_A100_CPU8.zip \
ECSP_V821_PREPACKAGE=1 \
PYTHON=/path/to/python \
  bash tools/validate_v8_2_1_cpu.sh /new/path/v8_2_1_prepackage_validation
~~~

## 최종 ZIP 생성

staging directory 이름은 반드시 다음과 같아야 한다.

~~~text
ECSP_v8_2_1_P0P1Fixed_BCNative_Hybrid_A100_CPU8
~~~

최종 validation metadata에서 모든 pending 항목을 실제 결과로 교체한 뒤 실행한다.

~~~bash
ECSP_V820_BASELINE_ZIP=/path/to/ECSP_v8_2_0_P0P1Fixed_BCNative_Hybrid_A100_CPU8.zip \
PYTHON=/path/to/python \
  bash tools/build_release.sh
~~~

builder는 v8.2.0 baseline hash 검증, v8.2.1 변경범위 감사, 전체 CPU/native
검증, `PACKAGE_SHA256_MANIFEST_V8_2_1.txt` 생성, ZIP CRC 및 fresh-extract
manifest 검사를 모두 통과한 경우에만 ZIP과 checksum을 공개한다. 역사적
`PACKAGE_SHA256_MANIFEST_V8_2.txt`는 payload에 그대로 남고 새 generic alias만
v8.2.1 authoritative manifest와 같아진다.
