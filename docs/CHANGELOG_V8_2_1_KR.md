# ECSP v8.2.1 변경 기록

## 릴리스 성격

v8.2.1은 immutable v8.2.0 P0/P1 수정본 위의 maintenance patch다. v8.2.0의
물리 의미나 v8.1에서 도입된 native/CUDA/hybrid/Vmin 성능 구조를 다시
baseline으로 삼지 않는다.

기준 archive와 해시는 다음과 같다.

- `ECSP_v8_2_0_P0P1Fixed_BCNative_Hybrid_A100_CPU8.zip`
- `799cfd001a8e28143e16e33f32424a9545584f2c95691370943c804f210832a6`
- 269 files, ZIP CRC 및 authoritative manifest 검증 통과

## 확정된 릴리스·감사 변경

- `VERSION.json`을 8.2.1과 새 release directory 이름으로 전환한다.
- v8.2.0 archive 전체 inventory를 `V820_BASELINE_MANIFEST.json`에 고정한다.
- v8.2.1 파일 변경만 `V821_EXPECTED_CHANGES.json`으로 허용한다.
- v8.2.0 historical authoritative manifest를 payload로 보존하고, v8.2.1의
  authoritative manifest는 `PACKAGE_SHA256_MANIFEST_V8_2_1.txt`로 분리한다.
- validator는 `ECSP_V820_BASELINE_ZIP`의 archive hash와 inventory가 모두
  일치해야 실행되며, pre-package mode는 현재 release의 생성 manifest 세
  항목만 유예한다.
- builder는 정확한 `ECSP_v8_2_1_P0P1Fixed_BCNative_Hybrid_A100_CPU8`
  staging name, 새 검증 보고서, v8.2.1 validator와 patch-level manifest를
  요구한다.

## 구현 patch

- Pillow mode-1 raster의 true 값이 raw `0xff` byte로 남을 수 있으므로 geometry
  생성과 nearest-neighbour resize에서 `uint8 != 0` 변환으로 canonical 0/1 bool
  storage를 만든다.
- batched CUDA solver도 anode/cathode mask를 stack하거나 Torch로 전송하기 전에
  `uint8 != 0` 값 비교를 적용해 Pillow `0xff` truth byte를 소유권이 있는 0/1 bool
  storage로 canonicalize한다.
- native adapter는 anode, cathode, fixed, propellant mask를 underlying byte 기준으로
  canonicalize하고 모든 mask의 device 일치를 확인한다.
- 직접 pybind 호출이 adapter를 우회해도 C++ entry point가 anode/cathode mask를
  byte 기준으로 canonicalize한 뒤 overlap/nonempty 검사와 geometry 생성을 한다.
- CPU contact operation은 state/property/contact/batch tensor 크기를 검증하고,
  ATen thread-pool closure가 immutable `Params` 값을 직접 소유하도록 한다.
- shared scalar finite helper는 host의 `std::isfinite`와 CUDA device의 global
  intrinsic을 compile path별로 분리한다.
- species/thermal update limiter는 모든 입력과 중간 계산의 유한성을 먼저
  검사하고, overflow/NaN을 clamp로 정상값처럼 숨기지 않고 기존 fail-closed
  상태 검사로 전달한다.
- potential solver의 잔차 norm은 scaled L2로 계산하고 모든 수렴 판정에 finite
  조건을 추가한다. 큰 유한 입력의 제곱이 `Inf`가 된 뒤 `Inf <= Inf`로 잘못
  수렴 처리되던 PCG/fallback/BiCG/direct/SOR 경로를 차단한다.
- NSGA-II native candidate 분류는 명시적인 `BCCandidateBatchError` 또는
  `BCCandidateGeometryError`만 후보 단위 실패로 처리한다. 일반
  `RuntimeError`의 메시지 substring을 근거로 메모리/수치 결함을 조용히 후보
  탈락으로 바꾸지 않는다.
- extension과 standalone이 native protocol version 8.2.1을 보고한다.
- 생성/resize/CUDA batch/adapter/direct binding의 `0xff` mask parity, host/CUDA
  helper source, CPU contact input validation, overflow fail-closed, PCG false
  convergence, typed-error boundary와 standalone version을 회귀시험한다.

이 변경은 v8.2.0의 물리식·objective·surface-contact footprint 의미를 변경하지
않고 언어/라이브러리 경계의 bool representation, native portability 및 기존
수치 실패의 fail-closed 처리를 강화한다.

## 검증 경계

v8.2.1 최종 CPU/native/parity/pipeline 결과는 아직 확정 전이다. 패키징
환경에서 실제 CUDA compile/runtime 또는 A100 검증은 수행되지 않았으며,
이 상태를 CPU 시험이나 CUDA source static check로 대체하지 않는다.
