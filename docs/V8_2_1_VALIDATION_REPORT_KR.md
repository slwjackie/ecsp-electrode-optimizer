# ECSP v8.2.1 검증 보고서

## 기준과 판정 원칙

v8.2.1 결과는 v8.2.0의 통과 숫자를 재사용하지 않는다. 기준 archive는
`ECSP_v8_2_0_P0P1Fixed_BCNative_Hybrid_A100_CPU8.zip`이며 SHA-256은
`799cfd001a8e28143e16e33f32424a9545584f2c95691370943c804f210832a6`이다.

최종 builder는 manifest-bearing tree에서 아래 검증을 다시 실행하고 모든
항목이 통과한 경우에만 ZIP/checksum을 원자적으로 공개한다.

## v8.2.1 최종 통합 검증

| 항목 | 상태 | 합격 조건 |
|---|---|---|
| Python compile | **passed: 78 sources** | package Python 전 파일 compile |
| Bash syntax | **passed: 34 scripts** | tools 및 top-level launcher 전수 |
| pytest 전체 suite | **passed: 284, skipped: 2** | failure/error 0; 두 skip은 선언된 CUDA 장치 전용 시험 |
| Pillow `0xff` mask 경계 회귀 | **passed** | generation/resize/CUDA staging/native adapter/direct binding에서 canonical 0/1 bool 및 동일 geometry |
| overflow/PCG/error-boundary 회귀 | **passed** | non-finite update fail-closed, scaled residual, typed candidate error 경계 |
| C++ native extension | **passed on macOS arm64/Apple Clang** | compile, load 및 시험 실행 |
| C++ native standalone | **passed on macOS arm64/Apple Clang** | build, protocol, extension parity 및 pipeline 실행 |
| Python/native strict parity | **passed** | 동일 물리·strict tolerance field/history/handoff 비교 |
| native end-to-end debug | **passed: 28/28** | Pareto → handoff → propagation → staggered 완료 |
| v8.2.0 파일범위 감사 | **passed** | immutable ZIP hash/inventory 일치; 수정 19, 추가 7, 삭제 0, undeclared 0 |
| release SHA manifest | **atomic builder gate** | historical V8_2 보존, V8_2_1 누락·불일치·미등재 0 |
| ZIP CRC 및 fresh-extract | **atomic builder gate** | CRC와 추출 후 authoritative manifest 모두 통과 |

통합 검증은 2026-09-07에 새 출력 디렉터리
`/private/tmp/ecsp-v821-integrated-final`에서 수행했다. 위 결과는 CPU 소프트웨어
회귀 검증이며 물성 보정이나 실험 검증을 뜻하지 않는다. 최종 builder는 생성된
manifest를 포함한 정확한 release tree에서 같은 전체 검증을 다시 실행한다.

## A100/CUDA 상태

패키징 host에는 NVIDIA GPU, CUDA-enabled PyTorch 및 nvcc가 없다. 따라서 실제
CUDA extension compile/load, A100 FP64 runtime, CPU/CUDA full-field parity,
batch 32/64 불변성, CPU8/A100 동시 사용 및 production 성능은 현재 **통과로
기록하지 않는다**. CUDA source static check가 통과하더라도 실제 장치 검증을
대체하지 않는다.

대상 A100에서 production wrapper의 fail-closed device/functional preflight를
별도로 통과해야 한다. 실험적 물성·kinetics calibration도 별도 검증 대상이다.

## 재현 명령

~~~bash
ECSP_V820_BASELINE_ZIP=/path/to/ECSP_v8_2_0_P0P1Fixed_BCNative_Hybrid_A100_CPU8.zip \
ECSP_V821_PREPACKAGE=1 \
  bash tools/validate_v8_2_1_cpu.sh /new/path/v8_2_1_prepackage_validation
~~~
