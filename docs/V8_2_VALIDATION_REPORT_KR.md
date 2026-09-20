# ECSP v8.2.0 검증 보고서

## 판정 원칙

이 문서는 v8.2.0 최종 통합 tree에 대해 실제로 실행한 결과만 기록한다.
v8.1.0의 80 passed / 2 CUDA skipped 기록은 역사적 기준이며 v8.2 통과
숫자로 재사용하지 않는다. 2026-09-06 pre-package 통합 검증의 완료 증거는
외부 검증 디렉터리에 보관한다. 최종 builder는 manifest가 생성된 정확한 tree에서
같은 gate를 다시 실행하고 `VALIDATION_COMPLETE.json`에 증거 파일별 hash를
기록하며, 모든 검사가 통과한 경우에만 ZIP과 checksum을 원자적으로 공개한다.

## 최종 통합 검증

| 항목 | 상태 | 합격 조건 |
|---|---|---|
| Python compile | **PASS — 77 sources** | package Python 전 파일 compile |
| Bash syntax | **PASS — 33 scripts** | tools 및 top-level launcher 전수 |
| pytest 전체 suite | **PASS — 256 collected, 254 passed, 2 skipped** | failure/error 0; skip은 GPU 부재로 실행할 수 없는 선언된 CUDA-only test 두 개뿐 |
| C++ native extension | **PASS** | 현재 host에서 hash-keyed compile/load 및 시험 실행 |
| C++ native standalone | **PASS** | build, protocol, extension 동일-engine parity 및 pipeline 시험 |
| Python/native strict parity | **PASS — 4 cases, 288 comparisons, 0 failures** | 수정된 동일 물리·strict common tolerance의 field/history/handoff 비교 |
| native end-to-end debug | **PASS — pipeline 28/28** | Pareto → handoff → propagation → staggered 완료 및 checked-in semantic evidence 일치 |
| v8.1 파일범위 감사 | **PASS — unexpected 0** | baseline 253, current 268; 동일 200, 수정 53, 추가 15, 삭제 0; 생성 예정 manifest 3개만 유예 |
| release SHA manifest | **원자적 공개 gate** | builder가 manifest-bearing tree에서 누락·불일치·미등재 0을 확인해야만 archive 공개 |
| ZIP CRC 및 fresh-extract 재검증 | **원자적 공개 gate** | builder가 CRC와 fresh-extract manifest를 확인해야만 ZIP/checksum 공개; 결과는 `RELEASE_BUILD_COMPLETE.json`에 기록 |

개발 중 별도 release-contract 및 conservation suite에서는 다음 항목을 직접
검사한다.

- B/C profile 여섯 개의 full-domain surface-overlay 계약
- 접촉면 아래 초기 species/temperature 존재
- cell-centred L/N raster spacing과 physics-grid resize 뒤 edge-to-edge gap,
  component topology/cap, contact area/balance 및 유효 최소폭 재검사
- legacy non-B/C direct loader의 기존 embedded-electrode 의미 격리
- harmonic-face finite-volume in-plane electrical power의 face 단위 보존과
  contact-normal resistor power 포함
- `conductive_sigma_E2`의 비음수성과 signed `total_j_dot_e`의 음수 diffusion
  cross-term 비-clipping
- chemical accepted extent의 alpha clipping·LP/PVA 공동 scaling·progress·
  species/product·heat 일치
- channel heat-release basis의 명시와 conversion weight 이중 적용 방지
- Faradaic LP/water sink가 inventory를 초과할 때 상태·전류·energy commit 전
  fail-closed하고, species sink만 제한한 결과를 수락하지 않음
- electrochemical LP consumption이 remaining inventory에 반영됨
- generated water와 mobile water 분리
- first-onset freeze와 history-tail 정책
- stale pre-flame heat replay 금지
- corrected propagation의 complete inventory 필수, partial/missing inventory
  정책, onset stoichiometry와 매-step finite/non-negative/progress invariant
- 닫힌 coupled solver가 없는 provided post-onset electrical-history 모드의
  무조건 fail-closed
- roundoff-safe step count, exact fractional final step, evaluation-time 및
  electrical-update grid 정렬
- B/C의 `maximumTimeSteps=100000` 및
  `maximumHistoryAllocationBytes=17179869184`, propagation의 동등한 snake-case
  상한을 allocation 전에 검사하여 비정상 step/history 요청을 명시적으로 거부
- current-congestion objective의 evaluation-horizon 절단과 legacy full-horizon
  peak 진단 분리
- B/C/propagation의 harmonic-face diffusion + convection/radiation loss Jacobian
  total explicit thermal stability CFL 한도 거부, propagation temperature-cap
  즉시 거부 및 corrected B/C profile의 zero cap-threshold 기반 trial 거부
- Python pre-flame/continuation/propagation과 native에서 accepted temperature의
  cp/k를 재평가하여 final step 물성 overflow도 commit 전 fail-closed
- 최종 solve뿐 아니라 모든 B/C electrical linear/nonlinear Robin solve의 누적
  수렴 판정
- AtEvaluationTime canonical / At2s same-value alias
- reference/native 공통 V_min state machine, 정상 단조성, 순서 독립
  same-voltage contradiction 탐지 및 최종 upper bracket full-horizon 검증

Faradaic inventory 검사의 합격 의미는 limiter가 작동한 결과를 수락한다는
뜻이 아니다. 현재 solver는 inventory-constrained BV/potential 재해석을 하지
않으므로 clipping이 필요해지는 trial을 명시적으로 실패시켜야 합격이다.
반면 chemical LP/PVA inventory limiter는 두 channel의 proposed extent를 함께
축소하고 그 accepted extent를 모든 reaction bookkeeping에 사용하는 것이
합격 조건이다.

## A100/CUDA 상태

패키징 host에는 NVIDIA GPU, CUDA-enabled PyTorch 및 nvcc가 없다. 따라서 다음
항목은 이 환경에서 통과로 기록할 수 없다.

- 실제 CUDA extension compile/load
- A100 FP64 runtime
- CPU/CUDA full-field parity
- batch 32와 64 수치 불변성
- CPU8/A100 동시 기능 사용
- production 처리율/peak VRAM benchmark

CUDA source static check는 별도 항목이며 실제 장치 검증의 대체가 아니다.
대상 A100에서 production wrapper를 실행하면 별도의 새 sibling 디렉터리에
preflight를 자동 실행하고, 완료 보고서를 재검증한 뒤에만 production을 시작한다.

~~~bash
bash tools/run_bc_native_a100_cpu8.sh "$PWD/runs/native_a100" 1000 5
~~~

이 preflight는 실제 CUDA compile/load, CPU/CUDA full-field parity, batch 32/64
수치 불변성과 짧은 corrected B/C CPU+CUDA hybrid scheduling을 검사한다. 또한
선택된 장치가 이름상 NVIDIA A100이고 compute capability 8.0인지 fail-closed로
확인하며, logical/physical index와 UUID(확인 가능한 경우), 메모리,
PyTorch/CUDA/driver 정보를 `a100_device_report.json`에 기록한다. production
workdir와 preflight workdir는 기존 경로를 삭제하거나 재사용하지 않는다.

`ECSP_ALLOW_NON_A100_NONCERTIFYING_PREFLIGHT=1`은 비-A100 기능 진단 전용이며
A100 검증으로 기록되지 않는다. `ECSP_UNSAFE_BYPASS_REQUIRED_A100_PREFLIGHT=1`은
필수 gate의 명시적 비상 우회로이고, 큰 경고와 함께 bypass 사실을
`A100_PRODUCTION_GATE.json` 및 `RUN_COMPLETE.json`에 남긴다. 어느 override도
A100 인증을 의미하지 않는다. production workload의 처리율과 peak VRAM
최적화는 별도 benchmark 대상이며 preflight의 짧은 smoke 결과로 인증되지 않는다.

## 과학적 검증 상태

이 릴리스는 소프트웨어 의미, 수치 일관성 및 reduced-model bookkeeping을
검증한다. literature-nominal 물성·수송·BV·kinetics를 실제 ECSP formulation,
수분함량, 두께, 전극 공정에 대해 보정하거나 검증한 것은 아니다. 따라서
실험 보정 전 결과를 정량적 설계 인증이나 안전 한계로 사용해서는 안 된다.

## 재현 명령

~~~bash
ECSP_V810_BASELINE_ZIP=/path/to/ECSP_v8_1_0_BCNative_Hybrid_A100_CPU8_Final.zip \
ECSP_V82_PREPACKAGE=1 \
  bash tools/validate_v8_2_cpu.sh /new/path/outside/package/v8_2_prepackage_validation

# 정확한 release-name staging tree에서 manifest, release validation,
# ZIP CRC와 fresh-extract manifest까지 수행한다.
ECSP_V810_BASELINE_ZIP=/path/to/ECSP_v8_1_0_BCNative_Hybrid_A100_CPU8_Final.zip \
  bash tools/build_release.sh
~~~
