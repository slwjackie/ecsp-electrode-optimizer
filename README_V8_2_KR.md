# ECSP v8.2.0 P0/P1 물리 수정 + v8.1 성능 경로

## 릴리스 범위

이 릴리스는 v8.1의 C++17-compatible FP64 core, CUDA FP64, batch 32/64, 후보 병렬
V_min, successful-trial early-stop, CPU 8-core/A100 hybrid 실행 구조를
유지하면서 B/C global pre-flame 및 condensed propagation의 알려진 P0/P1
물리·semantic 오류를 수정한다.

수정 경로는 물리적으로 일관된 상태를 만들 수 없는 경우 값을 조용히 잘라서
계속하지 않는다. 특히 요청된 Faradaic LP/water 소모가 가용 inventory를
초과하면 BV 전류와 potential을 다시 풀지 않은 채 sink만 clip할 수 없으므로,
상태·전류·energy를 commit하기 전에 명시적으로 실패한다.

수정된 production 계약은 다음 하나다.

~~~
full-domain condensed propellant
  + separate anode top-surface contact mask
  + separate cathode top-surface contact mask
  -> footprint-wide surface Butler-Volmer coupling
  -> first-onset conservative handoff
  -> post-onset condensed propagation without stale heat replay
~~~

전극 mask는 추진제 물질을 제거하지 않는다. 전극 아래 셀에도 초기 조성,
온도, 수송 및 반응 상태가 존재한다. B/C production/debug profile은 모두
boundaryCouplingModel: surface_overlay_bv를 명시하며, legacy perimeter 또는
embedded-electrode 의미로 조용히 되돌아가지 않는다.

## 수정된 항목

| 영역 | v8.2 동작 |
|---|---|
| 추진제 영역 | 모든 격자 셀이 추진제이고 전극은 별도 top-surface mask다. |
| 격자·제조성 | 물리 raster는 cell-centred `dx = L/N`을 사용한다. physics-grid resize 뒤 edge-to-edge gap, polarity별 component 수와 cap, 접촉면적·양극/음극 balance 및 유효 최소폭을 다시 검사하고 위반 후보를 거부한다. |
| BV 적용 위치 | 전극 perimeter가 아니라 실제 접촉 footprint 전체다. |
| 전류 적분 | 접촉 셀마다 j × dx²를 합산한다. |
| 체적 source | 표면 전류밀도를 j / surfaceLayerThickness로 변환한다. |
| 전기 power 이산화 | surface-overlay의 in-plane power는 harmonic face flux를 finite-volume face마다 한 번 계산하고 양쪽 control volume에 절반씩 배분한다. unresolved contact-normal 저항열도 별도 계산해 포함한다. |
| Joule heat 의미 | 기본 `conductive_sigma_E2`는 비음수 비가역 전도열이다. `total_j_dot_e`는 diffusion cross-term을 포함한 signed electrical energy transfer이며 음의 국소값을 clip하지 않는다. |
| 잔류 반응물 | 실제 mobile LP를 사용하므로 electrochemical LP 소모가 U_rem에 포함된다. |
| Faradaic inventory | LP/water sink clipping이 필요하면 self-consistent BV/potential 재해석 없이 결과를 수락하지 않고 commit 전에 fail-closed한다. |
| chemical bookkeeping | alpha clip 후 가용 LP/PVA에 맞춰 공동 scaling한 accepted extent 하나로 alpha, global progress, species, product, heat를 함께 갱신한다. |
| alpha clipping | chemical heat는 raw rate가 아니라 accepted delta-alpha / actual dt에서 계산한다. |
| 물 pool | 생성 H₂O product와 condensed mobile water를 별도 상태로 유지한다. |
| onset handoff | 최초로 채택된 onset 상태의 alpha, species, product, potential 및 inventory를 넘긴다. |
| onset 이후 열 | 기본값은 전기 가열 중단이며 pre-flame heat history를 propagation에서 재생하지 않는다. |
| 공통 평가시각 | onset이 먼저 발생한 후보는 그 시점에서 전기/BV 해석을 끝내고, 전기열 0인 응축상 전도·열손실·inventory-limited chemistry만 `evaluationTime_s`까지 이어서 모든 후보의 U_rem을 같은 시각에서 계산한다. |
| propagation 입력 | 완전한 LP/PVA/mobile-water/product-water/electrochemical-consumption inventory와 일관된 stoichiometry가 필수다. partial 또는 전체 누락 inventory는 모두 오류다. |
| 시간격자 | floating-point 오차로 0-duration 마지막 step을 만들지 않으며 실제 fractional `step_dt`로 정확히 `endTime_s`에 끝난다. 내부 evaluation time은 step grid에, electrical update interval은 `dt`의 정수배에 맞아야 한다. |
| 계산 자원 상한 | B/C는 `maximumTimeSteps: 100000`, `maximumHistoryAllocationBytes: 17179869184`를 사용하고 propagation은 동등한 snake-case 상한을 사용한다. 예상 step 수와 retained/transient history 크기를 allocation 전에 검사하며, 상한값만큼 메모리를 미리 예약하는 의미는 아니다. |
| 열 안정성 | harmonic-face diffusion 항과 convection/radiation loss Jacobian을 합친 total explicit thermal stability CFL을 매 step 검사해 한도를 넘으면 fail-closed한다. diffusive CFL은 별도 진단값으로 남긴다. propagation의 허용치를 넘는 temperature clipping은 즉시 거부하며, B/C trial의 cap fraction은 numerical-validity threshold로 거부한다. |
| profile 엄격화 | 여섯 corrected B/C profile은 temperature/species/chemical-rate cap 허용 fraction을 0으로 두고, channel heat를 initial bulk propellant 기준 기여량으로 명시해 conversion weight의 이중 적용을 막는다. debug profile의 physics grid는 design grid와 같은 32로 맞춘다. |
| solver 수렴 | 마지막 solve만 보지 않고 모든 electrical linear solve와 nonlinear Robin solve의 누적 수렴을 요구한다. 한 번이라도 실패한 B/C trial은 invalid다. |
| 설정 일관성 | initial temperature, onset temperature/progress의 충돌·비유한 값은 명시적 오류다. |
| V_min | reference/native가 같은 batched state machine을 사용한다. 실제 trial 단조성·동일 전압 모순을 검사하고, 선택된 upper igniting bracket을 full horizon으로 최종 재검증한다. |
| 평가시각 이름 | AtEvaluationTime이 정식 이름이다. 기존 At2s key는 같은 값을 갖는 deprecated alias다. |

고정 길이 `histories`는 각 후보의 pre-flame 구간만 나타낸다. onset 이후 tail은
상태·누적량 hold, 순간 source 0으로 표시되며 공통 평가시각 continuation을
재현하는 history가 아니다. 공통 시각의 scalar는 항상 canonical
`*AtEvaluationTime` metric에서 읽는다. 저수준 B/C API에서 full field 저장을
요청하고 `evaluationTime_s == endTime_s`인 경우에만 `evaluationFields`가 함께
반환된다. 일반 workflow가 저장하는 onset handoff NPZ는 공통 평가시각 field가
아니다.

## 실행

먼저 짧은 CPU 검증을 실행한다.

~~~bash
cd /path/to/ECSP_v8_2_0_P0P1Fixed_BCNative_Hybrid_A100_CPU8
ECSP_V810_BASELINE_ZIP=/path/to/ECSP_v8_1_0_BCNative_Hybrid_A100_CPU8_Final.zip \
  bash tools/validate_v8_2_cpu.sh /path/outside/package/v8_2_cpu_validation
~~~

검증 출력 경로는 기존 파일과 섞이지 않도록 반드시 존재하지 않는 package 외부
경로여야 한다. 개발 tree에서 아직 v8.2 SHA manifest를 만들기 전 검증할 때만
`ECSP_V82_PREPACKAGE=1`을 함께 지정한다. 이 모드도 immutable v8.1 ZIP과 모든
구현 변경을 엄격히 감사하고, 생성 단계에만 존재하는 세 manifest 차이만
명시적으로 유예한다.

CPU debug 전체 workflow:

~~~bash
bash tools/run_bc_native_debug.sh "$PWD/runs/native_debug"
~~~

CPU 8-core production profile:

~~~bash
bash tools/run_bc_native_cpu8.sh "$PWD/runs/native_cpu8" 1000 5
~~~

A100 production wrapper는 실제 장치 preflight를 **기본으로 자동 실행**하며,
preflight가 완전히 통과하지 않으면 production을 시작하지 않는다.

~~~bash
ECSP_BC_CUDA_BATCH=32 \
  bash tools/run_bc_native_a100_cpu8.sh "$PWD/runs/native_a100" 1000 5
~~~

wrapper는 sibling 경로 `runs/native_a100.a100_preflight`에 새 preflight를 만든다.
production과 preflight 경로는 모두 실행 전에 존재하지 않아야 하며, wrapper는
기존 파일이나 디렉터리를 삭제·재사용하지 않는다. standalone 진단만 원하면 새
경로를 지정해 다음을 직접 실행할 수 있다.

~~~bash
bash tools/run_bc_native_a100_preflight.sh "$PWD/runs/native_a100_diagnostic"
~~~

preflight는 선택된 CUDA 장치 이름에 `NVIDIA A100`이 포함되고 compute capability가
8.0인지 fail-closed로 확인한다. 가능한 경우 torch logical index, physical index,
UUID, 총 메모리와 함께 PyTorch/CUDA/driver 버전을 기록한다. 이어 실제 CUDA
compile/load와 CUDA entrypoint 존재, CPU/CUDA full-field parity, batch 32/64 수치
불변성 및 짧은 corrected B/C CPU+CUDA 동시 scheduling을 검사한다. 짧은 smoke의
경과시간은 기록하지만 production 처리율이나 peak VRAM benchmark를 대신하지
않는다. 과거 Python B/C global wrapper `tools/run_bc_global_a100.sh`도 같은 방식으로
자체 preflight를 자동 실행한다.

비-A100 CUDA 장치의 기능 진단이 꼭 필요한 경우에만
`ECSP_ALLOW_NON_A100_NONCERTIFYING_PREFLIGHT=1`을 명시할 수 있다. 결과에는
non-certifying override가 기록되며 A100 검증 또는 인증으로 해석해서는 안 된다.
필수 preflight 자체를 건너뛰는 유일한 escape hatch는
`ECSP_UNSAFE_BYPASS_REQUIRED_A100_PREFLIGHT=1`이다. 실행 시 큰 경고가 출력되고
`A100_PRODUCTION_GATE.json` 및 `RUN_COMPLETE.json`에 bypass가 기록되므로 정상
production 절차가 아니다.

batch 64 profile도 포함하지만 최적 batch 크기는 A100 모델, 사용 가능한 VRAM,
동시 CPU load와 grid 크기에 따라 직접 benchmark해야 한다.

native core source의 최소 언어 수준은 C++17이다. build helper는 설치된
LibTorch header의 요구조건에 맞춰 torch 2.14 미만에서는 C++17, torch 2.14
이상에서는 C++20 translation unit을 선택하고 실제 선택값을 build/runtime
metadata에 기록한다.

## 사용할 profile

수정된 B/C profile은 다음 여섯 개다.

- config/nsga2_bc_global_native_cpu8.yaml
- config/nsga2_bc_global_native_a100_batch32.yaml
- config/nsga2_bc_global_native_a100_batch64.yaml
- config/nsga2_bc_global_native_debug.yaml
- config/nsga2_bc_global_preflame_propagation_a100.yaml
- config/nsga2_bc_global_preflame_propagation_debug.yaml

config/nsga2_condensed_phase_no_f*.yaml과 과거 문서는 재현성을 위해 남아 있는
legacy 계열이다. v8.2 B/C production 결과를 만들 때 위 여섯 profile 중 하나를
사용한다.

## 결과 해석과 검증 한계

소프트웨어 회귀시험, 보존식·limiter 시험, Python/native 비교, archive hash
검증은 구현 오류의 위험을 크게 낮추지만 모든 입력과 모든 장치에서 결함이
없음을 수학적으로 보증하지는 않는다. 이 패키지에 포함된 계수는
literature-nominal이며 실제 추진제 batch와 전극 공정에 대해 실험 보정되지
않았다.

현재 패키징 환경에는 NVIDIA GPU/CUDA runtime이 없으므로 CUDA 소스의 정적
검사만 가능하다. 실제 nvcc build, A100 runtime, CPU/CUDA parity, batch 32/64
수치 불변성과 CPU+GPU 기능 동작은 대상 A100 preflight를 통과해야 한다.
production 처리율과 peak VRAM은 그 뒤 실제 workload로 별도 benchmark해야
한다. 자동 gate나 짧은 기능 preflight도 성능 인증을 뜻하지 않으며, 이 제한들을
CPU 시험으로 통과 처리하지 않는다.

최종 검증 결과와 미검증 항목은 docs/V8_2_VALIDATION_REPORT_KR.md, 파일별
변경 범위는 docs/V820_EXPECTED_CHANGES.json을 확인한다.

## 릴리스 ZIP 생성

릴리스 빌드는 정확히 `VERSION.json`의 `release_directory_name`과 같은 이름을
가진 새 staging 디렉터리에서 실행한다. immutable v8.1 입력 ZIP은 필수다.

~~~bash
cd /path/to/ECSP_v8_2_0_P0P1Fixed_BCNative_Hybrid_A100_CPU8
ECSP_V810_BASELINE_ZIP=/path/to/ECSP_v8_1_0_BCNative_Hybrid_A100_CPU8_Final.zip \
  bash tools/build_release.sh
~~~

빌더는 archive 대상과 검증 출력이 새 경로인지 먼저 확인하고, manifest를
원자적으로 만든 다음 정확히 archive에 들어갈 tree에 대해 CPU 전체 suite,
native build, strict Python/native parity, CUDA source 검사, executable pipeline
audit, v8.1 보존 감사와 manifest 검증을 다시 실행한다. 그 검증이 모두 통과한
경우에만 임시 ZIP의 CRC 및 fresh-extract manifest를 확인하고 기존 파일을
덮어쓰지 않는 방식으로 ZIP과 checksum을 공개한다. 외부 검증 디렉터리에는
`VALIDATION_COMPLETE.json`과 `RELEASE_BUILD_COMPLETE.json`이 남는다.

릴리스 무결성 검증의 정식 기준은 `PACKAGE_SHA256_MANIFEST_V8_2.txt`다.
`PACKAGE_SHA256_MANIFEST.txt`는 같은 내용을 담는 호환 alias이고,
`PACKAGE_SHA256_MANIFEST_V8_1.txt`는 과거 릴리스 재현용 historical manifest이며
v8.2 payload inventory에도 포함된다. 자기참조를 피하기 위해 제외되는 것은 현재
`PACKAGE_SHA256_MANIFEST_V8_2.txt`와 그 generic alias 두 파일뿐이다.
