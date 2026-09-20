# v8.4.1 검증 기록

## 결과

| 시험 | 결과 |
|---|---|
| 원본 v8.4.0 전체 시험 재실행 | 519 passed, 2 CUDA skips, 235.64s |
| 수정 v8.4.1 전체 시험 | 554 passed, 5 CUDA skips, 173.91s |
| 신규 tensor/backend/regression 파일 | 35 passed, 3 CUDA skips (전체 집계에 포함) |
| 기존 BC–Reactive 통합 시험 | 43 passed (전체 집계에 포함) |
| Python compileall / shell bash -n | 통과 |
| CUDA 없는 A100 preflight | 의도한 실패: CPU-only PyTorch를 감지하고 거부 |
| 실제 A100/CUDA Graph/8개 할당 CPU 완주 | 이 환경에서 미실행 |
| 시료 물성·실험적 순위 검증 | 미실행 |

시험시간 차이는 작업환경·캐시·컴파일 등도 영향을 주므로 코드 속도향상 배수로 해석하지 않습니다. 여기서 확인된 CPU count는 5이고 Torch 2.10.0+cpu, Python 3.13입니다. CPU 8개 예산은 quota/affinity-aware 코드와 단위시험으로 확인했으나 8코어 하드웨어 성능을 측정하지 않았습니다.

## 발견 및 수정 / 보호

| ID | 상태 | 문제 | 영향 | 수정 | 근거 | 시험 |
| --- | --- | --- | --- | --- | --- | --- |
| F01 | 수정·회귀시험 | 열안정성 bound가 4k/capacity 휴리스틱 | k와 열용량이 강하게 비균일한 사용자 물성에서 harmonic face 합보다 작아져 unsafe dt 가능 | 실제 FV face conductance row sum + 열손실 Jacobian; 매 RK stage 검사 | python/ecsp_reactive/condensed/solver.py:115-134 | test_numpy_thermal_bound_uses_face_conductance_not_4k_heuristic |
| F02 | 수정·회귀시험 | HLL 선택에도 stationary acoustic dt를 생략 | HLL은 정지 열/조성접촉을 수치 확산하므로 acoustic 제한 생략 부당 | exact stationary shortcut은 HLLC에만 허용; NumPy/torch 통일 | python/ecsp_reactive/condensed/solver.py:115-134; python/ecsp_reactive/condensed/tensor_math.py:289-315 | test_hll_stationary_contact_does_not_skip_acoustic_diffusion |
| F03 | 수정·회귀시험 | 상수 목적함수도 첫/끝 개체에 infinite crowding | 동률 목적에서 임의 개체를 우대해 탐색 선택 편향 | 목적 범위 0이면 endpoint crowding 부여 안 함 | python/ecsp_nsga2/nsga2.py:107-134 | test_constant_objectives_cannot_create_artificial_crowding_endpoints |
| F04 | 수정·회귀시험 | 문자열 false가 bool("false")로 True | 잘못된 전원/fast-path 설정을 조용히 반전 | 후속 스위치 strict boolean 검증 | python/ecsp_reactive/condensed/solver.py:31-103 | test_string_false_is_not_treated_as_true |
| F05 | 수정·회귀시험 | 공유 반응물이 소진된 셀의 raw rate도 dt 제한 | 실제 허용 반응량 0인데 불필요한 극소 dt 요구 | dt 제한만 가용 시약 셀 적용; raw-rate cap 검사는 유지 | python/ecsp_reactive/condensed/solver.py:115-134 | test_exhausted_stock_does_not_force_impossible_tiny_timestep |
| F06 | 수정 | 기존 launcher가 symlink/기존 log 보호 불완전 | 실행 출력/로그 덮어쓰기 위험 | workdir·symlink·log 존재시 시작 거부 | tools/run_bc_reactive.sh; tools/run_bc_reactive_a100_cpu8.sh | shell syntax + no-CUDA fail-closed |
| F07 | 최적화·시험 | CPU reference 매 단계 초기 수지 재합산 | 변하지 않는 초기장을 반복 reduction | 초기 mass/energy/reservoir 합 캐시 | python/ecsp_reactive/condensed/solver.py:183-223 | 전체 기존 시험 및 NumPy/torch parity |
| F08 | 신규 보호·시험 | GPU batch별 실패·메모리·시계 혼합 위험 | 한 후보 실패가 형상전체 실패 또는 timestep 왜곡 가능 | 독립 dt/거부/완료 mask; 동일 GPU batch 분할; 원본 handoff hash; CPU 대체 금지 | python/ecsp_nsga2/post_onset_batch.py; python/ecsp_reactive/condensed/tensor_solver.py | independent lanes/failure isolation/resource/CUDA required tests |

## 신규 시험의 내용

동일 synthetic U에서 NumPy와 Torch의 cp 적분/역변환, 보존 flux, HLL/HLLC, reflective/periodic/transmissive 경계를 대조합니다. Batch 1/2/4가 독립 NumPy 결과와 일치하는지, 두 후보의 서로 다른 dt가 유지되는지, 한 후보의 수치실패가 정상 형상을 오염하지 않는지 검사합니다. 실제 BC debug workflow에서 전달된 snapshot으로 비영 전류 NP/BV를 켜서 계면열/전력/수지를 대조했습니다.

CUDA 필수시험 세 개는 CPU 환경에서 skip되었습니다: stationary/dynamic+batch parity, changing-dt CUDA Graph replay parity, 실제 BC snapshot/NP/BV/전체 workflow GPU 경로. 기존 native CUDA 두 개와 함께 target A100 preflight에서 실제로 통과해야 합니다. 문서의 GPU 구현 설명은 소스 구현이며 하드웨어 성공 주장과 다릅니다.

## 성능 측정의 한계

`docs/validation/benchmark_cpu*.json`은 합성 FP64 연산 시험입니다. grid32/193, 여러 batch에서 최종 보존장은 NumPy와 일치했습니다. CPU tensor는 일부/다수 경우 NumPy보다 느렸습니다. 이것은 A100 최적화가 실패했다는 증명도, A100 속도향상 증명도 아닙니다. 실제GPU에서 wall time, peak VRAM, batch수, graph유무, power-on NP/BV 비용과 전체 workflow elapsed를 별도로 측정해야 합니다.

생산 프로필의 193² grid, 최대2s pre-onset+1s post-onset, population1000/generations5 완주 시간은 측정하지 않았습니다. nominal rate/transport 값에 따라 소요시간과 성공 후보 수가 바뀔 수 있습니다.

## 남아 있는 모델 한계 (코드 버그와 구별)

Tait thermal-pressure coupling 없음, 내부 reaction-front만 제공, 기화/외곽질량소모/상별EOS/전단응력/기체종·점성 CFD 없음. 미보정 LUT·계면 반응열·수분·물성으로 실험 예측 정확성을 주장하지 않습니다. 기존 코드에 남은 inactive 계수는 사용 경로에 따라 달라집니다. 모든 잠재적 버그가 제거되었다는 보장은 하지 않습니다.
