# v8.1.0 실행 검증 보고서

## 결론과 범위

업로드된 v8.0.0 ZIP에서 시작하여 native C++17 FP64 CPU/standalone 및 CUDA 소스, batched Vmin, early-stop, CPU pool/hybrid scheduler를 추가했다. **실행 가능한 CPU 시험을 완료했고 최종 소스 ZIP을 제공한다.** CUDA 소스 컴파일 및 실제 GPU 시험은 이 환경의 CUDA toolkit/nvcc/CUDA PyTorch/GPU 부재로 실행하지 않았다. 이는 CPU 시험으로 대체해서 통과 처리하지 않았다.

물리계수/조성을 재보정한 릴리스가 아니다. 단축 수치 포팅 비교는 실제 점화/반응률 실험 검증이나 1,000×5 full-horizon production 성능 측정을 의미하지 않는다.

## 시험환경

Linux x86_64, Python 3.13.5, PyTorch 2.10.0+cpu, g++ 14.2.0. cgroup CPU quota는 4개다. native CPU worker의 자동 축소와 spawn 동작을 시험했고, 실제 A100 또는 M2 Pro 측정은 없다. 원본 legacy hybrid C++은 clang++로 빌드했다.

| 항목 | 결과 | 증거 |
|---|---|---|
| 수정 전 원본 v8.0.0 tests | 56 passed | `v8_1_validation/baseline_pytest.log` |
| 새 코드 포함 전체 tests | 80 passed, 2 CUDA skipped | `v8_1_validation/final_pytest.log`, `final_pytest.xml`, `release_validation.log` |
| Python syntax | 68 files passed | `environment_and_syntax.json` |
| Bash syntax | 32 files passed | `environment_and_syntax.json` |
| C++/ATen CPU extension | 빌드 및 실제 실행 | `native_build.log`, `native_extension_build_commands.ninja.txt` |
| C++ standalone | 빌드 및 실제 실행 | `standalone_latest_build.log`, `standalone_linked_libraries.txt` |
| standalone vs extension | 시험한 field/history/handoff에서 bitwise equality | `test_bc_native_release.py` 전체시험 기록 |
| CPU/Python strict parity | 8 candidate trajectories, 228 comparison checks 통과 | `native_python_parity_strict.json` |
| legacy C++/Python parity | 33×33 단축 비교 통과 | `legacy_cpp_python_parity.json` |
| legacy C++ hybrid build | 통과 | `legacy_hybrid_build.log` |
| 기존 B/C 요구사항 자동감사 | 24/24 | `bc_global_pipeline_audit.json/md` |
| native 단축 전체 workflow | Pareto, handoff, propagation, staggered 완료 | `native_launcher_debug.log`; pytest end-to-end cases |
| 원본 파일 보존 | 173개 전수 비교: 동일170, 변경3, 삭제0 | `preservation/audit.json/md` |
| CUDA 소스 static readiness | 7/7 | `cuda_readiness.json` |
| 실제 nvcc build / load | 미검증: toolchain 없음 | 대상장치용 `run_bc_native_a100_preflight.sh` 포함 |
| CPU/GPU parity / GPU B32·B64 invariance | 미검증: GPU 없음, 2 tests skip | pytest skip 사유 명시 |
| A100 throughput / VRAM | 미검증: GPU 없음 | benchmark script 제공, 가속배수 주장 없음 |

## CPU/Python parity의 정확한 의미

v8 Python reference와 새 native를 동일한 조건에서 17×17 / 33×33 격자, 두 방향의 전극, 20 V / 260 V, 10 time steps로 비교했다. comparison 대상은 finalFields, histories, handoffFields 및 onset/잔류분율/전류혼잡도/에너지 metric이다.

기본 nonlinear tolerance가 느슨할 때, 두 수렴 알고리즘이 허용오차 안의 서로 다른 상태에서 멈추므로 거의 0인 일부 handoff 열원의 엄격한 점별 gate를 통과하지 못했다. **해당 실패 보고서를 삭제하지 않고** `native_python_parity_default_exploratory.json`으로 보존했다. 228 비교 중 9개 handoff heat 항목이 실패했고, key objective 및 final state 비교는 통과했다. 최대 절대 handoff 열원 차이는 약 1.11 W/m³ 이하였다. 이것만으로 전체 production 오차가 작다는 뜻은 아니다.

양쪽에 똑같이 더 엄격한 선형/비선형 수렴조건을 적용해 다시 검사했다. 새로운 물리계수나 임계값을 fitting한 것이 아니다. strict 비교 228/228이 통과했다. 대표 최대 절대차는 온도 약 1.43e-12 K, global progress 약 1.50e-18, 전위 약 7.32e-10 V였다. 전체 항목별 rtol/atol과 최대차는 JSON에 있다. 기본 production profile의 tolerance를 몰래 강화하거나 FP32를 사용하지 않았다.

재현:

```bash
PYTHONPATH="$PWD/python" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python python/compare_bc_native_python.py \
  --output "$PWD/runs/strict_parity.json" --strict-common-tolerances
```

## early-stop와 Vmin

합성 onset 시험에서 full run의 active steps [10,10,10]이 early-stop에서 [10,10,1]로 변하며 onset은 같았다. 성공 후보의 전기/반응/열 갱신을 mask로 중단하고 실패 후보는 끝까지 계산한다. CUDA는 공간을 매번 재압축하지 않으므로 grid launch나 일부 ATen 연산 비용이 완전히 없어지는 방식은 아니다.

reference/handoff는 전 시간구간을 계산한다. Vmin trial의 부분 적분 에너지/미분해율을 2초 결과로 쓰지 않는다. 최종 upper bracket 전체-horizon 재검사를 기본으로 켜서 조기 종료 이후 생기는 수치 cap 위반 가능성도 확인한다. 이 설정은 계산비용이 있으며 실제 가속배수는 A100에서 측정해야 한다.

시험에는 후보별 서로 다른 전압의 batch–single 일치, bisection 상태기계/left·right censor/invalid 처리, 합성 OOM 분할 후 후보 순서 유지, compiler 오류와 물리 실패 구분이 포함된다.

## 원본 보존에 관한 제한

원본 물리 및 파일 보존은 기존에 잘못되거나 미검증된 가정까지 과학적으로 승인한다는 뜻이 아니다. 대표적으로 v8 B/C 실제 propellant mask는 전극 셀 제외인데 일부 원본 문서는 surface overlay로 기술한다. 이번 버전은 parity와 기능보존을 위해 기존 실제 동작을 유지하고, 새 diagnostics/README에 이를 명시했다.

추가 코드가 지원하지 않는 saturation/프리컨디셔너/물리 옵션은 조용히 버리지 않는다. 원래 backend에서 쓰던 고급 옵션을 native에서 그대로 모두 지원한다고 간주하지 말고, 오류 안내에 따라 기존 backend를 명시적으로 선택해야 한다. 기존 설정과 기존 backend 파일은 byte-identical로 보존됐다.

## 최종 판정

**소스 구현·CPU 빌드·회귀시험·파일 보존 감사 완료. CUDA compile/runtime/성능 검증은 대상 A100에서 실행해야 한다.** 조건부 항목을 통과로 올려 적지 않았고, archive와 새 checksum manifest로 배포 내용을 검증한다.
