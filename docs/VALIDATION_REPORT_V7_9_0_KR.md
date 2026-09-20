# v7.9.0 검증 보고서

## 검증 범위

이 보고서는 소프트웨어·수치 배관 검증이다. LP/PVA/GLY/BA 절대 예측 정확도의 실험 검증은 아니다.

## 자동시험

- geometry template component-pair coverage
- independent component pose
- 목표 contact area 및 full propellant descriptor
- add/remove/split/merge mutation
- component crossover cap
- 저장 NPZ의 full propellant mask
- C++ surface-contact semantics와 추진제 cell 수 불변성
- 기존 regression/unit tests

최종 패키징 전에 전체 `pytest`, Python bytecode compile, C++ warning build, C++ 4형상 preflight를 다시 수행한다. 최종 수치는 `VERSION.json`과 이 문서 말미에 기록한다.

## 대표 수치검증 항목

1. 193×193에서 `propellantCellCount = 193² = 37,249`
2. contact 면적이 달라도 추진제 cell 수 동일
3. `propellantDomainAreaFraction = 1.0`
4. `electrodeMasksRemovePropellant = false`
5. `surfaceContactModel = true`
6. `hiddenBusConnectionAssumed = true`
7. 양극·음극 BV 전류수지와 FP64 전위 residual 수렴
8. 요청한 여섯 component-count 조합이 C++ one-step을 통과

## 해석상 주의

- 17.5%/극은 사용자가 지정한 설계조건이며 최적 면적의 실험적 증거가 아니다.
- hidden bus 연결이 실제 제작에서 구현되어야 multi-component 형상이 유효하다.
- 기상 CFD와 self-sustained flame은 포함되지 않는다.
- `retained_water_fraction=1.0` 등 nominal 값은 실험 보정이 필요하다.

## 재현 명령

```bash
bash tools/validate_surface_contact_cpp.sh "$PWD/runs/v790_surface_validation"
```

이 명령은 여섯 component-count 조합의 독립 C++ one-step gate를 포함한다. 결과는 `<workdir>_pairs/multi_component_pair_validation.json`에 저장된다.

## 최종 패키징 검증 결과

- 전체 Python test: **38/38 통과**
- Python bytecode compile: **41개 파일 통과**
- C++ `-Wall -Wextra -Wpedantic`: **warning 0개**
- AddressSanitizer + UndefinedBehaviorSanitizer: **대표 surface-contact case 통과**
- 193×193 C++ FP64 preflight: **4/4 성공, rejection 0**
- 요청 component 조합 C++ one-step: **6/6 성공**
- 193×193 추진제 cell 수: **37,249**, 모든 형상에서 동일
- 여섯 조합 전위 relative residual: 약 \(8.92\times10^{-10}\)–\(9.84\times10^{-10}\)
- 여섯 조합 최대 양·음극 전류 mismatch: 약 \(2.82\times10^{-4}\), 설정 허용치 0.005 이내
- 200개 Generation-0 직접 표본의 생성 mask feasible 비율: **173/200 = 86.5%**; production workflow는 topology별 retry 후 feasible 개체를 채움

검증 환경은 x86-64 Linux C++ toolchain이다. Apple M2 Pro에서는 패키지의 preflight를 다시 실행해야 ARM64 native build와 실제 wall-clock 성능이 확인된다.
