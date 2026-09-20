# ECSP v7.9.0 — M2 Pro C++ CPU FP64 구조

## 역할 분리

```text
Python
├─ surface-contact geometry grammar
├─ multi-component mutation/crossover
├─ manufacturability checks
├─ NSGA-II / Pareto / result I/O
└─ contact masks + scalar config를 후보당 한 번 전달
       ↓
standalone C++17 process — CPU FP64
├─ 20×20 mm full propellant state
├─ thin-layer potential equation
├─ area Robin/Butler–Volmer contacts
├─ Nernst–Planck species
├─ passivation / reduced gas coverage
├─ thermal / decomposition
└─ 2 s integration
       ↓
scalar objectives + diagnostics 반환
```

M2의 Metal/MPS는 condensed physics에 사용하지 않는다. C++ 실행파일은 Apple clang으로 ARM64 native build되며 FP64를 사용한다. 여러 후보는 독립 process로 병렬 실행한다.

## 데이터 이동

한 timestep마다 Python↔C++ 복사를 하지 않는다.

1. Python이 193×193 contact mask와 설정을 한 번 저장
2. C++가 전 시간적분을 내부에서 수행
3. 목적함수와 진단 JSON만 Python으로 반환

## surface-contact 의미

전극 footprint 아래도 추진제이므로 추진제 domain fraction은 항상 1이다. 전극 component가 분리되면 계산영역 밖 hidden bus 연결을 전제로 동일 terminal에 연결된다.

## 실행

```bash
bash tools/run_m2_cpp_fp64_preflight.sh "$PWD/runs/preflight"
bash tools/run_m2_cpp_fp64_100x2.sh "$PWD/runs/100x2"
```
