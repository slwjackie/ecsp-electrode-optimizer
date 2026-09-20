# v7.9.0 변경내역

## 형상 및 설계조건

- 설계·추진제 domain을 20×20 mm로 통일
- 목표 접촉면적을 양극 17.5%, 음극 17.5%로 변경
- `(Na,Nc)=(1,1),(1,2),(2,1),(2,2),(3,1),(1,3)` bootstrap 지원
- 각 component의 독립 start/heading 구현
- `add_component`, `remove_component`, `split_component`, `merge_component` mutation 구현
- component 단위 crossover 구현
- `Na≤3`, `Nc≤3`, `Na+Nc≤4` hard constraint 구현
- 96×96 생성 mask 및 193×193 resize 후 component topology 재검증
- NPZ에 `anode_contact_mask`, `cathode_contact_mask`, 전면 `propellant_domain_mask` 추가

## C++ 물리엔진

- electrode mask를 물질 제거영역에서 surface-contact label로 변경
- 전면 추진제에 T/species/liquid/alpha/gas/passivation 상태 초기화·적분
- 내부 fixed-potential electrode cell 방식을 제거
- contact footprint에 thin-layer Robin/Butler–Volmer source 결합
- contact 면적 기준 계면전류·발열 적분
- 외곽 전위 경계를 절연 Neumann으로 처리
- 양·음극 contact component의 hidden bus 연결 가정 명시
- 결과 JSON에 추진제 domain/contact 면적/semantics 진단값 추가

## Python–C++ 연결

- C++ 전달 설정에 contact normal length 및 surface-contact/hidden-bus flags 추가
- 193×193 resize 후 면적·component 수·최소간격을 physics 실행 전에 재검사
- intended component 수와 actual resized component 수 불일치 시 거부
- 출력 metadata와 diagnostics에 v7.9.0 semantics 기록

## 안전한 backend 정책

기존 MPS 직접 물리경로는 전극 mask를 hole로 해석하는 legacy formulation이므로 v7.9.0 production 경로에서 비활성화했다. M2 Pro에서는 C++ CPU FP64 backend를 사용한다.
