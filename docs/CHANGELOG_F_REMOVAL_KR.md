# F 제거 변경이력

## 제거

- `flameProgress` 상태변수
- surface reaction–diffusion equation
- flame seed/spread coefficients
- flame heat source
- F-threshold burned/unburned area
- F-threshold established ignition
- OpenFOAM/reduced-CFD 실행 모듈
- 기존 diffusion/surrogate/active-learning 실행 경로

## 추가·변경

- condensed-phase onset delay
- area-averaged \(1-\alpha\) at 2 s
- 2 s electrical input energy
- peak current congestion
- 20 topology × 50 variants bootstrap
- constrained NSGA-II
- 70+20+10 mating quota
- final normalized-utopia recommendation

## 호환성

기존 v7.7.2 파일명 일부와 `run_coupled_batch` 함수명은 adapter 호환을 위해 유지했지만, 내부 물리는 no-F condensed-phase solver로 교체했다.
