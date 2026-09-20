# ECSP v8.4.0 검증 기록

## 판정 범위

Linux x86_64, Python 3.13, NumPy/CPU PyTorch 환경에서 실행했다. 자세한 환경은
`validation_v8_4_0/environment.json`에 기록한다. **실험 검증, 실제 ECSP 점화/회귀율 재현,
A100 실행, 생산 규모 최적화의 완주·속도 검증은 하지 않았다.**

전체 결과: **519개 통과, 2개 CUDA 하드웨어 시험 미실행**.
새 BC→Reactive 시험은 **43개 통과**이며 전체 통과 수에 포함된다.
전체 JUnit은 `validation_v8_4_0/full_suite.xml`, 신규 시험 JUnit은
`validation_v8_4_0/bc_reactive_suite.xml`이다. 소스 binding과 배포 manifest로 실제
시험 소스/ZIP 내용의 일치 여부를 검사했다.

| 시험 | 실제 확인 결과 |
|---|---|
| 기존 모델/회귀 | 기존 numerical tests 유지. 릴리스 metadata 시험은 v8.4 lineage를 명시적으로 추가 |
| Python BC→두 backend | 32×32, 4후보 + 별도 staggered, 동일 onset 비교/8목적 최종 경로 완주 |
| Native C++ BC→두 backend | 같은 소규모 전체 경로 완주. 기존 C++ BC handoff가 새 adapter로 전달됨 |
| 개별 handoff CLI | BC 재실행 없이 저장 snapshot→두 backend 비교 완주 |
| Handoff | T/α1/α2 왕복 최대오차 0, 질량 유지; 변조·비승인·수지 불일치 입력 거부 |
| 두 채널 | 동일 X/서로 다른 α 상태 보존; 기존 Torch BC rate와 대조; 받아들인 반응량과 발열 일치 |
| 단일 에너지 | 별도 solid.py 적분 없음. qJ/qEchem/qChem/손실/경계유출 포함 보존 수지 검사 |
| 전원 ON | 실제 기존 NP/BV stage 재계산 6회, 비영 qJ/qEchem 및 contact support 검증 |
| 전원 OFF | 저장된 onset qJ/qEchem 값을 크게 바꿔도 이를 재생하지 않음 |
| Level set | 두 채널 X에서 경계를 구성; X 문턱 이후 α2가 계속 반응하고 continuum 질량이 삭제되지 않음 |
| 동적 Euler | 별도 작은 밀도 교란 시험에서 속도 생성, acoustic CFL 사용, 보존 수지 통과 |
| 정지 fast path | 정지 불변 조건에서 전체 flux 경로와 수치 결과 일치; 비균일 밀도면 비활성화 |
| 열전도 수렴 | Neumann Fourier 제조해 N=8/16/32에서 관측 차수 약 1.995/1.999 |
| 순위 비교 코드 | 동일 후보/8목적, synthetic 역순 예제에서 Spearman=-1. 실제 debug 형상은 동률이라 null |
| 실패 처리 | 잘못된 config/재고/수치 상태/자원 한도에서 실패. 성공한 다른 backend로 몰래 대체하지 않음 |

## 수치 기록

### 동일 onset 4후보 debug

Python BC 및 native BC 경로 모두:
- post-onset 시간 0.006 s; onset은 0.002 s의 강제 단축 기준.
- 최대 질량 상대잔차 0.
- 최대 에너지 상대잔차 `7.753715246968632e-16`.
- 최대 유속 0 m/s. **균일 rho/정지 handoff와 Tait p(rho)의 결과이지 화염 유동 검증이 아니다.**
- geometry rank 모두 동률. 이 smoke test로 형상 간 우열의 물리적 신뢰도를 입증하지 않는다.

### 전압 유지 재계산 시험

승인된 debug snapshot에서 공급전압을 5 V로 바꾼 짧은 수치 시험(실험 조건 제안 아님):
2 steps, 6 NP/BV 호출, 적분 시간 1e-5 s.

| 항목 | 값 |
|---|---:|
| 적분 qJ | 3.86294271201859e-9 J |
| 적분 qEchem | 3.5989065914796347e-10 J |
| 적분 qChem | 8.864711197591402e-5 J |
| 에너지 잔차 | 1.1705433179110142e-16 J |
| 최대 BC 전류 mismatch | 0.00350008539666646 |

qEchem은 contact footprint 밖에서 정확히 0이며 qJ는 기존 in-plane+contact-normal 열원의 합이다.
온도·진행도에 맞는 전기해석을 재호출했다. 기존 onset 발열장을 고정해 재생한 시험이 아니다.

### Fourier 제조해

| N | L2 온도오차 [K] | 관측 차수 |
|---:|---:|---:|
| 8 | 0.0019831666168318943 | — |
| 16 | 0.0004976528788094931 | 1.9945941971493333 |
| 32 | 0.00012452982315494483 | 1.9986485002099514 |

이는 **열전도 연산자의 수렴**이다. 전체 연성 ECSP 파동·전선의 5차 수렴이나 실제 점화시간
정확도를 검증한 것으로 읽으면 안 된다.

별도 밀도 교란 시험의 peak speed는 0.0003398212605340761 m/s이고 energy relative
residual은 1.909474489510875e-15이다. 동적 Euler 코드 경로가 실제 실행되는지를 확인한
합성 문제이며, 이 교란을 BC onset에 몰래 넣지는 않는다.

## 보존 잔차의 정의

질량 잔차는 initial mass와 경계 mass flux 적분을 기준으로 한다. 총에너지 잔차는 initial E,
경계 (E+p)u flux, accepted chemistry/electrical sources, 열전도 및 표면손실을 포함한다.
출력의 energy relative residual은 `max(|E0|, |Et|, |integrated sources/loss|, 1 J)`로
정규화한다. 따라서 아주 작은 에너지 시험에서는 함께 저장된 **절대 잔차 [J]**도 확인해야 한다.

기존 baseline은 cp(T)×dT 형태의 explicit 온도 적분을 유지하여 cp(T)를 적분한 에너지로
진단하면 시간차분 오차가 남을 수 있다. 그 오차를 수지 파일에서 지우거나 임의로 보정하지 않았다.
원래 baseline의 질량 일정은 고정 밀도/격자 가정이지 새 유동 질량방정식 검증이 아니다.

## 남아 있는 모델 한계

Tait 상수/BC 물성·반응표는 미보정이다. 반응·온도에 따른 압력 생성, 생성물별 EOS,
실제 외곽 표면 ablation, 기체상, 전극 이동·접촉 박리는 없다. Level set은 **내부 반응 상태
경계**이고 시료 외곽 소모면이 아니다. 전원 OFF에서는 별도 이온 NP 확산을 계속 적분하지 않는다.
새 Reactive는 CPU FP64이며 자체 CUDA kernel은 없다. 제조해 외의 생산 격자/시간/전선 문턱
독립성과 실험 형상 순위는 추가 확인이 필요하다.

## 재현

```bash
python -m pip install pytest
PYTHONPATH=python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q python/tests

python python/verify_release_manifest.py
```

원본 v8.3의 모든 파일을 보존하고, evaluator/workflow 연결 및 baseline 진단을 변경했다.
독립 Reactive reference 15파일은 byte-identical이다. 원본 SHA256 및 파일 대조는
`PARENT_PRESERVATION_V8_4_0.json`을 참고한다. 과거 `V8_3`, `V8_2` validation 및 builder는
그 릴리스의 기록으로 남겨두었으며 **현재의 권위 manifest는 PACKAGE_SHA256_MANIFEST_V8_4.txt**다.
