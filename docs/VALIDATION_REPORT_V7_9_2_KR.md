# v7.9.2 검증 보고서

## 자동시험

```text
pytest: 44 passed
```

검증 항목:

- 20×20 mm, 96→193 grid, 극당 20%, 폭 1–5 mm, 간격 2 mm
- 25×25 mm, 120→241 grid, 동일 조건
- design/physics grid 모두 양극 2 component + 음극 2 component
- 좌우 순서 A-C-A-C
- 양극 팔 두 개의 길이·폭 동일
- 음극 팔 두 개의 길이·폭 동일
- resize 후 네 팔의 길이·폭 동일
- hidden bus contact pixel 0
- 최소 수직 중첩비 0.45 이상
- 면적 및 최소간격 허용조건 만족
- 일반 NSGA-II candidate에는 component cap override가 적용되지 않음
- area-matched staggered 전용 metadata가 있을 때만 1+1 run의 baseline 2+2 contact 허용

## 20×20 mm 대표 결과

```text
contact order: A-C-A-C
finger width:  2.736842 mm
finger length: 14.947368 mm
area/design:   0.200303819 per polarity
area/physics:  0.199629520 per polarity
minimum gap/design:  2.315789 mm
minimum gap/physics: 2.187500 mm
vertical overlap/design:  0.479167
vertical overlap/physics: 0.481865
hidden-bus contact area: 0
```

## 25×25 mm 대표 결과

```text
contact order: A-C-A-C
finger width:  3.361345 mm
finger length: 18.907563 mm
area/design:   0.200000000 per polarity
area/physics:  0.199445602 per polarity
minimum gap/design:  2.310924 mm
minimum gap/physics: 2.187500 mm
vertical overlap/design:  0.500000
vertical overlap/physics: 0.502075
hidden-bus contact area: 0
```

## C++ FP64 one-step

20 mm/193 및 25 mm/241 hidden-bus 2+2 baseline을 동일 C++ FP64 surface-contact evaluator로 실행했다.

```text
20 mm: prepared=1, successful=1, rejected=0
25 mm: prepared=1, successful=1, rejected=0
```

이 검증은 수치 배선과 mask 의미를 확인하는 one-step 시험이며, 2초 성능 또는 실험 정확도를 검증한 것은 아니다.
