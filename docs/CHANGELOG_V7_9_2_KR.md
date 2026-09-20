# v7.9.2 변경내역

## 변경된 파일

- `python/ecsp_nsga2/baselines.py`
- `python/ecsp_nsga2/workflow.py`
- `python/ecsp_nsga2/evaluator.py`
- `python/tests/test_area_matched_staggered_baseline.py`
- `config/nsga2_condensed_phase_no_f*.yaml`
- `README_KR.md`
- `VERSION.json`

## 핵심 변경

1. 기존 좌우 contact bus + 수평 3쌍 finger comb를 제거했다.
2. area-matched staggered를 `A-C-A-C` 순서의 수직 2+2 finger로 고정했다.
3. 양극 두 팔은 top boundary에서 시작하고 동일 길이·폭을 사용한다.
4. 음극 두 팔은 bottom boundary에서 시작하고 동일 길이·폭을 사용한다.
5. 같은 극성 접합부는 hidden bus로만 표현하며 contact mask에는 넣지 않는다.
6. design 및 physics grid에서 면적, 간격, equal-dimension, component 수, 깊은 수직 중첩을 다시 검사한다.
7. 1+1 AI run에서도 baseline만 2+2 visible contacts를 허용하는 제한적 evaluator override를 추가했다.
8. 추천형상의 실제 극당 접촉면적을 baseline evaluator의 area target에도 전달해 raster 면적 비교를 일치시켰다.
9. baseline은 계속 NSGA-II와 Pareto 선택에서 완전히 제외된다.
10. C++ 물리식과 NSGA-II 생성·선택 로직은 변경하지 않았다.
