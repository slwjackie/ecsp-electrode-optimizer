> **Historical v7.9.1 document.** v7.9.2 replaces the connected comb baseline with the hidden-bus vertical 2-anode/2-cathode reference. See `AREA_MATCHED_STAGGERED_BASELINE_V7_9_2_KR.md`.

# v7.9.1 변경내역

- `python/ecsp_nsga2/baselines.py` 추가
  - 3-finger-pair area-matched staggered 생성
  - design grid와 physics grid에서 면적·간격·component 재검사
  - geometry-only parameter calibration
  - 비교 CSV/JSON/PNG 작성
- `python/ecsp_nsga2/workflow.py`
  - 최종 NSGA-II 완료 후 staggered 한 형상 자동 평가
  - baseline을 population/Pareto/recommendation에서 명시적으로 제외
  - AI 추천형상 대비 네 목적함수 개선율 출력
- `python/evaluate_area_matched_staggered.py`
  - 기존 완료 run에 baseline만 backfill
- `tools/evaluate_area_matched_staggered_existing_run.sh`
  - 기존 run용 실행 wrapper
- 모든 NSGA-II YAML에 `baselines.area_matched_staggered` 설정 추가
- 20×20/193 및 25×25/241, 극당 20%, 폭 1–5 mm, 간격 2 mm 조건 검증 추가
- 물리식과 C++ engine은 v7.9.0에서 변경하지 않음

- baseline 면적은 최종 추천형상의 실제 physics-grid 양·음극 접촉면적 평균을 우선 사용하고, 값이 없을 때만 설정 목표를 사용하도록 강화
