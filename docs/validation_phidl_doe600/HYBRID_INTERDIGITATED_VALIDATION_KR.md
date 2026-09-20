# Hybrid interdigitated geometry 검증

검증 run:

`runs/doe600_phidl_hybrid_geometry_validation_v2`

실행은 `--execute-physics`와 `--postflame` 없이 수행했다. 따라서 topology 100개와
Stage 1 geometry 300개만 생성했으며 Pre-flame/Post-flame physics는 실행하지 않았다.

## 결과

| 항목 | 결과 |
|---|---:|
| topology | 100 |
| interdigitated / separated | 50 / 50 |
| 1A1C / 2A2C | 50 / 50 |
| 네 style/component 조합 | 각 25 |
| Stage 1 geometry | 300 |
| interdigitated / separated variants | 150 / 150 |
| unique geometry hashes | 300 |
| unique physics-mask hashes | 300 |
| validation passed | 300 / 300 |
| maximum representative IoU | 0.7987736900780379 |
| minimum opposed-active-edge fraction | 0.5387506677268036 |
| minimum row alternation fraction | 1.0 |
| centerline segment range | 2.076351680279875–4.666183369155048 mm |
| minimum CAD opposite-polarity gap | 0.5018854444930271 mm |
| maximum design-grid area relative error | 0.009796626984126939 |
| maximum physics-grid area relative error | 0.009883295045312731 |
| physics output files | 0 |

설정 계약은 representative IoU ≤ 0.80, opposed-active-edge fraction ≥ 0.45,
centerline segment 0.5–5.0 mm, opposite-polarity gap ≥ 0.5 mm, 양 grid의
per-polarity area relative error ≤ 0.01이다.

PHIDL DOE 관련 테스트는 Desktop 설치 위치에서 60개 모두 통과했다.

```text
60 passed in 25.83s
```

동일 run을 geometry-only `--resume`으로 다시 실행했을 때 기존 300개를 읽어
`candidate_count: 300`으로 완료됐다.

## 산출물

- `topology_library/topology_contact_sheet.png`: topology 대표 100개
- `topology_library/topology_specs.json`: graph/placement spec과 대표 검증 결과
- `topology_library/topology_summary.csv`: style, component pair, orientation, IoU 요약
- `stage1/contact_sheet.png`: Stage 1 variant 300개
- `stage1/geometry_manifest.json`: 300개 artifact와 metadata hash
- `stage1/geometries/`: PNG/JSON/NPZ 원본
- `stage1/generation_rejections.csv`: fail-closed 거부 사유

## Pre-flame만 이어서 실행

```bash
cd /Users/kimjiin/Desktop/ECSP_v8_4_2_A100CPU8_GeometrySafe
export PATH="/Users/kimjiin/anaconda3/envs/ecsp-m2/bin:$PATH"

python tools/run_phidl_doe600.py \
  --run-dir runs/doe600_phidl_hybrid_geometry_validation_v2 \
  --resume \
  --execute-physics
```

이 명령은 저장된 Stage 1 geometry에서 이어 600개 Pre-flame, final20 선정,
동일 면적 staggered baseline의 Pre-flame 비교까지 수행한다. `--postflame`을
붙이지 않았으므로 Post-flame은 실행하지 않는다.
