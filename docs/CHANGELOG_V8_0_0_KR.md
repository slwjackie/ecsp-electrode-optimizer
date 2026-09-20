# v8.0.0 변경사항 — B/C global pre-flame + condensed propagation

## 추가

- `bc_global_preflame` opt-in evaluator backend
- B/C Eq. (22)–(24) + supporting equations contract
- strict Nernst–Planck + local BV/Faraday + charge/energy conservation
- 3번 논문 non-metallized wet recipe와 cured-water basis
- 3번 논문 global reaction + two conversion-dependent exothermic channels
- `Urem`: remaining reactive LP+PVA mass fraction
- T AND Xg AND minimum-area onset
- identical-criterion Vmin search
- Eq. (32) diagnostic output without heat-source double counting
- pre-flame Pareto + near-Pareto/max-min diversity selection
- full-field T/Xg/qJ/qechem onset handoff
- post-onset condensed reaction-progress/signed-distance level-set refinement
- A_unreacted, regression velocity, established time, front nonuniformity
- same-pipeline area-matched hidden-bus 2×2 staggered comparison
- executable 23-check implementation audit

## 명시적으로 제외

- Poisson PNP direct double-layer resolution
- paper Eq. (29)–(31) limiting-current closure when full NP+BV is active
- passivation, gas coverage, liquid-fraction/glycerol/BA transport multipliers
- external gas-phase CFD, flame plume, detailed gas chemistry

## v7.9.5 보존

기존 evaluator backends, A100+CPU48 hybrid scheduler, C++ FP64 engines, M2 profiles, CAD grammar, surface-contact/hidden-bus semantics, minimum ignition voltage search, area-matched staggered generator 및 모든 기존 config는 삭제하지 않았다. 새 물리는 별도 config/backend로만 활성화된다.
