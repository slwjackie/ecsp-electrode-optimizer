# v8.4.1 물리식·가정 검토

## 검토 범위와 출처

직접 기반 ZIP은 v8.4.0이며 이 자료는 실제 소스, 현재 production YAML, evaluator가 병합한 설정, 기존·신규 시험을 대조했습니다. 현재 인덱스는 Python/C++ 132개 파일·58,314줄, Python 함수 1,396개입니다. 이 수는 테스트·검증 도구를 포함합니다. 전체 구문·입력 인덱스와 핵심 물리/수지/스케줄 경로를 확인했으며, 모든 분기·모든 A100 환경·모든 조성에서 결함이 없음을 증명한 것은 아닙니다.

첨부 A/B의 지배방정식 형태를 사용한 부분과 코드가 선택한 구성식·운영상 기준을 분리했습니다. A/B의 원시 텍스트가 완결하지 않는 단위·계면 closure를 원문에 있는 것처럼 주장하지 않습니다. 타 문헌 prior의 좌표를 이번 작업에서 원 논문으로 재검증하지 못했습니다.

## 활성 핵심 방정식 (수치 limiter 적용 전의 골격)

\[
\boxed{\begin{aligned}
&\mathbf E=-\nabla_\parallel\phi,\quad
\mathbf N_i=-D_i\nabla_\parallel c_i-\frac{z_iFD_i}{RT}c_i\nabla_\parallel\phi,\quad
\partial_t c_i+\nabla_\parallel\cdot\mathbf N_i=S_i,\\
&\sigma_{\rm ion}=\frac{F^2}{RT}\sum_i z_i^2D_i c_i,\qquad
\nabla_\parallel\cdot\mathbf J=(\chi_a j_a-\chi_c j_c)/\delta_s,\\
&r_\ell=\exp[L_\ell(\alpha_\ell)-E_\ell(\alpha_\ell)/(RT)],\quad
X=w_1\alpha_1+w_2\alpha_2,\\
&\mathbf U_{\rm core}=(\rho,\rho u,\rho v,\rho E,\rho\alpha_1,\rho\alpha_2)^T,\\
&\partial_t\rho+\nabla\cdot(\rho\mathbf u)=0,\qquad
\partial_t(\rho\mathbf u)+\nabla\cdot(\rho\mathbf u\otimes\mathbf u+p\mathbf I)=0,\\
&\partial_t(\rho E)+\nabla\cdot[(\rho E+p)\mathbf u]
=\nabla\cdot(k\nabla T)+q_J+q_{\rm echem}+\rho\sum_{\ell=1}^2 Q_\ell\widetilde r_\ell-q_{\rm loss},\\
&\partial_t(\rho\alpha_\ell)+\nabla\cdot(\rho\mathbf u\alpha_\ell)=\rho\widetilde r_\ell,\\
&q_J=\sigma|\nabla_\parallel\phi|^2+q_{J,\rm normal},\quad
q_{\rm echem}=\sum_r\chi_r j_r\Delta H_r/(n_rF\delta_s),\\
&q_{\rm loss}=h(T-T_\infty)/\delta_s+\varepsilon\sigma_{SB}(T^4-T_\infty^4)/\delta_s,\\
&p=A+B[(\rho/\rho_0)^N-1],\quad
E=e+|\mathbf u|^2/2,\\
&e=\int_{\rho_0}^{\rho}\!\frac{p(r)}{r^2}\,dr+\int_{T_{ref}}^T\!c_p(\theta)\,d\theta.
\end{aligned}}
\]

\(\widetilde r_\ell\)는 alpha 상한과 공유 반응물 재고를 고려해 받아들인 반응률입니다. 두 채널의 accepted increment로만 발열을 넣습니다. LP/PVA/product-water/Faradaic stock의 6개 추가 수송변수를 포함하여 실제 state는 12열입니다. **화학열에 질량전환 가중치 w를 다시 곱하지 않습니다.**

BV 식은 일반식의 이름만으로 완전가역이라고 설명하지 않습니다. 구현은 \(\eta=[\Delta\phi-E_{eq}]_+\), 비음수 j, reverse availability 및 exp limit가 있는 채널 모델입니다.

## 식별 식·수치·진단 전체 목록

| ID | 물리항목 | 계산식 | 정립성 | 활성경로 | 논문_코드근거 | 적용한계 |
| --- | --- | --- | --- | --- | --- | --- |
| P01 | 전기장 | E = −∇φ | 정전기 정의 | BC/기존; Reactive 전원 ON callback | 첨부 논문 A: JKSC 29(4), 2024, 21–33; DOI 10.15231/jksc.2024.29.4.021 식(2); python/ecsp_v6/physics/electrochem.py:349-477 | 시간변화 자기유도·전자기파 제외 |
| P02 | 이온 수송 | Ni = −Di∇ci − zi F Di ci ∇φ/(RT); ∂t ci + ∇·Ni = Si | Nernst–Planck 형태는 정립 | BC; 후속 전원 ON에서 재계산 | 첨부 논문 A: JKSC 29(4), 2024, 21–33; DOI 10.15231/jksc.2024.29.4.021 식(11)–(15),(24); python/ecsp_v6/physics/species.py | 전기중성 2이온+물 모델; 응축상 BC 대류=0. 후속 보존 수송에는 ci u 추가; 농축계 보정 없음 |
| P03 | 이온 전도도 | σion = F² Σ zi² Di ci/(RT) | Nernst–Einstein 구성관계 | BC | 첨부 논문 A: JKSC 29(4), 2024, 21–33; DOI 10.15231/jksc.2024.29.4.021 식(13),(16); python/ecsp_v6/physics/bc_global.py:401-474 | 이상 수송 가정; σmin/max clamp는 수치/구성 제한 |
| P04 | 전류·전하보존 | J = −σ∇φ + Jdiff; ∇parallel·J = (χa ja − χc jc)/δs | 전류보존을 두께평균 | surface-contact BC | 첨부 논문 A: JKSC 29(4), 2024, 21–33; DOI 10.15231/jksc.2024.29.4.021 식(16),(17); python/ecsp_v6/physics/electrochem.py:2811-3052 | 2D 체적 source는 표면 주입을 평균한 것. 전극 아래 추진제 제거 안 함 |
| P05 | 수송계수 온도의존 | D(T)=Dref exp[−Ea/R(1/T−1/Tref)] | Arrhenius 형태의 경험 구성식 | BC | python/ecsp_v6/physics/bc_global.py:334-398 | 형태 자체가 LP/PVA 정확성을 보장하지 않음; 각 Dref/Ea nominal |
| P06 | 전극 계면반응 | j=[j0(a exp(βnFη/RT)−b exp(−(1−β)nFη/RT))]+; η=[Δφ−Eeq]+ | Butler–Volmer 기반 제한된 변형 | BC/기존 | 첨부 논문 A: JKSC 29(4), 2024, 21–33; DOI 10.15231/jksc.2024.29.4.021 식(7); python/ecsp_v6/physics/electrochem.py:1210-1272 | η·j 비음수, reverse availability 고정, exp clip; 완전 가역 다종 BV와 다름 |
| P07 | Faraday 반응률 | νe″=j/(nF); species sink=stoichiometry×j/F/δs | Faraday 법칙 | BC/기존/후속 ON | 첨부 논문 A: JKSC 29(4), 2024, 21–33; DOI 10.15231/jksc.2024.29.4.021 식(8),(15); python/ecsp_v6/physics/electrochem.py:3055-3327 | 각 채널 반응·전자수·소모계수의 실제 화학적 정당성은 별도 검증 |
| P08 | 벌크·법선 Joule 발열 | qJ‴=σ\|∇parallelφ\|² + Σcontact jn² ℓn/(σ δs) | Ohmic 손실 + 유효 접촉저항 | BC/기존/후속 ON | python/ecsp_v6/physics/electrochem.py:206-347; python/ecsp_v6/physics/electrochem.py:3329-3458 | 실제 구현은 harmonic face conductance의 보존 전력; 논문 A 식(9)의 \|∇φ\|\|Jtotal\|와 일반적으로 같지 않음 |
| P09 | 계면반응 발열 | qechem‴ = Σr χr jr ΔHr/(nr F δs) | 반응엔탈피×반응률 구조 | BC/기존/후속 ON | 첨부 논문 A: JKSC 29(4), 2024, 21–33; DOI 10.15231/jksc.2024.29.4.021 식(10),(23); python/ecsp_v6/physics/electrochem.py:855-867 | 채널 ΔH nominal·양수. BC는 별도 jη 활성화 열 OFF; 전체 전기/화학 에너지 기준 검증 필요 |
| P10 | 열전도 | qcond‴ = ∇·(k(T)∇T) | Fourier 법칙 | BC/기존/두 후속 | 첨부 논문 A: JKSC 29(4), 2024, 21–33; DOI 10.15231/jksc.2024.29.4.021 식(23); 첨부 논문 B: KSPE 28(3), 2024, 31–40; DOI 10.6108/KSPE.2024.28.3.031 식(2); python/ecsp_nsga2/propagation.py:288-323 | 평면 내 Fourier; 깊이 온도장 직접 계산하지 않음 |
| P11 | 열손실 | qloss‴ = h(T−T∞)/δs + εσSB(T⁴−T∞⁴)/δs | Newton 냉각·Stefan–Boltzmann | BC/기존/두 후속 | python/ecsp_reactive/condensed/solver.py:105-113 | h,ε는 유효 경계조건; 주변 기체 유동/복사 전달식 미해석 |
| P12 | BC 응축상 에너지 | ρ cp(T) ∂tT = ∇·(k∇T)+qJ+qechem+ρΣQi ri−qloss | 에너지보존 + 축약 구성식 | BC/기존 propagation | 첨부 논문 A: JKSC 29(4), 2024, 21–33; DOI 10.15231/jksc.2024.29.4.021 식(23); python/ecsp_v6/physics/bc_global.py:2591-3985; python/ecsp_nsga2/propagation.py:663-2160 | 고정 밀도·무대류. 기존 continuation explicit T 갱신; 새 모델은 cp 적분 에너지 |
| P13 | 두 채널 분해 | ri = exp[Li(αi)−Ei(αi)/(RT)]; αi≥1 → ri=0 | 변환율 의존 Arrhenius 경험 속도론 | BC/두 후속 | python/ecsp_v6/physics/bc_global.py:617-632; python/ecsp_reactive/condensed/chemistry.py:57-60 | Li=ln(Af/s⁻¹); (1−α)^n 추가 금지. 두 첨부 논문이 전체 LUT를 제공한 것은 아님 |
| P14 | 반응 진행과 재고 | X=w1α1+w2α2; ξ=ξmax X; ΔLP=−1.45Δξ, ΔPVA=−Δξ, ΔH2Oprod=2Δξ | 정해진 global stoichiometry의 수지 | BC/두 후속 | python/ecsp_v6/physics/bc_global.py:643-757; python/ecsp_reactive/condensed/chemistry.py:62-99 | 1.45 LiClO4+PVA repeat→2 CO2+2 H2O+0.4 O2+1.45 LiCl. 상세 반응경로/가스수송 아님 |
| P15 | 응축상 질량·운동량 | ∂tρ+∇·(ρu)=0; ∂t(ρu)+∇·(ρu⊗u+pI)=0 | Euler 보존식 | 새 Reactive/reference | 첨부 논문 B: KSPE 28(3), 2024, 31–40; DOI 10.6108/KSPE.2024.28.3.031 식(1); python/ecsp_reactive/condensed/finite_volume.py:134-158 | 비점성 연속체; 고체 전단응력/탄성/점성/기화 source 없음 |
| P16 | 단일 응축상 총에너지 | ∂t(ρE)+∇·[(ρE+p)u]=∇·(k∇T)+qJ+qechem+ρΣQi ri−qloss | Euler 에너지에 Fourier/열원 결합 | 새 Reactive | python/ecsp_reactive/condensed/solver.py:136-164; python/ecsp_reactive/condensed/tensor_math.py:317-337 | E=e+\|u\|²/2; 독립 solid.py 온도 중복 적분 없음; 단순 순수 Euler가 아니라 Euler–Fourier |
| P17 | 두 보존 진행변수 | ∂t(ραi)+∇·(ρuαi)=ρri; i=1,2 | 반응성 보존 스칼라 | 새 Reactive | python/ecsp_reactive/condensed/chemistry.py:62-99; python/ecsp_reactive/condensed/tensor_math.py:159-177 | α1·α2 별개 유지. 시약 부족 공통 제한율로 실제 반응량과 열 동시 제한 |
| P18 | 6개 재고 보존수송 | ∂tcj+∇·(u cj)=Sj (+ NP migration/diffusion and Faradaic source when powered) | 종/재고 수지 | 새 Reactive | python/ecsp_reactive/condensed/chemistry.py; python/ecsp_reactive/condensed/tensor_electrical.py | 추가 6성분은 mol/m³이며 추가 bulk density가 아님. LP/PVA/생성수 수지≠모든 생성물 종별 해석 |
| P19 | Tait EOS | p=A+B[(ρ/ρ0)^N−1]; c²=(BN/ρ0)(ρ/ρ0)^(N−1) | 정립된 barotropic EOS 형태 | 새 Reactive/reference | 첨부 논문 B: KSPE 28(3), 2024, 31–40; DOI 10.6108/KSPE.2024.28.3.031 식(3); python/ecsp_reactive/condensed/thermo.py:88-90 | A,B,N,ρ0는 미보정. p는 T/e에 독립; 열팽창·에너지 유도 압력파 없음 |
| P20 | caloric completion | e(ρ,T)=∫ρ0^ρ p(r)/r² dr + ∫Tref^T cp(θ)dθ | EOS를 닫는 명시적 추가 가정 | 새 Reactive | python/ecsp_reactive/condensed/thermo.py:103-104 | 첨부 논문 직접 제공식 아님. 이 barotropic 완성에서는 cp=cv, 열팽창 0; k/cp 표 유지 |
| P21 | 내부 반응경계 | φr=Xfront−X; Γr={X=Xfront}; signed distance 재구성 | 결과의 기하학적 정의/진단 | 새 Reactive/기존 propagation | python/ecsp_reactive/condensed/solver.py:297-379; python/ecsp_nsga2/propagation.py:605-660 | Xfront=.5는 운영 기준. 실제 외곽 material interface/ablation EOS분할이 아님 |
| P22 | 원 논문 material level set | ∂tψ+u∂xψ+v∂yψ=0 | 수동 경계 이류식 | 독립 paper-reference만 | 첨부 논문 B: KSPE 28(3), 2024, 31–40; DOI 10.6108/KSPE.2024.28.3.031 식(4); python/ecsp_reactive/level_set.py | 신규 2-channel 반응경계는 이 식을 독립 적분하지 않음 |
| P23 | 초기 혼합물 밀도·농도 | ρmix=(Σmi)/(Σmi/ρi); ci=(mi/Mi)/V | 부피 가산·완전 해리 근사 | 조성 초기화 | python/ecsp_v6/composition.py:60-134 | 기공/수축/비이상체적/염 해리도 미해석; 실제 경화 밀도 측정 권장 |
| N01 | 유한체적/공간·시간 적분 | harmonic face k, WENO5-JS, HLLC(or HLL), SSPRK(3,3) | 수치해법; 물리법칙 아님 | 새 Reactive | python/ecsp_reactive/condensed/finite_volume.py; python/ecsp_reactive/condensed/solver.py:166-181 | 음성면 1차 donor fallback, 거부 후 dt/2. 상수 목적·수치 제한과 물성을 구별 |
| D01 | 전선 유효 속도 | veff = ΔA/(Lbar_front Δt) | 반응면적 기반 진단 정의 | 두 후속 | python/ecsp_nsga2/propagation.py:557-602 | 실험적 표면 후퇴율/국부 법선 연소속도와 동일 아님; lab-frame 지표 |
| D02 | 전선 불균일도 | CV(tarrival \| observed) + (1−arrival coverage) | 운영상 평가 proxy | 두 후속 | python/ecsp_nsga2/propagation.py:521-541; python/ecsp_reactive/condensed/solver.py:297-379 | 관측되지 않은 셀에 coverage penalty. 물리 PDE 아님 |
| D03 | 전류집중/추천 순위 | CJ=J99/mean(J); Pareto sorting; normalized utopia distance | 최적화·진단 정의 | NSGA-II | python/ecsp_nsga2/nsga2.py; python/ecsp_nsga2/workflow.py | CJ는 절연파괴 모델 아님. 단일 추천은 정규화/선호 의존 |
| L01 | 구형 연화·전도 승수 | fliq,fast; Ddry/Dwet mixture × water/GLY/BA/fast-ion factors | 경험 proxy | 기존 모델만 | python/ecsp_v6/physics/electrochem.py:72-165 | BC 전자/액상/조성 승수 비활성; 일반적인 고체–액체 상변화 해석 아님 |
| L02 | 구형 단일 분해 | r=Aexp(−Ea/RT)(1−α)^n oxidizer^m sigmoid((T−Tact)/20K) | Arrhenius에 임의 gate 추가 | 기존 모델만 | python/ecsp_v6/physics/coupled.py | BC 2-channel에는 해당 sigmoid/A/E/n 미사용 |
| L03 | 구형 blocking/activity | passivation·gas coverage ODE, extended_debye_huckel_proxy | 경험적 폐쇄식 | 기존 모델만 | python/ecsp_v6/physics/electrochem.py:610-663; python/ecsp_v6/physics/electrochem.py:553-577 | BC에서 OFF; 계수 보정자료 미제공 |

## 가정·proxy·범위제한 전체 검토표

| ID | 경로 | 가정_선택 | 분류 | 의미_한계 | 코드근거 | 검증_보정 |
| --- | --- | --- | --- | --- | --- | --- |
| A01 | BC/두 후속 | 고정 유효두께 1 mm의 평면 표면 해석 | 차원축약 | 깊이별 온도·전위·전극 열용량 미해석 | python/ecsp_v6/physics/electrochem.py:2811-3052 | 실제 두께·접촉면·열조건 고정; 3D/두께 민감도 |
| A02 | BC/두 후속 | surface-contact overlay 전체 추진제 유지 | 기하 모델 | 전극 픽셀은 재료를 삭제하는 구멍이 아님 | python/ecsp_v6/physics/geometry.py | 제작 단면/실제 접촉 확인 |
| A03 | 전체 | 분리 전극 모두 hidden bus로 동일 전압 | 경계 가정 | 배선 저항/발열/전압강하/floating island 미해석 | config/nsga2_bc_reactive_a100_cpu8.yaml geometry | 배선 제작성/도통 확인 |
| A04 | BC | 두 단가 이온·전기중성 투영 | 전해질 축약 | Poisson/double-layer를 공간 해상하지 않음 | python/ecsp_v6/physics/species.py | 전류수지·이온수지·EIS 점검 |
| A05 | BC | Nernst–Einstein 이상 이온수송 | 구성 가정 | 농축고분자 전해질의 activity/ion correlation 미보정 | python/ecsp_v6/physics/bc_global.py:401-474 | 온도별 전도도; 필요시 transference |
| A06 | BC | σ∈[1e−4,0.5] S/m | 수치/구성 cap | 발동하면 실제 수송 관계 변경. 미작동이라고 가정 금지 | python/ecsp_v6/physics/bc_global.py:401-474 | cap 민감도·발동분율·전류 순위 |
| A07 | BC | BV η와 j 비음수, 역반응 availability=1 | 방향 제한 | 가역 BV 전체를 푸는 것과 다름 | python/ecsp_v6/physics/electrochem.py:1210-1272 | 극성별 I–V 및 기준전위 검증 |
| A08 | BC | 4개 반응채널 Eeq,n,ΔH 선택 | 반응 모델 | 보편 상수 대신 선택 반응의 가정; 물 1.23 V를 각 극에 적용한 기준 확인 | python/ecsp_v6/physics/electrochem.py:3055-3327 | 반응 정의/전위 기준/생성물 확인 |
| A09 | BC | 계면 ΔH 양수, 발열 음수 절삭; jη 추가열 OFF | 열분배 가정 | 전체 열역학적 전기일·엔탈피 분배 검증 필요 | python/ecsp_v6/physics/electrochem.py:855-867 | I(t)+IR/열량계로 net interface heat 확인 |
| A10 | BC | 접촉 법선 전도길이 30 µm | 유효 길이 | 1 mm 체적화 두께와 다름 | python/ecsp_v6/physics/electrochem.py:3329-3458 | 계면 전압강하/접촉저항 보정 |
| A11 | BC/두 후속 | 단일 global stoichiometry + 2단계 반응 | 화학 축약 | 두 채널이 상세 elementary chemistry가 아님 | python/ecsp_v6/physics/bc_global.py:643-757 | 다중 가열 DSC/TGA 및 생성물 정보 |
| A12 | BC/두 후속 | 그림에서 근사 추출한 E(α),L(α)와 문헌 열량 | 문헌 prior | 첨부 논문 두 편에 해당 LUT 원시값이 모두 있지 않음 | config/nsga2_bc_reactive_a100_cpu8.yaml bc_global.kinetics | 원 데이터 입수/동일 조성 DSC/TGA 보정 |
| A13 | BC/두 후속 | Q1,Q2는 initial bulk의 채널 기여량 | 단위·기준 가정 | w를 다시 곱하지 않음; 조성 바뀌면 전이 검증 필요 | python/ecsp_reactive/condensed/chemistry.py:62-99 | DSC 누적열과 단위 기준 확인 |
| A14 | BC/두 후속 | 두 채널이 같은 LP/PVA 재고를 공유 | resource closure | 시약 부족 시 공통 scale로 α와 q 동시 제한 | python/ecsp_reactive/condensed/chemistry.py:62-99 | 재고수지/열수지; 이 자체가 elementary 반응 추정은 아님 |
| A15 | BC/두 후속 | 생성수는 mobile BV water와 분리 | 화학 bookkeeping | 분해 생성수가 즉시 재전기분해되는 모델 아님 | python/ecsp_reactive/condensed/chemistry.py | 수분 수송/생성수 가용성 민감도 |
| A16 | 조성 | 경화 후 물 20 wt.% | nominal 초기조건 | 투입수 20% 잔존과 다름; 코드상 잔존율 약17.7936% | config/nsga2_bc_reactive_a100_cpu8.yaml physics | 경화 질량수지/물특이적 분석 |
| A17 | 조성 | 부피가산 밀도·완전 해리 | 혼합물 근사 | 기공·경화수축·농축계 해리 미고려 | python/ecsp_v6/composition.py | 경화 밀도·치수 측정 |
| A18 | BC/두 후속 | cp,k 표 및 범위 밖 끝값 유지 | 물성 근사 | T별 데이터 실측 아님; 잠열/기공/상변화 미포함 | python/ecsp_v6/physics/bc_global.py:334-398 | cp/열확산·냉각곡선 |
| A19 | BC/두 후속 | 대류 h=12,ε=.85 유효 열손실 | 경계 가정 | 기체 CFD/복사장/전극열전도 없음 | python/ecsp_reactive/condensed/solver.py:105-113 | IR 방사율 보정과 냉각곡선 |
| A20 | 후속 기본 | onset 직후 전원 OFF | 운영 정책 | 전압 유지 실험과 다른 문제 | config/nsga2_bc_reactive_a100_cpu8.yaml propagation_refinement | 시험 전원파형 일치; 유지시 recomputed+graph OFF |
| A21 | 후속 ON | 매 SSPRK stage NP/BV 재계산 | 분할 결합 | 전자기장을 연속 시간 implicit monolithic으로 푼 것은 아님 | python/ecsp_reactive/condensed/electrical.py; python/ecsp_reactive/condensed/tensor_electrical.py | dt/결합 주기/수지 수렴 |
| A22 | 새 Reactive | 단일 응축상 Euler, 전단·점성 제외 | 역학 가정 | 응축상 유체형 연속체이지 고체 탄성파/손상 해석 아님 | python/ecsp_reactive/condensed/finite_volume.py | 실제 기계거동이 필요한지 판단 |
| A23 | 새 Reactive | barotropic Tait p=p(ρ) | EOS 한계 | 동일ρ에서 가열은 p를 바꾸지 않음; 열팽창/열유도 압력파 없음 | python/ecsp_reactive/condensed/thermo.py | A/B/N 음향·압축자료 보정 또는 다른 EOS 별도연구 |
| A24 | 새 Reactive | BC cp 적분 + Tait cold energy | 추가 caloric closure | 첨부 논문 직접식 아님; cp=cv와 열팽창0를 내포 | python/ecsp_reactive/condensed/thermo.py | 열역학적 적용범위 명시 |
| A25 | 새 Reactive | 초기 rho 균일,u=v=0 | handoff 가정 | BC는 u,p를 계산하지 않아 초기 역학상태가 별도 가정 | python/ecsp_reactive/condensed/handoff.py:45-176 | 초기유동/압력 자료 없음을 명시 |
| A26 | 새 Reactive | HLLC exact stationary fast path | 정확 불변부분공간 최적화 | 이 EOS에서만 압력균일·정지 해는 역학이 안 변함; 인위적 감쇠 아님 | python/ecsp_reactive/condensed/finite_volume.py:126-131 | off/ON 결과대조; HLL은 절대 이 shortcut 적용 안 함 |
| A27 | 새 Reactive | 질량·에너지 단일 장; 생성물 continuum 잔류 | 보존 모델 | 가스 분사/압력 배출/외곽 ablation 없음 | python/ecsp_reactive/condensed/solver.py:136-164 | 고체→기체로 해석해서는 안 됨 |
| A28 | 새 Reactive | 반응/미반응 영역 같은 Tait와 물성 | material closure | 서로 다른 EOS를 선택하는 active material-interface가 아님 | python/ecsp_reactive/condensed/solver.py:297-379 | 외곽 경계/상전이와 내부 반응경계 구별 |
| A29 | 두 후속 | Xfront=.5, reacted area .5 established | 운영상 문턱 | 물리 반응식이 아닌 위치·완료 판정; 문턱 민감도 필요 | python/ecsp_nsga2/propagation.py:605-660 | 실험 영상과 동일한 전선 정의 |
| A30 | BC | 523.15K + X≥.01 + area≥.01 onset | 분해 개시 proxy | 가시 화염 점화/소화 판정 아님 | config/nsga2_bc_reactive_a100_cpu8.yaml bc_global.onsetCriterion | I(t)/IR/영상 동시 검증 |
| A31 | 두 후속 | 면적/전선길이 속도와 arrival CV | 평가 proxy | 진짜 표면 후퇴율/국부 연소속도가 아님 | python/ecsp_nsga2/propagation.py:557-602 | 질량감소·높이변화·전선속도를 별도구분 |
| A32 | 수치 | WENO5-JS+HLLC/HLL+SSPRK33, face positivity fallback | 수치 선택 | 논문 B가 모든 구체적 flux/variant를 지정한 것은 아님 | python/ecsp_reactive/condensed/finite_volume.py | 격자/시간/fallback/보존 convergence |
| A33 | 수치 | rate cap1000/s, exp clip[−100,60], temp bounds | 수치 제한 | raw-rate cap은 기본 후보거부; 명시 허용시 모델 변함 | python/ecsp_reactive/condensed/chemistry.py:62-99; python/ecsp_nsga2/propagation.py:412-426 | 발동율·상한 민감도 기록 |
| A34 | NSGA-II | 정규화 utopia 추천·Vmin 벌점·전류혼잡 | 알고리즘 선호 | 지배방정식 아님. 새로운 전극 전체 최적성 보장 아님 | python/ecsp_nsga2/workflow.py | 독립형상 검증/seed/예산/정규화 민감도 |
| A35 | 기존 경로 | 연화 sigmoid·조성전도 승수·단일열분해 gate | 경험 proxy | BC에서 비활성; 코드에 남아 있다고 활성으로 해석 금지 | python/ecsp_v6/physics/electrochem.py:72-165 | 사용 경로 명시 |
| A36 | 기존 경로 | passivation/gas coverage/activity proxy | 경험 proxy | BC는 OFF. 실제 기포/박막 CFD 아님 | python/ecsp_v6/physics/electrochem.py:610-663 | 사용시 별도 실험 보정 |
| A37 | 독립 reference | 별도 solid T,단일 λ,passive ψ | 미결합 reference | 이번 2-channel 통합에서는 실행하지 않음 | python/ecsp_reactive/solver.py | 논문재현용 실험 경로로 분리 |
| A38 | GPU | FP64, 독립 case dt, 같은 LUT·재고·경계 | 계산 구현 | 물리를 간소화하거나 정밀도를 줄인 최적화 아님 | python/ecsp_reactive/condensed/tensor_math.py | CPU↔GPU parity는 target GPU preflight 필요 |
| A39 | GPU | batch8·host2+worker6·graph 선택 | 운영 시작값 | A100 실측 최적 batch/속도 아님 | config/nsga2_bc_reactive_a100_cpu8.yaml post_onset.execution | 동일 물리 benchmark 후 batch만 조정 |
| A40 | 검증 | 실험 없이 수치 시험으로 순위 인증하지 않음 | 검증 범위 | unit pass ≠ 절대 점화/최적 형상 정확성 | docs/VALIDATION_V8_4_1_KR.md | 실측 보정+독립 형상 순위 검증 |

## 논문·공식 문서와의 대응

| 문헌 | 항목 | 위치 | 대응 | 차이 | 출처 |
| --- | --- | --- | --- | --- | --- |
| A | 전기장·NP·BV·Faraday | 식(2),(7),(8),(11)–(17) | BC 구조 기반 | 입력값·2D overlay·비음수 BV 등은 추가/변형 | 첨부 논문 A: JKSC 29(4), 2024, 21–33; DOI 10.15231/jksc.2024.29.4.021 |
| A | 고온 무대류 전기화학 열식 | 식(22)–(24) | BC pre-onset 대응 | 2채널 LUT/특정 조성 물성은 두 첨부파일이 완결하지 않음 | 첨부 논문 A: JKSC 29(4), 2024, 21–33; DOI 10.15231/jksc.2024.29.4.021 |
| A | 유동/기체 화학·추력 | 식(25)–(28), 후반 CEA | 새 프로필에서 미구현 | OpenFOAM/다종 가스/노즐추력 모델이라고 하지 않음 | 첨부 논문 A: JKSC 29(4), 2024, 21–33; DOI 10.15231/jksc.2024.29.4.021 |
| A | Joule term | 식(9) \|∇φ\|\|J\| | 코드는 conductive σE²/face power | diffusion current가 있을 때 일반적으로 동일식 아님 | 첨부 논문 A: JKSC 29(4), 2024, 21–33; DOI 10.15231/jksc.2024.29.4.021 |
| B | 반응 Euler·고체열·Tait | 식(1)–(3) | Euler/Tait 구조 차용 | 새 모델:2채널+Fourier+BC cp caloric completion; 원문 그대로 복제 아님 | 첨부 논문 B: KSPE 28(3), 2024, 31–40; DOI 10.6108/KSPE.2024.28.3.031 |
| B | material level-set | 식(4) | 독립 reference에만 그대로 수동 이류 | 새 모델은 X 등고선 진단; 외곽 material boundary 미완성 | 첨부 논문 B: KSPE 28(3), 2024, 31–40; DOI 10.6108/KSPE.2024.28.3.031 |
| B | 조성/격자/도메인 | 10×10mm 단면,100×100선정,W5% 조성 | 현재20×20mm surface overlay와 다름 | 논문 실험 수치가 현재 조성 검증값은 아님 | 첨부 논문 B: KSPE 28(3), 2024, 31–40; DOI 10.6108/KSPE.2024.28.3.031 |
| 외부 prior | 2nd_paper/3rd_paper/Fig8 | 설정파일의 출처 label | LUT/열량/수분의 nominal 근거표시 | 첨부 A/B의 순서를 뜻하지 않음; 원시 데이터 좌표 독립 확인 못함 | config/nsga2_bc_reactive_a100_cpu8.yaml |
| 공식 SW 문서 | CUDA graphs/synchronization | PyTorch CUDA semantics | 새 실행 구조 참조 | 물성근거 아님; 실제A100별도 시험 필요 | https://docs.pytorch.org/docs/stable/notes/cuda.html |
| 공식 SW 문서 | spawn/CPU oversubscription | PyTorch multiprocessing | CPU6+host2 budget / spawn 참조 | 8 core 기준 시작값; 실제quota따라 제한 | https://docs.pytorch.org/docs/stable/notes/multiprocessing.html |

## 상대 순위와 실험 보정

우선 필요한 자료는 경화 수분/밀도/두께, 온도별 벌크·계면 전기응답, 전류+공간 온도/냉각곡선, 복수 가열속도의 DSC/TGA입니다. 이는 현재 시료에 대한 nominal 수송·열·반응 조합을 식별하기 위한 제안이며 제공된 파일에 실제 실험결과는 없습니다. 단일 I–V로 모든 j0/β/반응엔탈피/D+/D−를 독립 식별할 수 있다고 가정하지 않습니다. 열분해 LUT의 Ea와 ln(Af)는 상관성을 유지하여 보정해야 합니다.

보정에 쓰지 않은 기준 staggered·대표 후보·근접 경쟁형의 동일 전원/두께/접촉 조건 시험으로 순위를 확인해야 합니다. 실제 물리식의 형태가 정립되었다고 해서 nominal 수치 순위를 실험 검증된 최적 형상이라고 부르면 안 됩니다.
