#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ecsp {


struct Config {
    int n = 193;
    double domain = 0.020;
    double layer = 0.001;
    double voltage = 260.0;
    double cathode_voltage = 0.0;
    double dt = 2.5e-4;
    double end_time = 2.0;
    double eval_time = 2.0;
    double electrical_interval = 2.5e-3;
    double ignition_T = 622.15;
    double ignition_progress_guard = 0.0;
    double ignition_rate_guard = 0.0;
    bool stop_on_ignition = false;

    // composition derived
    double density = 1261.36;
    double cation0 = 3000.0;
    double anion0 = 3000.0;
    double water0 = 30000.0;
    double lp0 = 3000.0;
    double glycerol_pva_ratio = 0.5;
    double boric_pva_ratio = 0.1;
    double effective_softening_T = 355.0;

    // electrical / transport
    double R = 8.314462618;
    double F = 96485.33212;
    double zplus = 1.0;
    double zminus = -1.0;
    double Dp_dry = 5e-15, Dm_dry = 3e-15;
    double Dp_wet = 2e-13, Dm_wet = 1.5e-13;
    double Dw0 = 2e-10;
    double Ea_Dp = 22000.0, Ea_Dm = 25000.0;
    double Tref = 298.15;
    double liquid_D_gain = 120.0;
    double water_activity_exp = 1.2;
    double glycerol_gain = 1.5;
    double crosslink_penalty = 0.8;
    double concentration_min_fraction = 1e-6;
    double concentration_max_multiple = 2.0;
    double electroneutral_relax = 1.0;
    double max_relative_concentration_change = 0.05;
    int species_substeps = 2;
    bool full_np = true;

    double sigma_e0 = 0.001;
    double sigma_e_temp_coeff = 0.006;
    double sigma_e_liquid_suppression = 0.98;
    double sigma_min = 1e-4, sigma_max = 0.5;
    bool diffusion_potential = true;

    // phase
    double transition_width = 15.0;
    double phase_relax_time = 0.08;
    double latent_heat = 0.0;
    double fast_ion_liquid_threshold = 0.1;

    // interface
    bool full_bv = true;
    bool mass_transfer_saturation = true;
    bool derive_km = true;
    double diffusion_layer = 3e-5;
    double reaction_layer_min = 3e-5;
    double contact_normal_length = 3e-5;
    double initial_contact_drop = 5.0;
    bool surface_contact_model = true;
    bool hidden_bus_assumed = true;
    bool activation_heat = true;
    double activation_heat_fraction = 1.0;
    double exp_limit = 45.0;
    double liquid_kinetics_gain = 40.0;
    double min_active_area = 0.02;

    struct Channel {
        double Eeq=0, j0=1, alpha=0.5, reverse=1, n=1, km=1e-5;
        double dH=0, water_stoich=0, salt_stoich=0, gas_yield=0;
        double ref_activity=1, nernst_exp=-1;
    } water_a, water_c, lp_a, lp_c;

    bool nernst_enabled = true;
    double activity_floor = 1e-6;
    double activity_A = 0.2, activity_B = 1.0, activity_linear = 0.0;
    double max_log10_gamma = 0.5;
    double max_nernst_shift = 0.25;

    // blocking
    bool passivation_enabled = true;
    double pass_form = 0.02, pass_remove=0.005, pass_jref=1000.0, pass_jexp=1.0;
    double pass_T0=360.0, pass_Tw=25.0;
    bool gas_coverage_enabled = true;
    double gas_cov_form=0.2, gas_cov_detach=1.0, gas_source_ref=100.0;
    double gas_thermal_insulation=0.5;

    // nonlinear robin
    int robin_min_iter = 2, robin_max_iter=120;
    double robin_phi_tol=0.01, robin_current_tol=0.005, robin_balance_tol=0.005;
    double robin_relax=0.35;
    double robin_balance_abs=1e-12;
    double gauge_margin=25.0, gauge_max_step=65.0, gauge_deriv_floor=1e-14;
    int gauge_min_iter=1, gauge_max_iter=24;
    double gauge_relax=1.0;
    int local_min_iter=8, local_max_iter=60, local_newton=2;
    double local_abs_tol=0.01, local_rel_tol=1e-4;

    // chemical
    bool chemical_enabled = true;
    double chem_A=5e9, chem_Ea=110000.0, chem_order=1.0, oxidizer_order=0.5;
    double oxidizer_consume=0.25, chem_max_rate=50.0, chem_heat=200000.0;
    double gas_molar_mass=0.03, gas_yield_mass=0.35, chem_activation_T=430.0;

    // thermal/gas
    double cp=2200.0, k=0.4, T0=298.15, Tamb=298.15, hconv=12.0;
    double emissivity=0.85, sigma_sb=5.670374419e-8, Tmin=250.0, Tmax=2500.0;
    double gas_D=3e-6, gas_loss=1.0, gas_max=5000.0;

    // solver
    int pcg_max_iter=1500;
    double pcg_rtol_static=1e-10, pcg_rtol_coupled=1e-9, pcg_atol=1e-12;
    double current_floor=1e-15, current_density_floor=1e-12;
};

static bool parse_bool(const std::string& s) {
    return s=="1" || s=="true" || s=="True" || s=="yes" || s=="on";
}
static std::map<std::string,std::string> read_kv(const std::string& path) {
    std::ifstream f(path);
    if(!f) throw std::runtime_error("cannot open config: "+path);
    std::map<std::string,std::string> m; std::string line;
    while(std::getline(f,line)) {
        if(line.empty() || line[0]=='#') continue;
        auto p=line.find('='); if(p==std::string::npos) continue;
        m[line.substr(0,p)] = line.substr(p+1);
    }
    return m;
}
template<class T> static void set_num(const std::map<std::string,std::string>& m,const char* k,T& v){auto it=m.find(k); if(it!=m.end()){std::istringstream ss(it->second); ss>>v;}}
static void set_bool(const std::map<std::string,std::string>& m,const char* k,bool& v){auto it=m.find(k); if(it!=m.end())v=parse_bool(it->second);}
static Config load_config(const std::string& path) {
    auto m=read_kv(path); Config c;
#define S(k,f) set_num(m,k,c.f)
#define B(k,f) set_bool(m,k,c.f)
    S("n",n);S("domain",domain);S("layer",layer);S("voltage",voltage);S("cathode_voltage",cathode_voltage);S("dt",dt);S("end_time",end_time);S("eval_time",eval_time);S("electrical_interval",electrical_interval);S("ignition_T",ignition_T);S("ignition_progress_guard",ignition_progress_guard);S("ignition_rate_guard",ignition_rate_guard);B("stop_on_ignition",stop_on_ignition);
    S("density",density);S("cation0",cation0);S("anion0",anion0);S("water0",water0);S("lp0",lp0);S("glycerol_pva_ratio",glycerol_pva_ratio);S("boric_pva_ratio",boric_pva_ratio);S("effective_softening_T",effective_softening_T);
    S("R",R);S("F",F);S("zplus",zplus);S("zminus",zminus);S("Dp_dry",Dp_dry);S("Dm_dry",Dm_dry);S("Dp_wet",Dp_wet);S("Dm_wet",Dm_wet);S("Dw0",Dw0);S("Ea_Dp",Ea_Dp);S("Ea_Dm",Ea_Dm);S("Tref",Tref);S("liquid_D_gain",liquid_D_gain);S("water_activity_exp",water_activity_exp);S("glycerol_gain",glycerol_gain);S("crosslink_penalty",crosslink_penalty);S("concentration_min_fraction",concentration_min_fraction);S("concentration_max_multiple",concentration_max_multiple);S("electroneutral_relax",electroneutral_relax);S("max_relative_concentration_change",max_relative_concentration_change);S("species_substeps",species_substeps);B("full_np",full_np);
    S("sigma_e0",sigma_e0);S("sigma_e_temp_coeff",sigma_e_temp_coeff);S("sigma_e_liquid_suppression",sigma_e_liquid_suppression);S("sigma_min",sigma_min);S("sigma_max",sigma_max);B("diffusion_potential",diffusion_potential);
    S("transition_width",transition_width);S("phase_relax_time",phase_relax_time);S("latent_heat",latent_heat);S("fast_ion_liquid_threshold",fast_ion_liquid_threshold);
    B("full_bv",full_bv);B("mass_transfer_saturation",mass_transfer_saturation);B("derive_km",derive_km);S("diffusion_layer",diffusion_layer);S("reaction_layer_min",reaction_layer_min);S("contact_normal_length",contact_normal_length);S("initial_contact_drop",initial_contact_drop);B("surface_contact_model",surface_contact_model);B("hidden_bus_assumed",hidden_bus_assumed);B("activation_heat",activation_heat);S("activation_heat_fraction",activation_heat_fraction);S("exp_limit",exp_limit);S("liquid_kinetics_gain",liquid_kinetics_gain);S("min_active_area",min_active_area);
    B("nernst_enabled",nernst_enabled);S("activity_floor",activity_floor);S("activity_A",activity_A);S("activity_B",activity_B);S("activity_linear",activity_linear);S("max_log10_gamma",max_log10_gamma);S("max_nernst_shift",max_nernst_shift);
    B("passivation_enabled",passivation_enabled);S("pass_form",pass_form);S("pass_remove",pass_remove);S("pass_jref",pass_jref);S("pass_jexp",pass_jexp);S("pass_T0",pass_T0);S("pass_Tw",pass_Tw);B("gas_coverage_enabled",gas_coverage_enabled);S("gas_cov_form",gas_cov_form);S("gas_cov_detach",gas_cov_detach);S("gas_source_ref",gas_source_ref);S("gas_thermal_insulation",gas_thermal_insulation);
    S("robin_min_iter",robin_min_iter);S("robin_max_iter",robin_max_iter);S("robin_phi_tol",robin_phi_tol);S("robin_current_tol",robin_current_tol);S("robin_balance_tol",robin_balance_tol);S("robin_relax",robin_relax);S("robin_balance_abs",robin_balance_abs);S("gauge_margin",gauge_margin);S("gauge_max_step",gauge_max_step);S("gauge_deriv_floor",gauge_deriv_floor);S("gauge_min_iter",gauge_min_iter);S("gauge_max_iter",gauge_max_iter);S("gauge_relax",gauge_relax);S("local_min_iter",local_min_iter);S("local_max_iter",local_max_iter);S("local_newton",local_newton);S("local_abs_tol",local_abs_tol);S("local_rel_tol",local_rel_tol);
    B("chemical_enabled",chemical_enabled);S("chem_A",chem_A);S("chem_Ea",chem_Ea);S("chem_order",chem_order);S("oxidizer_order",oxidizer_order);S("oxidizer_consume",oxidizer_consume);S("chem_max_rate",chem_max_rate);S("chem_heat",chem_heat);S("gas_molar_mass",gas_molar_mass);S("gas_yield_mass",gas_yield_mass);S("chem_activation_T",chem_activation_T);
    S("cp",cp);S("k",k);S("T0",T0);S("Tamb",Tamb);S("hconv",hconv);S("emissivity",emissivity);S("sigma_sb",sigma_sb);S("Tmin",Tmin);S("Tmax",Tmax);S("gas_D",gas_D);S("gas_loss",gas_loss);S("gas_max",gas_max);
    S("pcg_max_iter",pcg_max_iter);S("pcg_rtol_static",pcg_rtol_static);S("pcg_rtol_coupled",pcg_rtol_coupled);S("pcg_atol",pcg_atol);S("current_floor",current_floor);S("current_density_floor",current_density_floor);
    auto legacy_rtol=m.find("pcg_rtol");if(legacy_rtol!=m.end()){std::istringstream ss(legacy_rtol->second);double v=0;ss>>v;if(v>0){c.pcg_rtol_static=v;c.pcg_rtol_coupled=v;}}
#undef S
#undef B
    auto load_ch=[&](const std::string& p, Config::Channel& ch){
        set_num(m,(p+".Eeq").c_str(),ch.Eeq);set_num(m,(p+".j0").c_str(),ch.j0);set_num(m,(p+".alpha").c_str(),ch.alpha);set_num(m,(p+".reverse").c_str(),ch.reverse);set_num(m,(p+".n").c_str(),ch.n);set_num(m,(p+".km").c_str(),ch.km);set_num(m,(p+".dH").c_str(),ch.dH);set_num(m,(p+".water_stoich").c_str(),ch.water_stoich);set_num(m,(p+".salt_stoich").c_str(),ch.salt_stoich);set_num(m,(p+".gas_yield").c_str(),ch.gas_yield);set_num(m,(p+".ref_activity").c_str(),ch.ref_activity);set_num(m,(p+".nernst_exp").c_str(),ch.nernst_exp);
    };
    load_ch("water_a",c.water_a);load_ch("water_c",c.water_c);load_ch("lp_a",c.lp_a);load_ch("lp_c",c.lp_c);
    return c;
}

static void validate_config(const Config& c) {
    auto finite_positive = [](double value, const char* name) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::runtime_error(std::string(name) + " must be finite and positive");
        }
    };
    if (c.n < 5) throw std::runtime_error("grid size n must be at least 5");
    finite_positive(c.domain, "domain");
    finite_positive(c.layer, "layer");
    finite_positive(c.contact_normal_length, "contact_normal_length");
    finite_positive(c.dt, "dt");
    finite_positive(c.end_time, "end_time");
    finite_positive(c.electrical_interval, "electrical_interval");
    finite_positive(c.density, "density");
    finite_positive(c.cp, "cp");
    finite_positive(c.k, "k");
    finite_positive(c.pcg_rtol_static, "pcg_rtol_static");
    finite_positive(c.pcg_rtol_coupled, "pcg_rtol_coupled");
    finite_positive(c.pcg_atol, "pcg_atol");
    if (c.pcg_max_iter < 1) throw std::runtime_error("pcg_max_iter must be positive");
    if (c.robin_min_iter < 1 || c.robin_max_iter < c.robin_min_iter) {
        throw std::runtime_error("invalid nonlinear Robin iteration bounds");
    }
    if (c.gauge_min_iter < 1 || c.gauge_max_iter < c.gauge_min_iter) {
        throw std::runtime_error("invalid gauge iteration bounds");
    }
    if (c.local_min_iter < 1 || c.local_max_iter < c.local_min_iter) {
        throw std::runtime_error("invalid local interface iteration bounds");
    }
    if (!c.surface_contact_model) throw std::runtime_error("v7.9.4 C++ engine requires surface_contact_model=true");
    if (!c.hidden_bus_assumed) throw std::runtime_error("multi-component contacts require hidden_bus_assumed=true");
    if (!(c.voltage > c.cathode_voltage)) {
        throw std::runtime_error("voltage must exceed cathode_voltage");
    }
}

struct Grid {
    int n;
    size_t N;
    std::vector<uint8_t> anode, cathode, contact, prop;
    explicit Grid(int n_=0)
        : n(n_), N(size_t(n_)*n_), anode(N), cathode(N), contact(N), prop(N, uint8_t(1)) {}
    inline size_t id(int r,int col)const{return size_t(r)*n+col;}
};

static Grid read_mask(const std::string& path) {
    std::ifstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open mask: "+path);
    uint32_t n=0;
    f.read(reinterpret_cast<char*>(&n),sizeof(n));
    Grid g{int(n)};
    f.read(reinterpret_cast<char*>(g.anode.data()),g.N);
    f.read(reinterpret_cast<char*>(g.cathode.data()),g.N);
    if(!f) throw std::runtime_error("truncated mask file");
    for(size_t i=0;i<g.N;i++){
        if(g.anode[i]&&g.cathode[i]) throw std::runtime_error("mask polarity overlap");
        g.contact[i]=uint8_t(g.anode[i]||g.cathode[i]);
        // Surface electrodes are contact labels over the propellant.  They do
        // not remove condensed material from the 20 x 20 mm domain.
        g.prop[i]=uint8_t(1);
    }
    return g;
}

static void validate_grid(const Grid& g) {
    const auto anode_count = std::count(g.anode.begin(), g.anode.end(), uint8_t(1));
    const auto cathode_count = std::count(g.cathode.begin(), g.cathode.end(), uint8_t(1));
    const size_t propellant_count = static_cast<size_t>(
        std::count(g.prop.begin(), g.prop.end(), uint8_t(1))
    );
    if (anode_count == 0) throw std::runtime_error("mask contains no anode contact cells");
    if (cathode_count == 0) throw std::runtime_error("mask contains no cathode contact cells");
    if (propellant_count != g.N) throw std::runtime_error("surface-contact model requires a full propellant domain");
}

struct State {
    std::vector<double> T,cp,cm,water,liquid,alpha,gas,pass,coverage,phi;
    explicit State(size_t N=0):T(N),cp(N),cm(N),water(N),liquid(N),alpha(N),gas(N),pass(N),coverage(N),phi(N){}
};
struct Transport {std::vector<double> Dp,Dm,Dw,sigi,sige,sig,waterAct,fast; explicit Transport(size_t N=0):Dp(N),Dm(N),Dw(N),sigi(N),sige(N),sig(N),waterAct(N),fast(N){}};
struct Electrical {std::vector<double> Ex,Ey,Jx,Jy,Jmag,Jox,Joy,Jdx,Jdy,qj,ionicFrac; explicit Electrical(size_t N=0):Ex(N),Ey(N),Jx(N),Jy(N),Jmag(N),Jox(N),Joy(N),Jdx(N),Jdy(N),qj(N),ionicFrac(N){}};
struct Reaction {
    std::vector<double> jwa,jwc,jla,jlc,eta_wa,eta_wc,eta_la,eta_lc,q,saltSink,waterSink,gasSource;
    std::vector<uint8_t> anodeZone,cathodeZone;
    double rawA=0,rawC=0,totalI=0,mismatch=0,waterFrac=0,lpFrac=0,utilization=0,maxNernstShift=0;
    int outerIters=0,gaugeIters=0,linearSolves=0; long long linearIterations=0; double gaugeOffset=0,balanceCombined=0; bool converged=false;
    explicit Reaction(size_t N=0):jwa(N),jwc(N),jla(N),jlc(N),eta_wa(N),eta_wc(N),eta_la(N),eta_lc(N),q(N),saltSink(N),waterSink(N),gasSource(N),anodeZone(N),cathodeZone(N){}
};
struct SolverDiag {int iterations=0; double relres=0; bool converged=false; std::string method="cpp_fp64_pcg_jacobi_surface_contact";};

static inline double clampd(double x,double a,double b){return std::max(a,std::min(b,x));}
static inline double harmonic(double a,double b){double sum=a+b; return (a>0&&b>0&&sum>0)?2*a*b/sum:0.0;}
static inline double sigmoid(double x){ if(x>=0){double e=std::exp(-x);return 1/(1+e);} double e=std::exp(x);return e/(1+e); }

static std::pair<double,double> contact_centroid(const Grid& g, bool anode) {
    double sr=0.0, sc=0.0, count=0.0;
    const auto& mask=anode?g.anode:g.cathode;
    for(int r=0;r<g.n;r++) for(int col=0;col<g.n;col++) {
        const size_t i=g.id(r,col);
        if(mask[i]) {sr+=r; sc+=col; count+=1.0;}
    }
    if(count<=0.0) throw std::runtime_error("empty contact mask while computing centroid");
    return {sr/count, sc/count};
}

static void init_state(State& s,const Grid& g,const Config& c){
    const auto ac=contact_centroid(g,true);
    const auto cc=contact_centroid(g,false);
    const double vr=ac.first-cc.first, vc=ac.second-cc.second;
    const double norm2=std::max(vr*vr+vc*vc,1e-20);
    const double available=std::max((c.voltage-c.cathode_voltage)-2.0*c.initial_contact_drop,0.1);
    for(int r=0;r<g.n;r++) for(int col=0;col<g.n;col++) {
        const size_t i=g.id(r,col);
        const double projection=((r-cc.first)*vr+(col-cc.second)*vc)/norm2;
        const double xi=clampd(projection,0.0,1.0);
        s.T[i]=c.T0;
        s.cp[i]=c.cation0;
        s.cm[i]=c.anion0;
        s.water[i]=c.water0;
        s.liquid[i]=0.0;
        s.alpha[i]=0.0;
        s.gas[i]=0.0;
        s.pass[i]=0.0;
        s.coverage[i]=0.0;
        s.phi[i]=c.cathode_voltage+c.initial_contact_drop+xi*available;
    }
}

static void transport_fields(const State& s,const Grid& g,const Config& c,Transport& t){
    for(size_t i=0;i<g.N;i++){
        const double T=std::max(s.T[i],1.0);
        const double wa=clampd(s.water[i]/std::max(c.water0,1e-12),0,1);
        const double wf=std::pow(wa,c.water_activity_exp);
        double Dp=c.Dp_dry+(c.Dp_wet-c.Dp_dry)*wf;
        double Dm=c.Dm_dry+(c.Dm_wet-c.Dm_dry)*wf;
        const double pa=std::exp(-c.Ea_Dp/c.R*(1/T-1/c.Tref));
        const double ma=std::exp(-c.Ea_Dm/c.R*(1/T-1/c.Tref));
        const double fast=clampd((s.liquid[i]-c.fast_ion_liquid_threshold)/std::max(1-c.fast_ion_liquid_threshold,1e-30),0,1);
        const double phase_gain=1+c.liquid_D_gain*fast;
        const double plasticizer=1+c.glycerol_gain*c.glycerol_pva_ratio;
        const double crosslink=1/(1+c.crosslink_penalty*c.boric_pva_ratio);
        Dp*=pa*phase_gain*plasticizer*crosslink;
        Dm*=ma*phase_gain*plasticizer*crosslink;
        const double Dw=c.Dw0*std::exp(-12000/c.R*(1/T-1/c.Tref))*(1+4*fast);
        const double sigi=c.F*c.F/(c.R*T)*(c.zplus*c.zplus*Dp*std::max(s.cp[i],0.0)+c.zminus*c.zminus*Dm*std::max(s.cm[i],0.0));
        const double solid=std::max(1-c.sigma_e_liquid_suppression*s.liquid[i],0.01);
        const double sige=c.sigma_e0*std::exp(c.sigma_e_temp_coeff*(T-c.T0))*solid;
        t.Dp[i]=Dp;
        t.Dm[i]=Dm;
        t.Dw[i]=Dw;
        t.sigi[i]=sigi;
        t.sige[i]=sige;
        t.sig[i]=clampd(sige+sigi,c.sigma_min,c.sigma_max);
        t.waterAct[i]=wa;
        t.fast[i]=fast;
    }
}

static void neumann_edges(std::vector<double>& x,int n){
    for(int c=0;c<n;c++){x[size_t(c)]=x[size_t(n+c)];x[size_t(n-1)*n+c]=x[size_t(n-2)*n+c];}
    for(int r=0;r<n;r++){x[size_t(r)*n]=x[size_t(r)*n+1];x[size_t(r)*n+n-1]=x[size_t(r)*n+n-2];}
}

struct LinearSystem {
    int n; size_t N; std::vector<uint8_t> active; std::vector<double> ce,cw,cn,cs,diag,b;
    explicit LinearSystem(int n_=0):n(n_),N(size_t(n_)*n_),active(N),ce(N),cw(N),cn(N),cs(N),diag(N,0),b(N){}
};

static void diffusion_rhs(const State& s,const Transport& t,const Grid& g,const Config& c,double h,std::vector<double>& rhs){
    rhs.assign(g.N,0.0); if(!c.diffusion_potential)return; int n=g.n;
    std::vector<double> fx(size_t(n)*(n+1),0), fy(size_t(n+1)*n,0);
    for(int r=0;r<n;r++)for(int col=0;col<n-1;col++){size_t a=g.id(r,col),b=g.id(r,col+1);if(g.prop[a]&&g.prop[b]){double dp=harmonic(t.Dp[a],t.Dp[b]),dm=harmonic(t.Dm[a],t.Dm[b]);double j=-c.F*(c.zplus*dp*(s.cp[b]-s.cp[a])/h+c.zminus*dm*(s.cm[b]-s.cm[a])/h);fx[size_t(r)*(n+1)+col+1]=j;}}
    for(int r=0;r<n-1;r++)for(int col=0;col<n;col++){size_t a=g.id(r,col),b=g.id(r+1,col);if(g.prop[a]&&g.prop[b]){double dp=harmonic(t.Dp[a],t.Dp[b]),dm=harmonic(t.Dm[a],t.Dm[b]);double j=-c.F*(c.zplus*dp*(s.cp[b]-s.cp[a])/h+c.zminus*dm*(s.cm[b]-s.cm[a])/h);fy[size_t(r+1)*n+col]=j;}}
    for(int r=0;r<n;r++)for(int col=0;col<n;col++){size_t i=g.id(r,col);if(g.prop[i])rhs[i]=(fx[size_t(r)*(n+1)+col+1]-fx[size_t(r)*(n+1)+col])/h+(fy[size_t(r+1)*n+col]-fy[size_t(r)*n+col])/h;}
}

struct ContactLinearization {
    std::vector<double> anodeCurrent, cathodeCurrent, anodeSlope, cathodeSlope;
    explicit ContactLinearization(size_t N=0)
        : anodeCurrent(N), cathodeCurrent(N), anodeSlope(N), cathodeSlope(N) {}
};

static LinearSystem build_system(
    const State& s,const Transport& t,const Grid& g,const Config& c,double h,
    const ContactLinearization* contact
){
    LinearSystem A(g.n);
    std::vector<double> rhs;
    diffusion_rhs(s,t,g,c,h,rhs);
    const int n=g.n;
    const double surface_scale=h*h/std::max(c.layer,1e-30);
    for(int r=1;r<n-1;r++) for(int col=1;col<n-1;col++) {
        const size_t i=g.id(r,col);
        A.active[i]=1;
        const size_t east=i+1, west=i-1, north=i+n, south=i-n;
        // The outer perimeter is electrically insulating.  Boundary cells are
        // ghost values copied from the adjacent interior cell, so no face
        // conductance is added across the outer boundary.
        A.ce[i]=(col+1<n-1)?harmonic(t.sig[i],t.sig[east]):0.0;
        A.cw[i]=(col-1>0)?harmonic(t.sig[i],t.sig[west]):0.0;
        A.cn[i]=(r+1<n-1)?harmonic(t.sig[i],t.sig[north]):0.0;
        A.cs[i]=(r-1>0)?harmonic(t.sig[i],t.sig[south]):0.0;
        A.diag[i]=A.ce[i]+A.cw[i]+A.cn[i]+A.cs[i];
        A.b[i]=-rhs[i]*h*h;
        if(contact){
            const double ja=contact->anodeCurrent[i];
            const double jc=contact->cathodeCurrent[i];
            const double ka=std::max(contact->anodeSlope[i],0.0);
            const double kc=std::max(contact->cathodeSlope[i],0.0);
            const double slope=ka+kc;
            // Thin-layer charge conservation:
            // -div(sigma grad(phi)) = (j_anode-j_cathode)/layer.
            // Linearising j_a(phi) and j_c(phi) adds a non-negative diagonal,
            // preserving the symmetric positive-definite potential operator.
            A.diag[i]+=surface_scale*slope;
            A.b[i]+=surface_scale*(ja-jc+slope*s.phi[i]);
        }
        if(A.diag[i]<=0 || !std::isfinite(A.diag[i]))
            throw std::runtime_error("non-positive surface-contact potential diagonal");
    }
    return A;
}
static void apply_A(const LinearSystem& A,const std::vector<double>& x,std::vector<double>& y){int n=A.n;y.assign(A.N,0);for(int r=1;r<n-1;r++)for(int c=1;c<n-1;c++){size_t i=size_t(r)*n+c;if(!A.active[i])continue;double v=A.diag[i]*x[i];if(A.active[i+1])v-=A.ce[i]*x[i+1];if(A.active[i-1])v-=A.cw[i]*x[i-1];if(A.active[i+n])v-=A.cn[i]*x[i+n];if(A.active[i-n])v-=A.cs[i]*x[i-n];y[i]=v;}}
static double dot_active(const LinearSystem&A,const std::vector<double>&a,const std::vector<double>&b){double s0=0,s1=0,s2=0,s3=0;for(size_t i=0;i<A.N;i++)if(A.active[i]){double v=a[i]*b[i];switch(i&3u){case 0:s0+=v;break;case 1:s1+=v;break;case 2:s2+=v;break;default:s3+=v;break;}}return (s0+s1)+(s2+s3);}
static SolverDiag solve_pcg(const LinearSystem&A,std::vector<double>&x,const Config& c,double relative_tolerance){
    std::vector<double> Ax,r,z,p,Ap;apply_A(A,x,Ax);r.assign(A.N,0);z.assign(A.N,0);p.assign(A.N,0);for(size_t i=0;i<A.N;i++)if(A.active[i]){r[i]=A.b[i]-Ax[i];z[i]=r[i]/A.diag[i];p[i]=z[i];}
    double bnorm=std::sqrt(std::max(dot_active(A,A.b,A.b),0.0)), thr=std::max(relative_tolerance*std::max(bnorm,1.0),c.pcg_atol);double rz=dot_active(A,r,z);SolverDiag d;
    for(int it=0;it<c.pcg_max_iter;it++){double rn=std::sqrt(std::max(dot_active(A,r,r),0.0));if(rn<=thr){d.iterations=it;d.relres=rn/std::max(bnorm,1.0);d.converged=true;return d;}apply_A(A,p,Ap);double pap=dot_active(A,p,Ap);if(!(pap>0)&&std::isfinite(pap))break;if(!std::isfinite(pap)||std::abs(pap)<1e-300)break;double alpha=rz/pap;for(size_t i=0;i<A.N;i++)if(A.active[i]){x[i]+=alpha*p[i];r[i]-=alpha*Ap[i];z[i]=r[i]/A.diag[i];}double rz2=dot_active(A,r,z);if(!std::isfinite(rz2)||rz<=0)break;double beta=rz2/rz;for(size_t i=0;i<A.N;i++)if(A.active[i])p[i]=z[i]+beta*p[i];rz=rz2;d.iterations=it+1;}
    apply_A(A,x,Ax);for(size_t i=0;i<A.N;i++)if(A.active[i])r[i]=A.b[i]-Ax[i];double rn=std::sqrt(std::max(dot_active(A,r,r),0.0));d.relres=rn/std::max(bnorm,1.0);d.converged=rn<=thr;return d;
}
static SolverDiag solve_bulk(
    State& s,const Transport& t,const Grid& g,const Config& c,double h,
    const ContactLinearization* contact,double relative_tolerance
){
    LinearSystem A=build_system(s,t,g,c,h,contact);
    std::vector<double> x(g.N,0.0);
    for(size_t i=0;i<g.N;i++) if(A.active[i]) x[i]=s.phi[i];
    SolverDiag d=solve_pcg(A,x,c,relative_tolerance);
    if(!d.converged){
        // Deterministic FP64 red-black Gauss-Seidel fallback.
        for(int it=0;it<20000 && !d.converged;it++){
            double maxchg=0.0;
            for(int color=0;color<2;color++) for(int r=1;r<g.n-1;r++) for(int col=1;col<g.n-1;col++){
                if(((r+col)&1)!=color) continue;
                const size_t i=g.id(r,col);
                if(!A.active[i]) continue;
                double sum=A.b[i];
                if(A.active[i+1]) sum+=A.ce[i]*x[i+1];
                if(A.active[i-1]) sum+=A.cw[i]*x[i-1];
                if(A.active[i+g.n]) sum+=A.cn[i]*x[i+g.n];
                if(A.active[i-g.n]) sum+=A.cs[i]*x[i-g.n];
                const double nv=sum/A.diag[i];
                maxchg=std::max(maxchg,std::abs(nv-x[i]));
                x[i]=nv;
            }
            if(maxchg<1e-10){
                std::vector<double> Ax,r(A.N);
                apply_A(A,x,Ax);
                for(size_t i=0;i<A.N;i++) if(A.active[i]) r[i]=A.b[i]-Ax[i];
                const double bn=std::sqrt(std::max(dot_active(A,A.b,A.b),0.0));
                const double rn=std::sqrt(std::max(dot_active(A,r,r),0.0));
                d.relres=rn/std::max(bn,1.0);
                d.converged=rn<=std::max(relative_tolerance*std::max(bn,1.0),c.pcg_atol);
                d.iterations+=it+1;
                d.method="cpp_fp64_pcg_sor_fallback_surface_contact";
            }
        }
    }
    for(size_t i=0;i<g.N;i++) if(A.active[i]) s.phi[i]=x[i];
    neumann_edges(s.phi,g.n);
    return d;
}

struct ChannelEval {double j=0,dj=0,eta=0,shift=0;};
static double activity_gamma(double salt,double /*T*/,const Config&c){double I=std::max(salt/1000.0,0.0),sq=std::sqrt(I);double lg=-c.activity_A*sq/std::max(1+c.activity_B*sq,1e-12)+c.activity_linear*I;lg=clampd(lg,-std::abs(c.max_log10_gamma),std::abs(c.max_log10_gamma));return std::pow(10.0,lg);}
static std::pair<double,double> eeq(const Config::Channel&ch,double activity,double T,const Config&c){double shift=0;if(c.nernst_enabled){double ratio=std::max(activity/std::max(ch.ref_activity,c.activity_floor),c.activity_floor);shift=c.R*T/(ch.n*c.F)*ch.nernst_exp*std::log(ratio);shift=clampd(shift,-std::abs(c.max_nernst_shift),std::abs(c.max_nernst_shift));}return {ch.Eeq+shift,shift};}
static ChannelEval channel_eval(const Config::Channel&ch,double cf,double conc,double D,double eta,double T,double kinetic,const Config&c){ChannelEval e;e.eta=std::max(eta,0.0);double kf=ch.alpha*ch.n*c.F/(c.R*T),kr=(1-ch.alpha)*ch.n*c.F/(c.R*T);double rf=kf*e.eta,rr=-kr*e.eta;double f=clampd(rf,-c.exp_limit,c.exp_limit),r=clampd(rr,-c.exp_limit,c.exp_limit);double pref=ch.j0*kinetic,ef=std::exp(f),er=std::exp(r);double raw=c.full_bv?pref*(cf*ef-ch.reverse*er):pref*cf*ef;double draw=0;if(raw>0&&e.eta>0){draw=c.full_bv?pref*(cf*ef*kf + ch.reverse*er*kr):pref*cf*ef*kf;}double jk=std::max(raw,0.0);if(c.mass_transfer_saturation){double km=c.derive_km?D/std::max(c.diffusion_layer,1e-30):ch.km;double jl=ch.n*c.F*km*std::max(conc,0.0);if(jl>c.current_density_floor){double inv=1/(1+jk/jl);e.j=jk*inv;e.dj=draw*inv*inv;}else e.j=e.dj=0;}else{e.j=jk;e.dj=draw;}if(!std::isfinite(e.j)||!std::isfinite(e.dj))e.j=e.dj=0;return e;}

struct CellContext {double T,water_factor,salt_factor,salt_conc,kinetic,active;double eeq_wa,eeq_wc,eeq_la,eeq_lc;double shiftmax;};
static CellContext context_at(size_t i,const State&s,const Transport&t,const Config&c){(void)t;CellContext x{};x.T=std::max(s.T[i],1.0);x.water_factor=clampd(s.water[i]/std::max(c.water0,1e-30),0,1);x.salt_conc=std::min(s.cp[i],s.cm[i]);x.salt_factor=clampd(x.salt_conc/std::max(c.lp0,1e-30),0,1);x.kinetic=1+c.liquid_kinetics_gain*clampd((s.liquid[i]-c.fast_ion_liquid_threshold)/std::max(1-c.fast_ion_liquid_threshold,1e-30),0,1);x.active=clampd((1-s.pass[i])*(1-s.coverage[i]),c.min_active_area,1.0);double wa=std::max(x.water_factor,c.activity_floor),sa=std::max(x.salt_factor*activity_gamma(x.salt_conc,x.T,c),c.activity_floor);auto a=eeq(c.water_a,wa,x.T,c),b=eeq(c.water_c,wa,x.T,c),d=eeq(c.lp_a,sa,x.T,c),e=eeq(c.lp_c,sa,x.T,c);x.eeq_wa=a.first;x.eeq_wc=b.first;x.eeq_la=d.first;x.eeq_lc=e.first;x.shiftmax=std::max({std::abs(a.second),std::abs(b.second),std::abs(d.second),std::abs(e.second)});return x;}

struct LocalFace {double jw=0,jl=0,eta_w=0,eta_l=0,phi_face=0,sensitivity=0;bool ok=true;};
static LocalFace solve_face(size_t i,bool anode,double phi_cell,double sigma_face,const State&s,const Transport&t,const Config&c,double h){LocalFace out;CellContext x=context_at(i,s,t,c);double maxdrop=anode?std::max(c.voltage-phi_cell,0.0):std::max(phi_cell-c.cathode_voltage,0.0);if(maxdrop<=0){out.phi_face=anode?c.voltage:c.cathode_voltage;return out;}double G=std::max(sigma_face/h,c.sigma_min/h),lo=0,hi=maxdrop,gap=.5*maxdrop;auto eval=[&](double g,double&j,double&dj,double&residual,double&scale,double&ew,double&el){double eta_w=std::max(g-(anode?x.eeq_wa:x.eeq_wc),0.0),eta_l=std::max(g-(anode?x.eeq_la:x.eeq_lc),0.0);ChannelEval w=channel_eval(anode?c.water_a:c.water_c,x.water_factor,s.water[i],t.Dw[i],eta_w,x.T,1.0,c);double D=std::min(t.Dp[i],t.Dm[i]);ChannelEval l=channel_eval(anode?c.lp_a:c.lp_c,x.salt_factor,x.salt_conc,D,eta_l,x.T,x.kinetic,c);j=x.active*(w.j+l.j);dj=x.active*(w.dj+l.dj);double jb=G*std::max(maxdrop-g,0.0);residual=j-jb;scale=std::max({std::abs(j),std::abs(jb),1.0});ew=eta_w;el=eta_l;};
    double j=0,dj=0,sg=0,sc=1,ew=0,el=0;bool ok=false;for(int it=1;it<=c.local_max_iter;it++){eval(gap,j,dj,sg,sc,ew,el);if(it>=c.local_min_iter && std::abs(sg)<=std::min(c.local_abs_tol,c.local_rel_tol*sc)){ok=true;break;}if(sg<0)lo=gap;else hi=gap;gap=.5*(lo+hi);}for(int k=0;k<c.local_newton&&!ok;k++){eval(gap,j,dj,sg,sc,ew,el);double der=dj+G;if(std::isfinite(der)&&der>1e-12){double cand=clampd(gap-sg/der,lo,hi);gap=cand;eval(gap,j,dj,sg,sc,ew,el);if(sg<0)lo=gap;else hi=gap;if(std::abs(sg)<=std::min(c.local_abs_tol,c.local_rel_tol*sc))ok=true;}}
    eval(gap,j,dj,sg,sc,ew,el);ChannelEval w=channel_eval(anode?c.water_a:c.water_c,x.water_factor,s.water[i],t.Dw[i],ew,x.T,1.0,c);ChannelEval l=channel_eval(anode?c.lp_a:c.lp_c,x.salt_factor,x.salt_conc,std::min(t.Dp[i],t.Dm[i]),el,x.T,x.kinetic,c);out.jw=x.active*w.j;out.jl=x.active*l.j;out.eta_w=ew;out.eta_l=el;out.phi_face=anode?c.voltage-gap:c.cathode_voltage+gap;double ks=x.active*(w.dj+l.dj);double L=std::max(ks,G),sm=std::min(ks,G);out.sensitivity=(L>1e-12)?sm/(1+sm/L):0;out.ok=ok;return out;}

struct Boundary {
    ContactLinearization linearization;
    Reaction r;
    double sensitivity=0.0;
    explicit Boundary(size_t N=0):linearization(N),r(N){}
};

static Boundary eval_boundary(const State&s,const Transport&t,const Grid&g,const Config&c,double h){
    Boundary b(g.N);
    const double contact_area=h*h;
    for(size_t i=0;i<g.N;i++){
        const bool an=g.anode[i]!=0;
        const bool ca=g.cathode[i]!=0;
        if(!an&&!ca) continue;
        const LocalFace lf=solve_face(
            i,an,s.phi[i],t.sig[i],s,t,c,c.contact_normal_length
        );
        const double jw=lf.jw;
        const double jl=lf.jl;
        const double jt=jw+jl;
        const CellContext ctx=context_at(i,s,t,c);
        b.r.maxNernstShift=std::max(b.r.maxNernstShift,ctx.shiftmax);
        if(an){
            b.r.jwa[i]=jw;
            b.r.jla[i]=jl;
            b.r.eta_wa[i]=lf.eta_w;
            b.r.eta_la[i]=lf.eta_l;
            b.r.rawA+=jt*contact_area;
            b.r.anodeZone[i]=1;
            b.linearization.anodeCurrent[i]=jt;
            b.linearization.anodeSlope[i]=std::max(lf.sensitivity,0.0);
        }else{
            b.r.jwc[i]=jw;
            b.r.jlc[i]=jl;
            b.r.eta_wc[i]=lf.eta_w;
            b.r.eta_lc[i]=lf.eta_l;
            b.r.rawC+=jt*contact_area;
            b.r.cathodeZone[i]=1;
            b.linearization.cathodeCurrent[i]=jt;
            b.linearization.cathodeSlope[i]=std::max(lf.sensitivity,0.0);
        }
        b.sensitivity+=std::max(lf.sensitivity,0.0)*contact_area;
    }
    return b;
}

static void apply_offset(
    State&s,const Grid&g,const Config& /*c*/,
    const std::vector<double>&relative,double off
){
    for(size_t i=0;i<g.N;i++) s.phi[i]=relative[i]+off;
}

static Boundary gauge_balance(
    State&s,const Transport&t,const Grid&g,const Config&c,double h,
    int&iters,double&offset_out
){
    double sum=std::accumulate(s.phi.begin(),s.phi.end(),0.0);
    const double initial=sum/std::max<size_t>(g.N,1);
    std::vector<double> relative=s.phi;
    double rmin=1e300,rmax=-1e300;
    for(size_t i=0;i<g.N;i++){
        relative[i]-=initial;
        rmin=std::min(rmin,relative[i]);
        rmax=std::max(rmax,relative[i]);
    }
    double low=c.cathode_voltage-c.gauge_margin-rmax;
    double high=c.voltage+c.gauge_margin-rmin;
    double off=clampd(initial,low,high);
    Boundary b(g.N);
    for(int it=1;it<=c.gauge_max_iter;it++){
        apply_offset(s,g,c,relative,off);
        b=eval_boundary(s,t,g,c,h);
        const double diff=b.r.rawA-b.r.rawC;
        const double scale=std::max(std::abs(b.r.rawA),std::abs(b.r.rawC));
        const double threshold=std::max(c.robin_balance_tol*scale,c.robin_balance_abs);
        iters=it;
        if(it>=c.gauge_min_iter&&std::abs(diff)<=threshold) break;
        if(diff>0) low=off; else if(diff<0) high=off;
        const double midpoint=.5*(low+high);
        double target=midpoint;
        if(std::isfinite(b.sensitivity)&&b.sensitivity>c.gauge_deriv_floor){
            const double newton=off+diff/b.sensitivity;
            if(newton>low&&newton<high&&std::isfinite(newton)) target=newton;
        }
        const double step=clampd(target-off,-c.gauge_max_step,c.gauge_max_step);
        off=clampd(off+c.gauge_relax*step,low,high);
    }
    apply_offset(s,g,c,relative,off);
    b=eval_boundary(s,t,g,c,h);
    offset_out=off;
    return b;
}

static void finish_reaction(Reaction&r,const Config&c,double reaction_layer){
    double totalW=0,totalL=0,total=0;
    for(size_t i=0;i<r.q.size();i++){
        const double jwA=r.jwa[i],jwC=r.jwc[i],jlA=r.jla[i],jlC=r.jlc[i];
        auto heat=[&](double j,double eta,const Config::Channel&ch){
            const double mol=j/(ch.n*c.F);
            double q=mol*ch.dH/reaction_layer;
            if(c.activation_heat) q+=c.activation_heat_fraction*j*eta/reaction_layer;
            return std::max(q,0.0);
        };
        r.q[i]=heat(jwA,r.eta_wa[i],c.water_a)+heat(jwC,r.eta_wc[i],c.water_c)
              +heat(jlA,r.eta_la[i],c.lp_a)+heat(jlC,r.eta_lc[i],c.lp_c);
        r.waterSink[i]=c.water_a.water_stoich*jwA/(c.water_a.n*c.F*reaction_layer)
                     +c.water_c.water_stoich*jwC/(c.water_c.n*c.F*reaction_layer);
        r.saltSink[i]=c.lp_a.salt_stoich*jlA/(c.lp_a.n*c.F*reaction_layer)
                    +c.lp_c.salt_stoich*jlC/(c.lp_c.n*c.F*reaction_layer);
        r.gasSource[i]=c.water_a.gas_yield*jwA/(c.water_a.n*c.F*reaction_layer)
                     +c.water_c.gas_yield*jwC/(c.water_c.n*c.F*reaction_layer)
                     +c.lp_a.gas_yield*jlA/(c.lp_a.n*c.F*reaction_layer)
                     +c.lp_c.gas_yield*jlC/(c.lp_c.n*c.F*reaction_layer);
        totalW+=jwA+jwC;
        totalL+=jlA+jlC;
        total+=jwA+jwC+jlA+jlC;
    }
    r.totalI=.5*(r.rawA+r.rawC);
    r.mismatch=std::abs(r.rawA-r.rawC)/std::max({std::abs(r.rawA),std::abs(r.rawC),c.current_floor});
    r.waterFrac=totalW/std::max(total,c.current_density_floor);
    r.lpFrac=totalL/std::max(total,c.current_density_floor);
}

static std::pair<Reaction,SolverDiag> solve_electrochem(
    State&s,Transport&t,const Grid&g,const Config&c,double h
){
    // In the surface-contact formulation the Robin terms anchor the absolute
    // electrolyte potential.  The scalar offset is therefore not an extra
    // degree of freedom: the coupled SPD solve itself enforces global charge
    // balance.  Applying a separate post-solve gauge shift would over-constrain
    // the problem and create a solve/shift limit cycle.
    SolverDiag diag;
    long long total_linear_iterations=0;
    int total_linear_solves=0;
    Boundary boundary=eval_boundary(s,t,g,c,h);
    double previous_current=.5*(boundary.r.rawA+boundary.r.rawC);
    double dphi=1e300,dI=1e300;
    bool converged=false;
    int used=0;
    for(int iteration=1;iteration<=c.robin_max_iter;iteration++){
        State proposed=s;
        SolverDiag current_diag=solve_bulk(
            proposed,t,g,c,h,&boundary.linearization,c.pcg_rtol_coupled
        );
        total_linear_iterations+=current_diag.iterations;
        total_linear_solves++;
        if(!current_diag.converged)
            throw std::runtime_error("C++ FP64 surface-contact potential solve failed residual="+std::to_string(current_diag.relres));
        const std::vector<double> previous_phi=s.phi;
        for(size_t i=0;i<g.N;i++)
            s.phi[i]+=c.robin_relax*(proposed.phi[i]-s.phi[i]);
        neumann_edges(s.phi,g.n);
        boundary=eval_boundary(s,t,g,c,h);
        dphi=0.0;
        for(size_t i=0;i<g.N;i++) dphi=std::max(dphi,std::abs(s.phi[i]-previous_phi[i]));
        const double current=.5*(boundary.r.rawA+boundary.r.rawC);
        dI=std::abs(current-previous_current)/std::max({std::abs(current),std::abs(previous_current),1e-12});
        const double mismatch_abs=std::abs(boundary.r.rawA-boundary.r.rawC);
        const double threshold=std::max(
            c.robin_balance_tol*std::max(std::abs(boundary.r.rawA),std::abs(boundary.r.rawC)),
            c.robin_balance_abs
        );
        converged=(iteration>=c.robin_min_iter&&dphi<=c.robin_phi_tol&&dI<=c.robin_current_tol&&mismatch_abs<=threshold);
        previous_current=current;
        diag=current_diag;
        used=iteration;
        if(converged) break;
    }
    Reaction reaction=std::move(boundary.r);
    reaction.outerIters=used;
    reaction.gaugeIters=0;
    reaction.linearIterations=total_linear_iterations;
    reaction.linearSolves=total_linear_solves;
    reaction.gaugeOffset=0.0;
    reaction.converged=converged;
    const double scale=std::max(std::abs(reaction.rawA),std::abs(reaction.rawC));
    const double threshold=std::max(c.robin_balance_tol*scale,c.robin_balance_abs);
    reaction.balanceCombined=std::abs(reaction.rawA-reaction.rawC)/std::max(threshold,1e-300);
    finish_reaction(reaction,c,std::max({c.diffusion_layer,c.reaction_layer_min,h}));
    if(!converged)
        throw std::runtime_error(
            "C++ FP64 surface-contact nonlinear Robin did not converge dphi="+std::to_string(dphi)+
            " dI="+std::to_string(dI)+" balance="+std::to_string(reaction.mismatch)+
            " Ia="+std::to_string(reaction.rawA)+" Ic="+std::to_string(reaction.rawC)
        );
    return {std::move(reaction),diag};
}

static void masked_gradient(const std::vector<double>&f,const Grid&g,double h,std::vector<double>&gx,std::vector<double>&gy){gx.assign(g.N,0);gy.assign(g.N,0);int n=g.n;for(int r=0;r<n;r++)for(int c=0;c<n;c++){size_t i=g.id(r,c);if(!g.prop[i])continue;bool e=c+1<n&&g.prop[i+1],w=c>0&&g.prop[i-1],nn=r+1<n&&g.prop[i+n],ss=r>0&&g.prop[i-n];if(e&&w)gx[i]=(f[i+1]-f[i-1])/(2*h);else if(e)gx[i]=(f[i+1]-f[i])/h;else if(w)gx[i]=(f[i]-f[i-1])/h;if(nn&&ss)gy[i]=(f[i+n]-f[i-n])/(2*h);else if(nn)gy[i]=(f[i+n]-f[i])/h;else if(ss)gy[i]=(f[i]-f[i-n])/h;}}
static void compute_current(const State&s,const Transport&t,const Grid&g,const Config&c,double h,Electrical&e){std::vector<double>px,py,cpx,cpy,cmx,cmy;masked_gradient(s.phi,g,h,px,py);masked_gradient(s.cp,g,h,cpx,cpy);masked_gradient(s.cm,g,h,cmx,cmy);for(size_t i=0;i<g.N;i++){if(!g.prop[i])continue;e.Ex[i]=-px[i];e.Ey[i]=-py[i];e.Jdx[i]=c.diffusion_potential?-c.F*(c.zplus*t.Dp[i]*cpx[i]+c.zminus*t.Dm[i]*cmx[i]):0;e.Jdy[i]=c.diffusion_potential?-c.F*(c.zplus*t.Dp[i]*cpy[i]+c.zminus*t.Dm[i]*cmy[i]):0;e.Jox[i]=t.sig[i]*e.Ex[i];e.Joy[i]=t.sig[i]*e.Ey[i];e.Jx[i]=e.Jox[i]+e.Jdx[i];e.Jy[i]=e.Joy[i]+e.Jdy[i];e.Jmag[i]=std::hypot(e.Jx[i],e.Jy[i]);e.qj[i]=std::max(e.Jox[i]*e.Ex[i]+e.Joy[i]*e.Ey[i],0.0);double Em=std::hypot(e.Ex[i],e.Ey[i]),ii=t.sigi[i]*Em,ie=t.sige[i]*Em;e.ionicFrac[i]=ii/std::max(ii+ie,c.current_density_floor);}}

static double lap_at(const std::vector<double>&f,int n,int r,int c,double h){auto at=[&](int rr,int cc){rr=std::max(0,std::min(n-1,rr));cc=std::max(0,std::min(n-1,cc));return f[size_t(rr)*n+cc];};return (at(r,c+1)+at(r,c-1)+at(r+1,c)+at(r-1,c)-4*at(r,c))/(h*h);}
static void flux_div_charged(const std::vector<double>&conc,const std::vector<double>&D,const State&s,const Grid&g,const Config&c,double h,double z,std::vector<double>&div){int n=g.n;std::vector<double>fx(size_t(n)*(n+1),0),fy(size_t(n+1)*n,0);for(int r=0;r<n;r++)for(int col=0;col<n-1;col++){size_t a=g.id(r,col),b=g.id(r,col+1);if(g.prop[a]&&g.prop[b]){double df=harmonic(D[a],D[b]),cf=.5*(conc[a]+conc[b]),Tf=std::max(.5*(s.T[a]+s.T[b]),1.0);fx[size_t(r)*(n+1)+col+1]=-df*(conc[b]-conc[a])/h-z*df*c.F/(c.R*Tf)*cf*(s.phi[b]-s.phi[a])/h;}}for(int r=0;r<n-1;r++)for(int col=0;col<n;col++){size_t a=g.id(r,col),b=g.id(r+1,col);if(g.prop[a]&&g.prop[b]){double df=harmonic(D[a],D[b]),cf=.5*(conc[a]+conc[b]),Tf=std::max(.5*(s.T[a]+s.T[b]),1.0);fy[size_t(r+1)*n+col]=-df*(conc[b]-conc[a])/h-z*df*c.F/(c.R*Tf)*cf*(s.phi[b]-s.phi[a])/h;}}div.assign(g.N,0);for(int r=0;r<n;r++)for(int col=0;col<n;col++){size_t i=g.id(r,col);if(g.prop[i])div[i]=(fx[size_t(r)*(n+1)+col+1]-fx[size_t(r)*(n+1)+col])/h+(fy[size_t(r+1)*n+col]-fy[size_t(r)*n+col])/h;}}
static void flux_div_neutral(const std::vector<double>&conc,const std::vector<double>&D,const Grid&g,double h,std::vector<double>&div){int n=g.n;std::vector<double>fx(size_t(n)*(n+1),0),fy(size_t(n+1)*n,0);for(int r=0;r<n;r++)for(int col=0;col<n-1;col++){size_t a=g.id(r,col),b=g.id(r,col+1);if(g.prop[a]&&g.prop[b])fx[size_t(r)*(n+1)+col+1]=-harmonic(D[a],D[b])*(conc[b]-conc[a])/h;}for(int r=0;r<n-1;r++)for(int col=0;col<n;col++){size_t a=g.id(r,col),b=g.id(r+1,col);if(g.prop[a]&&g.prop[b])fy[size_t(r+1)*n+col]=-harmonic(D[a],D[b])*(conc[b]-conc[a])/h;}div.assign(g.N,0);for(int r=0;r<n;r++)for(int col=0;col<n;col++){size_t i=g.id(r,col);if(g.prop[i])div[i]=(fx[size_t(r)*(n+1)+col+1]-fx[size_t(r)*(n+1)+col])/h+(fy[size_t(r+1)*n+col]-fy[size_t(r)*n+col])/h;}}
static bool limited_update(double&field,double rate,double dt,double ref,const Config&c){double raw=dt*rate,lim=c.max_relative_concentration_change*std::max(field,ref*c.concentration_min_fraction),ld=clampd(raw,-lim,lim);double rawu=field+ld,lo=ref*c.concentration_min_fraction,hi=ref*c.concentration_max_multiple,up=clampd(rawu,lo,hi);bool l=std::abs(ld-raw)>1e-12+1e-7*std::max({std::abs(raw),std::abs(ld),std::abs(field),lo})||std::abs(up-rawu)>1e-12+1e-7*std::max({std::abs(up),std::abs(rawu),lo});field=up;return l;}
static double update_species(State&s,Transport&t,const Reaction&r,const std::vector<double>&chem,const Grid&g,const Config&c,double h,double dt){int subs=std::max(1,c.species_substeps);double limiter=0;size_t propcnt=std::count(g.prop.begin(),g.prop.end(),uint8_t(1));for(int sub=0;sub<subs;sub++){if(sub>0)transport_fields(s,g,c,t);std::vector<double>dc,dm,dw;if(c.full_np){flux_div_charged(s.cp,t.Dp,s,g,c,h,c.zplus,dc);flux_div_charged(s.cm,t.Dm,s,g,c,h,c.zminus,dm);flux_div_neutral(s.water,t.Dw,g,h,dw);}else{dc.assign(g.N,0);dm.assign(g.N,0);dw.assign(g.N,0);}size_t limcount=0;for(size_t i=0;i<g.N;i++)if(g.prop[i]){double cs=chem[i]*c.lp0*c.oxidizer_consume;double rc=-dc[i]-r.saltSink[i]-cs,rm=-dm[i]-r.saltSink[i]-cs,rw=-dw[i]-r.waterSink[i];bool a=limited_update(s.cp[i],rc,dt/subs,c.cation0,c),b=limited_update(s.cm[i],rm,dt/subs,c.anion0,c),d=limited_update(s.water[i],rw,dt/subs,c.water0,c);double neutral=.5*(s.cp[i]+s.cm[i]);s.cp[i]=(1-c.electroneutral_relax)*s.cp[i]+c.electroneutral_relax*neutral;s.cm[i]=(1-c.electroneutral_relax)*s.cm[i]+c.electroneutral_relax*neutral;if(a||b||d)limcount++;}limiter+=double(limcount)/std::max<size_t>(propcnt,1);}return limiter/subs;}
static void update_blocking(State&s,const Reaction&r,const Grid&g,const Config&c,double dt){for(size_t i=0;i<g.N;i++){bool zone=r.anodeZone[i]||r.cathodeZone[i];if(!zone||!g.prop[i]){s.pass[i]=s.coverage[i]=0;continue;}double jt=r.jwa[i]+r.jla[i]+r.jwc[i]+r.jlc[i];if(c.passivation_enabled){double gate=sigmoid((s.T[i]-c.pass_T0)/std::max(c.pass_Tw,1e-12));double form=c.pass_form*std::pow(std::max(jt/c.pass_jref,0.0),c.pass_jexp)*gate*(1-s.pass[i]);s.pass[i]=clampd(s.pass[i]+dt*(form-c.pass_remove*s.pass[i]),0,1);}if(c.gas_coverage_enabled){double source=std::max(r.gasSource[i]/c.gas_source_ref,0.0);double form=c.gas_cov_form*source*(1-s.coverage[i]);s.coverage[i]=clampd(s.coverage[i]+dt*(form-c.gas_cov_detach*s.coverage[i]),0,1);}}}

static double percentile99(std::vector<double> v){if(v.empty())return 0;std::sort(v.begin(),v.end());double pos=.99*(v.size()-1),lo=std::floor(pos),hi=std::ceil(pos),w=pos-lo;return v[size_t(lo)]*(1-w)+v[size_t(hi)]*w;}
struct Metrics {
    double ignition=std::numeric_limits<double>::quiet_NaN();
    double undecomp=1.0;
    double energy_at_eval=0.0;
    double energy_to_ignition=std::numeric_limits<double>::quiet_NaN();
    double congestion=0.0,peakT=0.0,peakI=0.0,finalR=0.0,solver_res=0.0;
    bool ignited=false,converged=true;
    int electrical_solves=0,total_linear_solves=0,total_robin_outer_iterations=0,total_gauge_iterations=0;
    long long total_linear_iterations=0;
    double wall=0.0,max_species_limiter=0.0,max_temp_cap=0.0,max_gas_cap=0.0,max_chem_cap=0.0;
    double simulated_time=0.0;
    bool terminated_at_ignition=false;
    Reaction last_reaction;
    SolverDiag last_solver;
    explicit Metrics(size_t N=0):last_reaction(N){}
};

static Metrics simulate(const Grid&g,const Config&c){
    auto start=std::chrono::steady_clock::now();
    State s(g.N); Transport t(g.N); Electrical e(g.N);
    init_state(s,g,c); transport_fields(s,g,c,t);
    double h=c.domain/std::max(g.n-1,1),alphaT=c.k/(c.density*c.cp);
    int steps=int(std::llround(c.end_time/c.dt));
    int solve_every=std::max(1,int(std::llround(c.electrical_interval/c.dt)));
    Metrics M(g.N);
    auto pr=solve_electrochem(s,t,g,c,h);
    Reaction reaction=std::move(pr.first);
    M.last_solver=pr.second; M.electrical_solves++;
    M.total_linear_iterations+=reaction.linearIterations;
    M.total_linear_solves+=reaction.linearSolves;
    M.total_robin_outer_iterations+=reaction.outerIters;
    M.total_gauge_iterations+=reaction.gaugeIters;
    compute_current(s,t,g,c,h,e);

    std::vector<double>Jvals;
    double meanJ=0.0; size_t pcnt=0;
    for(size_t i=0;i<g.N;i++) if(g.prop[i]){Jvals.push_back(e.Jmag[i]);meanJ+=e.Jmag[i];pcnt++;}
    meanJ/=std::max<size_t>(pcnt,1);
    double congestion=percentile99(Jvals)/std::max(meanJ,c.current_density_floor);

    double energy=0.0,prev_power=0.0;
    bool have_prev=false;
    double eval_undecomp=1.0,eval_energy=0.0;
    double ignition_energy=std::numeric_limits<double>::quiet_NaN();

    for(int step=0;step<steps;step++){
        double time=(step+1)*c.dt;
        double liqrate;
        transport_fields(s,g,c,t);
        if(step>0&&step%solve_every==0){
            auto x=solve_electrochem(s,t,g,c,h);
            reaction=std::move(x.first); M.last_solver=x.second; M.electrical_solves++;
            M.total_linear_iterations+=reaction.linearIterations;
            M.total_linear_solves+=reaction.linearSolves;
            M.total_robin_outer_iterations+=reaction.outerIters;
            M.total_gauge_iterations+=reaction.gaugeIters;
            compute_current(s,t,g,c,h,e);
            Jvals.clear(); meanJ=0.0;
            for(size_t i=0;i<g.N;i++) if(g.prop[i]){Jvals.push_back(e.Jmag[i]);meanJ+=e.Jmag[i];}
            meanJ/=std::max<size_t>(pcnt,1);
            congestion=percentile99(Jvals)/std::max(meanJ,c.current_density_floor);
        }

        std::vector<double>chem(g.N,0.0);
        for(size_t i=0;i<g.N;i++) if(g.prop[i]&&c.chemical_enabled){
            double ox=std::min(s.cp[i],s.cm[i])/std::max(c.lp0,1e-12);
            double gate=sigmoid((s.T[i]-c.chem_activation_T)/20.0);
            double rate=c.chem_A*std::exp(-c.chem_Ea/(c.R*std::max(s.T[i],1.0)))
                *std::pow(std::max(1-s.alpha[i],0.0),c.chem_order)
                *std::pow(std::max(ox,0.0),c.oxidizer_order)*gate;
            chem[i]=clampd(rate,0,c.chem_max_rate);
        }
        double lim=update_species(s,t,reaction,chem,g,c,h,c.dt);
        M.max_species_limiter=std::max(M.max_species_limiter,lim);
        update_blocking(s,reaction,g,c,c.dt);

        size_t tempcap=0,gascap=0,chemcap=0;
        double maxT=0.0,sumAlpha=0.0;
        bool ignited_this_step=false;
        for(int r=0;r<g.n;r++) for(int col=0;col<g.n;col++){
            size_t i=g.id(r,col); if(!g.prop[i]) continue;
            double leq=1/(1+std::exp(-(s.T[i]-c.effective_softening_T)/c.transition_width));
            liqrate=(leq-s.liquid[i])/std::max(c.phase_relax_time,1e-30);
            double chemgas=c.density*c.gas_yield_mass*chem[i]/std::max(c.gas_molar_mass,1e-30);
            double gasrate=c.gas_D*lap_at(s.gas,g.n,r,col,h)+reaction.gasSource[i]+chemgas-c.gas_loss*s.gas[i];
            double cheath=c.density*c.chem_heat*chem[i];
            double convfac=clampd(1-c.gas_thermal_insulation*s.coverage[i],0.05,1);
            double loss=convfac*c.hconv/c.layer*(s.T[i]-c.Tamb)
                +c.emissivity*c.sigma_sb/c.layer*(std::pow(s.T[i],4)-std::pow(c.Tamb,4));
            double latent=c.density*c.latent_heat*liqrate;
            double Trate=alphaT*lap_at(s.T,g.n,r,col,h)
                +(e.qj[i]+reaction.q[i]+cheath-loss-latent)/(c.density*c.cp);
            double rawT=s.T[i]+c.dt*Trate,rawG=s.gas[i]+c.dt*gasrate;
            if(rawT<c.Tmin||rawT>c.Tmax) tempcap++;
            if(rawG<0||rawG>c.gas_max) gascap++;
            if(chem[i]>=c.chem_max_rate*(1-1e-12)) chemcap++;
            s.T[i]=clampd(rawT,c.Tmin,c.Tmax);
            s.liquid[i]=clampd(s.liquid[i]+c.dt*liqrate,0,1);
            s.alpha[i]=clampd(s.alpha[i]+c.dt*chem[i],0,1);
            s.gas[i]=clampd(rawG,0,c.gas_max);
            maxT=std::max(maxT,s.T[i]); sumAlpha+=s.alpha[i];
            if(!M.ignited&&s.T[i]>=c.ignition_T&&s.alpha[i]>=c.ignition_progress_guard&&chem[i]>=c.ignition_rate_guard){
                M.ignited=true; M.ignition=time; ignited_this_step=true;
            }
        }

        double power=reaction.totalI*(c.voltage-c.cathode_voltage);
        if(!have_prev){energy=power*c.dt;have_prev=true;}
        else energy+=0.5*c.dt*(power+prev_power);
        prev_power=power;
        if(ignited_this_step && !std::isfinite(ignition_energy)) ignition_energy=energy;

        double undecomp=1-sumAlpha/std::max<size_t>(pcnt,1);
        if(time>=c.eval_time-0.5*c.dt){eval_undecomp=undecomp;eval_energy=energy;}
        M.peakT=std::max(M.peakT,maxT);
        M.peakI=std::max(M.peakI,reaction.totalI);
        M.congestion=std::max(M.congestion,congestion);
        M.max_temp_cap=std::max(M.max_temp_cap,double(tempcap)/std::max<size_t>(pcnt,1));
        M.max_gas_cap=std::max(M.max_gas_cap,double(gascap)/std::max<size_t>(pcnt,1));
        M.max_chem_cap=std::max(M.max_chem_cap,double(chemcap)/std::max<size_t>(pcnt,1));
        M.simulated_time=time;
        if(c.stop_on_ignition && M.ignited){
            M.terminated_at_ignition=true;
            break;
        }
    }

    M.undecomp=eval_undecomp;
    M.energy_at_eval=eval_energy;
    M.energy_to_ignition=ignition_energy;
    M.last_reaction=reaction;
    M.solver_res=M.last_solver.relres;
    M.converged=M.last_solver.converged&&reaction.converged;
    M.finalR=(c.voltage-c.cathode_voltage)/std::max(reaction.totalI,c.current_floor);
    M.wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    return M;
}

static void write_json(const std::string&path,const Metrics&m,const Grid&g,const Config&c){
    std::ofstream f(path);
    if(!f) throw std::runtime_error("cannot write output");
    const double invN=1.0/std::max<size_t>(g.N,1);
    const double anode_fraction=std::count(g.anode.begin(),g.anode.end(),uint8_t(1))*invN;
    const double cathode_fraction=std::count(g.cathode.begin(),g.cathode.end(),uint8_t(1))*invN;
    f<<std::setprecision(17)<<"{\n";
    auto num=[&](const char*k,double v,bool comma=true){f<<"  \""<<k<<"\": ";if(std::isfinite(v))f<<v;else f<<"null";f<<(comma?",\n":"\n");};
    num("appliedVoltage_V",c.voltage);
    num("cathodeVoltage_V",c.cathode_voltage);
    num("ignitionDelay_s",m.ignition);
    num("condensedPhaseIgnitionDelay_s",m.ignition);
    f<<"  \"ignitionSucceeded\": "<<(m.ignited?"true":"false")<<",\n";
    num("areaAveragedUndecomposedFractionAt2s",m.undecomp);
    num("areaWeightedUndecomposedFractionAt2s",m.undecomp);
    num("inputElectricalEnergyToIgnition_J",m.energy_to_ignition);
    num("inputElectricalEnergyAt2s_J",m.energy_at_eval);
    num("inputElectricalEnergyAtEvaluationTime_J",m.energy_at_eval);
    // Legacy alias remains evaluation-horizon energy for backward compatibility.
    num("inputElectricalEnergy_J",m.energy_at_eval);
    num("peakCurrentCongestion",m.congestion);
    num("peakCurrentCongestionTo2s",m.congestion);
    num("peakMaximumTemperature_K",m.peakT);
    num("peakCurrent_A",m.peakI);
    num("finalEffectiveResistance_ohm",m.finalR);
    num("finalElectricalRelativeResidual",m.solver_res);
    f<<"  \"finalElectricalConverged\": "<<(m.converged?"true":"false")<<",\n";
    f<<"  \"converged\": "<<(m.converged?"true":"false")<<",\n";
    f<<"  \"finalElectricalSolverMethod\": \""<<m.last_solver.method<<"\",\n";
    f<<"  \"finalElectricalIterations\": "<<m.last_solver.iterations<<",\n";
    f<<"  \"electricalSolveCount\": "<<m.electrical_solves<<",\n";
    f<<"  \"totalLinearSolveCount\": "<<m.total_linear_solves<<",\n";
    f<<"  \"totalLinearIterations\": "<<m.total_linear_iterations<<",\n";
    f<<"  \"totalNonlinearRobinOuterIterations\": "<<m.total_robin_outer_iterations<<",\n";
    f<<"  \"totalGaugeIterations\": "<<m.total_gauge_iterations<<",\n";
    num("finalAnodeCathodeCurrentMismatch",m.last_reaction.mismatch);
    num("finalNonlinearRobinAnodeCurrent_A",m.last_reaction.rawA);
    num("finalNonlinearRobinCathodeCurrent_A",m.last_reaction.rawC);
    num("finalNonlinearRobinGaugeOffset_V",m.last_reaction.gaugeOffset);
    num("finalNonlinearRobinCurrentBalanceCombinedResidual",m.last_reaction.balanceCombined);
    num("maximumSpeciesLimiterFraction",m.max_species_limiter);
    num("maximumTemperatureCapFraction",m.max_temp_cap);
    num("maximumGasCapFraction",m.max_gas_cap);
    num("maximumChemicalRateCapFraction",m.max_chem_cap);
    num("simulatedTime_s",m.simulated_time);
    f<<"  \"stopOnIgnitionRequested\": "<<(c.stop_on_ignition?"true":"false")<<",\n";
    f<<"  \"terminatedAtIgnition\": "<<(m.terminated_at_ignition?"true":"false")<<",\n";
    num("wallClockTime_s",m.wall);
    num("propellantDomainAreaFraction",1.0);
    num("anodeContactAreaFraction",anode_fraction);
    num("cathodeContactAreaFraction",cathode_fraction);
    num("totalContactAreaFraction",anode_fraction+cathode_fraction);
    num("contactNormalConductionLength_m",c.contact_normal_length);
    f<<"  \"propellantCellCount\": "<<g.N<<",\n";
    f<<"  \"surfaceContactModel\": true,\n";
    f<<"  \"electrodeMasksRemovePropellant\": false,\n";
    f<<"  \"hiddenBusConnectionAssumed\": "<<(c.hidden_bus_assumed?"true":"false")<<",\n";
    f<<"  \"physicsDevice\": \"cpu\",\n"
      <<"  \"physicsDtype\": \"float64\",\n"
      <<"  \"backend\": \"cpp_fp64_cpu\",\n"
      <<"  \"solverRevision\": \"v7_9_4_cpp_fp64_surface_contact_vmin_search_ready\",\n"
      <<"  \"empiricalSurfaceReactionProgressUsed\": false\n}\n";
}



static void debug_gauge_from_potential(const Grid&g,const Config&c,const std::string&potential_path,const std::string&out_path){
    State s(g.N); Transport t(g.N); init_state(s,g,c);
    std::ifstream f(potential_path,std::ios::binary); if(!f)throw std::runtime_error("cannot open debug potential");
    f.read(reinterpret_cast<char*>(s.phi.data()),std::streamsize(sizeof(double)*g.N));
    if(!f)throw std::runtime_error("truncated debug potential");
    transport_fields(s,g,c,t); double h=c.domain/std::max(g.n-1,1); int gi=0; double go=0;
    Boundary b=gauge_balance(s,t,g,c,h,gi,go);
    std::ofstream o(out_path); o<<std::setprecision(17)<<"{\n"
      <<"  \"gaugeOffset_V\": "<<go<<",\n"
      <<"  \"gaugeIterations\": "<<gi<<",\n"
      <<"  \"rawAnodeCurrent_A\": "<<b.r.rawA<<",\n"
      <<"  \"rawCathodeCurrent_A\": "<<b.r.rawC<<",\n"
      <<"  \"sensitivity_A_per_V\": "<<b.sensitivity<<"\n}\n";
}

} // namespace ecsp

int main(int argc,char**argv){try{std::string mask,cfg,out,debug_potential;for(int i=1;i<argc;i++){std::string a=argv[i];if(a=="--mask"&&i+1<argc)mask=argv[++i];else if(a=="--config"&&i+1<argc)cfg=argv[++i];else if(a=="--output"&&i+1<argc)out=argv[++i];else if(a=="--debug-gauge-potential"&&i+1<argc)debug_potential=argv[++i];else if(a=="--version"){std::cout<<"ecsp_cpp_solver 7.9.4\n";return 0;}}if(mask.empty()||cfg.empty()||out.empty())throw std::runtime_error("usage: ecsp_cpp_solver --mask mask.bin --config config.kv --output metrics.json");auto c=ecsp::load_config(cfg);ecsp::validate_config(c);auto g=ecsp::read_mask(mask);if(g.n!=c.n)throw std::runtime_error("mask/config grid mismatch");ecsp::validate_grid(g);if(!debug_potential.empty()){ecsp::debug_gauge_from_potential(g,c,debug_potential,out);return 0;}auto m=ecsp::simulate(g,c);ecsp::write_json(out,m,g,c);return 0;}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 2;}}
