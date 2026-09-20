"""Synthetic conservation/portability fixtures, NOT ECSP material calibration."""
from __future__ import annotations
import numpy as np
import math

def synthetic_condensed_case(shape=(8,8),rates=(.2,.1),heats=(1000.,2000.),alpha=(.2,.2)):
    f=lambda v:np.full(shape,v,dtype=float)
    a=np.zeros(shape,bool);c=a.copy();a[:,0]=True;c[:,-1]=True
    w=np.asarray([2/3,1/3]);X=w@alpha;xi=100*X
    h={'handoffSchemaVersion':'ecsp_bc_surface_onset_v8.2.0','onsetSucceeded':True,'onsetReportedByPhysics':True,
       'numericallyValidForPropagationHandoff':True,'propagationHandoffAuthorizationReason':'ignition_and_numerics_valid',
       'ignitionDelay_s':.1,'continuedElectricalHeating':False,'temperatureAtOnset_K':f(300),'propellantMask':f(1),
       'anodeContactMask':a,'cathodeContactMask':c,'alphaChannel1AtOnset':f(alpha[0]),'alphaChannel2AtOnset':f(alpha[1]),
       'globalProgressAtOnset':f(X),'xiMax_mol_per_m3':100.,'initialMobileLP_mol_per_m3':145.,
       'initialPVARepeat_mol_per_m3':100.,'initialMobileWater_mol_per_m3':50.,'molarMassLP_kg_per_mol':.1,
       'molarMassPVARepeat_kg_per_mol':.05,'initialReactiveMass_kg_per_m3':19.5,
       'cationAtOnset_mol_per_m3':f(145-1.45*xi),'anionAtOnset_mol_per_m3':f(145-1.45*xi),
       'mobileLPAtOnset_mol_per_m3':f(145-1.45*xi),'pvaReactiveRepeatAtOnset_mol_per_m3':f(100-xi),
       'generatedWaterProductAtOnset_mol_per_m3':f(2*xi),'mobileWaterAtOnset_mol_per_m3':f(50),
       'electrochemicalLPConsumedAtOnset_mol_per_m3':f(0),'potentialAtOnset_V':f(0),
       'qJAtOnset_W_per_m3':f(0),'qEchemAtOnset_W_per_m3':f(0)}
    p={'duration_s':.01,'time_step_s':.001,'snapshot_interval_s':.005,'domain_size_m':.02,
       'density_kg_per_m3':1000.,'surface_layer_thickness_m':.001,'gas_constant_J_per_molK':8.31446261815324,
       'front_progress_threshold':.5,'established_reacted_area_fraction':.5,'continued_electrical_heating':False,
       'electrical_heating_mode':'off'}
    ch=lambda k,q:dict(alpha_grid=[0.,1.],activation_energy_J_per_mol=[0.,0.],ln_Af_per_s=[math.log(k)]*2,heat_release_J_per_kg=q)
    b={'endTime_s':1.,'onsetCriterion':{'temperature_K':300.,'minimum_progress':.001,'minimum_area_fraction':.5},
       'thermal':{'heat_capacity':2200.,'thermal_conductivity':.4,'initialTemperature_K':298.15,
                  'ambientTemperature_K':298.15,'minimumTemperature_K':1,'maximumTemperature_K':10000},
       'kinetics':{'mass_conversion_weights':w.tolist(),'channels':[ch(rates[0],heats[0]),ch(rates[1],heats[1])]}}
    r={'eos':{'A_Pa':101325.,'B_Pa':2e9,'N':7.,'provenance':'synthetic numerical test not a material calibration'}}
    return h,p,b,r

