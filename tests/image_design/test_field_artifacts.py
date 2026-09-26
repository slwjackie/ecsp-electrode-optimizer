from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))

from ecsp_nsga2.field_diagnostics import write_snapshot_artifacts


def test_representative_snapshot_writes_raw_fields_images_and_uniformity_metrics(tmp_path):
    n=17
    y,x=np.mgrid[0:n,0:n]
    temperature=298.15 + 40.0*(x+y)/(2*(n-1))
    jx=np.full((n,n),2.0)
    jy=np.full((n,n),3.0)
    jmag=np.sqrt(jx*jx+jy*jy)
    potential=260.0*(1.0-x/(n-1))
    progress=np.clip((temperature-298.15)/40.0,0.0,1.0)
    qj=np.full((n,n),1000.0)
    anode=np.zeros((n,n),bool);anode[:,:2]=True
    cathode=np.zeros((n,n),bool);cathode[:,-2:]=True

    artifact,metrics=write_snapshot_artifacts(
        tmp_path,'eval',
        temperature=temperature,current_x=jx,current_y=jy,
        current_magnitude=jmag,potential=potential,progress=progress,
        joule_heat=qj,anode=anode,cathode=cathode,
        domain_mm=25.0,time_s=2.0,initial_temperature_K=298.15,
        semantics='test_evaluation_state',
    )

    assert (tmp_path/'fields_eval.npz').is_file()
    for name in ('geometry_eval.png','temperature_eval.png','current_eval.png',
                 'potential_eval.png','progress_eval.png','overview_panel.png'):
        assert (tmp_path/name).is_file()
    with np.load(tmp_path/'fields_eval.npz',allow_pickle=False) as data:
        assert data['temperature_K'].dtype==np.float32
        assert data['temperature_K'].shape==(n,n)
        assert data['current_density_magnitude_A_per_m2'].shape==(n,n)
        assert data['x_mm'].shape==(n,) and data['y_mm'].shape==(n,)
        assert float(data['time_s'])==2.0
    assert artifact['state_semantics']=='test_evaluation_state'
    assert metrics['temperatureP95MinusP05_K']>0
    assert metrics['temperatureRiseCV'] is not None
    assert metrics['currentDensityCV']==0.0
