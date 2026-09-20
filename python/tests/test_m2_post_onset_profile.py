from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from ecsp_nsga2.geometry import RasterizedGeometry
from ecsp_nsga2.nsga2 import Individual
from ecsp_nsga2.post_onset import validate_post_onset_config
from ecsp_nsga2.propagation import PropagationCandidateInputError
from ecsp_nsga2.workflow import (
    NSGA2ElectricalSolidWorkflow,OBJECTIVE_NAMES,REFINED_OBJECTIVE_NAMES,
)
from ecsp_reactive.condensed.validation_cases import synthetic_condensed_case
from ecsp_reactive.condensed.handoff import BCReactiveHandoffAdapter, model_digest


ROOT = Path(__file__).resolve().parents[2]
M2_CONFIG = ROOT / "config" / "nsga2_bc_reactive_m2cpu_200x3.yaml"
A100_CONFIG = ROOT / "config" / "nsga2_bc_reactive_a100_gpuonly_200x3.yaml"

PRE_FLAME_KEYS = (
    "project",
    "workflow",
    "optimization",
    "geometry",
    "physics",
    "condensed_ignition",
    "minimum_ignition_voltage_search",
    "baselines",
    "bc_global",
    "evaluator",
)

EXPECTED_PRE_FLAME_SHA256 = {
    M2_CONFIG.name: "416134903da7d345aae2a5f3e71b54021cbcce9c976f1dba03a14eb0313efd63",
    A100_CONFIG.name: "fca22383dab372af97edc3bcc9535c42ebe9ee330a83b8814ae486c708a33e09",
}
EXPECTED_BC_GLOBAL_SHA256 = (
    "b481bfe7308f917b7fe6559a01bb4a4f41b76aab38823f3f9a2a22cb37521730"
)
EXPECTED_PROPAGATION_REFINEMENT_SHA256 = (
    "5365d9b0ddcb5e123ad9d364dc5ac2388c3ca5f45c81211853930b2aecb6e96a"
)

EXPECTED_CHEMISTRY_CONTROLS = {
    "chemistry_integration_mode": "local_adaptive_thermochemical",
    "chemistry_relative_tolerance": 1.0e-7,
    "chemistry_absolute_tolerance": 1.0e-10,
    "chemistry_temperature_tolerance_K": 1.0e-4,
    "maximum_chemistry_corrector_iterations": 8,
    "maximum_chemistry_depletion_iterations": 40,
    "maximum_chemistry_local_refinements": 10,
    "maximum_chemistry_reaction_coordinate_steps": 64,
    "progress_log_interval_steps": 250,
    "progress_log_interval_wall_s": 30.0,
}

REMOVED_POST_ONSET_CAP_OR_SUBCYCLE_KEYS = {
    "maximum_chemical_rate_cap_fraction",
    "maximum_channel_increment",
    "maximum_chemistry_subcycles_per_pde_step",
    "maximum_chemistry_panel_attempts_per_cell",
}


def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("path", [M2_CONFIG, A100_CONFIG])
def test_production_preflame_sections_are_immutable(path: Path) -> None:
    config = _load(path)
    protected = {key: config[key] for key in PRE_FLAME_KEYS}

    assert _digest(protected) == EXPECTED_PRE_FLAME_SHA256[path.name]
    assert model_digest(config["bc_global"]) == EXPECTED_BC_GLOBAL_SHA256
    assert config["optimization"]["population_size"] == 200
    assert config["optimization"]["generations"] == 3
    assert tuple(config["optimization"]["objectives_minimise"]) == OBJECTIVE_NAMES
    assert config["condensed_ignition"]["onset_temperature_K"] == 523.15
    assert config["bc_global"]["kinetics"]["maximum_rate_per_s"] == 1000.0


def test_m2_propagation_objectives_and_settings_are_immutable() -> None:
    config = _load(M2_CONFIG)
    propagation = config["propagation_refinement"]

    assert _digest(propagation) == EXPECTED_PROPAGATION_REFINEMENT_SHA256
    assert tuple(propagation["combined_final_objectives"]) == REFINED_OBJECTIVE_NAMES
    assert propagation["duration_s"] == 1.0
    assert propagation["time_step_s"] == 0.00025
    assert propagation["continued_electrical_heating"] is False
    assert propagation["electrical_heating_mode"] == "off"


@pytest.mark.parametrize("path", [M2_CONFIG, A100_CONFIG])
def test_production_uses_only_local_adaptive_chemistry_controls(path: Path) -> None:
    config = _load(path)
    reactive = config["post_onset"]["reactive_euler"]

    assert {
        key: reactive[key] for key in EXPECTED_CHEMISTRY_CONTROLS
    } == EXPECTED_CHEMISTRY_CONTROLS
    assert REMOVED_POST_ONSET_CAP_OR_SUBCYCLE_KEYS.isdisjoint(reactive)
    assert reactive["eos"]["A_Pa"] == 101325.0
    assert reactive["eos"]["B_Pa"] == 2.0e9
    assert reactive["eos"]["N"] == 7.0
    assert reactive["caloric_closure"] == "bc_cp_integral_plus_tait_cold_energy"
    assert reactive["boundary"] == "reflective"
    assert reactive["riemann_solver"] == "hllc"
    assert reactive["cfl"] == 0.35
    assert reactive["thermal_cfl"] == 0.7

    validated = validate_post_onset_config(
        config["post_onset"],
        for_optimization=True,
    )
    assert validated["reactive_euler"]["chemistry_integration_mode"] == (
        "local_adaptive_thermochemical"
    )


def test_m2_and_a100_keep_only_the_intended_runtime_differences() -> None:
    m2 = _load(M2_CONFIG)["post_onset"]
    a100 = _load(A100_CONFIG)["post_onset"]

    assert m2["reactive_euler"] == a100["reactive_euler"]
    assert m2["compare_backends"] is False
    assert a100["compare_backends"] is True
    assert m2["execution"]["backend"] == "numpy_cpu"
    assert m2["execution"]["device"] == "cpu"
    assert m2["execution"]["batch_size"] == 1
    assert m2["execution"]["cpu_workers"] == 6
    assert a100["execution"]["backend"] == "torch_batch"
    assert a100["execution"]["device"] == "cuda"
    assert a100["execution"]["batch_size"] == 8


def _compatible_handoff(bc_global: dict) -> tuple[dict, dict]:
    shape = (8, 8)
    field = lambda value: np.full(shape, value, dtype=np.float64)
    anode = np.zeros(shape, dtype=bool)
    cathode = np.zeros(shape, dtype=bool)
    anode[:, 0] = True
    cathode[:, -1] = True

    weights = np.asarray(bc_global["kinetics"]["mass_conversion_weights"])
    alpha = np.asarray([0.2, 0.2])
    progress = float(weights @ alpha)
    xi_max = 100.0
    extent = xi_max * progress
    initial_lp = 145.0
    initial_pva = 100.0
    molar_mass_lp = 0.1
    molar_mass_pva = 0.05

    handoff = {
        "handoffSchemaVersion": "ecsp_bc_surface_onset_v8.2.0",
        "onsetSucceeded": True,
        "onsetReportedByPhysics": True,
        "numericallyValidForPropagationHandoff": True,
        "propagationHandoffAuthorizationReason": "ignition_and_numerics_valid",
        "ignitionDelay_s": 0.1,
        "continuedElectricalHeating": False,
        "bcGlobalConfigSHA256": EXPECTED_BC_GLOBAL_SHA256,
        "temperatureAtOnset_K": field(523.15),
        "propellantMask": np.ones(shape, dtype=bool),
        "anodeContactMask": anode,
        "cathodeContactMask": cathode,
        "alphaChannel1AtOnset": field(alpha[0]),
        "alphaChannel2AtOnset": field(alpha[1]),
        "globalProgressAtOnset": field(progress),
        "xiMax_mol_per_m3": xi_max,
        "initialMobileLP_mol_per_m3": initial_lp,
        "initialPVARepeat_mol_per_m3": initial_pva,
        "initialMobileWater_mol_per_m3": 50.0,
        "molarMassLP_kg_per_mol": molar_mass_lp,
        "molarMassPVARepeat_kg_per_mol": molar_mass_pva,
        "initialReactiveMass_kg_per_m3": (
            molar_mass_lp * initial_lp + molar_mass_pva * initial_pva
        ),
        "cationAtOnset_mol_per_m3": field(initial_lp - 1.45 * extent),
        "anionAtOnset_mol_per_m3": field(initial_lp - 1.45 * extent),
        "mobileLPAtOnset_mol_per_m3": field(initial_lp - 1.45 * extent),
        "pvaReactiveRepeatAtOnset_mol_per_m3": field(initial_pva - extent),
        "generatedWaterProductAtOnset_mol_per_m3": field(2.0 * extent),
        "mobileWaterAtOnset_mol_per_m3": field(50.0),
        "electrochemicalLPConsumedAtOnset_mol_per_m3": field(0.0),
        "potentialAtOnset_V": field(0.0),
        "qJAtOnset_W_per_m3": field(0.0),
        "qEchemAtOnset_W_per_m3": field(0.0),
    }
    propagation = {
        "duration_s": 0.001,
        "time_step_s": 0.001,
        "snapshot_interval_s": 0.001,
        "domain_size_m": 0.02,
        "density_kg_per_m3": 1000.0,
        "surface_layer_thickness_m": 0.001,
        "gas_constant_J_per_molK": 8.314462618,
        "continued_electrical_heating": False,
        "electrical_heating_mode": "off",
    }
    return handoff, propagation


def test_authorized_v82_handoff_digest_remains_compatible() -> None:
    config = _load(M2_CONFIG)
    bc_global = config["bc_global"]
    reactive = config["post_onset"]["reactive_euler"]
    handoff, propagation = _compatible_handoff(bc_global)

    adapted = BCReactiveHandoffAdapter(
        propagation,
        bc_global,
        reactive,
    ).adapt(handoff)
    assert adapted.audit["bc_config_sha256"] == EXPECTED_BC_GLOBAL_SHA256
    assert adapted.audit["onset_time_s"] == 0.1

    changed_bc = copy.deepcopy(bc_global)
    changed_bc["kinetics"]["channels"][0]["ln_Af_per_s"][0] += 1.0e-12
    with pytest.raises(PropagationCandidateInputError, match="hash mismatch"):
        BCReactiveHandoffAdapter(
            propagation,
            changed_bc,
            reactive,
        ).adapt(handoff)

    wrong_schema = copy.deepcopy(handoff)
    wrong_schema["handoffSchemaVersion"] = "changed"
    with pytest.raises(PropagationCandidateInputError, match="authorized v8.2"):
        BCReactiveHandoffAdapter(
            propagation,
            bc_global,
            reactive,
        ).adapt(wrong_schema)


def test_resume_final_only_reuse_flag_bypasses_bc_physics_and_stubs_post_onset(
        tmp_path,monkeypatch) -> None:
    """Exercise the real reuse patch and workflow batch-handoff call site."""
    import ecsp_nsga2.bc_native as bc_native
    import ecsp_nsga2.evaluator as evaluator_module
    import ecsp_nsga2.post_onset_batch as post_onset_batch
    import ecsp_nsga2.workflow as workflow_module

    tool=ROOT/'tools'/'resume_final_only.py'
    final_dir=tmp_path/'final'
    handoff_dir=(final_dir/'propagation_candidates'/'route_probe'
                 /'bc_handoff')
    handoff_dir.mkdir(parents=True)
    handoff, _propagation, _bc, _reactive = synthetic_condensed_case(
        shape=(5,5)
    )
    fields={
        key:value for key,value in handoff.items()
        if isinstance(value,np.ndarray)
    }
    metadata={
        key:value for key,value in handoff.items()
        if not isinstance(value,np.ndarray)
    }
    metadata.update(
        field_file='bc_handoff_fields.npz',geometry_id='route_probe'
    )
    fields_path=handoff_dir/'bc_handoff_fields.npz'
    metadata_path=handoff_dir/'bc_handoff_metadata.json'
    np.savez_compressed(fields_path,**fields)
    metadata_path.write_text(
        json.dumps(metadata,sort_keys=True),encoding='utf-8'
    )

    def fingerprint(path):
        stat=path.stat()
        return {
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'size':stat.st_size,
            'mtime_ns':stat.st_mtime_ns,
        }

    before={path.name:fingerprint(path)
            for path in (fields_path,metadata_path)}
    candidate=Individual(
        geometry_id='route_probe',genome={},
        objectives=np.asarray([0.1,0.2,0.3,0.4]),
        metrics={},topology_id='route_topology',
    )
    raster=RasterizedGeometry(
        anode_mask=np.zeros((5,5),dtype=bool),
        cathode_mask=np.zeros((5,5),dtype=bool),
        descriptors={},constraint_violation=0.0,violation_details={},
    )
    post_metrics={
        'status':'complete','onsetSucceeded':True,
        'propagationSucceeded':True,
        'finalUnreactedAreaFraction':0.75,
        'establishedTimeAfterOnset_s':0.01,
        'meanEffectiveRegressionVelocity_m_per_s':0.02,
        'reactionFrontNonuniformity':0.03,
    }
    raw_evaluator_calls=[]
    validated=[]
    post_onset_calls=[]

    def forbidden_evaluator(*_args,_label,**_kwargs):
        raw_evaluator_calls.append(_label)
        raise AssertionError('B/C evaluator was called in reuse mode')

    with monkeypatch.context() as patch:
        # resume_final_only assigns these class attributes directly.  Register
        # their originals with pytest first so no test leaks the wrapper patch.
        patch.setattr(
            NSGA2ElectricalSolidWorkflow,'run',
            NSGA2ElectricalSolidWorkflow.run,
        )
        patch.setattr(
            NSGA2ElectricalSolidWorkflow,'_validate_handoff_reevaluation',
            NSGA2ElectricalSolidWorkflow._validate_handoff_reevaluation,
        )
        seen=set()
        for module in (evaluator_module,bc_native):
            for name,cls in vars(module).items():
                if not isinstance(cls,type):
                    continue
                for method_name in ('evaluate_handoff_batch',
                                    'evaluate_handoff'):
                    method=getattr(cls,method_name,None)
                    identity=(cls,method_name)
                    if not callable(method) or identity in seen:
                        continue
                    seen.add(identity)
                    patch.setattr(
                        cls,method_name,
                        lambda *args,_label=(f'{module.__name__}.{name}.'
                                             f'{method_name}'),**kwargs:
                            forbidden_evaluator(
                                *args,_label=_label,**kwargs
                            ),
                    )
        patch.setattr(sys,'argv',[str(tool),'--reuse-handoff'])
        patch.setattr(sys,'path',list(sys.path))
        namespace=runpy.run_path(
            str(tool),run_name='resume_reuse_route_test'
        )
        assert namespace['_REUSE_HANDOFF'] is True
        assert '--reuse-handoff' not in sys.argv
        from ecsp_nsga2.bc_native import NativeBCHybridEvaluator
        assert NativeBCHybridEvaluator.evaluate_handoff_batch is (
            namespace['_rh_reuse_batch']
        )

        def post_stub(handoffs,_propagation_config,_bc_config,output_dirs,
                      **_kwargs):
            post_onset_calls.append({
                'handoff_count':len(handoffs),
                'output_count':len(output_dirs),
            })
            return [post_metrics]

        patch.setattr(post_onset_batch,'run_post_onset_batch',post_stub)
        patch.setattr(
            workflow_module,'rasterize_and_validate',
            lambda _genome,_limits:raster,
        )
        evaluator=object.__new__(NativeBCHybridEvaluator)
        evaluator.config={'bcGlobal':{'probe':True}}

        def validate_reused(_individual,result):
            validated.append(result)
            assert result['reusedPersistedHandoff'] is True
            return {'preflameHandoffReevaluationConsistent':True}

        def refine_stub(clone,_raster,_directory,*,handoff_result,
                        precomputed_result,**_kwargs):
            assert handoff_result is validated[0]
            assert precomputed_result is post_metrics
            clone.metrics.update(precomputed_result)
            return precomputed_result

        fake=SimpleNamespace(
            evaluator=evaluator,
            post_onset_cfg={'execution':{},'compare_backends':False},
            propagation_cfg={},limits=None,
            _select_propagation_candidates=lambda *_args:[candidate],
            _propagation_metadata=(
                lambda individual,_raster:{
                    'geometry_id':individual.geometry_id
                }
            ),
            _resolved_propagation_config=lambda:{},
            _write_population_csv=lambda *_args,**_kwargs:None,
            _copy_recommendation_geometry=lambda *_args,**_kwargs:None,
            _evaluate_area_matched_staggered=lambda *_args,**_kwargs:None,
            _validate_handoff_reevaluation=validate_reused,
            _run_propagation_refinement=refine_stub,
        )
        recommendation=(
            NSGA2ElectricalSolidWorkflow._finalise_with_propagation(
                fake,[candidate],[candidate],final_dir
            )
        )

        assert raw_evaluator_calls==[]
        assert len(validated)==1
        assert validated[0]['reusedPersistedHandoff'] is True
        assert validated[0]['handoffPreparationFailed'] is False
        assert post_onset_calls==[{'handoff_count':1,'output_count':1}]
        assert recommendation.geometry_id=='route_probe'

    after={path.name:fingerprint(path)
           for path in (fields_path,metadata_path)}
    assert after==before
