"""Configuration migration preserves physical inputs and rejects ambiguity."""
from __future__ import annotations
import copy
from pathlib import Path
import pytest
import yaml
from ecsp_preflame.configuration import normalize_preflame_config, normalize_evaluator_config
from ecsp_preflame.evaluator import create_evaluator

ROOT = Path(__file__).resolve().parents[2]


def test_legacy_aliases_preserve_nested_inputs_without_mutating_caller():
    legacy = {
        'bc_global': {'kinetics': {'mass_conversion_weights': [2 / 3, 1 / 3]},
                      'thermal': {'heat_capacity': {'value': 2200.0}}},
        'optimization': {'numerical_cap_thresholds': {'temperature': 0.0}},
        'evaluator': {'backend': 'bc_global_native', 'native': {'cpu_workers': 2}},
    }
    before = copy.deepcopy(legacy)
    canonical = normalize_preflame_config(legacy)
    assert canonical['preflame_model'] == legacy['bc_global']
    assert canonical['evaluation'] == legacy['optimization']
    assert canonical['evaluator'] == {
        'backend': 'preflame_cpp_cuda', 'cpp_cuda': {'cpu_workers': 2}}
    canonical['preflame_model']['kinetics']['mass_conversion_weights'][0] = 0.0
    assert legacy == before


@pytest.mark.parametrize('legacy_key', ['bc_global', 'bcGlobal', 'preflameModel'])
def test_conflicting_model_aliases_fail_closed(legacy_key):
    with pytest.raises(ValueError, match='Conflicting configuration aliases'):
        normalize_preflame_config({
            'preflame_model': {'timeStep_s': 0.001}, legacy_key: {'timeStep_s': 0.002}})


def test_conflicting_execution_aliases_fail_closed():
    with pytest.raises(ValueError, match='Conflicting configuration aliases'):
        normalize_evaluator_config({'cpp_cuda': {'cpu_workers': 2}, 'native': {'cpu_workers': 6}})


def test_legacy_and_canonical_workflows_resolve_identical_physics(tmp_path):
    canonical = yaml.safe_load((ROOT / 'config/preflame_cpp_cuda_debug.yaml').read_text())
    canonical['evaluator']['backend'] = 'preflame_torch'
    legacy = copy.deepcopy(canonical)
    legacy['bc_global'] = legacy.pop('preflame_model')
    legacy['optimization'] = legacy.pop('evaluation')
    legacy['evaluator']['backend'] = 'bc_global_preflame'
    legacy['evaluator']['native'] = legacy['evaluator'].pop('cpp_cuda')
    before = copy.deepcopy(legacy)
    evaluators = []
    for index, raw in enumerate((canonical, legacy)):
        adapter = dict(raw['evaluator'])
        adapter['physics_config'] = raw
        evaluators.append(create_evaluator(ROOT, adapter, tmp_path / str(index), False))
    assert evaluators[0].config == evaluators[1].config
    assert evaluators[0].composition == evaluators[1].composition
    assert legacy == before
