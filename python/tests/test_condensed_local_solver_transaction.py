from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from ecsp_nsga2.propagation import PropagationCandidateNumericalError
from ecsp_reactive.condensed.chemistry import ENERGY, NCONS
from ecsp_reactive.condensed.handoff import BCReactiveHandoffAdapter
from ecsp_reactive.condensed.solver import BCReactiveSolver
from ecsp_reactive.condensed.validation_cases import synthetic_condensed_case


def _local_case(*, hot_spot=False):
    handoff, propagation, bc, reactive = synthetic_condensed_case(
        shape=(5, 5), rates=(20.0, 8.0), heats=(2.0e5, 3.0e5),
        alpha=(0.2, 0.2),
    )
    handoff["temperatureAtOnset_K"][:] = 600.0
    if hot_spot:
        handoff["temperatureAtOnset_K"][:] = 450.0
        handoff["temperatureAtOnset_K"][2, 2] = 1000.0
        for channel in bc["kinetics"]["channels"]:
            channel.update(
                activation_energy_J_per_mol=[150000.0, 150000.0],
                ln_Af_per_s=[25.0, 25.0],
                heat_release_J_per_kg=8.0e5,
            )
    propagation.update(
        duration_s=1.0e-4 if hot_spot else 2.0e-3,
        time_step_s=1.0e-4 if hot_spot else 2.0e-3,
        snapshot_interval_s=1.0e-4 if hot_spot else 2.0e-3,
    )
    reactive.update(
        chemistry_integration_mode="local_adaptive_thermochemical",
        chemistry_relative_tolerance=1.0e-7 if hot_spot else 1.0e-9,
        chemistry_absolute_tolerance=1.0e-10 if hot_spot else 1.0e-12,
        chemistry_temperature_tolerance_K=1.0e-4 if hot_spot else 1.0e-6,
        maximum_chemistry_corrector_iterations=8 if hot_spot else 12,
        maximum_chemistry_depletion_iterations=40 if hot_spot else 50,
        maximum_chemistry_local_refinements=16,
        maximum_step_retries=12,
        minimum_time_step_s=1.0e-14,
        stationary_mechanics_fast_path=True,
        riemann_solver="hllc",
        progress_log_interval_steps=1000,
        progress_log_interval_wall_s=1.0e9,
    )
    adapted = BCReactiveHandoffAdapter(
        propagation, bc, reactive
    ).adapt(handoff)
    return adapted, propagation, bc, reactive


def test_local_chemistry_work_exhaustion_is_terminal_and_uncommitted(
        tmp_path, monkeypatch):
    adapted, propagation, bc, reactive = _local_case(hot_spot=True)
    solver = BCReactiveSolver(
        adapted, propagation, bc, reactive, initialize_electrical=False
    )

    def exhausted_local_map(*_args, **_kwargs):
        raise PropagationCandidateNumericalError(
            "Local chemistry chronological scheduler exceeded its derived "
            "panel-attempt work bound"
        )

    # Work exhaustion is deliberately injected at the local-map boundary.
    # The production bound is derived from the refinement depth rather than a
    # user trajectory knob, so making a real case exhaust it would couple this
    # transactional test to the local integrator's implementation details.
    monkeypatch.setattr(solver.chem, "advance_local", exhausted_local_map)

    with pytest.raises(
            PropagationCandidateNumericalError,
            match="derived panel-attempt work bound") as failure:
        solver.run(tmp_path / "work_exhaustion")

    assert failure.value.retry_reason == "local_chemistry"
    # A local accuracy/work failure is terminal, not a reason to keep halving
    # the global PDE step until the numerical work happens to become cheaper.
    assert solver.rejected_steps == 1
    assert solver.step_number == 0
    assert solver.chemistry_half_step_attempts == 1
    assert solver.chemistry_failed_half_steps == 1
    np.testing.assert_array_equal(solver.U, adapted.U)
    assert not (tmp_path / "work_exhaustion" / "propagation_metrics.json").exists()
    assert not (tmp_path / "work_exhaustion" / "INCOMPLETE_STATE.npz").exists()


def test_qchem_source_and_ledger_are_the_accepted_state_energy_increment():
    adapted, propagation, bc, reactive = _local_case()
    propagation = dict(propagation, time_step_s=1.0e-3)
    solver = BCReactiveSolver(
        adapted, propagation, bc, reactive, initialize_electrical=False
    )
    dt = 1.0e-3

    after, integrated, _, sources = solver.advance(0.0, adapted.U, dt)
    energy_increment = after[..., ENERGY] - adapted.U[..., ENERGY]
    source_increment = sources["qChem_W_per_m3"] * dt

    # This uniform stationary case has no net conductive, boundary, loss, or
    # electrical energy.  qChem is therefore the exact conservative-state
    # increment returned by the two accepted chemistry maps, not a separately
    # sampled kinetic rate or a duplicate solid-energy update.
    np.testing.assert_array_equal(source_increment, energy_increment)
    expected = float(np.sum(energy_increment) * solver.vol)
    assert float(integrated[NCONS + 2]) == expected
    assert float(np.sum(source_increment) * solver.vol) == expected
    assert np.count_nonzero(integrated) == 1


def test_post_chemistry_stability_retry_rolls_back_state_and_ledgers(tmp_path):
    adapted, propagation, bc, reactive = _local_case()
    retry_solver = BCReactiveSolver(
        adapted, propagation, bc, reactive, initialize_electrical=False
    )
    original_limit = retry_solver._nonchemical_time_step_limit
    limit_calls = 0

    def reject_only_the_first_post_chemistry_check(state):
        nonlocal limit_calls
        limit_calls += 1
        # Call 1 selects the initial PDE step.  Call 2 is the stability check
        # after its first chemistry half-step and forces the sole retry.
        if limit_calls == 2:
            return propagation["duration_s"] / 2.0
        return original_limit(state)

    retry_solver._nonchemical_time_step_limit = (
        reject_only_the_first_post_chemistry_check
    )
    metrics = retry_solver.run(tmp_path / "retry")

    reference_propagation = dict(
        propagation, time_step_s=propagation["duration_s"] / 2.0
    )
    reference_solver = BCReactiveSolver(
        adapted, reference_propagation, bc, reactive,
        initialize_electrical=False,
    )
    reference_metrics = reference_solver.run(tmp_path / "reference")

    # Both accepted trajectories consist of the same two half-sized PDE
    # steps.  Equality proves that the provisional chemistry map from the
    # rejected large step did not leak into state or the accepted ledger.
    np.testing.assert_array_equal(retry_solver.U, reference_solver.U)
    assert metrics["integratedChemicalHeat_J"] \
        == reference_metrics["integratedChemicalHeat_J"]
    assert metrics["acceptedTimeSteps"] == 2
    assert metrics["rejectedTimeSteps"] == 1
    assert metrics["postChemistryTimestepRecheckRejections"] == 1
    assert metrics["acceptedChemistryHalfSteps"] == 4
    assert metrics["chemistryHalfStepAttempts"] == 5
    assert metrics["failedChemistryHalfSteps"] == 0
    assert metrics["chemistryEndpointEvaluationCount"] \
        == reference_metrics["chemistryEndpointEvaluationCount"]
    assert metrics["attemptedChemistryEndpointEvaluationCount"] \
        > metrics["chemistryEndpointEvaluationCount"]

    total_energy_increment = float(np.sum(
        retry_solver.U[..., ENERGY] - adapted.U[..., ENERGY]
    ) * retry_solver.vol)
    assert metrics["integratedChemicalHeat_J"] == total_energy_increment
    with (tmp_path / "retry" / "propagation_history.csv").open(
            newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    assert float(history[-1]["chemical_energy_J"]) \
        == metrics["integratedChemicalHeat_J"]

    assert metrics["chemistryIntegrationMode"] \
        == "local_adaptive_thermochemical"
    assert metrics["physicalChemicalRatesClipped"] is False
    assert metrics["chemicalRateThresholdDiagnosticOnly"] is True
    assert metrics["transactionalStepRetries"] is True
    assert metrics["chemistryPhysicalKinetics"] \
        == "raw_uncapped_two_channel_BC_tables"
    assert all("cap" not in name.casefold() for name in metrics)

    required_residual_extrema = (
        "maximumChemistryCaloricInverseResidual_J_per_kg",
        "maximumChemistryHeatClosureResidual_J_per_m3",
        "maximumChemistryInventoryEventResidual_mol_per_m3",
        "maximumAttemptedChemistryCaloricInverseResidual_J_per_kg",
        "maximumAttemptedChemistryHeatClosureResidual_J_per_m3",
        "maximumAttemptedChemistryInventoryEventResidual_mol_per_m3",
    )
    for name in required_residual_extrema:
        assert name in metrics
        assert math.isfinite(metrics[name])
        assert metrics[name] >= 0.0
