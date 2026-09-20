"""Experimental paper-reactive ECSP solver components."""

from .configuration import ReactiveCaseConfig, load_reactive_config, parse_reactive_config
from .core import (
    PrimitiveState,
    conservative_to_primitive,
    flux_x,
    flux_y,
    physical_state_mask,
    primitive_to_conservative,
)
from .eos import IdealGasEOS, TaitEOS
from .electrical import (
    ElectricalSourceFields,
    SurfaceContactGeometry,
    canonical_bool_mask,
    joule_heating_sigma_e2,
    surface_heat_flux_to_volume,
)
from .front import (
    NormalRegressionResult,
    NormalSamplingFrame,
    axis_aligned_sampling_frame,
    evaluate_species_threshold_sensitivity,
    track_species_isofront_motion,
)
from .level_set import (
    BoundaryCondition2D,
    MaterialLevelSet,
    ReinitializationConfig,
    advect_material_level_set,
)
from .numerics import (
    BoundaryType,
    FluxDiagnostics,
    cfl_time_step,
    finite_volume_rhs,
    hll_flux,
    interface_fluxes,
    max_signal_speed,
    ssprk3_step,
    weno5_js_reconstruct,
)
from .parallel import (
    MPIRowDomain,
    add_outflow_row_halos,
    add_periodic_row_halos,
    all_row_partitions,
    row_partition,
)
from .reaction import ArrheniusReaction, arrhenius_rate, reaction_source
from .solver import ReactiveResult, ReactiveSolver, run_reactive_case
from .solid import SolidThermalModel, solid_thermal_rhs

__all__ = [
    "ArrheniusReaction",
    "BoundaryCondition2D",
    "BoundaryType",
    "ElectricalSourceFields",
    "FluxDiagnostics",
    "IdealGasEOS",
    "MaterialLevelSet",
    "MPIRowDomain",
    "NormalRegressionResult",
    "NormalSamplingFrame",
    "PrimitiveState",
    "ReactiveCaseConfig",
    "ReactiveResult",
    "ReactiveSolver",
    "ReinitializationConfig",
    "SolidThermalModel",
    "SurfaceContactGeometry",
    "TaitEOS",
    "advect_material_level_set",
    "add_outflow_row_halos",
    "add_periodic_row_halos",
    "all_row_partitions",
    "arrhenius_rate",
    "axis_aligned_sampling_frame",
    "canonical_bool_mask",
    "cfl_time_step",
    "conservative_to_primitive",
    "evaluate_species_threshold_sensitivity",
    "finite_volume_rhs",
    "flux_x",
    "flux_y",
    "hll_flux",
    "interface_fluxes",
    "joule_heating_sigma_e2",
    "load_reactive_config",
    "max_signal_speed",
    "parse_reactive_config",
    "physical_state_mask",
    "primitive_to_conservative",
    "reaction_source",
    "row_partition",
    "run_reactive_case",
    "solid_thermal_rhs",
    "ssprk3_step",
    "surface_heat_flux_to_volume",
    "track_species_isofront_motion",
    "weno5_js_reconstruct",
]
