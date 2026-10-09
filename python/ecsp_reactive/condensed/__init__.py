"""BC two-channel continuation in one condensed, conservative continuum.

The original single-progress paper-reference solver remains in ecsp_reactive.
This package is the opt-in BC-connected path; it does not solve a gas domain.
"""
from .handoff import PreflameReactiveHandoffAdapter
from .solver import CondensedReactiveSolver, run_reactive_propagation

__all__ = ["PreflameReactiveHandoffAdapter", "CondensedReactiveSolver", "run_reactive_propagation"]
