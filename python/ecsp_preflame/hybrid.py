from __future__ import annotations

"""A100 + CPU heterogeneous evaluator for ECSP v7.9.5.

The existing standalone C++/FP64 solver is preserved.  The hybrid backend adds
an equation-matched batched Torch/CUDA/FP64 engine and a work-stealing
scheduler.  CUDA receives full batches, while independent C++ processes consume
other candidates on the CPU.  A CUDA failure is isolated and re-queued to the
CPU rather than aborting the generation.
"""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import os
import subprocess
import time

import numpy as np

from .evaluator import (
    CppCondensedFp64Evaluator,
    EvaluatorError,
    _strict_json_value,
)
from ecsp_cuda import CudaBatchCase, TorchCudaSurfaceContactBatchRunner
from ecsp_cpp.backend import CppPhysicsRunner


@dataclass(frozen=True)
class _HybridTask:
    task_id: int
    anode: np.ndarray
    cathode: np.ndarray
    metadata: Mapping[str, Any]
    output_dir: Path
    config_values: Mapping[str, Any]
    force_cpu: bool = False
    cuda_attempts: int = 0


class HybridCudaCpuFp64Evaluator(CppCondensedFp64Evaluator):
    """Dynamic A100/48-vCPU surface-contact evaluator.

    The reference CPU executable from v7.9.4 remains untouched at
    ``cpp/ecsp_cpp_solver.cpp`` and is still selected by ``cpp_fp64_cpu``.
    Hybrid mode uses ``cpp/ecsp_cpp_solver_hybrid.cpp`` for CPU workers because
    its explicit thermal/gas step is buffered and therefore order-independent,
    matching the batched CUDA discretisation.
    """

    def __init__(self, package_root: Path, config: Mapping[str, Any], workdir: Path) -> None:
        adapter = dict(config)
        hybrid_executable = Path(
            str(adapter.get("hybrid_cpu_executable", "build/ecsp_cpp_solver_hybrid"))
        )
        hybrid_path = (Path(package_root).resolve() / hybrid_executable).resolve()
        if not hybrid_path.is_file() and bool(adapter.get("cpp_auto_build", True)):
            script = Path(package_root).resolve() / "tools" / "build_cpp_hybrid_cpu.sh"
            completed = subprocess.run(
                ["bash", str(script)], cwd=Path(package_root).resolve(),
                capture_output=True, text=True,
            )
            if completed.returncode != 0:
                raise EvaluatorError(
                    "Hybrid CPU reference build failed:\n"
                    + (completed.stderr or completed.stdout)
                )
        adapter["cpp_executable"] = str(hybrid_executable)
        adapter["cpp_auto_build"] = False
        # CppCondensedFp64Evaluator initializes all model/configuration fields
        # and the CPU runner.  The original cpp_fp64_cpu backend is not changed.
        super().__init__(package_root, adapter, workdir)
        self.cpp_runner = CppPhysicsRunner(self.package_root, hybrid_path)
        self.cpp_version = self.cpp_runner.verify()

        def _env_or_config_int(env_name: str, config_name: str, default: int) -> int:
            raw = os.environ.get(env_name)
            if raw is None or not raw.strip():
                raw = adapter.get(config_name, default)
            try:
                return int(raw)
            except (TypeError, ValueError) as exc:
                raise EvaluatorError(
                    f"{env_name}/{config_name} must be an integer, got {raw!r}"
                ) from exc

        self.requested_cpu_workers = max(1, _env_or_config_int(
            "ECSP_HYBRID_CPU_WORKERS", "hybrid_cpu_worker_cases", 40
        ))
        self.reserved_host_vcpus = max(1, _env_or_config_int(
            "ECSP_HYBRID_HOST_RESERVE", "hybrid_reserved_host_vcpus", 8
        ))
        try:
            self.visible_cpu_count = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            self.visible_cpu_count = int(os.cpu_count() or 1)
        safe_cpu_workers = max(1, self.visible_cpu_count - self.reserved_host_vcpus)
        if bool(adapter.get("hybrid_allow_cpu_oversubscription", False)):
            self.cpu_worker_cases = self.requested_cpu_workers
        else:
            self.cpu_worker_cases = min(self.requested_cpu_workers, safe_cpu_workers)
        self.cuda_batch_size = max(1, _env_or_config_int(
            "ECSP_HYBRID_CUDA_BATCH_SIZE", "hybrid_cuda_batch_size", 64
        ))
        self.cuda_min_batch_size = max(1, _env_or_config_int(
            "ECSP_HYBRID_CUDA_MIN_BATCH_SIZE", "hybrid_cuda_min_batch_size", 8
        ))
        if self.cuda_min_batch_size > self.cuda_batch_size:
            raise EvaluatorError(
                "hybrid_cuda_min_batch_size cannot exceed hybrid_cuda_batch_size"
            )
        self.cuda_device = str(adapter.get("hybrid_cuda_device", "cuda:0"))
        self.cuda_enabled = bool(adapter.get("hybrid_cuda_enabled", True))
        self.cuda_fallback_to_cpu = bool(adapter.get("hybrid_cuda_fallback_to_cpu", True))
        self.cuda_max_retries = max(0, int(adapter.get("hybrid_cuda_max_retries", 1)))
        self.cuda_allow_partial_final_batch = bool(
            adapter.get("hybrid_cuda_allow_partial_final_batch", True)
        )
        # When a V_min wave is smaller than one full CUDA batch, reserve a
        # configurable handful of trials for CPU workers so both devices still
        # make progress. Large waves always give CUDA a full batch first.
        self.cpu_reserve_per_wave = max(0, _env_or_config_int(
            "ECSP_HYBRID_CPU_RESERVE_PER_WAVE",
            "hybrid_cpu_reserve_per_wave",
            16,
        ))
        self.cuda_runner: TorchCudaSurfaceContactBatchRunner | None = None
        self.cuda_info: dict[str, Any] = {}
        self._cuda_runtime_disabled_reason: str | None = None
        if self.cuda_enabled:
            try:
                self.cuda_runner = TorchCudaSurfaceContactBatchRunner(
                    self.cuda_device,
                    dtype_name="float64",
                    deterministic=bool(adapter.get("hybrid_cuda_deterministic", True)),
                    allow_tf32=False,
                )
                self.cuda_info = self.cuda_runner.verify()
            except Exception as exc:
                if bool(adapter.get("hybrid_require_cuda", True)):
                    raise EvaluatorError(f"Hybrid backend requires CUDA: {exc}") from exc
                self.cuda_runner = None
                self._cuda_runtime_disabled_reason = f"{type(exc).__name__}: {exc}"

        self.hybrid_poll_interval_s = max(
            0.01, float(adapter.get("hybrid_poll_interval_s", 0.05))
        )
        self._hybrid_stats = {
            "cpu_completed": 0,
            "cuda_completed": 0,
            "cuda_batches": 0,
            "cuda_failures_requeued_to_cpu": 0,
            "cpu_failures": 0,
            "cuda_failures": 0,
        }
        self._write_hybrid_diagnostics(adapter)

    def _write_hybrid_diagnostics(self, adapter: Mapping[str, Any]) -> None:
        payload = {
            "backend": "hybrid_cuda_fp64_a100_cpp_fp64_cpu",
            "cpu_engine": self.cpp_version,
            "cpu_executable": str(self.cpp_runner.executable),
            "cpu_worker_cases": self.cpu_worker_cases,
            "requested_cpu_worker_cases": self.requested_cpu_workers,
            "visible_cpu_count": self.visible_cpu_count,
            "reserved_host_vcpus": self.reserved_host_vcpus,
            "cuda_enabled": self.cuda_runner is not None,
            "cuda_device": self.cuda_device,
            "cuda_batch_size": self.cuda_batch_size,
            "cuda_min_batch_size": self.cuda_min_batch_size,
            "cuda_fallback_to_cpu": self.cuda_fallback_to_cpu,
            "cpu_reserve_per_wave": self.cpu_reserve_per_wave,
            "cuda_runtime_disabled_reason": self._cuda_runtime_disabled_reason,
            "cuda_info": self.cuda_info,
            "configured_48vcpu_target_partition": {
                "cpu_solver_processes": self.requested_cpu_workers,
                "reserved_for_python_cuda_host_io_os": self.reserved_host_vcpus,
            },
            "effective_runtime_partition": {
                "cpu_solver_processes": self.cpu_worker_cases,
                "visible_logical_cpus": self.visible_cpu_count,
                "remaining_visible_cpus": max(0, self.visible_cpu_count - self.cpu_worker_cases),
            },
            "model_consistency": {
                "cpu_hybrid_update": "buffered_explicit",
                "cuda_update": "buffered_explicit",
                "legacy_cpu_solver_preserved": "cpp/ecsp_cpp_solver.cpp",
                "legacy_cpu_backend_preserved": "cpp_fp64_cpu",
            },
            "active_objectives": [
                "ignition_delay_s_at_reference_voltage",
                "area_undecomposed_fraction_at_2s_at_reference_voltage",
                "minimum_ignition_voltage_V",
                "current_congestion_at_reference_voltage",
            ],
        }
        (self.workdir / "hybrid_evaluator_diagnostics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _decorate_row(
        self,
        row: Mapping[str, Any],
        metadata: Mapping[str, Any],
        *,
        backend: str,
        device: str,
    ) -> dict[str, Any]:
        out = dict(row)
        out.update(
            {
                "geometry_id": str(metadata.get("geometry_id")),
                "backend": backend,
                "solverRevision": (
                    "v7_9_5_hybrid_cuda_cpu_surface_contact_minimum_ignition_voltage"
                ),
                "empiricalSurfaceReactionProgressUsed": False,
                "modelStatus": self.config["project"]["modelStatus"],
                "postIgnitionClosure": "none_condensed_phase_only",
                "evaluationTime_s": float(min(2.0, self.end_time_s)),
                "ignitionCriterionType": self.config["condensedIgnition"]["criterion"],
                "ignitionOnsetTemperature_K": float(
                    self.config["condensedIgnition"]["decompositionOnsetTemperature_K"]
                ),
                "ignitionInterpretation": self.config["condensedIgnition"]["interpretation"],
                "physicsDevice": device,
                "physicsDtype": "float64",
                "gridSize": self.grid_size,
                "timeStep_s": float(self.config["coupled"]["timeStep_s"]),
                "surfaceContactModel": True,
                "electrodeMasksRemovePropellant": False,
                "propellantDomainAreaFraction": 1.0,
                "hiddenBusConnectionAssumed": True,
                "hybridExecutionBackend": backend,
            }
        )
        return out

    def _cpp_failed_geometry_row(
        self, metadata: Mapping[str, Any], output_dir: Path, exc: Exception
    ) -> dict[str, Any]:
        row = super()._cpp_failed_geometry_row(metadata, output_dir, exc)
        row.update({
            "backend": "hybrid_cuda_fp64_a100_cpp_fp64_cpu",
            "solverRevision": (
                "v7_9_5_hybrid_cuda_cpu_surface_contact_minimum_ignition_voltage"
            ),
            "physicsDevice": "hybrid",
            "physicsDtype": "float64",
        })
        return row

    def _cpu_collect(
        self,
        task: _HybridTask,
        prepared_case: Any,
        process: Any,
        started: float,
    ) -> dict[str, Any]:
        row = self.cpp_runner.collect_prepared_case(prepared_case, process, started)
        row = self._decorate_row(
            row, task.metadata,
            backend="cpp_fp64_cpu_hybrid_buffered",
            device="cpu",
        )
        task.output_dir.mkdir(parents=True, exist_ok=True)
        (task.output_dir / "condensed_metrics.json").write_text(
            json.dumps(_strict_json_value(row), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return row

    def _gpu_run(self, tasks: Sequence[_HybridTask]) -> list[dict[str, Any]]:
        if self.cuda_runner is None:
            raise EvaluatorError("CUDA runner is unavailable")
        cases = [
            CudaBatchCase(
                anode=t.anode,
                cathode=t.cathode,
                config=t.config_values,
                output_dir=t.output_dir,
                geometry_id=str(t.metadata.get("geometry_id")),
            )
            for t in tasks
        ]
        return self.cuda_runner.run_batch(cases)

    @staticmethod
    def _pop_gpu_batch(
        pending: deque[_HybridTask],
        batch_size: int,
        minimum_size: int,
        allow_partial: bool,
        cpu_running_count: int,
        cpu_worker_capacity: int,
        cpu_reserve_per_wave: int,
    ) -> list[_HybridTask]:
        eligible = sum(1 for task in pending if not task.force_cpu)
        if eligible < minimum_size and not (allow_partial and cpu_running_count == 0 and eligible > 0):
            return []
        free_cpu = max(0, int(cpu_worker_capacity) - int(cpu_running_count))
        reserve = 0
        if eligible <= batch_size and eligible > minimum_size and free_cpu > 0:
            reserve = min(
                max(0, int(cpu_reserve_per_wave)),
                free_cpu,
                eligible - minimum_size,
            )
        take = min(batch_size, eligible - reserve)
        if take < minimum_size and eligible >= minimum_size:
            take = minimum_size
        chosen: list[_HybridTask] = []
        kept: deque[_HybridTask] = deque()
        while pending and len(chosen) < take:
            task = pending.popleft()
            if task.force_cpu:
                kept.append(task)
            else:
                chosen.append(task)
        while pending:
            kept.append(pending.popleft())
        pending.extend(kept)
        return chosen

    def _execute_hybrid_tasks(
        self,
        tasks: Sequence[_HybridTask],
        *,
        stage: str,
    ) -> dict[int, dict[str, Any]]:
        """Work-stealing CPU/GPU scheduler for independent physics trials."""
        pending: deque[_HybridTask] = deque(tasks)
        results: dict[int, dict[str, Any]] = {}
        cpu_running: dict[int, tuple[_HybridTask, Any, Any, float]] = {}
        gpu_future: Future[list[dict[str, Any]]] | None = None
        gpu_tasks: list[_HybridTask] = []
        started_all = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ecsp-a100")
        dynamic_cuda_batch = self.cuda_batch_size

        print(
            f"[hybrid:{stage}] start tasks={len(tasks)} cpu_workers={self.cpu_worker_cases} "
            f"cuda={'on' if self.cuda_runner is not None else 'off'} "
            f"cuda_batch={dynamic_cuda_batch}",
            flush=True,
        )
        try:
            while pending or cpu_running or gpu_future is not None:
                # GPU receives the first full/eligible batch before CPU workers
                # are filled.  Subsequent batches are submitted immediately on
                # completion, while CPU processes independently work-steal.
                if gpu_future is None and self.cuda_runner is not None:
                    gpu_tasks = self._pop_gpu_batch(
                        pending,
                        dynamic_cuda_batch,
                        self.cuda_min_batch_size,
                        self.cuda_allow_partial_final_batch,
                        len(cpu_running),
                        self.cpu_worker_cases,
                        self.cpu_reserve_per_wave,
                    )
                    if gpu_tasks:
                        gpu_future = executor.submit(self._gpu_run, gpu_tasks)
                        self._hybrid_stats["cuda_batches"] += 1

                while pending and len(cpu_running) < self.cpu_worker_cases:
                    task = pending.popleft()
                    try:
                        prepared_case = self.cpp_runner.prepare_case(
                            task.anode, task.cathode, task.config_values, task.output_dir
                        )
                        started = time.perf_counter()
                        process = self.cpp_runner.launch_prepared_case(prepared_case)
                        cpu_running[task.task_id] = (task, prepared_case, process, started)
                    except Exception as exc:
                        self._hybrid_stats["cpu_failures"] += 1
                        results[task.task_id] = self._cpp_failed_geometry_row(
                            task.metadata, task.output_dir, exc
                        )

                progressed = False
                for task_id, (task, prepared_case, process, started) in list(cpu_running.items()):
                    if process.poll() is None:
                        continue
                    progressed = True
                    del cpu_running[task_id]
                    try:
                        results[task_id] = self._cpu_collect(
                            task, prepared_case, process, started
                        )
                        self._hybrid_stats["cpu_completed"] += 1
                    except Exception as exc:
                        self._hybrid_stats["cpu_failures"] += 1
                        results[task_id] = self._cpp_failed_geometry_row(
                            task.metadata, task.output_dir, exc
                        )

                if gpu_future is not None and gpu_future.done():
                    progressed = True
                    try:
                        gpu_rows = gpu_future.result()
                        if len(gpu_rows) != len(gpu_tasks):
                            raise RuntimeError(
                                f"CUDA row count mismatch: {len(gpu_rows)} vs {len(gpu_tasks)}"
                            )
                        for task, raw in zip(gpu_tasks, gpu_rows):
                            if bool(raw.get("physicsRejected", False)):
                                self._hybrid_stats["cuda_failures"] += 1
                                reason = str(raw.get("physicsRejectionReason", "CUDA physics rejected"))
                                (task.output_dir / "cuda_fallback_reason.txt").write_text(
                                    reason + "\n", encoding="utf-8"
                                )
                                if self.cuda_fallback_to_cpu:
                                    pending.appendleft(
                                        replace(task, force_cpu=True, cuda_attempts=task.cuda_attempts + 1)
                                    )
                                    self._hybrid_stats["cuda_failures_requeued_to_cpu"] += 1
                                else:
                                    results[task.task_id] = self._cpp_failed_geometry_row(
                                        task.metadata, task.output_dir, RuntimeError(reason)
                                    )
                            else:
                                row = self._decorate_row(
                                    raw, task.metadata,
                                    backend="torch_cuda_fp64_a100",
                                    device=self.cuda_device,
                                )
                                task.output_dir.mkdir(parents=True, exist_ok=True)
                                (task.output_dir / "condensed_metrics.json").write_text(
                                    json.dumps(
                                        _strict_json_value(row), indent=2, allow_nan=False
                                    ),
                                    encoding="utf-8",
                                )
                                results[task.task_id] = row
                                self._hybrid_stats["cuda_completed"] += 1
                    except Exception as exc:
                        self._hybrid_stats["cuda_failures"] += len(gpu_tasks)
                        if self.cuda_runner is not None:
                            try:
                                self.cuda_runner.torch.cuda.empty_cache()
                            except Exception:
                                pass
                        # Out-of-memory or a batch-level CUDA failure: retry once
                        # with smaller GPU batches, then move cases to CPU.
                        can_shrink = dynamic_cuda_batch > self.cuda_min_batch_size
                        if can_shrink:
                            dynamic_cuda_batch = max(
                                self.cuda_min_batch_size, dynamic_cuda_batch // 2
                            )
                        for task in reversed(gpu_tasks):
                            task.output_dir.mkdir(parents=True, exist_ok=True)
                            (task.output_dir / "cuda_fallback_reason.txt").write_text(
                                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                            )
                            retry_allowed = (
                                can_shrink
                                and task.cuda_attempts < self.cuda_max_retries
                            )
                            if retry_allowed:
                                pending.appendleft(
                                    replace(
                                        task,
                                        cuda_attempts=task.cuda_attempts + 1,
                                    )
                                )
                            elif self.cuda_fallback_to_cpu:
                                pending.appendleft(
                                    replace(
                                        task,
                                        force_cpu=True,
                                        cuda_attempts=task.cuda_attempts + 1,
                                    )
                                )
                                self._hybrid_stats[
                                    "cuda_failures_requeued_to_cpu"
                                ] += 1
                            else:
                                results[task.task_id] = self._cpp_failed_geometry_row(
                                    task.metadata, task.output_dir, exc
                                )
                        if not can_shrink:
                            self._cuda_runtime_disabled_reason = (
                                f"runtime_disabled_after_batch_failure: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            self.cuda_runner = None
                    finally:
                        gpu_future = None
                        gpu_tasks = []

                complete = len(results)
                if progressed and (complete == len(tasks) or complete % max(1, len(tasks) // 10) == 0):
                    print(
                        f"[hybrid:{stage}] complete={complete}/{len(tasks)} "
                        f"cpu_active={len(cpu_running)} cuda_active={int(gpu_future is not None)} "
                        f"pending={len(pending)} elapsed={time.perf_counter()-started_all:.1f}s",
                        flush=True,
                    )
                if not progressed and (cpu_running or gpu_future is not None):
                    time.sleep(self.hybrid_poll_interval_s)
        except BaseException:
            for _, _, process, _ in cpu_running.values():
                if process.poll() is None:
                    process.terminate()
            for _, _, process, _ in cpu_running.values():
                try:
                    process.wait(timeout=2.0)
                except Exception:
                    if process.poll() is None:
                        process.kill()
            if gpu_future is not None:
                gpu_future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=False)
            stats_path = self.workdir / "hybrid_runtime_stats.json"
            payload = dict(self._hybrid_stats)
            payload.update(
                {
                    "last_stage": stage,
                    "last_stage_task_count": len(tasks),
                    "last_stage_wall_s": time.perf_counter() - started_all,
                }
            )
            stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return results

    def _attach_minimum_ignition_voltage_search_hybrid(
        self,
        prepared_items: Mapping[int, tuple[np.ndarray, np.ndarray, Mapping[str, Any], Path]],
        results: list[dict[str, Any] | None],
    ) -> None:
        if not self.vmin_enabled:
            raise EvaluatorError("V_min is active and the search cannot be disabled")
        states: dict[int, dict[str, Any]] = {}

        def finalise(
            i: int,
            *,
            status: str,
            search_valid: bool,
            objective_value: float,
            physical_value: float | None,
            low_nonigniting: float | None,
            high_igniting: float | None,
            left_censored: bool = False,
            right_censored: bool = False,
            failure_reason: str | None = None,
        ) -> None:
            state = states[i]
            row = results[i]
            if row is None:
                return
            trials = state["trials"]
            monotonic = self._vmin_monotonicity_ok(trials)
            if not monotonic and search_valid:
                search_valid = False
                status = "invalid_nonmonotonic_sampled_ignition_response"
                failure_reason = "sampled_ignition_response_is_nonmonotonic_in_voltage"
                objective_value = self.vmin_upper_bound_V + self.vmin_invalid_penalty_V
                physical_value = None
            width = (
                float(high_igniting - low_nonigniting)
                if high_igniting is not None and low_nonigniting is not None
                else None
            )
            row.update(
                {
                    "referenceAppliedVoltage_V": float(self.voltage),
                    "minimumIgnitionVoltageObjective_V": float(objective_value),
                    "minimumIgnitionVoltage_V": physical_value,
                    "minimumIgnitionVoltageSearchStatus": status,
                    "minimumIgnitionVoltageSearchValid": bool(search_valid),
                    "minimumIgnitionVoltageSearchFailureReason": failure_reason,
                    "minimumIgnitionVoltageLeftCensored": bool(left_censored),
                    "minimumIgnitionVoltageRightCensored": bool(right_censored),
                    "minimumIgnitionVoltageLowerSearchBound_V": float(self.vmin_lower_bound_V),
                    "minimumIgnitionVoltageUpperSearchBound_V": float(self.vmin_upper_bound_V),
                    "minimumIgnitionVoltageLowerNonIgnitingBound_V": low_nonigniting,
                    "minimumIgnitionVoltageUpperIgnitingBound_V": high_igniting,
                    "minimumIgnitionVoltageBracketWidth_V": width,
                    "minimumIgnitionVoltageTargetTolerance_V": float(self.vmin_tolerance_V),
                    "minimumIgnitionVoltageBisectionIterations": int(state["bisection_iterations"]),
                    "minimumIgnitionVoltageTrialCount": len(trials),
                    "minimumIgnitionVoltageMonotonicityObserved": bool(monotonic),
                    "minimumIgnitionVoltageNumericalThresholds": dict(self.vmin_numerical_thresholds),
                    "minimumIgnitionVoltageTrials": trials,
                    "minimumIgnitionVoltageDefinition": (
                        "minimum numerically-valid applied voltage within the configured "
                        "search interval that reaches the condensed-phase ignition criterion "
                        "by the finite simulation horizon; conservative upper/igniting bracket"
                    ),
                    "minimumIgnitionVoltageOtherObjectivesReferenceVoltage_V": float(self.voltage),
                }
            )
            state["done"] = True

        active: list[int] = []
        for i in prepared_items:
            row = results[i]
            if row is None:
                continue
            ref = self._vmin_trial_summary(
                row, self.vmin_upper_bound_V, "reference_voltage_upper_bound"
            )
            states[i] = {
                "trials": [ref], "low": None, "high": self.vmin_upper_bound_V,
                "bisection_iterations": 0, "done": False,
            }
            if not bool(ref["numerically_valid_for_threshold_classification"]):
                finalise(
                    i, status="invalid_reference_voltage_numerics", search_valid=False,
                    objective_value=self.vmin_upper_bound_V + self.vmin_invalid_penalty_V,
                    physical_value=None, low_nonigniting=None, high_igniting=None,
                    failure_reason=str(ref["validity_reason"]),
                )
            elif not bool(ref["ignition_succeeded"]):
                finalise(
                    i, status="right_censored_no_ignition_at_reference_voltage",
                    search_valid=True,
                    objective_value=self.vmin_upper_bound_V + self.vmin_right_censor_penalty_V,
                    physical_value=None, low_nonigniting=self.vmin_upper_bound_V,
                    high_igniting=None, right_censored=True,
                )
            else:
                active.append(i)

        wave = 0
        while active:
            tasks: list[_HybridTask] = []
            task_to_candidate: dict[int, tuple[int, float, str]] = {}
            next_task_id = 0
            for i in active:
                state = states[i]
                if state["low"] is None:
                    voltage = self.vmin_lower_bound_V
                    role = "lower_bound"
                else:
                    low = float(state["low"])
                    high = float(state["high"])
                    voltage = 0.5 * (low + high)
                    role = "bisection"
                anode, cathode, metadata, output_dir = prepared_items[i]
                cfg = dict(self.cpp_config_values)
                cfg["voltage"] = float(voltage)
                cfg["stop_on_ignition"] = bool(
                    self.vmin_stop_successful_trials_at_ignition
                )
                token = f"{voltage:09.4f}".replace("-", "m").replace(".", "p")
                trial_dir = Path(output_dir) / "vmin_search" / f"V_{token}"
                trial_meta = dict(metadata)
                trial_meta["geometry_id"] = f"{metadata.get('geometry_id')}__V{voltage:.4f}"
                tasks.append(
                    _HybridTask(
                        task_id=next_task_id,
                        anode=anode,
                        cathode=cathode,
                        metadata=trial_meta,
                        output_dir=trial_dir,
                        config_values=cfg,
                    )
                )
                task_to_candidate[next_task_id] = (i, float(voltage), role)
                next_task_id += 1

            wave_rows = self._execute_hybrid_tasks(tasks, stage=f"vmin-wave-{wave}")
            next_active: list[int] = []
            for task_id, (i, voltage, role) in task_to_candidate.items():
                state = states[i]
                trial_row = wave_rows[task_id]
                trial_row["appliedVoltage_V"] = voltage
                trial_row["vminTrialRole"] = role
                summary = self._vmin_trial_summary(trial_row, voltage, role)
                state["trials"].append(summary)
                if not bool(summary["numerically_valid_for_threshold_classification"]):
                    finalise(
                        i, status="invalid_trial_numerics", search_valid=False,
                        objective_value=self.vmin_upper_bound_V + self.vmin_invalid_penalty_V,
                        physical_value=None, low_nonigniting=state.get("low"),
                        high_igniting=state.get("high"),
                        failure_reason=str(summary["validity_reason"]),
                    )
                    continue
                if not self._vmin_monotonicity_ok(state["trials"]):
                    finalise(
                        i, status="invalid_nonmonotonic_sampled_ignition_response",
                        search_valid=False,
                        objective_value=self.vmin_upper_bound_V + self.vmin_invalid_penalty_V,
                        physical_value=None, low_nonigniting=state.get("low"),
                        high_igniting=state.get("high"),
                        failure_reason="sampled_ignition_response_is_nonmonotonic_in_voltage",
                    )
                    continue
                ignited = bool(summary["ignition_succeeded"])
                if role == "lower_bound":
                    if ignited:
                        finalise(
                            i, status="left_censored_ignites_at_lower_search_bound",
                            search_valid=True, objective_value=self.vmin_lower_bound_V,
                            physical_value=self.vmin_lower_bound_V,
                            low_nonigniting=None, high_igniting=self.vmin_lower_bound_V,
                            left_censored=True,
                        )
                        continue
                    state["low"] = voltage
                    state["high"] = self.vmin_upper_bound_V
                else:
                    state["bisection_iterations"] += 1
                    if ignited:
                        state["high"] = voltage
                    else:
                        state["low"] = voltage
                low = float(state["low"])
                high = float(state["high"])
                if high - low <= self.vmin_tolerance_V + 1.0e-12:
                    finalise(
                        i, status="bracketed_converged", search_valid=True,
                        objective_value=high, physical_value=high,
                        low_nonigniting=low, high_igniting=high,
                    )
                elif state["bisection_iterations"] >= self.vmin_max_iterations:
                    finalise(
                        i, status="bracketed_max_iterations_reached",
                        search_valid=True, objective_value=high, physical_value=high,
                        low_nonigniting=low, high_igniting=high,
                        failure_reason=(
                            "target_voltage_tolerance_not_reached_before_maximum_bisection_iterations"
                        ),
                    )
                else:
                    next_active.append(i)
            active = next_active
            wave += 1

        for i, prepared in prepared_items.items():
            row = results[i]
            if row is None:
                continue
            output_dir = Path(prepared[3])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "condensed_metrics.json").write_text(
                json.dumps(_strict_json_value(row), indent=2, allow_nan=False),
                encoding="utf-8",
            )

    def evaluate_batch(
        self,
        items: Sequence[tuple[np.ndarray, np.ndarray, Mapping[str, Any], Path]],
    ) -> list[dict[str, Any]]:
        if not items:
            return []
        results: list[dict[str, Any] | None] = [None] * len(items)
        prepared_items: dict[int, tuple[np.ndarray, np.ndarray, Mapping[str, Any], Path]] = {}
        tasks: list[_HybridTask] = []
        for i, item in enumerate(items):
            try:
                prepared = self._prepare_cpp_item(item)
                prepared_items[i] = prepared
                anode, cathode, metadata, output_dir = prepared
                tasks.append(
                    _HybridTask(
                        task_id=i, anode=anode, cathode=cathode,
                        metadata=metadata, output_dir=Path(output_dir),
                        config_values=dict(self.cpp_config_values),
                    )
                )
            except Exception as exc:
                results[i] = self._cpp_failed_geometry_row(item[2], item[3], exc)
        rows = self._execute_hybrid_tasks(tasks, stage="reference-260V")
        for i, row in rows.items():
            results[i] = row
        self._attach_minimum_ignition_voltage_search_hybrid(prepared_items, results)
        return [row for row in results if row is not None]

    def evaluate(
        self,
        anode_mask: np.ndarray,
        cathode_mask: np.ndarray,
        metadata: Mapping[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        return self.evaluate_batch(
            [(anode_mask, cathode_mask, metadata, output_dir)]
        )[0]
