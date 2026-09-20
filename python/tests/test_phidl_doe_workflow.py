from pathlib import Path
from types import SimpleNamespace
import copy
import json

import numpy as np
import pytest

from ecsp_doe.selection import OBJECTIVES, rank_designs, select_final, select_topologies
from ecsp_doe.storage import digest, read_json, write_json
from ecsp_doe.workflow import DOEOptions, DOEWorkflow


def metric_row(name, values, topology=None):
    return dict(geometry_id=name, topology_id=topology or name, status="success", **dict(zip(OBJECTIVES, values)))


def test_pareto_fronts_and_crowding():
    rows = [metric_row("a", [1, 3, 1, 1]), metric_row("b", [2, 2, 1, 1]),
            metric_row("c", [3, 1, 1, 1]), metric_row("d", [4, 4, 2, 2])]
    ranked = {r["geometry_id"]: r for r in rank_designs(rows)}
    assert [ranked[n]["pareto_rank"] for n in "abcd"] == [1, 1, 1, 2]
    assert np.isinf(ranked["a"]["crowding_distance"])
    assert ranked["b"]["crowding_distance"] == 2
    assert ranked["b"]["normalized_utopia_distance"] < ranked["d"]["normalized_utopia_distance"]
    assert rank_designs(rows) == rank_designs(list(reversed(rows)))


def test_constant_objectives_and_ties_are_deterministic():
    rows = [metric_row(n, [1, 1, 1, 1]) for n in ["d", "b", "a", "c"]]
    ranked = rank_designs(rows)
    assert [r["geometry_id"] for r in ranked] == ["a", "b", "c", "d"]
    assert all(r["crowding_distance"] == 0 and r["normalized_utopia_distance"] == 0 for r in ranked)


def test_unique_topology_best_variant_scan():
    rows = [metric_row("a", [1, 1, 1, 1], "T0"), metric_row("b", [2, 2, 2, 2], "T0"),
            metric_row("c", [3, 3, 3, 3], "T1"), metric_row("d", [4, 4, 4, 4], "T2")]
    assert [r["geometry_id"] for r in select_topologies(rows, 2)] == ["a", "c"]
    with pytest.raises(ValueError):
        select_topologies(rows, 4)


def test_final20_fills_front_then_crowding_and_never_weighted_sum():
    rows = [metric_row(f"g{i:02}", [i, 29-i, 0, 0]) for i in range(30)]
    rows += [metric_row("dominated", [50, 50, 50, 50])]
    selected = select_final(rows, 20)
    assert len(selected) == 20
    assert {"g00", "g29"} <= {r["geometry_id"] for r in selected}
    assert all(r["pareto_rank"] == 1 for r in selected)
    # Different physical units leave Pareto/crowding order unchanged.
    scaled = copy.deepcopy(rows)
    for row in scaled:
        row[OBJECTIVES[0]] *= 100000
    assert [r["geometry_id"] for r in selected] == [r["geometry_id"] for r in select_final(scaled, 20)]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_selection_rejects_nonfinite(value):
    with pytest.raises(ValueError):
        rank_designs([metric_row("a", [value, 1, 1, 1])])


def fake_library(limits, count, seed, physics_grid_size, options, rejection_callback):
    return [{"topology_id": f"T{i:03}", "topology_graph_signature": f"signature-{i}",
             "component_pair": [1+i%2, 1+i%2]} for i in range(count)]


def fake_variants(spec, limits, count, seed, physics_grid_size, options, excluded_parameters, rejection_callback):
    output = []
    for i in range(count):
        params = {"seed": seed, "sample": i}
        assert params not in excluded_parameters
        a = np.zeros((8, 8), bool)
        c = np.zeros((8, 8), bool)
        a[1:3, 1:3] = True
        c[5:7, 5:7] = True
        metadata = {**spec, "parameter_vector": params, "geometry_hash": digest([spec, params]),
                    "generator_seed": seed, "validation": {"valid": True}}
        output.append(SimpleNamespace(anode_mask=a, cathode_mask=c, design_anode_mask=a,
                                      design_cathode_mask=c, metadata=metadata, polygons={}))
    return output


class CountingPhysics:
    def __init__(self):
        self.calls = []
        self.baseline_calls = 0
        self.post_calls = []
        self.fail = set()

    def evaluate(self, a, c, metadata, output):
        geometry_id = metadata["geometry_id"]
        self.calls.append(geometry_id)
        if geometry_id in self.fail:
            raise RuntimeError("mock numerical failure")
        i = int(geometry_id[4:7]) + int(geometry_id[-3:])
        return {**dict(zip(OBJECTIVES, [i+1, .8/(i+1), 100+i, i+2])), "physics_success": True}

    def evaluate_baseline(self, final):
        if not (final / "mock_baseline.json").exists():
            self.baseline_calls += 1
            write_json(final / "mock_baseline.json", {"complete": True})
        return ("baseline", "raster")

    def write_preflame_comparison(self, selected, baseline, final):
        assert baseline[0] == "baseline"

    def refine_selected(self, selected, baseline, final):
        before = (final / "preflame_top20.csv").read_bytes()
        self.post_calls = [r["metadata"]["geometry_id"] for r in selected] + [baseline[0]]
        assert (final / "preflame_top20.csv").read_bytes() == before


def mini_workflow(tmp_path, physics=None, root_name="run"):
    config = {"project": {"seed": 123}, "geometry": {"grid_size": 8}, "evaluator": {"grid_size": 8}}
    return DOEWorkflow(Path(__file__).resolve().parents[2], config, tmp_path / root_name,
        options=DOEOptions(4, 2, 2, 3, 3), physics=physics or CountingPhysics(),
        library_generator=fake_library, variant_generator=fake_variants)


def test_mini_two_stages_resume_no_calls_and_immutable_selection(tmp_path):
    physics = CountingPhysics()
    workflow = mini_workflow(tmp_path, physics)
    result = workflow.run(execute_physics=True, postflame=True)
    assert result["candidate_count"] == 14
    assert len([n for n in physics.calls if n.startswith("S1")]) == 8
    assert len([n for n in physics.calls if n.startswith("S2")]) == 6
    assert physics.baseline_calls == 1
    assert len(physics.post_calls) == 4  # exactly selected3 plus external baseline
    immutable = (workflow.run_root / "final/preflame_top20.csv").read_bytes()
    assert all("baseline" not in n for n in physics.calls)
    physics.calls.clear()
    workflow.run(resume=True, execute_physics=True)
    assert physics.calls == []
    assert physics.baseline_calls == 1
    assert (workflow.run_root / "final/preflame_top20.csv").read_bytes() == immutable
    params = []
    for path in workflow.run_root.glob("stage*/geometries/*.json"):
        params.append(json.dumps(read_json(path)["parameter_vector"], sort_keys=True))
    assert len(params) == len(set(params)) == 14


def test_deterministic_geometry_and_stage2_selection(tmp_path):
    one, two = mini_workflow(tmp_path, root_name="one"), mini_workflow(tmp_path, root_name="two")
    one.run(execute_physics=True)
    two.run(execute_physics=True)
    for artifact in ["stage1/top30_topologies.csv", "stage1/parameter_samples.csv", "stage2/parameter_samples.csv", "final/preflame_top20.csv"]:
        assert (one.run_root / artifact).read_bytes() == (two.run_root / artifact).read_bytes()


def test_geometry_only_and_resume_fingerprint_failure(tmp_path):
    physics = CountingPhysics()
    workflow = mini_workflow(tmp_path, physics)
    assert workflow.run()["phase"] == "stage1_geometry"
    assert physics.calls == [] and physics.baseline_calls == 0
    with pytest.raises(RuntimeError, match="already initialized"):
        workflow.run()
    workflow.config_hash = "changed"
    with pytest.raises(RuntimeError, match="fingerprint"):
        workflow.run(resume=True)


def test_failed_sample_persisted_and_retry_only_failed(tmp_path):
    physics = CountingPhysics()
    physics.fail.add("S1_T001_V000")
    workflow = mini_workflow(tmp_path, physics)
    with pytest.raises(RuntimeError, match="failed physics"):
        workflow.run(execute_physics=True)
    assert len(physics.calls) == 8
    assert not (workflow.run_root / "stage1/top30_topologies.csv").exists()
    checkpoint = workflow.run_root / "stage1/physics/S1_T001_V000/evaluation.json"
    assert read_json(checkpoint)["status"] == "failed"
    physics.calls.clear()
    with pytest.raises(RuntimeError, match="failed physics"):
        workflow.run(resume=True, execute_physics=True)
    assert physics.calls == []
    physics.fail.clear()
    workflow.run(resume=True, execute_physics=True, retry_failed=True)
    assert [n for n in physics.calls if n.startswith("S1")] == ["S1_T001_V000"]
    assert len(physics.calls) == 7


def test_interrupted_sample_resumes_automatically(tmp_path):
    physics = CountingPhysics()
    workflow = mini_workflow(tmp_path, physics)
    workflow.run(execute_physics=True)
    checkpoint = workflow.run_root / "stage1/physics/S1_T001_V000/evaluation.json"
    value = read_json(checkpoint)
    value["status"] = "running"
    write_json(checkpoint, value)
    physics.calls.clear()
    workflow.run(resume=True, execute_physics=True)
    assert physics.calls == ["S1_T001_V000"]


def test_completed_cache_hash_mismatch_fails_closed(tmp_path):
    workflow = mini_workflow(tmp_path)
    workflow.run(execute_physics=True)
    checkpoint = workflow.run_root / "stage1/physics/S1_T001_V000/evaluation.json"
    value = read_json(checkpoint)
    value["geometry_hash"] = "tampered"
    write_json(checkpoint, value)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        workflow.run(resume=True, execute_physics=True)


def test_real_phidl_mini_workflow_with_mock_physics(tmp_path):
    """Exercise actual CAD/LHS/storage/Pareto together, without a physics solve."""
    import yaml
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "config/nsga2_preflame_only_200x3.yaml").read_text())
    config["phidl_doe"] = {"representative_iou_threshold": 0.80,
                           "maximum_attempts_per_topology": 1000}
    physics = CountingPhysics()
    workflow = DOEWorkflow(root, config, tmp_path / "real_cad_mini",
                           options=DOEOptions(4, 2, 2, 2, 3), physics=physics)
    assert workflow.run(execute_physics=True, postflame=True)["candidate_count"] == 12
    assert len(physics.calls) == 12 and physics.baseline_calls == 1
    masks = []
    for path in workflow.run_root.glob("stage*/geometries/*.json"):
        metadata = read_json(path)
        assert metadata["validation"]["passed"]
        assert metadata["validation"]["existing_solver_grid"]["violations"] == []
        masks.append(metadata["physics_mask_hash"])
    assert len(set(masks)) == 12
    physics.calls.clear()
    workflow.run(resume=True, execute_physics=True)
    assert physics.calls == []
