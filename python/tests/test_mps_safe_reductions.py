from __future__ import annotations

import torch

from ecsp_v6.physics.electrochem import _batch_scatter_max, _batch_scatter_sum
from ecsp_v6.physics.numerics import batch_quantile_masked


def test_fixed_shape_batch_reductions_match_reference() -> None:
    batch_index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64)
    integers = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
    values = integers.to(torch.float32) * 0.5

    assert torch.equal(
        _batch_scatter_sum(integers, batch_index, 4),
        torch.tensor([3, 3, 15, 0], dtype=torch.int64),
    )
    assert torch.equal(
        _batch_scatter_sum(values, batch_index, 4),
        torch.tensor([1.5, 1.5, 7.5, 0.0]),
    )
    assert torch.equal(
        _batch_scatter_max(values, batch_index, 4),
        torch.tensor([1.0, 1.5, 3.0, 0.0]),
    )


def test_fixed_shape_masked_quantile_matches_linear_reference() -> None:
    field = torch.tensor(
        [
            [[1.0, 7.0, 3.0], [5.0, 9.0, 11.0]],
            [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]],
        ]
    )
    mask = torch.tensor(
        [
            [[True, False, True], [True, False, True]],
            [[False, True, False], [True, True, False]],
        ]
    )
    result = batch_quantile_masked(field, mask, 0.5)
    assert torch.allclose(result, torch.tensor([4.0, 8.0]), atol=1.0e-7, rtol=0.0)


def test_fixed_shape_quantile_accepts_single_2d_field() -> None:
    field = torch.tensor([[1.0, 8.0], [3.0, 5.0]])
    mask = torch.tensor([[True, False], [True, True]])
    result = batch_quantile_masked(field, mask, 0.5)
    assert result.shape == (1,)
    assert torch.allclose(result, torch.tensor([3.0]), atol=1.0e-7, rtol=0.0)
