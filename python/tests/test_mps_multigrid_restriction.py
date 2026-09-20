from __future__ import annotations

import torch

from ecsp_v6.physics import potential


def _full_weight_reference(field: torch.Tensor) -> torch.Tensor:
    kernel = field.new_tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
    ) / 16.0
    return torch.nn.functional.conv2d(
        field[:, None], kernel[None, None], stride=2, padding=1
    )[:, 0]


def test_full_weighting_odd_grid_sizes_match_expected_hierarchy():
    # Exercise the exact hierarchy used by the 193x193 M2 profile without
    # requiring Apple hardware in CI.
    for n in (193, 97, 49, 25, 13, 7, 3):
        x = torch.arange(n * n, dtype=torch.float32).reshape(1, n, n)
        y = _full_weight_reference(x)
        target = (n + 1) // 2
        assert y.shape == (1, target, target)
        assert torch.isfinite(y).all()


def test_historical_cpu_area_restriction_is_unchanged():
    x = torch.randn(2, 193, 193, dtype=torch.float64)
    y = potential._mg_restrict(x, 97)
    expected = torch.nn.functional.interpolate(
        x[:, None], size=(97, 97), mode="area"
    )[:, 0]
    assert torch.equal(y, expected)
