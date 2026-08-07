import pytest
import torch
from neuralop.layers.shape_enforcer import ShapeEnforcer


@pytest.mark.parametrize(
    "ndim", [1, 2, 3]
)  # Number of dimensions we want to enforce a shape on
@pytest.mark.parametrize(
    "start_dim", [0, 1, 2]
)  # Dimension at which we want to start enforcing a shape
@pytest.mark.parametrize("op", ["none", "crop", "pad", "mixed"])  # Operation
def test_shape_enforcer(ndim, start_dim, op):
    # input shape
    shape = [10] * (
        ndim + 2
    )  # 10 in each dimension - represents batch, channel, and ndim spatial dimensions
    x = torch.randn(shape)
    # start index for enforcing shape
    start_idx = start_dim
    end_idx = start_idx + ndim

    # decide output_shape
    if op == "none":
        out = None
        output_shape = None
    else:
        out = list(x.shape)
        if op == "crop":
            out[start_idx:end_idx] = [s // 2 for s in out[start_idx:end_idx]]
        elif op == "pad":
            out[start_idx:end_idx] = [s * 2 for s in out[start_idx:end_idx]]
        else:  # mixed
            out[start_idx:end_idx] = [
                (s // 2 if i % 2 == 0 else s * 2)
                for i, s in enumerate(out[start_idx:end_idx])
            ]
        output_shape = out[start_idx:end_idx]  # Only pass the dimensions to enforce

    # apply ShapeEnforcer
    enforcer = ShapeEnforcer(start_dim)
    y = enforcer(x, output_shape=output_shape)

    # shape checks
    if op == "none":
        assert y.shape == x.shape
    else:
        assert list(y.shape) == out
