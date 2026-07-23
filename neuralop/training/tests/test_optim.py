import torch
from torch.nn import Parameter
from torch.testing import assert_close
import pytest
import math
import neuralop.training.tensorgrad as tensorgrad_module

from ..adamw import AdamW
from ..tensorgrad import (
    TensorGRaD,
    TensorGRaDProjector,
    UnstructuredSparseProjector,
    fno_tensorgrad_param_groups,
)

from ...models import FNO

from tensorly.tucker_tensor import validate_tucker_rank


@pytest.mark.parametrize("adam_optimizer_cls", [AdamW])
def test_correct_complex_adam_momentum(adam_optimizer_cls):
    # param = x * 2j
    x = torch.randn((3, 3), dtype=torch.float64)
    param = Parameter(((0.0 + 1.0j) * x).to(torch.cfloat))

    optimizer = adam_optimizer_cls(params=[param], betas=(0.5, 0.5))

    loss = torch.view_as_real((param * param.conj())).sum()
    # grad x^2 = 2x, grads are all 0 + 2j * x

    loss.backward()
    optimizer.step()

    # momentum value should be elemwise (2jx * -2jx * (1 - 0.5)) = 4x**2 * 0.5 = 2x**2
    # exp_avg_sq should be empty, meaning it is just momentum * (1-beta2)
    momentum = optimizer.state[param]["exp_avg_sq"]
    assert_close(momentum, (2 * x**2).to(torch.cfloat))


@pytest.mark.parametrize("tensorgrad_param_pct", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("n_dim", [2, 3, 4])
def test_tensorgrad_low_rank_projection(tensorgrad_param_pct, n_dim):
    """TensorGRaD low-rank projection should preserve the requested rank size."""

    full_rank_tensor = torch.randn([8] * n_dim)

    tensorgrad_rank = validate_tucker_rank(
        full_rank_tensor.shape,
        tensorgrad_param_pct,
    )
    projector = TensorGRaDProjector(rank=tensorgrad_rank, warm_restart=True)

    # test that the projection is the correct size
    # test 0th iter for full computation
    low_rank_tensor = projector.project(full_rank_tensor, 0)
    assert low_rank_tensor.numel() == math.prod(tensorgrad_rank)


def test_tensorgrad_low_rank_projection_updates_after_gap():
    """TensorGRaD should refresh the low-rank projection at the configured gap."""

    full_rank_tensor = torch.randn(4, 4)
    projector = TensorGRaDProjector(rank=(2, 2), update_proj_gap=2)
    projection_calls = []

    def get_projection_tensor(weights, rank):
        projection_calls.append((weights.shape, rank))
        return [torch.eye(4)[:, :2], torch.eye(4)[:, :2]]

    projector.get_projection_tensor = get_projection_tensor

    projector.project(full_rank_tensor, 0)
    projector.project(full_rank_tensor, 1)
    projector.project(full_rank_tensor, 2)

    assert projection_calls == [
        (full_rank_tensor.shape, (2, 2)),
        (full_rank_tensor.shape, (2, 2)),
    ]


def test_tensorgrad_low_rank_projection_warm_restart_updates():
    """Warm restart should pass TensorLy a valid Tucker initialization."""

    full_rank_tensor = torch.randn(4, 4)
    projector = TensorGRaDProjector(
        rank=(2, 2),
        update_proj_gap=1,
        warm_restart=True,
    )

    first_projection = projector.project(full_rank_tensor, 0)
    second_projection = projector.project(full_rank_tensor, 1)

    assert first_projection.shape == (2, 2)
    assert second_projection.shape == (2, 2)


def test_tensorgrad_activation_checkpoint_projection():
    """Checkpointed TensorLy calls should preserve projection behavior."""

    full_rank_tensor = torch.randn(4, 4)
    projector = TensorGRaDProjector(
        rank=(2, 2),
        activation_checkpoint=True,
    )

    low_rank_tensor = projector.project(full_rank_tensor, 0)

    assert low_rank_tensor.shape == (2, 2)


@pytest.mark.parametrize("activation_checkpoint", [False, True])
def test_tensorgrad_projector_passes_tucker_n_iter_max(
    monkeypatch,
    activation_checkpoint,
):
    """The projector should wire tucker_n_iter_max into TensorLy Tucker calls."""

    n_iter_max_values = []

    def fake_tucker(tensor, *, rank, init, n_iter_max):
        n_iter_max_values.append(n_iter_max)
        factors = [
            torch.eye(dim, rank_dim, dtype=tensor.dtype, device=tensor.device)
            for dim, rank_dim in zip(tensor.shape, rank)
        ]
        return tensor, factors

    monkeypatch.setattr(tensorgrad_module, "tucker", fake_tucker)

    projector = TensorGRaDProjector(
        rank=(2, 2),
        tucker_n_iter_max=3,
        activation_checkpoint=activation_checkpoint,
    )

    projector.get_projection_tensor(torch.randn(4, 4), (2, 2))

    assert n_iter_max_values == [3]


@pytest.mark.parametrize("galore_param_pct", [0.25, 0.5, 1.0])
def test_galore_adamw_rank(galore_param_pct):
    """AdamW's legacy low-rank path should store optimizer state in projected size."""

    x = torch.randn((8, 8, 8, 8), dtype=torch.float64)
    param = Parameter(((0.0 + 1.0j) * x).to(torch.cfloat))
    galore_param = Parameter(((1.0 + 0j) * x).to(torch.cfloat))

    # pick mode-wise rank so that low_rank_grad.numel() / full_rank_grad.numel() == galore_param_pct
    galore_rank = validate_tucker_rank(galore_param.shape, galore_param_pct)

    optimizer = AdamW(
        params=[param],
        galore_params=[galore_param],
        galore_rank=galore_rank,
        betas=(0.5, 0.5),
    )

    loss = torch.view_as_real((param * galore_param.conj())).sum()

    loss.backward()
    optimizer.step()

    momentum = optimizer.state[galore_param]["exp_avg_sq"]
    # make sure low-rank params are being stored
    assert momentum.numel() == math.prod(galore_rank)


def test_unstructured_sparse_projector_topk():
    """Sparse top-k projection should keep only the requested largest entries."""

    grad = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    projector = UnstructuredSparseProjector(
        sparse_ratio=0.25,
        sparse_type="topk",
        update_proj_gap=1,
    )

    sparse_grad = projector.project(grad, step=0)
    projected_back = projector.project_back(sparse_grad)

    assert sparse_grad.numel() == 4
    assert torch.count_nonzero(projected_back) == 4
    assert_close(projected_back.max(), grad.max())


def test_tensorgrad_low_rank_stores_compressed_state():
    """Low-rank TensorGRaD should keep Adam moments in compressed coordinates."""

    param = Parameter(torch.randn(4, 4, 4, dtype=torch.float32))
    optimizer = TensorGRaD(
        [
            {
                "params": [param],
                "tensorgrad": True,
                "rank": 0.5,
                "update_proj_gap": 1,
            }
        ],
        betas=(0.5, 0.5),
    )

    loss = (param**2).sum()
    loss.backward()
    optimizer.step()

    state = optimizer.state[param]
    assert state["step"] == 1
    assert "low_rank_exp_avg_sq" in state
    assert state["low_rank_exp_avg_sq"].numel() < param.numel()


def test_tensorgrad_low_rank_sparse_stores_two_compressed_states():
    """Low-rank+sparse TensorGRaD should keep separate compressed moment states."""

    param = Parameter(torch.randn(4, 4, 4, dtype=torch.float32))
    optimizer = TensorGRaD(
        [
            {
                "params": [param],
                "tensorgrad": True,
                "rank": 0.5,
                "sparse_ratio": 0.25,
                "sparse_type": "topk",
                "update_proj_gap": 1,
            }
        ],
        betas=(0.5, 0.5),
    )

    loss = (param**2).sum()
    loss.backward()
    optimizer.step()

    state = optimizer.state[param]
    assert state["step"] == 1
    assert "low_rank_exp_avg_sq" in state
    assert "sparse_exp_avg_sq" in state
    assert state["sparse_exp_avg_sq"].numel() == int(0.25 * param.numel())


def test_fno_tensorgrad_param_groups_select_spectral_params():
    """FNO helper should split spectral-convolution params from regular params."""

    model = FNO(
        n_modes=(4, 4),
        hidden_channels=8,
        in_channels=1,
        out_channels=1,
        projection_channel_ratio=2,
    )

    groups = fno_tensorgrad_param_groups(
        model,
        rank=0.25,
        sparse_ratio=0.05,
        min_params=0,
    )

    regular_params = groups[0]["params"]
    tensorgrad_params = groups[1]["params"]
    regular_ids = {id(param) for param in regular_params}
    tensorgrad_ids = {id(param) for param in tensorgrad_params}

    assert groups[1]["tensorgrad"]
    assert groups[1]["sparse_ratio"] == 0.05
    assert tensorgrad_params
    assert regular_ids.isdisjoint(tensorgrad_ids)
