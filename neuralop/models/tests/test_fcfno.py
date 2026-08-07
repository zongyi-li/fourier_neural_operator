import torch
import pytest

import torch.nn.functional as F

from neuralop.models.fc_fno import FC_FNO
from neuralop.layers.fourier_continuation import FCGram, FCLegendre


@pytest.fixture(autouse=True)
def fcfno_tests_run_at_fp64():
    """Run FC_FNO tests at float64; restore previous default dtype after each test."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


# Fixed variables
in_channels = 1
out_channels = 1
hidden_channels = 7
n_layers = 2

test_modes = [10, 11]
test_lengths = [1, 2 * torch.pi]
projection_nonlinearities = [F.tanh, F.silu]
fc_objects = [FCGram(5, 50), FCLegendre(5, 40)]


def _run_fcfno(
    modes_num,
    lengths_num,
    n_dim,
    projection_nonlinearity,
    derivs_to_compute,
    FC_object,
    **kwargs,
):
    n_modes = (modes_num,) * n_dim
    domain_lengths = (lengths_num,) * n_dim
    model = FC_FNO(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        projection_nonlinearity=projection_nonlinearity,
        FC_object=FC_object,
        domain_lengths=domain_lengths,
        **kwargs,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    input_resolution = (2 * modes_num,) * n_dim
    x = torch.randn(1, 1, *input_resolution, dtype=torch.float64).to(device)
    output, dx_arr = model(x, derivs_to_compute=derivs_to_compute)
    return output, dx_arr, input_resolution


def _assert_model_outputs(output, dx_arr, expected_spatial_shape, FC_object):
    message = f"FC object {FC_object.__class__.__name__}"
    expected_shape = (1, 1, *expected_spatial_shape)
    
    assert (
        output.shape == expected_shape
    ), f"unexpected output shape {output.shape}, expected {expected_shape} ({message})"
    
    assert output.dtype == torch.float64, f"wrong dtype {output.dtype} ({message})"
    
    for deriv in dx_arr:
        assert (
            deriv.shape == expected_shape
        ), f"unexpected derivative shape {deriv.shape}, expected {expected_shape} ({message})"
        assert (
            deriv.shape == output.shape
        ), f"derivative shape {deriv.shape} does not match output shape {output.shape} ({message})"
        assert (
            deriv.dtype == torch.float64
        ), f"wrong derivative dtype {deriv.dtype} ({message})"


single_deriv_combos = [
    (["dx"], 1),
    (["dyy"], 2),
    (["dzz"], 3),
    (["dy"], 2),
]

multiple_deriv_combos = [
    (["dx", "dxx"], 1),
    (["dxx", "dyy"], 2),
    (["dxx", "dyy", "dzz"], 3),
]


@pytest.mark.parametrize("derivs_to_compute, n_dim", single_deriv_combos)
@pytest.mark.parametrize("modes_num", test_modes)
@pytest.mark.parametrize("lengths_num", test_lengths)
@pytest.mark.parametrize("projection_nonlinearity", projection_nonlinearities)
@pytest.mark.parametrize("FC_object", fc_objects)
@pytest.mark.parametrize("enforce_hermitian_symmetry", [True, False])
def test_fcfno_single_derivative_combinations(
    modes_num,
    lengths_num,
    projection_nonlinearity,
    derivs_to_compute,
    n_dim,
    FC_object,
    enforce_hermitian_symmetry,
):
    output, dx_arr, input_resolution = _run_fcfno(
        modes_num=modes_num,
        lengths_num=lengths_num,
        n_dim=n_dim,
        projection_nonlinearity=projection_nonlinearity,
        derivs_to_compute=derivs_to_compute,
        FC_object=FC_object,
        enforce_hermitian_symmetry=enforce_hermitian_symmetry,
    )
    _assert_model_outputs(output, dx_arr, input_resolution, FC_object)


@pytest.mark.parametrize("derivs_to_compute, n_dim", multiple_deriv_combos)
@pytest.mark.parametrize("modes_num", test_modes)
@pytest.mark.parametrize("lengths_num", test_lengths)
@pytest.mark.parametrize("projection_nonlinearity", projection_nonlinearities)
@pytest.mark.parametrize("FC_object", fc_objects)
def test_fcfno_multiple_derivative_combinations(
    modes_num,
    lengths_num,
    projection_nonlinearity,
    derivs_to_compute,
    n_dim,
    FC_object,
):
    output, dx_arr, input_resolution = _run_fcfno(
        modes_num=modes_num,
        lengths_num=lengths_num,
        n_dim=n_dim,
        projection_nonlinearity=projection_nonlinearity,
        derivs_to_compute=derivs_to_compute,
        FC_object=FC_object,
    )
    _assert_model_outputs(output, dx_arr, input_resolution, FC_object)


all_deriv_combos = single_deriv_combos + multiple_deriv_combos


@pytest.mark.parametrize("derivs_to_compute, n_dim", all_deriv_combos)
@pytest.mark.parametrize("modes_num", test_modes)
@pytest.mark.parametrize("lengths_num", test_lengths)
@pytest.mark.parametrize("projection_nonlinearity", projection_nonlinearities)
@pytest.mark.parametrize("FC_object", fc_objects)
@pytest.mark.parametrize("implementation", ["factorized", "reconstructed"])
def test_fcfno_implementation(
    modes_num,
    lengths_num,
    projection_nonlinearity,
    derivs_to_compute,
    n_dim,
    FC_object,
    implementation,
):
    output, dx_arr, input_resolution = _run_fcfno(
        modes_num=modes_num,
        lengths_num=lengths_num,
        n_dim=n_dim,
        projection_nonlinearity=projection_nonlinearity,
        derivs_to_compute=derivs_to_compute,
        FC_object=FC_object,
        implementation=implementation,
    )
    _assert_model_outputs(output, dx_arr, input_resolution, FC_object)
