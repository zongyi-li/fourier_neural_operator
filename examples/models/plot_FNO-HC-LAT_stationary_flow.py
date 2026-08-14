"""
Training a FNO using the hyperbolic cross index set and rank-1 lattices
=======================================================================

We train a FNO using the hyperbolic cross index set and rank-1 lattices on a stationary flow dataset using a series expansion for the random field.
The output is compared to the output of a FNO using a hyperrectangle index set and a grid discretization.

This tutorial demonstrates both how the hyperbolic cross index set and rank-1 lattices can be used for the FNO.

There are two versions of the experiment:
 - A small model that can be trained on a CPU in a few minutes.
 - A practical model that can be trained on a GPU in under an hour, allowing the comparison of the different index set and discretization to that of a grid FNO.

References
----------
.. [1] Dilen, J., Keller, A., Kuo, F. Y., Nuyens, D. "Fourier Neural Operators
    with Rank-1 Lattice Points and Hyperbolic Cross" (2026).
    https://arxiv.org/abs/2606.08871. 
"""

from neuralop.data.datasets.tensor_dataset import TensorDataset
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.losses.data_losses import LpLoss
from neuralop.models.fno import FNO
from neuralop.layers.index_sets import HyperRectangleIndexSet, HyperbolicCrossIndexSet
from neuralop.layers.spectral_transforms import RegularGridFFT, Rank1LatticeFFT
from neuralop.layers.embeddings import GridEmbedding2D, LatticeEmbedding
from neuralop.training.trainer import Trainer

from neuralop.utils import count_model_params
import torch
import os
import gc
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


"""
Stationary flow dataset
=======================

Functions that can be used to efficiently calculate the stationary flow dataset by solving a system of the Fourier coefficients.
"""

def generate_stationary_flow_dataset(
    n_samples: int,
    batchsize = 64,
    seed: int = 0,
):
    """
    Actually calculation of the samples and generation of the specified dataset.

    Parameters:
    -----------
    n_samples : int
        number of samples
    batchsize : int
        number of samples calculated at the same time
    seed : int
        random seed
    """

    # Dataset parameters
    res = 32
    n = 1024
    z = torch.tensor([1, 721])
    M = 9
    alpha = 1

    # Calculation parameters
    M_calc = 15
    alpha_calc = 0
    data_type = torch.complex64
    grid_data_type = torch.float32

    # Set seed
    np.random.seed(seed)

    # Generate meshgrid
    X1_reg, X2_reg = torch.meshgrid(
        torch.as_tensor(np.linspace(0, 1, res, endpoint=False), dtype=grid_data_type), torch.as_tensor(np.linspace(0, 1, res, endpoint=False), dtype=grid_data_type), indexing="xy"
    )

    # Allocate datasets
    data = dict()
    x = torch.zeros([n_samples, res, res], dtype=data_type)
    y = torch.zeros([n_samples, res, res], dtype=data_type)

    # Define f
    f_eval_reg = f_checker_board(X1_reg, X2_reg, True)
    f_coeff_reg = torch.fft.fft2(f_eval_reg, norm="forward")

    # Generate lattice point set
    X1_lattice = ((torch.arange(0, n, dtype=grid_data_type) * z[0]) % n) / n
    X2_lattice = ((torch.arange(0, n, dtype=grid_data_type) * z[1]) % n) / n

    # Allocate datasets
    lattice_data = dict()
    x_lattice = torch.zeros([n_samples, n], dtype=data_type)
    y_lattice = torch.zeros([n_samples, n], dtype=data_type)

    # Define f
    f_eval_lattice = f_checker_board(X1_lattice, X2_lattice, True)
    f_coeff_lattice = torch.fft.fft(f_eval_lattice, norm="forward")

    # Index set
    h_r_pairs = createIndexSet(M=M, alpha=alpha, complex=True)
    s = 2 * len(h_r_pairs)

    # Random coefficients using embedded lattice
    random_coefficients = torch.randn((n_samples, s)) - 0.5

    # Calculation modes
    i = torch.arange(-M_calc, M_calc + 1, dtype=torch.int32)
    grids = torch.meshgrid([i] * 2, indexing="ij")
    modes = torch.stack(grids, dim=-1).reshape(-1, 2)

    # Remove [0, 0]
    indices = list(range(0, (2 * M_calc + 1)**2))
    indices.pop(M_calc * (2 * M_calc+1) + M_calc)
    modes = modes[indices]

    h_modes = torch.as_tensor(list(map(lambda x: x[0], h_r_pairs)))
    h_modes_lattice = torch.inner(h_modes, z.flip(dims=[0])) % n

    r_values = torch.as_tensor(list(map(lambda x: x[1], h_r_pairs)))

    # Actual data generation
    iteration = 0
    while n_samples > 0:
        if n_samples < batchsize:
            batchsize = n_samples
        n_samples -= batchsize

        # Define a
        a_coeff_grid = torch.zeros(batchsize, res, res, dtype=data_type)
        a_coeff_grid[:, 0, 0] = 1


        a_coeff_lattice = torch.zeros(batchsize, n, dtype=data_type)
        a_coeff_lattice[:, 0] = 1

        coeff_vector_real= (random_coefficients[iteration * batchsize:(iteration + 1) * batchsize, ::2])
        coeff_vector_complex = (random_coefficients[iteration * batchsize:(iteration + 1) * batchsize, 1::2])
        coeff_vector = coeff_vector_real + 1j * coeff_vector_complex

        scaled_coeff_vector = 0.1 * coeff_vector / r_values

        # Grid calculation
        a_coeff_grid[:, h_modes[:, 1], h_modes[:, 0]] += scaled_coeff_vector
        x[(batchsize * iteration):(batchsize * (iteration + 1))] = torch.fft.ifftn(a_coeff_grid, dim=[1, 2], norm="forward")

        A_reg, f_reg = createMatrixVectorSystem_grid(a_coeff_grid, f_coeff_reg, modes, M_calc, alpha_calc)
        u_coeff_reg = torch.linalg.solve(A_reg, f_reg)

        del A_reg, f_reg
        gc.collect()

        y[(batchsize * iteration):(batchsize * (iteration + 1))] = reconstructU_grid(u_coeff_reg, modes, res)

        # Lattice calculation
        a_coeff_lattice[:, h_modes_lattice] += scaled_coeff_vector
        x_lattice[(batchsize * iteration):(batchsize * (iteration + 1))] = torch.fft.ifftn(a_coeff_lattice, dim=[-1], norm="forward")

        A_lattice, f_lattice = createMatrixVectorSystem_lattice(a_coeff_lattice, f_coeff_lattice, modes, n, z, M_calc, alpha_calc)
        u_coeff_lattice = torch.linalg.solve(A_lattice, f_lattice)

        del A_lattice, f_lattice
        gc.collect()

        y_lattice[(batchsize * iteration):(batchsize * (iteration + 1))] = reconstructU_lattice(u_coeff_lattice, modes, n, z)

        iteration += 1

    data["x"] = x
    data["y"] = y

    lattice_data["x"] = x_lattice
    lattice_data["y"] = y_lattice

    return data, lattice_data

def sigmoid(x, k=1, s=1, complex=False):
    """
    Complex sigmoid function: \sigma(x) = \sigma(x.real) + \sigma(x.imag)
    """
    if complex:
        return s * (1 / (1 + torch.exp(-k * x.real)) -0.5) + 1j * s * (1 / (1 + torch.exp(-k * x.imag)) -0.5)
    return s / (1 + torch.exp(-k * x)) -0.5

def f_checker_board(x, y, k=3, omega=2, s=500, complex=False):
    """
    Checker board forcing term
    """
    if complex:
        result = torch.zeros(x.shape, dtype=torch.complex64)
    else:
        result = torch.zeros(x.shape, dtype=torch.float32)
    result -= 1 * (sigmoid(torch.sin(omega * torch.pi * (x)), k=k) + sigmoid(torch.sin(omega * torch.pi * (y)), k=k))
    if complex:
        result += 1j * (sigmoid(torch.sin(omega * torch.pi * (x)), k=k) + sigmoid(torch.sin(omega * torch.pi * (y)), k=k))
    return result * s

def createIndexSet(M: int, alpha: float, complex: bool = False):
    """
    Creates the index set for a given index bound M and a smoothness parameter alpha.

    Parameters:
    -----------
    M : int
            M value of index set
    alpha : float
            alpha value of index set
    complex : bool
            indicates if all or only half coefficients have to be kept,
            due to the complex conjugate nature of Fourier coefficients of real functions
    """
    h_r_pairs = []
    h_i_max = int(np.floor(M ** (1 / (2 * alpha))))
    for i in range(-h_i_max, h_i_max + 1):
        start = 0
        if complex:
            start = -h_i_max
        for j in range(start, h_i_max + 1):
            if complex or (not (i < 0 and j == 0)):
                r_i = (max(abs(i), 1) * max(abs(j), 1)) ** (2 * alpha)
                if r_i <= M:
                    h_i = [i, j]
                    h_r_pairs.append((h_i, r_i))
    h_r_pairs.sort(key=lambda pair: pair[1])
    return h_r_pairs

def createMatrixVectorSystem_grid(a_coeff, f_coeff, modes, M, alpha=0):
    """
    Creates the matrix 'A' and vector 'f' needed to solve the PDE using the Fourier coefficients
    in a batched manner, assuming the coefficients are given in a regular grid format.

    Parameters:
    -----------
    -a_coeff : torch.Tensor
        Fourier coefficients of a
    -f_coeff : torch.Tensor
        Fourier coefficients of f
    -modes : torch.Tensor
        corresponding modes
    -M : int
        the M of the index set
    -alpha : float
        the alpha of the hyperbolic cross (alpha=0 gives hyperrectangle)
    """
    batchsize, *dims = a_coeff.shape
    n_dims = len(dims)
    assert(len(modes.shape) == 2)
    assert(n_dims == modes.shape[1])
    assert(n_dims == len(f_coeff.shape))

    if modes.dtype != torch.int:
        modes = modes.to(dtype=torch.int)

    device = a_coeff.device
    if modes.device != device:
        modes = modes.to(device=device)
    if f_coeff.device != device:
        f_coeff = f_coeff.to(device=device)

    n_modes = len(modes)

    L = torch.repeat_interleave(modes.unsqueeze(0), n_modes, axis=0)
    H = torch.repeat_interleave(modes.unsqueeze(0), n_modes, axis=1).reshape(n_modes, n_modes, n_dims)

    # Generate A
    a_indices = (L - H)
    if alpha != 0:
        mask = abs(torch.prod(torch.max(abs(a_indices), torch.ones_like(a_indices))[:], dim=-1))**(2 * alpha) <= M
    else:
        mask = (torch.prod(a_indices[:] >= -M, dim=-1)) * (torch.prod(a_indices[:] <= M, dim=-1))
    a_indices_masked = torch.repeat_interleave(mask.unsqueeze(-1), n_dims, dim=-1) * a_indices

    A_shape = [n_modes] * n_dims
    A_shape.insert(0, batchsize)
    dtype = a_coeff.dtype
    if (dtype == torch.complex64 or dtype == torch.float32):
        A = torch.zeros(A_shape, device=device).to(torch.complex64)
    elif (dtype == torch.complex128 or dtype == torch.float64):
        A = torch.zeros(A_shape, device=device).to(torch.complex128)
    else:
        raise Exception("Wrong datatype for 'a_coeff'.")

    A += torch.repeat_interleave((torch.sum(L * H, axis=-1)).unsqueeze(0), batchsize, axis=0)
    A *= a_coeff[(slice(None), *(a_indices_masked[:].unbind(dim=-1)))]
    A -= a_coeff[(slice(None), *torch.Tensor([0] * n_dims).to(dtype=torch.int))].unsqueeze(-1).unsqueeze(-1) * torch.repeat_interleave(((~mask) * torch.sum(L * H, axis=-1)).unsqueeze(0), batchsize, axis=0)
    A *= 4 * torch.pi**2

    # Generate f
    f = f_coeff[L[0, :, 0], L[0, :, 1]]

    if batchsize == 1:
        f = f.unsqueeze(0)
    else:
        f = torch.repeat_interleave(f.unsqueeze(0), batchsize, axis=0)

    return A, f

def createMatrixVectorSystem_lattice(a_coeff, f_coeff, modes, n, z, M, alpha=0):
    """
    Creates the matrix 'A' and vector 'f' needed to solve the PDE using the Fourier coefficients
    in a batched manner, assuming the coefficients gained through applying the 1-dimensional FFT

    Parameters:
    -----------
    -a_coeff : torch.Tensor
        Fourier coefficients of a
    -f_coeff : torch.Tensor
        Fourier coefficients of f
    -modes : torch.Tensor
        corresponding modes
    -n : int
        number of lattice points
    -z : torch.Tensor
        generating vector
    -M : int
        the M of the index set
    -alpha : float
        the alpha of the hyperbolic cross (alpha=0 gives hyperrectangle)
    """
    assert(len(f_coeff.shape) == 1)
    assert(len(a_coeff.shape) == 2)
    assert(modes.shape[-1] == len(z))

    batchsize = a_coeff.shape[0]

    if modes.dtype != torch.int:
        modes = modes.to(dtype=torch.int)

    device = a_coeff.device
    if modes.device != device:
        modes = modes.to(device=device)
    if f_coeff.device != device:
        f_coeff = f_coeff.to(device=device)

    n_modes, n_dim = modes.shape

    L = torch.repeat_interleave(modes.unsqueeze(0), n_modes, axis=0)
    H = torch.repeat_interleave(modes.unsqueeze(0), n_modes, axis=1).reshape(n_modes, n_modes, 2)

    a_indices = (L - H)

    if alpha != 0:
        mask = abs(torch.prod(torch.max(abs(a_indices), torch.ones_like(a_indices))[:], dim=-1))**(2 * alpha) <= M
    else:
        mask = (torch.prod(a_indices[:] >= -M, dim=-1)) * (torch.prod(a_indices[:] <= M, dim=-1))
    a_indices_masked = torch.repeat_interleave(mask.unsqueeze(-1), n_dim, dim=-1) * a_indices
    a_indices_masked = (torch.linalg.matmul(a_indices_masked.to(dtype=torch.int32), z.to(dtype=torch.int32)) % n).to(dtype=torch.int)

    if (a_coeff.dtype == torch.complex64) or (a_coeff.dtype == torch.float32):
        A = torch.zeros((batchsize, n_modes, n_modes)).to(torch.complex64)
    elif (a_coeff.dtype == torch.complex128) or (a_coeff.dtype == torch.float64):
        A = torch.zeros((batchsize, n_modes, n_modes)).to(torch.complex128)

    A += torch.repeat_interleave((torch.sum(L * H, axis=-1)).unsqueeze(0), batchsize, axis=0)
    A *= a_coeff[:, a_indices_masked[:, :]]
    A -= a_coeff[:, 0].unsqueeze(-1).unsqueeze(-1) * torch.repeat_interleave(((~mask) * torch.sum(L * H, axis=-1)).unsqueeze(0), batchsize, axis=0)
    A *= 4 * torch.pi**2

    f = f_coeff[(torch.linalg.matmul(L[0, :], z.to(dtype=torch.int32))%n).to(dtype=torch.int)]
    return A, f

def reconstructU_grid(u_coeff: torch.Tensor, modes: torch.Tensor, res: int):
    """
    Reconstructs 'u' on a regular grid with given resolution
    using the provided Fourier coefficients and there position.

    Parameters:
    -----------
    -u_coeff : torch.Tensor
        Fourier coefficients of u
    -modes : torch.Tensor
        corresponding modes
    -res : int
        resolution of the grid
    """

    assert(len(modes.shape) == 2)
    assert((len(u_coeff.shape) == 1) or (len(u_coeff.shape) == 2))
    assert(u_coeff.shape[-1] == len(modes))

    # Set-up
    if modes.dtype != torch.int:
        modes = modes.to(dtype=int)
    device = u_coeff.device
    if modes.device != device:
        modes = modes.to(device=device)
    n_modes, n_dim = modes.shape
    fft_dim = tuple(torch.arange(-n_dim, 0, dtype=torch.int).numpy())

    # Decide batchsize
    if len(u_coeff.shape) == 1:
        u_coeff = u_coeff.unsqueeze(0)
        batchsize = 1
    else:
        batchsize = u_coeff.shape[0]

    dtype = u_coeff.dtype
    if (dtype == torch.complex64 or dtype == torch.float32):
        u_fft = torch.zeros((batchsize, res, res), device=device).to(dtype=torch.complex64)
    elif (dtype == torch.complex128 or dtype == torch.float64):
        u_fft = torch.zeros((batchsize, res, res), device=device).to(dtype=torch.complex128)
    else:
        raise Exception("Wrong datatype for 'u_coeff'.")

    # Asign coefficients
    u_fft[(slice(None), *modes.unbind(dim=-1))] = u_coeff

    return torch.fft.ifftn(u_fft, dim=fft_dim, norm="forward")

def reconstructU_lattice(u_coeff, modes, n, z):
    """
    Reconstructs 'u' on the given rank-1 lattice
    using the provided Fourier coefficients and there position.

    Parameters:
    -----------
    -u_coeff : torch.Tensor
        Fourier coefficients of u
    -modes : torch.Tensor
        corresponding modes
    -n : int
        number of lattice points
    -z : torch.Tensor
        generating vector
    """
    assert((len(u_coeff.shape) == 1) or (len(u_coeff.shape) == 2))
    assert(u_coeff.shape[-1] == len(modes))

    # Set-up
    if modes.dtype != torch.int:
        modes = modes.to(dtype=torch.int)
    device = u_coeff.device
    if modes.device != device:
        modes = modes.to(device=device)

    # Decide batchsize
    if len(u_coeff.shape) == 1:
        u_coef = u_coeff.unsqueeze(0)
        batchsize = 1
    else:
        batchsize = u_coeff.shape[0]

    dtype = u_coeff.dtype
    if (dtype == torch.complex64 or dtype == torch.float32):
        u_fft = torch.zeros((batchsize, n), device=device).to(dtype=torch.complex64)
    elif (dtype == torch.complex128 or dtype == torch.float64):
        u_fft = torch.zeros((batchsize, n), device=device).to(dtype=torch.complex128)
    else:
        raise Exception("Wrong datatype for 'u_coeff'.")

    lattice_coefficients = (torch.linalg.matmul(modes, z.to(dtype=torch.int)) % n).to(dtype=torch.int64)
    u_out_fft = torch.scatter(
        u_fft, 1, lattice_coefficients.repeat(batchsize, 1), u_coeff
    )

    u = torch.fft.ifftn(u_out_fft, dim=[-1], norm="forward")
    return u

"""
Stationary flow models
=======================

Functions to run the example
"""

def run_experiment(epochs, n_training_samples, n_test_samples, n_generalization_samples):
    res = 32
    n = 1024
    z = torch.tensor([1, 721])

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    path = "stationary_flow_example"
    if not os.path.exists(path):
        os.mkdir(path)


    # Generate data
    print("Started generating training dataset")
    grid_train_loader, lattice_train_loader = generate_datasets(n_samples=n_training_samples, seed=0)
    print("Finished generating training dataset")

    print("Started generating test dataset")
    grid_test_loader, lattice_test_loader = generate_datasets(n_samples=n_test_samples, seed=1)
    print("Finished generating test dataset")

    print("Started generating generalization dataset")
    grid_generalization_loader, lattice_generalization_loader = generate_datasets(n_samples=n_generalization_samples, seed=2)
    print("Finished generating generalization dataset")

    # Create models
    n_modes = 9
    hyper_rectangle_index_set = HyperRectangleIndexSet(radius=9.5, n_dim=2)
    grid_transform = RegularGridFFT(order=2, complex_data=True)
    grid_model = FNO(n_modes=[n_modes * 2 + 1] * 2, in_channels=1, out_channels=1, hidden_channels=16, n_layers=2, lifting_channel_ratio=0, projection_channel_ratio=0, positional_embedding=GridEmbedding2D(in_channels=1), complex_data=True, use_channel_mlp=False, index_set=hyper_rectangle_index_set, spectral_transform=grid_transform).to(device)
    grid_model.eval()
    print(f"Grid model parameter count: {count_model_params(grid_model)}")

    hyperbolic_cross_index_set = HyperbolicCrossIndexSet(radius=9.0, n_dim=2, beta=1)
    lattice_transform = Rank1LatticeFFT(n, z, complex_data=True)
    lattice_model = FNO(n_modes=[n_modes * 2 + 1], in_channels=1, out_channels=1, hidden_channels=16, n_layers=2, lifting_channel_ratio=0, projection_channel_ratio=0, positional_embedding=LatticeEmbedding(in_channels=1, z=z), complex_data=True, use_channel_mlp=False, index_set=hyperbolic_cross_index_set, spectral_transform=lattice_transform).to(device)
    lattice_model.eval()
    print(f"Lattice model parameter count: {count_model_params(lattice_model)}")


    # Train grid model
    optimizer = torch.optim.Adam(grid_model.parameters(), lr=0.01, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.8, patience=50)
    loss = ComplexLpLoss(d=2)

    trainer = Trainer(
        model=grid_model,
        n_epochs=epochs,
        device=device,
        data_processor=DefaultDataProcessor().to(device),
        eval_interval=epochs,
        verbose=True,
    )

    print("Started training grid model")
    trainer.train(
        train_loader=grid_train_loader,
        test_loaders={res: grid_test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=path,
        training_loss=loss,
        save_every=epochs,
    )
    print("Finished training grid model")

    # Train grid model
    optimizer = torch.optim.Adam(grid_model.parameters(), lr=0.01, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.8, patience=50)
    loss = ComplexLpLoss(d=2)

    trainer = Trainer(
        model=lattice_model,
        n_epochs=epochs,
        device=device,
        data_processor=DefaultDataProcessor().to(device),
        eval_interval=epochs,
        verbose=True,
    )
    optimizer = torch.optim.Adam(lattice_model.parameters(), lr=0.01, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.8, patience=50)


    print("Started training lattice model")
    trainer.train(
        train_loader=lattice_train_loader,
        test_loaders={n: lattice_test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=path,
        training_loss=loss,
        save_every=epochs,
    )
    print("Finished training lattice model")

    # Calculate generalization error
    with torch.no_grad():
        error = 0
        denominator = 0
        for sample in grid_generalization_loader:
            x_sample = sample["x"].to(device)
            y_sample = sample["y"].to(device)
            y_pred = grid_model(x_sample)
            sample_error = torch.sqrt(torch.sum(torch.abs(y_pred - y_sample), dim=(-3, -2, -1)))
            error += sample_error
            denominator += torch.sqrt(torch.sum(torch.abs(y_sample), dim=(-3, -2, -1)))
        relative_error = torch.sum(error) / torch.sum(denominator)
        error = torch.sum(error) / len(grid_generalization_loader)
        print(f"Error FNO : {error}")
        print(f"Relative error FNO : {relative_error}")

        error = 0
        denominator = 0
        for sample in lattice_generalization_loader:
            x_sample = sample["x"].to(device)
            y_sample = sample["y"].to(device)
            y_pred = lattice_model(x_sample)
            sample_error = torch.sqrt(torch.sum(torch.abs(y_pred - y_sample), dim=(-2, -1)))
            error += sample_error
            denominator += torch.sqrt(torch.sum(torch.abs(y_sample), dim=(-2, -1)))
        relative_error = torch.sum(error) / torch.sum(denominator)
        error = torch.sum(error) / len(grid_generalization_loader)
        print(f"Error FNO-HC-LAT : {error}")
        print(f"Relative error FNO-HC-LAT : {relative_error}")

    # Plot result
    with torch.no_grad():
        x_sample = grid_train_loader.dataset[0]["x"].unsqueeze(0).to(device)
        grid_y_sample = grid_train_loader.dataset[0]["y"].squeeze(0).squeeze(0).cpu()
        grid_y_pred = grid_model(x_sample).squeeze(0).squeeze(0).cpu()
        del x_sample

        x_sample = lattice_train_loader.dataset[0]["x"].unsqueeze(0).to(device)
        lattice_y_pred = lattice_model(x_sample).squeeze(0).squeeze(0).cpu()
        del x_sample

        fig, axs = plt.subplots(3, 2, subplot_kw={"projection": "3d"})
        X, Y = np.meshgrid(np.linspace(0, 1, res, endpoint=False), np.linspace(0, 1, res, endpoint=False), indexing="ij")
        X_lattice = (np.arange(0, n) % n) / n
        Y_lattice = ((np.arange(0, n) * z[1].cpu().numpy()) % n) / n
        axs[0, 0].plot_surface(X, Y, grid_y_sample.real, cmap=cm.viridis)
        axs[0, 1].plot_surface(X, Y, grid_y_sample.imag, cmap=cm.viridis)
        axs[1, 0].plot_surface(X, Y, grid_y_pred.real, cmap=cm.viridis)
        axs[1, 1].plot_surface(X, Y, grid_y_pred.imag, cmap=cm.viridis)
        axs[2, 0].plot_trisurf(X_lattice, Y_lattice, lattice_y_pred.real, cmap=cm.viridis)
        axs[2, 1].plot_trisurf(X_lattice, Y_lattice, lattice_y_pred.imag, cmap=cm.viridis)
        plt.show()

def generate_datasets(n_samples, batch_size=16, seed=0):
    """
    Calculate the grid and lattice datasets and return the respective DataLoaders.
    """
    grid_data, lattice_data = generate_stationary_flow_dataset(n_samples=n_samples, seed=seed)
    x_data = grid_data["x"][0:n_samples].unsqueeze(1).type(torch.complex64).clone()
    y_data = grid_data["y"][0:n_samples].unsqueeze(1).type(torch.complex64).clone()

    grid_dataset = TensorDataset(x_data, y_data)
    grid_loader = torch.utils.data.DataLoader(
        grid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    del x_data
    del y_data

    x_data = lattice_data["x"][0:n_samples].unsqueeze(1).type(torch.complex64).clone()
    y_data = lattice_data["y"][0:n_samples].unsqueeze(1).type(torch.complex64).clone()

    lattice_dataset = TensorDataset(x_data, y_data)
    lattice_loader = torch.utils.data.DataLoader(
        lattice_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    return grid_loader, lattice_loader

class ComplexLpLoss(LpLoss):
    """
    Wrapper for the LpLoss implemented in the 'neuralop' package to allow the use of complex losses.
    """
    def __init__(self, d=1, p=1, measure=1.0, reduction="sum", eps=1e-8, take_root=False):
        self.take_root=take_root
        super().__init__(d, p, measure, reduction, eps)

    def __call__(self, y_pred, y, **kwargs):
        real_error = super().rel(y_pred.real, y.real, take_root=self.take_root)
        imag_err = super().rel(y_pred.imag, y.imag, take_root=self.take_root)
        return real_error + imag_err

def main(full_experiment: bool = False):
    if full_experiment:
        n_training_samples = 1024
        n_test_samples = 256
        n_generalization_samples = 4096
        epochs = 2000
    else:
        n_training_samples = 16
        n_test_samples = 4
        n_generalization_samples = 16
        epochs = 1000
    run_experiment(epochs=epochs, n_training_samples=n_training_samples, n_test_samples=n_test_samples, n_generalization_samples=n_generalization_samples)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        full_experiment = sys.argv[1] == "True"
    else:
        full_experiment = False
    main(full_experiment)

