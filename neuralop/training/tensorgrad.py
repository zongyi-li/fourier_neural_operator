import math
from typing import Callable, Iterable, Optional

import torch
from tensorly import tenalg
from tensorly.decomposition import tucker
from torch import nn
from torch.optim import Optimizer
from torch.utils.checkpoint import checkpoint
from tensorly.tucker_tensor import validate_tucker_rank


class TensorGRaDProjector:
    """Low-rank gradient projector used by TensorGRaD.

    The original tensor is projected into a low-rank subspace using low-rank
    mode-wise factors obtained by Tucker decomposition. The parameters are
    optimized in this space to save memory and are then projected back into the
    full-rank space for use in a model.

    Parameters
    ----------
    rank : float, int or tuple[int, ...]
        Goal rank of the transformed gradient tensor. A float is interpreted as
        a target fraction of parameters to preserve.
    update_proj_gap : int, optional
        Number of optimizer steps between projection-tensor updates.
    scale : float, optional
        Scalar applied after projecting low-rank updates back to full size.
    warm_restart : bool, optional
        If ``True``, uses the previous projection tensor as the initializer for
        the next Tucker decomposition.
    activation_checkpoint : bool, optional
        Whether to use activation checkpointing for Tucker decomposition.

    References
    ----------
    .. _[1] : Loeschcke, S., Pitt, D., George, R., Zhao, J., Luo, C.,
        Tian, Y., Kossaifi, J., Anandkumar, A. (2025).
        TensorGRaD: Tensor Gradient Robust Decomposition for Memory-Efficient
        Neural Operator Training. arXiv preprint,
        https://arxiv.org/abs/2501.02379.
    """

    def __init__(
        self,
        rank,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        tucker_n_iter_max: int = 10,
        warm_restart: bool = False,
        activation_checkpoint: bool = False,
    ):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.warm_restart = warm_restart
        self.tucker_n_iter_max = tucker_n_iter_max
        self.activation_checkpoint = activation_checkpoint
        self.proj_tensor = None

    def project(self, full_rank_grad, iter):
        if self.proj_tensor is None or iter % self.update_proj_gap == 0:
            self.proj_tensor = self.get_projection_tensor(full_rank_grad, self.rank)
        return self.transform(self.proj_tensor, full_rank_grad)

    def project_back(self, low_rank_grad):
        full_rank_grad = self.inverse_transform(self.proj_tensor, low_rank_grad)
        return full_rank_grad * self.scale

    def get_projection_tensor(self, weights, rank):
        if torch.is_complex(weights.data) and weights.data.dtype != torch.cfloat:
            matrix = weights.data.cfloat()
        else:
            matrix = weights.data

        if self.warm_restart and self.proj_tensor is not None:
            init_factors = [factor.to(matrix.device) for factor in self.proj_tensor]
            init_core = self.transform(init_factors, matrix)
            init = (init_core, init_factors)
        else:
            init = "svd"
        if self.activation_checkpoint:
            core, factors = checkpoint(tucker, matrix, rank=rank, init=init, n_iter_max=self.tucker_n_iter_max, use_reentrant=False)
        else:
            core, factors = tucker(matrix, rank=rank, init=init, n_iter_max=self.tucker_n_iter_max)
        del core
        return factors

    def transform(self, factors, x):
        if self.activation_checkpoint:
            return checkpoint(tenalg.multi_mode_dot, x, factors, transpose=True, use_reentrant=False)
        return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, factors, x):
        if self.activation_checkpoint:
            return checkpoint(tenalg.multi_mode_dot, x, factors)
        return tenalg.multi_mode_dot(x, factors)


class UnstructuredSparseProjector:
    """Project gradients by keeping a sparse set of entries in flattened order.

    Parameters
    ----------
    sparse_ratio : float
        Fraction of entries to keep in the sparse gradient.
    update_proj_gap : int, optional
        Number of optimizer steps between sparse mask updates.
    sparse_type : {"randk", "randomk", "topk"}, optional
        Strategy used to choose sparse entries. ``topk`` keeps entries with the
        largest absolute gradient values, while ``randk``/``randomk`` sample
        entries uniformly at random.
    scale : float, optional
        Scalar applied after projecting the sparse update back to full size.
    scale_by_mask_ratio : bool, optional
        If ``True``, additionally scales by the square root of the inverse
        sparse mask ratio after projection back.
    """

    def __init__(
        self,
        sparse_ratio: float,
        update_proj_gap: int = 50,
        sparse_type: str = "randk",
        scale: float = 1.0,
        scale_by_mask_ratio: bool = False,
    ):
        if not 0 < sparse_ratio <= 1:
            raise ValueError(f"Expected sparse_ratio in (0, 1], got {sparse_ratio}.")
        if update_proj_gap < 1:
            raise ValueError(f"Expected update_proj_gap >= 1, got {update_proj_gap}.")
        if sparse_type not in {"randk", "randomk", "topk"}:
            raise ValueError(f"Unsupported sparse_type={sparse_type}.")

        self.sparse_ratio = sparse_ratio
        self.update_proj_gap = update_proj_gap
        self.sparse_type = sparse_type
        self.scale = scale
        self.scale_by_mask_ratio = scale_by_mask_ratio
        self.indices = None
        self.orig_shape = None

    def project(self, grad: torch.Tensor, step: int):
        if self.orig_shape is None:
            self.orig_shape = grad.shape
        if self.indices is None or step % self.update_proj_gap == 0:
            self.indices = self._build_indices(grad)
        return grad.reshape(-1)[self.indices]

    def project_back(self, sparse_grad: torch.Tensor):
        out = torch.zeros(
            self.orig_shape,
            dtype=sparse_grad.dtype,
            device=sparse_grad.device,
        )
        values = sparse_grad * self.scale
        if self.scale_by_mask_ratio:
            values = values * math.sqrt(out.numel() / sparse_grad.numel())
        out.reshape(-1).scatter_(0, self.indices, values.to(out.dtype))
        return out

    def _build_indices(self, grad: torch.Tensor):
        flat = grad.reshape(-1)
        k = max(1, int(self.sparse_ratio * flat.numel()))
        if self.sparse_type == "topk":
            _, indices = torch.topk(flat.abs(), k)
        else:
            indices = torch.randperm(flat.numel(), device=grad.device)[:k]
        return indices.sort().values


class TensorGRaD(Optimizer):
    """AdamW with optional low-rank and sparse gradient projection.

    Parameters with a ``tensorgrad`` parameter-group flag are optimized in a
    compressed gradient space. Groups with only ``rank`` use a single
    TensorGRaD/Tucker low-rank branch. Groups that also set
    ``sparse_ratio`` use a TensorGRaD-style residual branch: low-rank project
    the gradient, sparse-project the low-rank residual, run Adam on both
    compressed states, then project both updates back and combine them.

    Parameters
    ----------
    params : iterable
        Iterable of parameters or parameter groups. Parameter groups with
        ``tensorgrad=True`` use compressed-gradient updates.
    lr : float, optional
        Learning rate.
    betas : tuple[float, float], optional
        Adam exponential moving average coefficients.
    eps : float, optional
        Epsilon added to the denominator for numerical stability.
    weight_decay : float, optional
        Decoupled weight decay.
    correct_bias : bool, optional
        Whether to apply Adam bias correction.

    TensorGRaD parameter-group options
    ----------------------------------
    rank : float or tuple[int, ...]
        Tucker low-rank budget for the low-rank branch. A float gives the
        target fraction of low-rank gradient entries.
    sparse_ratio : float, optional
        If provided, enables the sparse residual branch with this entry budget.
    update_proj_gap : int, optional
        Number of optimizer steps between low-rank basis and sparse mask
        updates.
    scale : float, optional
        Scalar applied to the projected-back low-rank update.
    sparse_scale : float, optional
        Scalar applied to the projected-back sparse update.
    lambda_sparse : float, optional
        Weight used when adding the sparse update to the low-rank update.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0 <= betas[0] < 1:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0 <= betas[1] < 1:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("TensorGRaD does not support sparse gradients.")

                state = self.state[param]
                if "step" not in state:
                    state["step"] = 0

                step = state["step"]
                if group.get("tensorgrad", False):
                    update, step_size = self._tensorgrad_update(
                        param,
                        grad,
                        group,
                        state,
                    )
                else:
                    update, step_size = self._adamw_update(
                        grad,
                        group,
                        state,
                        "exp_avg",
                        "exp_avg_sq",
                        step + 1,
                    )

                param.add_(update, alpha=-step_size)
                if group["weight_decay"] > 0:
                    param.add_(param, alpha=-group["lr"] * group["weight_decay"])
                state["step"] = step + 1

        return loss

    def _tensorgrad_update(self, param, grad, group, state):
        if "low_rank_projector" not in state:
            state["low_rank_projector"] = self._make_low_rank_projector(param, group)
            if group.get("sparse_ratio") is not None:
                state["sparse_projector"] = self._make_sparse_projector(group)

        low_rank_projector = state["low_rank_projector"]
        low_rank_grad = low_rank_projector.project(grad, state["step"])

        if "sparse_projector" not in state:
            update, step_size = self._adamw_update(
                low_rank_grad,
                group,
                state,
                "low_rank_exp_avg",
                "low_rank_exp_avg_sq",
                state["step"] + 1,
            )
            return low_rank_projector.project_back(update), step_size

        # Build the sparse branch from the residual left by the unscaled
        # low-rank reconstruction. Scaling is applied only to the final update.
        low_rank_reconstruction = low_rank_projector.inverse_transform(
            low_rank_projector.proj_tensor,
            low_rank_grad,
        )
        residual = grad - low_rank_reconstruction
        sparse_projector = state["sparse_projector"]
        sparse_grad = sparse_projector.project(residual, state["step"])

        low_rank_update, step_size = self._adamw_update(
            low_rank_grad,
            group,
            state,
            "low_rank_exp_avg",
            "low_rank_exp_avg_sq",
            state["step"] + 1,
        )
        sparse_update, _ = self._adamw_update(
            sparse_grad,
            group,
            state,
            "sparse_exp_avg",
            "sparse_exp_avg_sq",
            state["step"] + 1,
        )
        sparse_weight = group.get("lambda_sparse", 1.0)
        full_update = low_rank_projector.project_back(low_rank_update)
        full_update.add_(
            sparse_projector.project_back(sparse_update),
            alpha=sparse_weight,
        )
        return full_update, step_size

    def _adamw_update(self, grad, group, state, exp_avg_key, exp_avg_sq_key, step):
        if exp_avg_key not in state:
            state[exp_avg_key] = torch.zeros_like(grad)
            state[exp_avg_sq_key] = torch.zeros_like(grad)

        exp_avg = state[exp_avg_key]
        exp_avg_sq = state[exp_avg_sq_key]
        beta1, beta2 = group["betas"]

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        if torch.is_complex(grad):
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        denom = exp_avg_sq.sqrt().add_(group["eps"])
        step_size = group["lr"]
        if group["correct_bias"]:
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size *= math.sqrt(bias_correction2) / bias_correction1
        return exp_avg / denom, step_size

    @staticmethod
    def _make_low_rank_projector(param, group):
        rank = validate_tucker_rank(param.shape, group.get("rank", 0.25))
        return TensorGRaDProjector(
            rank=rank,
            update_proj_gap=group.get("update_proj_gap", 50),
            scale=group.get("scale", 1.0),
            warm_restart=group.get("warm_restart", True),
            tucker_n_iter_max=group.get("tucker_n_iter_max", 10),
            activation_checkpoint=group.get("activation_checkpoint", False),
        )

    @staticmethod
    def _make_sparse_projector(group):
        return UnstructuredSparseProjector(
            sparse_ratio=group["sparse_ratio"],
            update_proj_gap=group.get("update_proj_gap", 50),
            sparse_type=group.get("sparse_type", "randk"),
            scale=group.get("sparse_scale", 1.0),
            scale_by_mask_ratio=group.get("scale_by_mask_ratio", False),
        )


def fno_tensorgrad_param_groups(
    model: nn.Module,
    rank: float = 0.25,
    sparse_ratio: Optional[float] = None,
    min_params: int = 1000,
    update_proj_gap: int = 50,
    scale: float = 1.0,
    sparse_scale: float = 1.0,
    sparse_type: str = "randk",
    lambda_sparse: float = 1.0,
    warm_restart: bool = True,
):
    """Return AdamW and TensorGRaD parameter groups for FNO spectral weights.

    The returned groups put large parameters from ``model.fno_blocks.convs`` in
    a TensorGRaD group and all remaining trainable parameters in a regular
    AdamW-style group.

    Parameters
    ----------
    model : nn.Module
        FNO-like model with ``fno_blocks.convs``.
    rank : float, optional
        Tucker low-rank budget for selected spectral-convolution parameters.
    sparse_ratio : float, optional
        Sparse residual budget. If ``None``, only the low-rank branch is used.
    min_params : int, optional
        Minimum number of entries a spectral-convolution parameter must have to
        be optimized with TensorGRaD.
    update_proj_gap : int, optional
        Number of optimizer steps between projection updates.
    scale : float, optional
        Scalar applied to projected-back low-rank updates.
    sparse_scale : float, optional
        Scalar applied to projected-back sparse updates.
    sparse_type : {"randk", "randomk", "topk"}, optional
        Strategy used by the sparse residual projector.
    lambda_sparse : float, optional
        Weight used when adding the sparse update to the low-rank update.
    warm_restart : bool, optional
        Whether Tucker decomposition reuses the previous projection factors
        when updating the low-rank basis.
    """

    module = model.module if hasattr(model, "module") else model
    if not hasattr(module, "fno_blocks") or not hasattr(module.fno_blocks, "convs"):
        raise ValueError("Expected an FNO-like model with fno_blocks.convs.")

    tensorgrad_params = [
        param
        for param in module.fno_blocks.convs.parameters()
        if param.requires_grad and param.numel() >= min_params
    ]
    tensorgrad_param_ids = {id(param) for param in tensorgrad_params}
    regular_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in tensorgrad_param_ids
    ]
    if not tensorgrad_params:
        raise ValueError(
            "No FNO spectral convolution parameters matched TensorGRaD filtering."
        )

    tensorgrad_group = {
        "params": tensorgrad_params,
        "tensorgrad": True,
        "rank": rank,
        "update_proj_gap": update_proj_gap,
        "scale": scale,
        "warm_restart": warm_restart,
    }
    if sparse_ratio is not None:
        tensorgrad_group.update(
            {
                "sparse_ratio": sparse_ratio,
                "sparse_type": sparse_type,
                "sparse_scale": sparse_scale,
                "lambda_sparse": lambda_sparse,
            }
        )
    return [{"params": regular_params}, tensorgrad_group]
