import torch
from neuralop.data.transforms.rescale import (
    extract_boundary_2D,
    extract_boundary_1D_Time,
    extract_boundary_2D_complex,
)
from neuralop.data.transforms.rescale import (
    RandomCropResize2D,
    RandomCropResize1DTime,
    RandomCropResize2DTimeAR,
)

def LossSelfconsistency(
    model,
    x,
    loss_fn,
    y=None,
    re=None,
    rate=None,
    new_y=None,
    size_min=64,
    type="darcy",
    plot=False,
    group_action=None,
    align_corner=False,
):
    """
    Selfconsistency loss:
    Enforces that the model evaluated on the entire domain, and restricted to a subdomain
    be equal to the model directly evaluated on the subdomain. The subdomain is chosen randomly each time.

    Code adapted from paper https://arxiv.org/abs/2507.18813

    Use example:
    ----------
    loss_consistency = LossSelfconsistency(model, x, loss_fn, y=y)

    Attributes
    ----------
        model : nn.Module
            The neural operator.
        x : torch.Tensor
            The input tensor of shape (B, C, H, W) or (B, C, T, H, W).
        loss_fn : callable
            The loss function to compute the difference between the two evaluations (usually Lploss).
        y : torch.Tensor, optional
            The ground truth tensor of shape (B, C, H, W) or (B, C, T, H, W). If provided,
            the loss is computed between model(x_small) and y_small.
            If not provided, the loss is computed between model(x_small) and y_small where
            y = model(x).
        re : torch.Tensor, optional
            Additional input tensor of shape (B, 1) if required by the model.
        rate : float, optional
            The rate at which to crop the input tensor. If None, a random rate is chosen.
        new_y : torch.Tensor, optional
            Not used in this function.
        size_min : int, optional
            The minimum size of the cropped subdomain.
        type : str, optional
            The type of PDE ('darcy', 'helmholtz', 'NS', 'burgers') to determine boundary extraction.
        plot : bool, optional
            If True, plots the cropped regions (not implemented here).
        group_action : callable, optional
            A function to apply group actions on (x_small, y_small) if needed.
        align_corner : bool, optional
            If True, aligns the corners of the cropped regions.

    Returns:
        torch.Tensor
            The computed self-consistency loss.
    """
    #

    batch_size = x.shape[0]

    if type == "darcy":
        ExtractBD = extract_boundary_2D
        transform_xy = RandomCropResize2D(p=1.0, size_min=size_min)
    elif type == "helmholtz":
        ExtractBD = extract_boundary_2D_complex
        transform_xy = RandomCropResize2D(p=1.0, size_min=size_min)
    elif type == "NS":
        ExtractBD = lambda x, y: x  # No boundary
        transform_xy = RandomCropResize2DTimeAR(p=1.0, size_min=size_min)
    elif type == "burgers":
        ExtractBD = lambda x, y: BurgersExtractBC(y)
        transform_xy = RandomCropResize1DTime(p=1.0, size_min=size_min)
    else:
        print("boundary type not supported")

    if re == None:
        re = torch.ones(batch_size, 1, requires_grad=False).to(x.device)

    # If y is given, we use it as the ground truth. the gradient only flow to y_small_
    if y is not None:

        # resample on smaller domain
        i, j, h, w, re = transform_xy.get_params(x, y, re=re, rate=rate)

        x_small = transform_xy.crop(x, i, j, h, w)
        y_small = transform_xy.crop(y, i, j, h, w)
        x_small = ExtractBD(x_small, y_small)

        if group_action is not None:
            x_small, y_small = group_action(x_small, y_small)

        if type == "darcy":
            y_small_ = model(x_small)
        else:
            y_small_ = model(x_small, re)

        return loss_fn(y_small_, y_small)

    # If y is not given, we set y=model(x). We treat the subdomain as ground truth can detach y_small_
    else:
        mode = "self_supervised"

        y = model(x, re)

        # resample on smaller domain
        i, j, h, w, re = transform_xy.get_params(x, y, re=re, rate=rate)
        if align_corner:
            i = 0
            j = 0

        x_small = transform_xy.crop(x, i, j, h, w)
        y_small = transform_xy.crop(y, i, j, h, w)
        x_small = ExtractBD(x_small, y_small)

        # if group_action is not None:
        #     x_small, y_small = group_action(x_small, y_small)
        y_small_ = model(x_small.detach(), re)

        if align_corner:
            H = int(y_small.shape[-2] // 2)
            W = int(y_small.shape[-1] // 2)
            y_small = y_small[..., :H, :W]
            y_small_ = y_small_[..., :H, :W]

        return loss_fn.truncated(y_small_, y_small)
