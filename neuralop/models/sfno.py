"""
SFNO - Spherical Fourier Neural Operator
Replaces the default SpectralConv (a convolution in the frequency domain
over Fourier basis functions) with a SphericalConv (a convolution over the
spherical harmonic basis functions)
"""

from ..layers.spherical_convolution import SphericalConv
from .fno import (
    FNO,
    partialclass,
    _T_EMB_DEFAULT_EMBED,
    _T_EMB_DEFAULT_MODE_MOD,
)

SFNO = partialclass("SFNO", FNO, factorization="dense", conv_module=SphericalConv)
SFNO.__doc__ = SFNO.__doc__.replace("Fourier", "Spherical Fourier", 1)
SFNO.__doc__ = SFNO.__doc__.replace("FNO", "SFNO")
SFNO.__doc__ = SFNO.__doc__.replace("fno", "sfno")
SFNO.__doc__ = SFNO.__doc__.replace(":ref:`sfno_intro`", ":ref:`fno_intro`")


class t_emb_SFNO(SFNO):
    """Time-conditioned Spherical Fourier Neural Operator.

    An :class:`SFNO` pre-configured with default ``embed`` and
    ``mode_modulation`` dicts so that ``forward(x, t=...)`` does mode
    modulation out of the box. Any of ``embed``, ``mode_modulation``, or
    ``norm_modulation`` passed by the caller override the defaults.

    Each instance receives a fresh copy of the default dicts, so mutations
    by one instance never leak into another.

    Requires ``torch-harmonics``.

    Examples
    --------
    >>> from neuralop.models import t_emb_SFNO
    >>> model = t_emb_SFNO(n_modes=(16, 16), in_channels=1, out_channels=1,
    ...                    hidden_channels=64)
    >>> y = model(torch.randn(2, 1, 32, 32), t=0.5)
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("embed", _T_EMB_DEFAULT_EMBED.copy())
        kwargs.setdefault("mode_modulation", _T_EMB_DEFAULT_MODE_MOD.copy())
        super().__init__(*args, **kwargs)
