from typing import Optional
from zencfg import ConfigBase


class WandbConfig(ConfigBase):
    """WandbConfig provides config options for setting up
    an interface with [Weights and Biases](https://wandb.ai).


    Parameters
    ----------
    log: bool, default False
        whether to log outputs to W&B
    entity: Optional[str], default None
        W&B username/entity to which to log
    project: Optional[str], default None
        Project name within W&B account to which to log.
    name: Optional[str], default None
        Name of the logged run on W&B
    group: str, default None
        If provided, will group this run along with all other
        runs tagged to the same group.
    sweep: bool, default False
        whether to perform an automatic W&B sweep. Deprecated.
    log_output: bool, default True
        Whether to optionally log model outputs at each eval step
        to W&B, if logging to W&B (``log = True``)

    """

    log: bool = False
    entity: Optional[str] = None
    project: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    sweep: bool = False
    log_output: bool = True