import torch
from .trainer import Trainer
from torch.nn.parallel import DistributedDataParallel as DDP

class SelfConsistencyTrainer(Trainer):
    def __init__(self, *args,
                 selfconsistency_loss=None,
                 selfconsistency_weight: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.selfconsistency_loss = selfconsistency_loss
        self.selfconsistency_weight = float(selfconsistency_weight)

    def _unwrap_model(self):
        # if using DDP, call the underlying module for auxiliary forwards
        return self.model.module if isinstance(self.model, DDP) else self.model

    def train_one_batch(self, idx, sample, training_loss):
        """Run one batch and return total training loss:
           base training_loss(out, **sample)
           + selfconsistency_weight * selfconsistency_loss(model, out, sample)
           + (optional) regularizer
        """
        self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()

        # preprocess / move to device
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            sample = {
                k: v.to(self.device)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }

        # accounting for number of samples
        if isinstance(sample.get("y", None), torch.Tensor):
            self.n_samples += sample["y"].shape[0]
        else:
            self.n_samples += 1

        # forward pass
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**sample)
        else:
            out = self.model(**sample)

        if self.epoch == 0 and idx == 0 and self.verbose and isinstance(out, torch.Tensor):
            print(f"Raw outputs of shape {out.shape}")

        # optional postprocess
        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        # base loss
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                loss = training_loss(out, **sample)
        else:
            loss = training_loss(out, **sample)

        # self-consistency term (optional)
        if self.selfconsistency_loss is not None and self.selfconsistency_weight != 0.0:
            model = self._unwrap_model()
            if self.mixed_precision:
                with torch.autocast(device_type=self.autocast_device_type):
                    sc_loss = self.selfconsistency_loss(model, sample["x"], loss_fn=training_loss, y=sample["y"])
            else:
                sc_loss = self.selfconsistency_loss(model, sample["x"], loss_fn=training_loss, y=sample["y"])
            loss = loss + self.selfconsistency_weight * sc_loss

        # regularizer (optional)
        if self.regularizer:
            loss = loss + self.regularizer.loss

        return loss
