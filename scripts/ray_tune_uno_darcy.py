"""
Hyperparameter optimization for UNO on Darcy Flow using HyperNOs (https://github.com/MaxGhi8/HyperNOs) and Ray Tune.
"""

import torch
from ray import tune
from hypernos.datasets import NO_load_data_model
from hypernos.loss_fun import loss_selector
from hypernos.tune import tune_hyperparameters
from hypernos.wrappers import wrap_model_builder
from neuralop.models import UNO

def run_hpo_uno_darcy():
    # Base hyperparameters
    default_hyper_params = {
        "training_samples": 1024,
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "weight_decay": 1e-5,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "beta": 1,
        "hidden_channels": 32,
        "modes": 16,
        "input_dim": 1,
        "out_dim": 1,
        "problem_dim": 2,
        "FourierF": 0,
        "retrain": 4,
    }

    # Define the search space
    config_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "hidden_channels": tune.choice([16, 32, 64]),
        "uno_out_channels": tune.choice([[32, 64, 64, 32], [16, 32, 32, 16]]),
        "uno_n_modes": tune.choice([[16, 16], [8, 8]]),
    }

    # Merge fixed params
    fixed_params = {**default_hyper_params}
    for param in config_space.keys():
        fixed_params.pop(param, None)
    config_space.update(fixed_params)

    # Model builder for NeuralOperator's UNO
    def model_builder(config):
        return UNO(
            in_channels=config["input_dim"],
            out_channels=config["out_dim"],
            hidden_channels=config["hidden_channels"],
            uno_out_channels=config["uno_out_channels"],
            uno_n_modes=config["uno_n_modes"],
            uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0]],
            n_layers=4,
            horizontal_skips_map={4: 0, 3: 1}
        )
    
    model_builder = wrap_model_builder(model_builder, "darcy_uno_operator")

    def dataset_builder(config):
        return NO_load_data_model(
            which_example="darcy",
            no_architecture={
                "FourierF": config["FourierF"],
                "retrain": config["retrain"],
            },
            batch_size=config["batch_size"],
            training_samples=config["training_samples"],
        )

    loss_fn = loss_selector("L2", problem_dim=2, beta=1)

    tune_hyperparameters(
        config_space=config_space,
        model_builder=model_builder,
        dataset_builder=dataset_builder,
        loss_fn=loss_fn,
        default_hyper_params=[default_hyper_params],
        num_samples=10,
        max_epochs=default_hyper_params["epochs"],
        checkpoint_freq=default_hyper_params["epochs"],
        runs_per_cpu=1.0,
        runs_per_gpu=1.0 if torch.cuda.is_available() else 0.0,
    )

if __name__ == "__main__":
    run_hpo_uno_darcy()
