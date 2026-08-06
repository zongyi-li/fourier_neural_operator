"""
Hyperparameter optimization for TFNO on Darcy Flow using HyperNOs (https://github.com/MaxGhi8/HyperNOs) and Ray Tune.
"""

import torch
from ray import tune
from hypernos.datasets import NO_load_data_model
from hypernos.loss_fun import loss_selector
from hypernos.tune import tune_hyperparameters
from hypernos.wrappers import wrap_model_builder
from neuralop.models import TFNO

def run_hpo_tfno_darcy():
    # Base hyperparameters
    default_hyper_params = {
        "training_samples": 100,
        "learning_rate": 0.001,
        "epochs": 50, 
        "batch_size": 32,
        "weight_decay": 1e-5,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "beta": 1,
        "width": 32,
        "modes": 16,
        "n_layers": 4,
        "input_dim": 1,
        "out_dim": 1,
        "rank": 0.05,
        "FourierF": 0,
        "retrain": 4,
        "problem_dim": 2,
    }

    # Define the search space
    config_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "rank": tune.choice([0.05, 0.1, 0.15]),
        "width": tune.choice([16, 32, 64]),
        "modes": tune.choice([8, 12, 16]),
    }

    # Merge fixed params into config_space
    fixed_params = {**default_hyper_params}
    for param in config_space.keys():
        fixed_params.pop(param, None)
    config_space.update(fixed_params)

    # Model builder for NeuralOperator's TFNO
    def model_builder(config):
        return TFNO(
            n_modes=(config["modes"], config["modes"]),
            hidden_channels=config["width"],
            n_layers=config["n_layers"],
            in_channels=config["input_dim"],
            out_channels=config["out_dim"],
            factorization="tucker",
            implementation="factorized",
            rank=config["rank"],
        )
    
    # Wrap model builder for HyperNOs compatibility
    model_builder = wrap_model_builder(model_builder, "darcy_neural_operator")

    # Dataset builder using HyperNOs standard loader
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

    # Loss function (relative L2 loss)
    loss_fn = loss_selector(
        loss_fn_str="L2",
        problem_dim=2,
        beta=1,
    )

    # Run Ray Tune
    tune_hyperparameters(
        config_space=config_space,
        model_builder=model_builder,
        dataset_builder=dataset_builder,
        loss_fn=loss_fn,
        default_hyper_params=[default_hyper_params],
        num_samples=10,
        max_epochs=default_hyper_params["epochs"],
        checkpoint_freq=default_hyper_params["epochs"],
        grace_period=10,
        reduction_factor=2,
        runs_per_cpu=1.0,
        runs_per_gpu=1.0 if torch.cuda.is_available() else 0.0,
    )

if __name__ == "__main__":
    run_hpo_tfno_darcy()
