"""
Hyper-Parameter Optimization (HPO) with Synetune locally.
"""

import os
import pathlib
from synetune.backend import LocalBackend
from synetune.optimizer.schedulers import FIFOScheduler
from synetune import Tuner, StoppingCriterion, config_space

# Assuming these flags are defined similarly to your original script
FLAGS = flags.FLAGS

def hpo_synetune_local(config_fn: str, smi_method: str) -> None:
    """Hyper-parameter optimisation with Synetune locally."""

    training_job_name = config_fn.replace('.py', '').replace('/', '-').replace('_', '-')
    metric = "mean_dist_anchor_val_min"
    mode = "min"
    searcher = 'bayesopt'
    n_workers = 1  # Adjust based on your local machine's capabilities
    stop_criterion = StoppingCriterion(max_num_trials_started=20)

    # Configuration space adapted from your script, customized for local execution
    config_space_dict = {
        # Define your configuration space here
    }

    # The local backend replaces SageMakerBackend for running trials
    backend = LocalBackend(entry_point="path/to/your_training_script.py")

    scheduler = FIFOScheduler(
        config_space=config_space_dict,
        searcher=searcher,
        metric=metric,
        mode=mode,
    )

    tuner = Tuner(
        trial_backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )

    tuner.run()

def main():
    # Assuming you have a way to parse arguments or set them directly
    config_fn = 'path/to/your_config_file.py'  # Example path to your configuration file
    smi_method = 'vmp_flow'  # Example method

    hpo_synetune_local(config_fn=config_fn, smi_method=smi_method)

if __name__ == "__main__":
    main()