
#%%
import debugpy
#%%
debugpy.listen(5678)
print('Waiting for debugger')
debugpy.wait_for_client()
print('Debugger attached')
#%%
from absl import app
from absl import flags
from absl import logging

from functools import partial
import jax
from jax import numpy as jnp

import haiku as hk
import math
import matplotlib.pyplot as plt 
from ml_collections import config_flags, ConfigDict
import numpy as np
import pandas as pd
import pathlib
import re
from train_flow_allhp import (load_data, error_locations_estimate, make_optimizer)
from train_vmp_flow_allhp_smallcondval import (sample_all_flows, get_cond_values, loss, PriorHparams, q_distr_global, q_distr_loc_floating, get_inducing_points)
from log_prob_fun_allhp import sample_priorhparams_values, PriorHparams
from modularbayes._src.typing import (Any, Array, Batch, Callable, ConfigDict, Dict, List, NamedTuple,
                                      Optional, PRNGKey, SmiEta, SummaryWriter,
                                      Tuple)
from modularbayes import initial_state_ckpt
from modularbayes._src.utils.training import TrainState

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def amortisation_plot(config: ConfigDict, 
                      workdir: str,
                      loss_type: str = 'ELBO') -> None:
    # Remove trailing slash
    workdir = workdir.rstrip("/")

    # Initialize random keys
    prng_seq = hk.PRNGSequence(config.seed)

    # Load and process LALME dataset
    lalme_dataset = load_data(
        prng_key=next(prng_seq),  # use fixed seed for data loading
        config=config,
    )

        
    config.num_profiles = lalme_dataset['num_profiles']
    config.num_profiles_anchor = lalme_dataset['num_profiles_anchor']
    config.num_profiles_floating = lalme_dataset['num_profiles_floating']
    config.num_forms_tuple = lalme_dataset['num_forms_tuple']
    config.num_inducing_points = math.prod(config.flow_kwargs.inducing_grid_shape)

    # For training, we need a Dictionary compatible with jit
    # we remove string vectors
    train_ds = {
        k: v for k, v in lalme_dataset.items() if k not in ['items', 'forms']
    }

    train_ds = get_inducing_points(
        dataset=train_ds,
        inducing_grid_shape=config.flow_kwargs.inducing_grid_shape,
    )

    LPs = np.split(
        train_ds['LP'],
        np.cumsum(train_ds['num_profiles_split']),#np.cumsum(batch['num_profiles_split'][1:]),
    )[:-1]
    print(f"TRAIN LPs: {LPs[0] if config.num_lp_anchor_train>0 else 'NONE'} \n VAL LPs: {LPs[1] if config.num_lp_anchor_val>0 else 'NONE'} \n TEST LPs: {LPs[2] if config.num_lp_anchor_test>0 else 'NONE'} \n FLOATING LPs: {LPs[3] }")
    
    num_profiles_split = train_ds['num_profiles_split']

 
    # These parameters affect the dimension of the flow
    # so they are also part of the flow parameters
    config.flow_kwargs.num_profiles_anchor = lalme_dataset['num_profiles_anchor']
    config.flow_kwargs.num_profiles_floating = lalme_dataset[
        'num_profiles_floating']
    config.flow_kwargs.num_forms_tuple = lalme_dataset['num_forms_tuple']
    config.flow_kwargs.num_inducing_points = int(
        math.prod(config.flow_kwargs.inducing_grid_shape))
    config.flow_kwargs.is_smi = True

    # Get locations bounds
    # These define the range of values produced by the posterior of locations
    loc_bounds = np.stack(
        [lalme_dataset['loc'].min(axis=0), lalme_dataset['loc'].max(axis=0)],
        axis=1).astype(np.float32)
    config.flow_kwargs.loc_x_range = tuple(loc_bounds[0])
    config.flow_kwargs.loc_y_range = tuple(loc_bounds[1])

    eta_fixed = 1.
    prior_hparams_fixed = PriorHparams(*config.prior_hparams_fixed)

    etas = config.etas
    dirs = {}
    if config.workdir_VMP:
        VMP_dir = str(pathlib.Path(config.workdir_VMP) / 'checkpoints')
        dirs['VMP'] = VMP_dir
    if config.workdir_AdditiveVMP:
        AdditiveVMP_dir = str(pathlib.Path(config.workdir_AdditiveVMP) / 'checkpoints')
        dirs['AdditiveVMP'] = AdditiveVMP_dir
    if config.workdirs_VP:
        VP_dirs = [str(pathlib.Path(workdir) / 'checkpoints') for workdir in config.workdirs_VP]
        dirs.update({f'VP_{k}':v for k,v in zip(etas,VP_dirs)})
    optim_prior_hparams = pd.read_csv(config.optim_prior_hparams_dir + '/fixed_eta_opt.csv', index_col='eta_fixed')
    optim_prior_hparams =  optim_prior_hparams.to_dict(orient='index')


    if loss_type == 'ELBO':
        partial_ELBO_loss_eval = partial(loss,
                                        batch=train_ds,
                                        prng_key=next(prng_seq),
                                        flow_name=config.flow_name,
                                        flow_kwargs=config.flow_kwargs,
                                        include_random_anchor=False,
                                        num_samples=config.num_samples_amortisation_plot,
                                        profile_is_anchor=False,
                                        kernel_name=config.kernel_name,
                                        sample_priorhparams_fn=sample_priorhparams_values,
                                        sample_priorhparams_kwargs=config.prior_hparams_hparams,
                                        num_samples_gamma_profiles=config.num_samples_gamma_profiles,
                                        gp_jitter=config.gp_jitter,
                                        num_profiles_anchor=config.num_profiles_anchor,
                                        num_inducing_points=config.num_inducing_points,
                                        eta_sampling_a=config.eta_sampling_a,
                                        eta_sampling_b=config.eta_sampling_b,
                                        training=False,
            )

    elif loss_type == 'MSE':
        error_locations_estimate_jit = lambda locations_sample, loc: error_locations_estimate(
            locations_sample=locations_sample,
            num_profiles_split=num_profiles_split,
            loc=loc,
            floating_anchor_copies=None,
            train_idxs=None,
            ad_hoc_val_profiles=None, 
            val_idxs=None,
          )
        error_locations_estimate_jit = jax.jit(error_locations_estimate_jit)

          
        def mse_fixedhp(
            hp_params:Array,
            state_list: List[TrainState],
            batch: Optional[Batch],
            prng_key: PRNGKey,
            flow_name: str,
            flow_kwargs: Dict[str, Any],
            include_random_anchor:bool,
            num_samples: int,
            eta_fixed:float = None,
        ) -> Dict[str, Array]:

            if eta_fixed is not None:
              cond_values = jnp.hstack([hp_params, eta_fixed])
            else:
              cond_values = hp_params

            q_distr_out = sample_all_flows(
                params_tuple=[state.params for state in state_list],
                prng_key=prng_key,
                flow_name=flow_name,
                flow_kwargs=flow_kwargs,
                cond_values=jnp.broadcast_to(cond_values, (num_samples, len(cond_values))),
                # smi_eta=smi_eta_,
                include_random_anchor=include_random_anchor,
                num_samples=num_samples,
            )

            error_loc_dict = error_locations_estimate_jit(
                    locations_sample=q_distr_out['locations_sample'],
                    loc=batch['loc'],
                )
            return error_loc_dict['mean_dist_anchor_val'] #- logprobs_rho.sum()
          
        partial_mse_fixedhp = partial(mse_fixedhp, 
                batch=train_ds,
                prng_key=next(prng_seq),
                flow_name=config.flow_name,
                flow_kwargs=config.flow_kwargs,
                include_random_anchor=False,
                num_samples=config.num_samples_amortisation_plot,
            )
        partial_mse_fixedhp = jax.jit(partial_mse_fixedhp)

    amortisation_plot_points = {}

    for i, (dir_name, dir) in enumerate(dirs.items()):
        # cond values
        if 'VP' in dir_name:
            cond_values_init = None
        elif ((dir_name=='VMP') or (dir_name=='AdditiveVMP')):
            cond_values_init = get_cond_values(cond_hparams_names=config.cond_hparams_names,
                                num_samples=config.num_samples_elbo,
                                eta_init=eta_fixed,
                                prior_hparams_init=prior_hparams_fixed,
                                )
        else:
            raise ValueError(f"Invalid dir_name: {dir_name}")
    

        # Global parameters
        checkpoint_dir = dir
        state_name_list = [
            'global', 'loc_floating', 'loc_floating_aux', 'loc_random_anchor'
        ]
        state_list = []

        state_list.append(
            initial_state_ckpt(
                checkpoint_dir=f'{checkpoint_dir}/{state_name_list[0]}',
                forward_fn=hk.transform(q_distr_global),
                forward_fn_kwargs={
                    'flow_name': config.flow_name,
                    'flow_kwargs': config.flow_kwargs,
                    'cond_values':cond_values_init,
                    #   'eta': smi_eta_init['profiles'],
                    'num_samples':config.num_samples_elbo,
                },
                prng_key=next(prng_seq),
                optimizer=make_optimizer(**config.optim_kwargs),
            ))

        # Get an initial sample of global parameters
        # (used below to initialize floating locations)
        global_sample_base_ = hk.transform(q_distr_global).apply(
            state_list[0].params,
            next(prng_seq),
            flow_name=config.flow_name,
            flow_kwargs=config.flow_kwargs,
            cond_values=cond_values_init,
            #   eta=smi_eta_init['profiles'],
            num_samples=config.num_samples_elbo,
        )['sample_base']

        state_list.append(
            initial_state_ckpt(
                checkpoint_dir=f'{checkpoint_dir}/{state_name_list[1]}',
                forward_fn=hk.transform(q_distr_loc_floating),
                forward_fn_kwargs={
                    'flow_name': config.flow_name,
                    'flow_kwargs': config.flow_kwargs,
                    'global_params_base_sample': global_sample_base_,
                    'cond_values':cond_values_init,
                    #   'eta': smi_eta_init['profiles'],
                    'name': 'loc_floating',
                },
                prng_key=next(prng_seq),
                optimizer=make_optimizer(**config.optim_kwargs),
            ))

        state_list.append(
            initial_state_ckpt(
                checkpoint_dir=f'{checkpoint_dir}/{state_name_list[2]}',
                forward_fn=hk.transform(q_distr_loc_floating),
                forward_fn_kwargs={
                    'flow_name': config.flow_name,
                    'flow_kwargs': config.flow_kwargs,
                    'global_params_base_sample': global_sample_base_,
                    'cond_values':cond_values_init,
                    #   'eta': smi_eta_init['profiles'],
                    'name': 'loc_floating_aux',
                },
                prng_key=next(prng_seq),
                optimizer=make_optimizer(**config.optim_kwargs),
            ))

        if loss_type == 'ELBO':
            
            if ((dir_name=='AdditiveVMP') or (dir_name=='VMP')):
                amortised_points = []
                for eta in etas:
                    cond_hparams_values_evaluation = optim_prior_hparams[eta].copy()
                    cond_hparams_values_evaluation['eta'] = eta 
                    amortised_points.append(partial_ELBO_loss_eval(cond_hparams=config.cond_hparams_names,
                                                        cond_hparams_values_evaluation=cond_hparams_values_evaluation,
                                                params_tuple=[state.params for state in state_list]))
                amortisation_plot_points[dir_name] = amortised_points  

            elif 'VP' in dir_name:
                VP_points = []
                eta = float(re.search(r'\d+(\.\d+)?', dir_name).group())
                cond_hparams_values_evaluation = optim_prior_hparams[eta].copy()
                cond_hparams_values_evaluation['eta'] = eta 
                VP_points.append(partial_ELBO_loss_eval(cond_hparams=[],
                                                cond_hparams_values_evaluation=cond_hparams_values_evaluation,
                                            params_tuple=[state.params for state in state_list]))
                amortisation_plot_points[f'VP_{eta}'] = VP_points

            else:
                raise ValueError(f"Invalid dir_name: {dir_name}")
                
        elif loss_type == 'MSE':
            if ((dir_name=='AdditiveVMP') or (dir_name=='VMP')):
                amortised_points = [partial_mse_fixedhp(hp_params=get_cond_values(cond_hparams_names=config.cond_hparams_names,
                                                                num_samples=1,
                                                                eta_init=eta,
                                                                prior_hparams_init=prior_hparams_fixed,
                                                                ), 
                                                state_list=state_list) for eta in etas]
                amortisation_plot_points[dir_name] = amortised_points
            
            elif 'VP' in dir_name: 
                eta = float(re.search(r'\d+(\.\d+)?', dir_name).group()) 
                VP_points = partial_mse_fixedhp(hp_params=None, state_list=state_list)
                amortisation_plot_points[f'VP_{eta}'] = VP_points
            
            else:
                raise ValueError(f"Invalid dir_name: {dir_name}") 
            
            del state_list
    return amortisation_plot_points

def main(_):
    etas = FLAGS.config.etas
    # Get the results
    amortisation_plot_points = amortisation_plot(FLAGS.config, 
                                                 FLAGS.workdir, 
                                                 loss_type=FLAGS.config.loss_type)

    # Plot the results
    plt.figure(figsize=(10, 5))
    for key, points in amortisation_plot_points.items():
        plt.plot(etas, points, label=key)
    plt.xlabel('eta')
    plt.ylabel(FLAGS.config.loss_type)
    plt.legend()
    plt.savefig(f'{FLAGS.workdir}/{FLAGS.config.loss_type}_vs_eta.png')

if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)


