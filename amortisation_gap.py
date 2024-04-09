from functools import partial
import jax
from jax import numpy as jnp

import haiku as hk
import math
import matplotlib.pyplot as plt 
import numpy as np
import pathlib
from train_flow_allhp import (load_data, error_locations_estimate)
from train_vmp_flow_allhp_smallcondval import (sample_all_flows, get_cond_values, PriorHparams, q_distr_global, q_distr_loc_floating, get_inducing_points)
from modularbayes._src.typing import (Any, Array, Batch, Callable, ConfigDict, Dict, List, NamedTuple,
                                      Optional, PRNGKey, SmiEta, SummaryWriter,
                                      Tuple)
from modularbayes._src.utils import (initial_state_ckpt, make_optimizer)
from modularbayes._src.utils.training import TrainState




def amortisation_plot(config: ConfigDict, workdir: str) -> None:
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
    if config.cond_prior_hparams:
        prior_hparams_fixed = PriorHparams(config.prior_hparams_fixed)
    else:
        prior_hparams_fixed = None

    etas = config.etas
    VMP_dir = str(pathlib.Path(config.workdir_VMP) / 'checkpoints')
    AdditiveVMP_dir = str(pathlib.Path(config.workdir_AdditiveVMP) / 'checkpoints')
    VP_dirs = [str(pathlib.Path(workdir) / 'checkpoints') for workdir in config.workdirs_VP]
    dirs = {f'VP_{k}':v for k,v in zip(etas,VP_dirs)}
    dirs.update({'VMP':VMP_dir, 'AdditiveVMP':AdditiveVMP_dir})


    states = {k:0 for k in dirs.keys()}

    for i, (dir_name, dir) in enumerate(dirs.items()):
        # cond values
        if 'VP' in dir_name:
            cond_values_init = None
        elif ((dir_name=='VMP') or (dir_name=='AdditiveVMP')):
            cond_values_init = get_cond_values(cond_hparams_names=config.cond_hparams_names,
                                num_samples=config.num_samples_amortisation_plot,
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
                    'num_samples':config.num_samples_amortisation_plot,
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
            num_samples=config.num_samples_amortisation_plot,
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
        states[dir_name] = state_list


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
        ) -> Dict[str, Array]:

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
            return error_loc_dict['mean_dist_anchor_val'] 

    mse_fixedhp = jax.jit(mse_fixedhp)

    partial_mse_fixedhp = partial(mse_fixedhp, 
        batch=train_ds,
        prng_key=next(prng_seq),
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        include_random_anchor=True,
        num_samples=config.num_samples_amortisation_plot,
    )

    VMP_points = [partial_mse_fixedhp(hp_params=get_cond_values(cond_hparams_names=config.cond_hparams_names,
                                                    num_samples=1,
                                                    eta_init=eta,
                                                    prior_hparams_init=prior_hparams_fixed,
                                                    ), 
                                      state_list=states['VMP']) for eta in etas]
    
    AdditiveVMP_points = [partial_mse_fixedhp(hp_params=get_cond_values(cond_hparams_names=config.cond_hparams_names,
                                                    num_samples=1,
                                                    eta_init=eta,
                                                    prior_hparams_init=prior_hparams_fixed,
                                                    ), 
                                              state_list=states['AdditiveVMP']) for eta in etas]
    
    VP_points = [partial_mse_fixedhp(hp_params=None, state_list=states[f'VP_{eta}']) for eta in etas]

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(etas, VMP_points, label='VMP')
    plt.plot(etas, AdditiveVMP_points, label='AdditiveVMP')
    plt.plot(etas, VP_points, label='VP')
    plt.xlabel('eta')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(f'{workdir}/mse_vs_eta.png')
