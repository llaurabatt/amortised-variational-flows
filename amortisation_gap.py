from functools import partial
import matplotlib.pyplot as plt 
import pathlib
from train_flow_allhp import (error_locations_estimate)
from train_vmp_flow_allhp_smallcondval import (sample_all_flows)
from modularbayes._src.typing import (Any, Array, Batch, Callable, ConfigDict, Dict, List, NamedTuple,
                                      Optional, PRNGKey, SmiEta, SummaryWriter,
                                      Tuple)
from modularbayes._src.utils import (initial_state_ckpt, make_optimizer)



etas = [0., 0.25, 0.5, 0.75, 1.]
VMP_dir = str(pathlib.Path(workdir) / 'checkpoints')
AdditiveVMP_dir = str(pathlib.Path(workdir) / 'checkpoints')
VP_dirs = [str(pathlib.Path(workdir) / 'checkpoints'), str(pathlib.Path(workdir) / 'checkpoints')]
dirs = {f'VP_{k}':v for k,v in zip(etas,VP_dirs)}
dirs.update({'VMP':VMP_dir, 'AdditiveVMP':AdditiveVMP_dir})


states = {k:0 for k in dirs.keys()}

for i, (dir_name, dir) in enumerate(dirs.items()):
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
  states[dir_name] = state_list


error_locations_estimate_jit = lambda locations_sample, loc: error_locations_estimate(
    locations_sample=locations_sample,
    num_profiles_split=num_profiles_split,
    loc=loc,
    floating_anchor_copies=floating_anchor_copies,
    train_idxs=train_idxs,
    ad_hoc_val_profiles=ad_hoc_val_profiles,
    val_idxs=val_idxs,
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
    batch=batch,
    prng_key=prng_key,
    flow_name=config.flow_name,
    flow_kwargs=config.flow_kwargs,
    include_random_anchor=True,
    num_samples=config.num_samples_elbo,
)

VMP_points = [partial_mse_fixedhp(hp_params=cond_values_init, state_list=states['VMP']) for eta in etas]
AdditiveVMP_points = [partial_mse_fixedhp(hp_params=cond_values_init, state_list=states['AdditiveVMP']) for eta in etas]
VP_points = [partial_mse_fixedhp(hp_params=cond_values_init, state_list=states[f'VP_{eta}']) for eta in etas]

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(etas, VMP_points, label='VMP')
plt.plot(etas, AdditiveVMP_points, label='AdditiveVMP')
plt.plot(etas, VP_points, label='VP')
plt.xlabel('eta')
plt.ylabel('MSE')
plt.legend()
plt.savefig(f'{workdir}/mse_vs_eta.png')
