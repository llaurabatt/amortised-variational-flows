#!/bin/bash
# Eta-only SGD: restore ckpt_100000 -> skip training -> optimise ETA ONLY against
# held-out PMSE (prior scales held at PriorHparams/set_defaults values) -> writes
# tune_eta/hp_info_etaonly_*_new.sav + eta_tuning/ tensorboard. Leaves the
# full-tune artifacts (tune_all_hparams/, hparam_tuning/) untouched.
exec > "/home/llaurabat/geoff_project/vmp-output/lp-all/3ELBO/40val/eta_tune_run.log" 2>&1
rm -f /tmp/libtpu_lockfile
cd "/home/llaurabat/geoff_project/amortised-variational-flows"
exec /home/llaurabat/.local/bin/micromamba run -p "/mnt/disks/geoff/envs/spatial-smi-oldv-micromamba" python "/home/llaurabat/geoff_project/amortised-variational-flows/linguistic_profiles/main.py" \
  --config "/home/llaurabat/geoff_project/amortised-variational-flows/linguistic_profiles/configs/all_items_flow_nsf_vmp_flow_3ELBO_40val.py" \
  --workdir "/home/llaurabat/geoff_project/vmp-output/lp-all/3ELBO/40val" \
  --log_dir "/home/llaurabat/geoff_project/vmp-output/lp-all/3ELBO/40val/log_dir" \
  --config.tune_vmp_hparams=False \
  --config.tune_vmp_hparams_fix_eta=False \
  --config.scan_loss_surface=False \
  --config.tune_vmp_eta_only=True \
  --alsologtostderr
