## Reproduce experiments using Sagemaker

### Run single experiments
1. Install aditional dependencies locally
```bash
pip install -r sagemaker/requirements.txt
```
2. Build Docker image with CUDA support and upload it to AWS ECR
```bash
sh sagemaker/push_ecr_image.sh
```
4. Send Training Jobs using Sagemaker
```bash
python sagemaker/run_sm.py
```

### Hyper-Parameter Optimization (HPO) with Syne-tune and Sagemaker

Single eta optimization
```bash
pip install -r sagemaker/requirements.txt
python sagemaker/run_hpo_sm.py --config_fn='configs/flow_mf_like_mcmc.py' --smi_method='flow'
python sagemaker/run_hpo_sm.py --config_fn='configs/flow_nsf_like_mcmc.py' --smi_method='flow'
python sagemaker/run_hpo_sm.py --config_fn='configs/flow_mf.py' --smi_method='flow'
python sagemaker/run_hpo_sm.py --config_fn='configs/flow_nsf.py' --smi_method='flow'
python sagemaker/run_hpo_sm.py --config_fn='configs/flow_nsf_vmp_flow_like_mcmc.py' --smi_method='vmp_flow'
python sagemaker/run_hpo_sm.py --config_fn='configs/flow_nsf_vmp_flow.py' --smi_method='vmp_flow'
```
