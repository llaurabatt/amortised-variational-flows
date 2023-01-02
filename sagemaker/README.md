## Reproduce experiments using Sagemaker

### Before sending jobs...
Install required modules
```bash
pip install -r requirements.txt
pip install -r sagemaker/requirements.txt
```
Build Docker image with CUDA support and upload it to AWS ECR
```bash
bash sagemaker/push_ecr_image.sh
```
You can run the docker container interactively to test the image
```bash
docker run --gpus all -u ubuntu -it jax-tf
```

### Run all experiments
Send Training Jobs using Sagemaker
```bash
python sagemaker/run_sm.py
```

### Download all results
Send Training Jobs using Sagemaker
```bash
bash sagemaker/download_sm_res.sh
```

### Hyper-Parameter Optimization (HPO) with Syne-tune and Sagemaker

```bash
pip install -r sagemaker/requirements.txt
python sagemaker/run_hpo_sm.py --config_fn='configs/8_items_flow_mf.py' --smi_method='flow'
python sagemaker/run_hpo_sm.py --config_fn='configs/8_items_flow_nsf.py' --smi_method='flow'
python sagemaker/run_hpo_sm.py --config_fn='configs/8_items_flow_nsf_vmp_flow.py' --smi_method='vmp_flow'
python sagemaker/run_hpo_sm.py --config_fn='configs/all_items_flow_mf.py' --smi_method='flow'
python sagemaker/run_hpo_sm.py --config_fn='configs/all_items_flow_nsf.py' --smi_method='flow'
python sagemaker/run_hpo_sm.py --config_fn='configs/all_items_flow_nsf_vmp_flow.py' --smi_method='vmp_flow' --log_dir $HOME/spatial-smi-output/hpo_log_20230101 --alsologtostderr &
```
