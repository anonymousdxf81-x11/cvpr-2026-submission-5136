# CUTS: Corrective Unlearning in Task Space (CVPR 2026 Submission ID 5136)
### Implementation of *"Subtract the Corruption: Training-Data-Free Corrective Machine Unlearning using Task Arithmetic"*

This repository contains the implementation of CUTS, a training-data-free
corrective machine unlearning method that uses task arithmetic to remove
the effect of label noise and backdoor triggers from trained models.

<div align="center" style="margin-top: 16px; margin-bottom: 8px;">
  <p><b>PCA trajectory of the penultimate features during CUTS correction.</b></p>
  <img src="media/pca_evol.gif" width="800" alt="Merged GIF"/>
</div>

<table align="center" width="800">
  <tr>
    <td align="center"><b>(a) CIFAR10</b></td>
    <td align="center"><b>(b) MNIST</b></td>
    <td align="center"><b>(c) CIFAR10</b></td>
  </tr>
  <tr>
    <td align="center">60% Symmetric Noise</td>
    <td align="center">40% Symmetric Noise</td>
    <td align="center">2% Poison Trigger</td>
  </tr>
</table>


## Repository structure

- `configs/` — experiment configuration files (datasets, architectures, noise rates, triggers)
- `scripts/` — shell scripts to reproduce the results reported in the tables/figures in the paper.
- `src/` — core implementation (datasets, models, CUTS algorithm, baselines)
- `run_experiment.py/` — main script for training models on corrupted datasets and applying correction methods
- `estiamte_alpha.py/` — the implementation of alpha estimator bsed on KNN-selfagreement for label corruption

## Installation and Requirements
Install the requirements by:
```bash
git clone https://github.com/anonymousdxf81-x11/cvpr-2026-submission-5136.git
cd cvpr-2026-submission-5136
pip install -r requirements.txt
```

## Models and Datasets

We use MNIST, CIFAR10, CIFAR100, and Clothing1M.

- MNIST / CIFAR10 / CIFAR100 are downloaded automatically by `torchvision`.
- Clothing1M: will be downloaded from `openxlab`. For this you will need to add `OPENXLAB_AK` and `OPENXLAB_SK` in a file named `.env` in the root directory of the workspace:
    ```bash
    OPENXLAB_AK=<your openxlab AK>
    OPENXLAB_SK=<your openxlab SK>
    ```

For CLIP we use the official implementation form [OpenAI CLIP](https://github.com/openai/CLIP). For DINOv3 we use [Facebook Huggingface](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) and the weights are downloaded automatically if `HF_TOKEN` is provided in `.env`. Other models use weights from `torchvision` model hub (if initialized with pre-trained weights.)



## Reproducing results
To reproduce the results shown in the tables in the paper, you can run the bash scripts inside `scripts/` directory:
```bash
bash ./scripts/run_table1.bash
```

To run for a customized configuration, create a yaml file inside `configs` directory, following the structure of other files, and run the experiment by:
```bash
python run_experiment.py --experiment <noise or poison> --arch <clip or dino or regular> --config <config name> --finetune --tv --sap or --potion
```

if you are running for a real-world dataset like `Clothing1M`, use with `--real-world` and to run with Pytorch DDP
```bash
CUDA_VISIBLE_DEVICES=<gpu IDs> torchrun --nproc_per_node=<num processes> run_experiment.py -e noise -a <clip or dino or regular> --real-world -c <config name> -f -t
```

