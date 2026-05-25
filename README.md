# REACT: Relaxed Efficient Acquisition of Context and Temporal Features

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2603.11370-blue)](https://arxiv.org/abs/2603.11370)
[![ACM BCB 2026](https://img.shields.io/badge/ACM%20BCB-2026-green)](https://acm-bcb.org)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)

Official PyTorch implementation of the paper:

> **REACT: Relaxed Efficient Acquisition of Context and Temporal Features**  
> Accepted as a full paper at ACM BCB 2026  
> [https://arxiv.org/abs/2603.11370](https://arxiv.org/abs/2603.11370)

---

## Overview

REACT addresses the problem of **longitudinal active feature acquisition (LAFA)** — deciding *which* features to measure and *when*, across multiple time points, given that acquiring features has a cost. Rather than observing all features at every visit, REACT learns a policy (actor) that selects the most informative subset of features at each timestep to support downstream classification, while minimizing acquisition cost.

The key technical contribution is a **Gumbel-Sigmoid relaxation** that makes the discrete feature-selection problem differentiable, enabling end-to-end training of the acquisition policy. The pipeline consists of:

1. **Classifier pre-training** — a masked classifier trained to predict outcomes from arbitrary subsets of observed features.
2. **Oracle rollout generation** — a search procedure that produces near-optimal acquisition trajectories to supervise the actor.
3. **Actor training** — an iterative policy learner (with optional joint fine-tuning of the classifier) that uses Gumbel-Sigmoid gates to select features at each timestep.
4. **Evaluation** — assessment of acquisition quality, classification performance, and feature cost.

---

## Repository Structure

```
.
├── main.py                        # Entry point: runs the full pipeline
├── config.py                      # All hyperparameters and dataset settings
├── dataset.py                     # Dataset loading and preprocessing
├── models.py                      # Classifier and actor model definitions
├── gumbel_actor.py                # Gumbel-Sigmoid acquisition actor
├── classifier_trainer.py          # Lightning module for classifier training
├── train_classifier.py            # Script: train the masked classifier
├── oracle_generator.py            # Logic for generating oracle rollouts
├── generate_oracle.py             # Script: generate oracle rollouts
├── train_actor_iterative_joint.py # Script: train actor with joint classifier fine-tuning
├── train_classifier.py            # Script: standalone classifier training
├── evaluate.py                    # Script: evaluate the trained actor
├── evaluate_vanilla.py            # Script: evaluate without acquisition (baseline)
├── utils.py                       # Utility functions
└── requirements.txt               # Python dependencies
```

---

## Installation

**Requirements:** Python 3.8+, a CUDA-capable GPU is recommended.

```bash
# 1. Clone the repository
git clone https://github.com/quyunniii/Longitudinal_Active_Feature_Acquisition_with_Gumble_Sigmoid_Relaxation.git
cd Longitudinal_Active_Feature_Acquisition_with_Gumble_Sigmoid_Relaxation

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Dependencies** (from `requirements.txt`):
```
torch>=2.0.0
torchmetrics>=0.11.0
pytorch-lightning>=2.0.0
numpy>=1.21.0
tqdm>=4.62.0
```

---

## Data Preparation

The codebase supports several datasets. Place your data in a subfolder matching the dataset name under the project root:

| Dataset key | Expected folder | Description |
|---|---|---|
| `womac` / `klg` | `./oai_data/` | OAI (Osteoarthritis Initiative) data |
| `adni` | `./adni/` | ADNI longitudinal biomarker data |

Each folder should contain `train`, `val`, and `test` splits in the format expected by `dataset.py`. The default dataset is `womac` (set in `config.py`).

---

## Quick Start

### Option A: Run the full pipeline with `main.py`

`main.py` orchestrates all four stages in sequence.

```bash
# Run everything with default settings (dataset = womac)
python main.py

# Specify a dataset
python main.py --data klg

# Override cost weights
python main.py --data womac --cw 0.05 --acw 0.001

# Use joint actor training (with classifier fine-tuning, joint is used for all REACT results in the paper)
python main.py --data womac --joint

# Skip stages you've already completed
python main.py --data womac --skip classifier oracle
```

**Pipeline arguments:**

| Argument | Description | Default |
|---|---|---|
| `--data` | Dataset: `synthetic`, `cheears_demog`, `klg`, `womac`, etc. | Config default (`womac`) |
| `--cw` | Cost weight — controls the acquisition cost penalty | Config default |
| `--acw` | Auxiliary cost weight — penalty for static/demographic feature cost | Config default |
| `--joint` | Use joint iterative actor (fine-tunes classifier during actor training) | `False` |
| `--skip` | Space-separated list of steps to skip: `classifier oracle actor evaluate` | None |

---

### Option B: Run individual stages

Each stage can also be run as a standalone script.

#### Stage 1 — Train the masked classifier
```bash
python train_classifier.py
```

#### Stage 2 — Generate oracle rollouts
```bash
python generate_oracle.py
# Or with custom cost weight:
python generate_oracle.py --cost_weight 0.05
```

#### Stage 3 — Train the actor
```bash
# Standard iterative actor
python train_actor_iterative_joint.py --cost_weight 0.05 --aux_cost_weight 0.001

# Joint actor (fine-tunes classifier simultaneously)
python train_actor_iterative_joint.py --cost_weight 0.05 --aux_cost_weight 0.001
```

#### Stage 4 — Evaluate
```bash
python evaluate.py --cost_weight 0.05 --aux_cost_weight 0.001

# Evaluate the joint model
python evaluate.py --cost_weight 0.05 --aux_cost_weight 0.001 --joint

# Evaluate vanilla baseline (no acquisition)
python evaluate_vanilla.py
```

---

## Configuration

All hyperparameters live in `config.py`. The most commonly tuned settings:

```python
# Select dataset (can also be set via environment variable)
DATASET = 'womac'   # or 'klg', 'cheears_demog', 'synthetic', 'adni', ...

# Mask type for classifier pre-training
MASK_TYPE = 'uniform'  # 'uniform' or 'bernoulli'
```

**Key hyperparameters in `ACTOR_CONFIG`:**

| Key | Description |
|---|---|
| `cost_weight` | Weight on longitudinal feature acquisition cost in the actor loss |
| `aux_cost_weight` | Weight on static/auxiliary feature acquisition cost |
| `gate_tau` | Gumbel-Sigmoid temperature (lower = more discrete) |
| `threshold` | Decision threshold for converting soft gates to hard selections at inference |
| `planner_hidden` | Hidden layer sizes of the acquisition planner MLP |
| `time_emb_dim` | Dimension of temporal embedding |
| `patience` | Early stopping patience (epochs) |

**Feature costs** can be edited directly in `config.py` under `LONGITUDINAL_FEATURE_COSTS` and `STATIC_FEATURE_COSTS` to reflect real-world measurement costs for your application.

You can also override the dataset at runtime without editing `config.py`:
```bash
ACTOR_DATASET=klg python main.py
```

---

## Outputs

Results are saved to `./outputs/<dataset>/`:

| File | Description |
|---|---|
| `classifier.ckpt` | Trained masked classifier checkpoint |
| `oracle_rollout_cw<cw>.npz` | Oracle trajectories for the given cost weight |
| `actor_iterative_cw<cw>_acw<acw>.ckpt` | Trained actor checkpoint |
| `actor_iterative_joint_cw<cw>_acw<acw>.ckpt` | Trained joint actor checkpoint |
| `evaluation_results_cw<cw>_acw<acw>.npz` | Evaluation metrics |
| `trajectory_cw<cw>_acw<acw>.npz` | Acquisition trajectories on the test set |

---

## Reproducing Paper Results

To reproduce the main results reported in the paper on the OAI dataset (WOMAC / KLG targets):

```bash
# WOMAC target
python main.py --data womac --cw 0.05 --acw 0.001 --joint

# KLG target
python main.py --data klg --cw 0.05 --acw 0.001 --joint
```

For the CHEEARS dataset with demographic features:
```bash
python main.py --data cheears_demog --cw 0.01 --acw 0.001 --joint
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{react2026,
  title         = {REACT: Relaxed Efficient Acquisition of Context and Temporal Features},
  author        = {Qu, Yunni and Dinh, Dzung and King, Grant and Ringwald, Whitney and Kok, Bing Cai and Gates, Kathleen and Wright, Aidan and Oliva, Junier},
  year          = {2026},
  eprint        = {2603.11370},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2603.11370},
}
```

---

## License

This project is released as open-source for research use. Please see the repository for license details.

---

## Contact

For questions about the code or paper, please open a [GitHub Issue](https://github.com/quyunniii/Longitudinal_Active_Feature_Acquisition_with_Gumble_Sigmoid_Relaxation/issues).
