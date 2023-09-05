<div align="center">

# TOPOLOGICAL NEURAL NETWORKS OVER THE AIR

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://wandb.ai/site"><img alt="Weights & Biases" src="https://img.shields.io/badge/Weights%20%26%20Biases-ffbe00?logo=weightsandbiases&logoColor=white"></a>

[![Paper](.svg)](mylinktopaper)
    
</div>

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setting up a Conda Environment](#setting-up-a-conda-environment)
3. [Usage](#usage)

## Introduction

Topological neural networks (TNNs) are information processing architectures that model representations from data lying over topological spaces (e.g., simplicial or cell complexes) and allow for decentralized implementation through localized communications over different neighborhoods. Existing TNN architectures have not yet been considered in realistic communication scenarios, where channel effects typically introduce disturbances such as fading and noise. This paper aims to propose a novel TNN design, operating on regular cell complexes, that performs over-the-air computation, incorporating the wireless communication model into its architecture. Specifically, during training and inference, the proposed method considers channel impairments such as fading and noise in the topological convolutional filtering operation, which takes place over different signal orders and neighborhoods. Numerical results illustrate the architecture's robustness to channel impairments during testing and the superior performance with respect to existing architectures, which are either communication-agnostic or graph-based. 

## Installation

### Prerequisites

- [Anaconda or Miniconda](https://docs.anaconda.com/anaconda/install/)

### Setting up a Conda Environment

1. **Clone the repository**
    ```bash
    git clone https://github.com/your_username/your_project_name.git
    ```

2. **Navigate to the project directory**
    ```bash
    cd your_project_name
    ```

3. **Create a new Conda environment**
    ```bash
    conda create --name your_environment_name python=3.x
    ```

4. **Activate the Conda environment**
    ```bash
    conda activate your_environment_name
    ```

5. **Install the required packages**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Train model with default configuration

```bash
python src/train.py
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=10 datamodule.batch_size=64
```
