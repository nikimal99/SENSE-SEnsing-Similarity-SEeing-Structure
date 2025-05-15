# SENSE: SEnsing Similarity, SEeing Structure

**SENSE** is a unified, geometry-aware framework for privacy-preserving decentralized neighbor embedding. It supports a wide range of settings depending on the visibility of anchor points and distribution of client data, and can handle both Euclidean and hyperbolic spaces.

---

## 📦 Setup

Clone the repository and install the required dependencies, using conda:


```bash
conda env create -f environment.yml
conda activate sense-env

## 📁 Repository Structure

SENSE: SEnsing Similarity, SEeing Structure/
│
├── create_dataset_sense.py # Dataset preprocessing and Dirichlet partitioning
├── Pointwise_Full.py # SENSE configuration: Pointwise Full
├── Multisite_Full.py # SENSE configuration: Multisite Full
├── Multisite_Partial.py # SENSE configuration: Multisite Partial
├── bash_sense.sh # Master script to run all experiments
├── README.md # Project overview and usage
└── environment.yml # Conda environment with dependencies

##SENSE Configurations
Each .py file represents a distinct decentralized setting.

## Running the Full Pipeline
To run all configurations across multiple datasets and partition settings, use the bash script:
./bash_sense.sh

The script iterates over all datasets, applies IID, Balanced, and Unbalanced partitioning, runs all SENSE configurations and saves results in a structured directory:
results/<dataset>/<partition_type>/<config>/output.txt

Example output path:
results/RetinaMNIST/balanced/pointwise_full/output.txt

You can modify bash_sense.sh to include or exclude specific datasets or configurations.

##Dataset Preparation
The script create_dataset_sense.py prepares the dataset and splits it across clients:

IID: Uniform random splitting

Non-IID Balanced: Dirichlet partitioning with balanced client sizes

Non-IID Unbalanced: Dirichlet partitioning with variable client sizes

##Notes
Results include both the embedding performance metrics and the log output for traceability.

Code is modular and easily extendable to new datasets or embedding backends.

Works with image datasets (MNIST, fashionmnist), medical image datasets (e.g., MedMNIST) and tabular datasets (e.g., German Credit).
