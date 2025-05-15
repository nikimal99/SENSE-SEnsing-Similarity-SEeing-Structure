# SENSE: SEnsing Similarity, SEeing Structure

**SENSE** is a unified, geometry-aware framework for privacy-preserving decentralized neighbor embedding. It supports a wide range of settings depending on the visibility of anchor points and distribution of client data, and can handle both Euclidean and hyperbolic spaces.

---

## 📦 Setup

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/sense-framework.git
cd sense-framework
pip install -r requirements.txt

sense-framework/
│
├── create_dataset_sense.py      # Dataset preprocessing and client splitting (IID, Non-IID Balanced/Unbalanced)
├── Pointwise_full.py      # SENSE in Pointwise Full configuration
├── Multisite_Full.py      # SENSE in Multisite Full configuration
├── Multisite_Partial.py   # SENSE in Multisite Partial configuration
├── bash_sense.sh                # Bash script to launch specific SENSE configurations
└── requirements.txt             # Python dependencies


## To execute the full pipeline for any SENSE configuration, run:
./bash_sense.sh

Dataset Preparation
The script create_dataset_sense.py prepares the dataset and splits it across clients:

IID: Uniform random splitting

Non-IID Balanced: Dirichlet partitioning with balanced client sizes

Non-IID Unbalanced: Dirichlet partitioning with variable client sizes
