# Predicting unseen chemical perturbation effects by integrating multi-source prior knowledge with foundation model
scMCP is a chemical perturbation framework that integrate multi-source biological prior knowledge for drug discovery.
## Overview of scMCP:

## QuickStart:
:satisfied:Requirements are shown in the requirements.txt. Please use the requirements.txt to create the environment for running scMCP.
<img width="2737" height="3813" alt="Figure1" src="https://github.com/user-attachments/assets/9f28bc5b-961a-492a-b934-30ba0e440814" />

:satisfied:Please use the molecule fingerprint generated procession to pregenerate the embedding before inference using scMCP which will help to load faster.

The scMCP must conduct the following process before prediction:
"**Gene set preprocessing**", relevant files can be found in data/gene_list to reserve the co genes of 17911 in unseen datasets.

"**Knowledge emmbedding**", relevant files can be found in embeddings. PPI and GRN embedding are save in embeddings/grn and embedding/ppi Gene graph networks for GO pathway embedding can be found in models/graph_networks.py

"**Molecule emmbedding**", relevant files can be found in drugs.

### Training scMCP
Before training, preprocess adata to shard is needed which can be found in preprocess_data_npy.ipynb.

Run trainer/run_scMCP_pretraining.py to begin the training process.

### Perturbation prediction based on scMCP

Example about how to predict responses to unseen drugs are saved under sample/inference.py



### Drug screening pipeline based on scMCP
The pipeline is saved in sample/drug_screening.py

### Large scale trained version for scMCP (Will be released soon)
This model is trained using about 1M perturbed cell pairs in Tahoe. A version in larger scale trained wil be released soon.
