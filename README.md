# Predicting unseen chemical perturbation effects by integrating multi-source prior knowledge with foundation model
scMCP is a chemical perturbation framework that integrate multi-source biological prior knowledge for drug discovery.
## Overview of scMCP:
<img width="2739" height="4008" alt="Figure1" src="https://github.com/user-attachments/assets/6c974a52-a497-4782-9987-ae312abe00f8" />

## QuickStart:
:satisfied:Requirements are shown in the requirements.txt. Please use the requirements.txt to create the environment for running scMCP.

:satisfied:Please use the molecule fingerprint generated procession to pregenerate the embedding before inference using scMCP which will help to load faster.

:satisfied:Please use the gene_list to filter out the genes that can not be identified by scMCP.
The scMCP must conduct the following process before prediction:
"**Gene set preprocessing**", relevant files can be found in preprocess/gene_list to reserve the co genes of 17911 in unseen datasets.

"**Knowledge emmbedding**", relevant files can be found in embeddings. PPI and GRN embedding are save in embeddings/grn and embedding/ppi Gene graph networks for GO pathway embedding can be found in models/graph_networks.py

"**Molecule emmbedding**", relevant files can be found in drugs.




### Perturbation prediction based on scMCP

Example about how to predict responses to unseen drugs are saved under sample/inference.py



### Drug screening pipeline based on scMCP
The pipeline is saved in sample/drug_screening.py

### Large scale pretraining processing for scMCP (Will be released soon)
This model is pretrained using about 1M perturbed cell pairs in Tahoe. A process for larger scale pretraining wil be released soon.
