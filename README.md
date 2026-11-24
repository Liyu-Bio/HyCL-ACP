# HyCL-ACP: A Hybrid Cross-modal Learning framework Incorporating Semantic and Chemical Insights for Identifying Anticancer Peptides and Their Cancer-type Activities

## Overview

HyCL-ACP is a hybrid cross-modality learning framework for anticancer peptide (ACP) identification and cancer-type activity prediction. It combines peptide sequence semantics, Martini coarse-grained structural topology, and pretrained ESM-2 representations. The repository supports binary ACP classification and multi-label prediction of cancer-type-specific peptide activities.

## Main Components

- **Multi-scale Sequence Context Encoding (MSCE):** combines depthwise convolutions with a Transformer to capture local peptide motifs and long-range sequence dependencies.
- **Coarse-grained Structure Topology Learning (CSTL):** constructs Martini bead-level peptide graphs and learns residue-aligned structural features with GraphSAGE.
- **Cross-modality Information Fusion (CIF):** applies bidirectional cross-attention between sequence and graph representations, followed by attention-based feature fusion.
- **Pretrained Semantic Representation Augmentation (PSRA):** combines frozen ESM-2 embeddings with AAC20 and DPC400 descriptors in a class-balanced RBF-SVM branch.
- **Prior-regularized Integration and Prediction (PPI):** learns validation-based weights for the CIF and PSRA prediction pathways.

## Repository Structure

```text
HyCL-ACP/
├── model.py                 
├── train.py                
├── test.py                  
├── requirements.txt         
├── data/
    ├── binary/ACP-Mixed-80/
    │   ├── train.csv
    │   └── test.csv
    └── multilabel/
        └── CancerPPD.tsv
```

## Requirements

The project was tested with Python 3.10.8 and CUDA 12.1. Install the pinned
dependencies with:

```bash
pip install -r requirements.txt
```

## Datasets
In terms of datasets, three datasets were used in this study:

- The ACP-Mixed-80 is used for the binary classification task, training set contains 484 samples and independent test set contains 122 samples.
  `data/binary/ACP-Mixed-80/train.csv`
  `data/binary/ACP-Mixed-80/test.csv`
- The CancerPPD is used for the multi-label classification task, contains 211 samples across seven cancer types:
  - Blood
  - Breast
  - Cervix
  - Colon
  - Lung
  - Prostate
  - Skin
  
  `data/multilabel/CancerPPD.tsv`.

## Running the Models

**Binary Classification Task**

```bash
python train.py ACP-Mixed-80 \
  --epochs 300 \
  --batch-size 512 \
  --micro-batch-size 512 \
  --esm-batch-size 8
```

**Multi-label Classification Task**

```bash
python train.py CancerPPD \
  --epochs 300 \
  --patience 40 \
  --batch-size 512 \
  --esm-batch-size 8
```

