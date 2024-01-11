# Project Directory Structure

## 1. Directory Structure
- `data`: Dataset
  - `test`: Test set can be obtained from CDCS.
    - `solidity`: Solidity test set.
    - `sql`: SQL test set.
  - `train_valid`: Training and validation sets.
    - `java`: Training set of the semantically closest programming language, obtained from codesearchnet.
    - `solidity`: Solidity training and validation set.
    - `sql`: SQL training and validation set.

- `graph_code_bert`: Code pre-training model (needs to be downloaded from huggingface and saved in this directory).

- `results`: Inference results of the model.
- `save_model`: Saved trained models.
- `src`: Source code
  - Main program: `main_run.py`

## 2. Execution
- Run `src/adapter_spos/main_run.py`.
- Note: Data must be prepared in advance. Large datasets cannot be uploaded due to GitHub restrictions. Instructions for downloading and processing are provided.
