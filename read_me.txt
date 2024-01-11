1. Directory Structure
   - data: Dataset
     - test: Test set can be obtained from CDCS.
       - solidity: solidity test set
       - sql: sql test set
     - train_valid
       - java: Training set of the semantically closest programming language, obtained from codesearchnet.
       - solidity: solidity training and validation set
       - sql: sql training and validation set

   - graph_code_bert: Code pre-training model. It needs to be downloaded from huggingface and saved in this directory.

   - results: The inference results of the model
   - save_model: Saved trained models
   - src: Source code
       - The main program is main_run.py

2. Run
   Run src/adapter_spos/main_run.py.
   Note: Data must be prepared in advance. Due to GitHub restrictions, large datasets cannot be uploaded. Instructions for downloading and processing are provided.