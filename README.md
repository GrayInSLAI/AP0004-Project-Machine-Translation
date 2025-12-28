# AP0004 Project: Machine-Translation

**NOTE: ** Due to file size limitations, all tokenizers and model checkpoints have not been uploaded. Apart from this, all code, data, and runtime logs related to this project have been uploaded.



#### Python (Code) File structure

+ Related to inference with the best model
  + `inference.py`: interactive translation interface 
+ Related to visual analysis 
  + `analysis.ipynb`: data analysis 
+ Related to baseline tokenization + RNN-based NMT from scratch
  + `data_utils.py`: data cleaning & tokenization
  + `vocab.py`: vocabulary construction
  + `datasets.py`: data modeling
  + `models.py`: RNN modeling
  + `train.py`: word embedding initialization (with pretrained word vector), training & evaluation 
+ Related to Byte-level BPE + RNN-based NMT from scratch 
  + `bpe_vocab.py`: vocabulary construction
  + `bpe_datasets.py`: data modeling
  + `bpe_models.py`: RNN modeling
  + `bpre_train.py`: training & evaluation 
+ Related to Byte-level BPE + Transfomer-based NMT from scratch 
  + `bpe_vocab.py`: vocabulary construction
  + `bpe_datasets.py`: data modeling
  + `transfomer_model.py`: Transformer modeling
  + `transfomer_train.py`: training & evaluation 
+ Related to fine-tuning
  + `mt5_train.py`: fine-tuning & evaluation



#### Log File Structure 

+ Related to baseline tokenization + RNN-based NMT from scratch
  + file with prefix `train_` 
+ Related to Byte-level BPE + RNN-based NMT from scratch 
  + all file with prefix `bpe_train`
+ Related to Byte-level BPE + Transfomer-based NMT from scratch 
  + all file with prefix `transformer_train`
+ Related to fine-tuning
  + file with prefix `mt5_train`