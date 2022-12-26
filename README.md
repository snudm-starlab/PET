# PET (Parameter-Efficient Knowledge Distillation on Transformer)
This project is a PyTorch implementation of PET (Parameter-Efficient Knowledge Distillation on Transformer). PET proposes a novel approach that compresses transformer both the encoder and decoder, improving the performance of knowledge distillation(KD) on Transformer.

## Overview
#### Brief Explanation of PET. 
PET improves an overall process of Transformer KD as follows:

#### 1. Defining Parameter-Efficient Architecture of the Transformer Encoder and Decoder

We find a replaceable pairs of modules in each encoder and decoder. 
Replaceable pair indicates that paired modules are robust to weight-sharing.

#### 2. Pre-training the Compressed Model with Simplified Task

To initialize a compressed model that well adapts to challenging tasks, 
we pre-train the model with a simplified version of the original task, less challenging enough for its reduced sizes.
We propose a method to improve the performance of pre-training task by modeling the predictions of the model.

#### 3. Layer-wise Attention Head Sampling 

For further optimization, we train a wider compressed model, which has more attention heads than our compression target,
and sample an efficient attention heads by layers.

#### Code Description
This repository is written based on the codes in [FAIRSEQ](https://github.com/facebookresearch/fairseq).
Here's an overview of our codes.

``` Unicode
PET
  │
  ├──  src   
  │     ├── criterions
  │     │    └── pet_cross_entropy.py: customized criterion for PET
  │     ├── models
  │     │    └── pet_transformer.py: the implementation of the PET models
  │     ├── modules
  │     │    ├── pet_multihead_attention.py: multihead attention for PET
  │     │    └── pet_transformer_layer.py : layers of encoder and decoder of PET
  │     │    
  │     ├── pet_train.py : codes for training a new model 
  │     └── pet_trainer.py : codes for managing training process 
  │     
  │     
  ├──  scripts
  │     ├── iwslt_preprocess.sh: a script for downloading and preprocess iwslt14
  │     ├── prepare-iwslt14.sh : a script for preparing iwslt14 which is run by iwslt_preprocess.sh
  │     ├── iwslt_pet_test.sh: a script for testing the trained model
  │     └── demo.sh : a script for running demo  
  │     
  ├──  data-bin : a directory for saving datasets
  ├──  checkpoints : a directory for saving checkpoints 
  │  
  ├── Makefile
  ├── LICENSE
  └── README.md

```

## Install 

#### Environment 
* Ubuntu
* CUDA 11.6
* numpy 1.23.4
* torch 1.12.1
* sacrebleu 2.0.0
* pandas 

## Installing Requirements
Install [PyTorch](http://pytorch.org/) (version >= 1.5.0) and install fairseq with following instructions:
```
git clone https://github.com/pytorch/fairseq 
cd fairseq
git reset --hard 0b54d9fb2e42c2f40db3449ca34586952b8abe94
pip install --editable ./
pip install sacremoses
```

# Getting Started

## Preprocess
Download IWSLT'14 German to English dataset by running script:
```
cd scripts
bash iwslt_preprocess.sh
```

## Demo 
you can run the demo version.
```
make
```

## Run Your Own Training
* We provide scripts for training and testing.
Followings are key arguments:
    * arch: architecture type
    * compression-rate: target compreesion rate
    * is-SPS: whether use PET or not
    * save-dir: path for saving checkpoints
    * pretrained-model: path for pre-trained model to be pruned
    

* First, we begin with training a Transformer model
```
CUDA_VISIBLE_DEVICES=0 python ../src/train.py \
    ../data-bin/iwslt14.tokenized.de-en \
    --user-dir ../src \
    --arch pet_iwslt_de_en --share-decoder-input-output-embed \
    --task translation \
    --optimizer srp_adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion srp --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ../checkpoints/base \
```
This code is also saved in scripts/iwslt_pet.sh

## Reference
* FAIRSEQ: https://github.com/facebookresearch/fairseq
