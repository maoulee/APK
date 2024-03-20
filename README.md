## ASPKGC: Analogical Soft Prompts for Knowledge Graph Completion

In this paper,
we identify that one key issue for text-based knowledge graph completion is efficient contrastive learning.
By combining large number of negatives and hardness-aware InfoNCE loss,
SimKGC can substantially outperform existing methods on popular benchmark datasets.

## Requirements
* python>=3.7
* torch>=1.6 (for mixed precision training)
* transformers>=4.15
* lightning >2.0
* networkit >11.0

All experiments are run with 1 A100(40GB) GPUs.

## How to Run

It involves 3 steps: dataset preprocessing, model training, and model evaluation.

We also provide the predictions from our models in [predictions](predictions/) directory.

For WN18RR and FB15k237 datasets, we use files from [KG-BERT](https://github.com/yao8839836/kg-bert).

### WN18RR dataset

Step 1, preprocess the dataset
```
bash scripts/preprocess.sh WN18RR
bash scripts/precode.sh ./data/FB20K/ FB20K
```

Step 2, training the model and (optionally) specify the output directory (< 3 hours)
```
OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn.sh
OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn_src.sh
OUTPUT_DIR=./checkpoint/wn18rr_ke1/ bash scripts/train_wn_src.sh
OUTPUT_DIR=./checkpoint/wn18rr_ori/ bash scripts/train_wn_src.sh
OUTPUT_DIR=./checkpoint/wn18rr_multi/ bash scripts/train_wn.sh


OUTPUT_DIR=./checkpoint/db50k/ bash scripts/train_db.sh
OUTPUT_DIR=./checkpoint/fb20k/ bash scripts/train_fb20.sh
OUTPUT_DIR=./checkpoint/fb15k_owe/ bash scripts/train_fb_owe.sh
 bash scripts/precode.sh ./checkpoint/wn18rr_ori/model_last.mdl WN18RR
```
OUTPUT_DIR=./checkpoint/dbpedia/ bash scripts/train_db.sh
Step 3, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/wn18rr/model_last.mdl WN18RR
bash scripts/eval.sh ./checkpoint/wn18rr_ke1/model_last.mdl WN18RR
bash scripts/eval.sh ./checkpoint/wn18rr_ori/model_last.mdl WN18RR
bash scripts/eval_ori.sh ./checkpoint/wn18rr_ori/model_last.mdl WN18RR
bash scripts/eval.sh ./checkpoint/wn18rr_ke/model_last.mdl WN18RR
bash scripts/eval.sh ./checkpoint/wn18rr_multi/model_last.mdl WN18RR
```
bash scripts/eval.sh ./checkpoint/db50k/model_last.mdl DBPEDIA50K
bash scripts/eval.sh ./checkpoint/fb20k/model_last.mdl FB20K
bash scripts/eval.sh ./checkpoint/fb15k_owe/model_last.mdl FB15K237OWE
Feel free to change the output directory to any path you think appropriate.

### FB15k-237 dataset

Step 1, preprocess the dataset
```
bash scripts/preprocess.sh FB15k237
```

Step 2, training the model and (optionally) specify the output directory (< 3 hours)

```
OUTPUT_DIR=./checkpoint/fb15k237/ bash scripts/train_fb.sh
OUTPUT_DIR=./checkpoint/fb15k237_ke/ bash scripts/train_fb.sh

```

Step 3, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/fb15k237/model_last.mdl FB15k237
bash scripts/eval.sh ./checkpoint/fb15k237_ke/model_last.mdl FB15k237
```

### Wikidata5M transductive dataset

Step 0, download the dataset. 
We provide a script to download the [Wikidata5M dataset](https://deepgraphlearning.github.io/project/wikidata5m) from its official website.
This will download data for both transductive and inductive settings.
```
bash ./scripts/download_wikidata5m.sh
```

Step 1, preprocess the dataset
```
bash scripts/preprocess.sh wiki5m_trans
```

Step 2, training the model and (optionally) specify the output directory (about 12 hours)
```
OUTPUT_DIR=./checkpoint/wiki5m_trans/ bash scripts/train_wiki.sh wiki5m_trans
```

Step 3, evaluate a trained model (it takes about 1 hour due to the large number of entities)
```
bash scripts/eval_wiki5m_trans.sh ./checkpoint/wiki5m_trans/model_last.mdl
```

### Wikidata5M inductive dataset

Make sure you have run `scripts/download_wikidata5m.sh` to download Wikidata5M dataset.

Step 1, preprocess the dataset
```
bash scripts/preprocess.sh wiki5m_ind
```

Step 2, training the model and (optionally) specify the output directory (about 11 hours)
```
OUTPUT_DIR=./checkpoint/wiki5m_ind/ bash scripts/train_wiki.sh wiki5m_ind
```

Step 3, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/wiki5m_ind/model_last.mdl wiki5m_ind
```

## Citation

If you find our paper or code repository helpful, please consider citing as follows: