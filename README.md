## APK： Analogical Enhanced Pre-trained Model for Knowledge Graph Completion
## Requirements
* python>=3.7
* torch>=1.6 (for mixed precision training)
* transformers>=4.15
* lightning >2.0
* networkit >11.0

All experiments are run with 1 A100(40GB) GPUs.

## How to Run

It involves 3 steps: dataset preprocessing, model training, and model evaluation.

For WN18RR and FB15k237 datasets, we use files from [KG-BERT](https://github.com/yao8839836/kg-bert).

For DBPEDIA50k and FB20K datasets, we use files from [Conmask](https://github.com/bxshi/ConMask).

### WN18RR dataset

Step 1, preprocess the dataset (Process training data and encode the enitity through the PLMS without any finetune. We have provided processed training data in the data folder.)
```
bash scripts/preprocess.sh WN18RR 
bash scripts/precode.sh ./data/WN18RR/ WN18RR 
```

Step 2, training the model and (optionally) specify the output directory 
```
OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn.sh
```

Step 3, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/wn18rr/model_last.mdl WN18RR
```
Feel free to change the output directory to any path you think appropriate.

### FB15k-237 dataset

Step 1, preprocess the dataset (Process training data and encode the enitity through the PLMS without any finetune. We have provided processed training data in the data folder.)

```
bash scripts/preprocess.sh FB15K237  
bash scripts/precode.sh ./data/FB15K237/ FB15K237  
```

Step 2, training the model and (optionally) specify the output directory 
```
OUTPUT_DIR=./checkpoint/FB15K237/ bash scripts/train_fb.sh
```

Step 3, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/FB15K237/model_last.mdl FB15K237
```

### FB20K dataset

Step 1, preprocess the dataset (Process training data and encode the enitity through the PLMS without any finetune. We have provided processed training data in the data folder.)

```
bash scripts/preprocess.sh FB20K  
bash scripts/precode.sh ./data/FB20K/ FB20K  
```

Step 2, training the model and (optionally) specify the output directory 
```
OUTPUT_DIR=./checkpoint/FB20K/ bash scripts/train_fb20.sh
```

Step 3, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/FB20K/model_last.mdl FB20K
```

### DBPEDIA50K dataset

Step 1, preprocess the dataset (Process training data and encode the enitity through the PLMS without any finetune. We have provided processed training data in the data folder.)

```
bash scripts/preprocess.sh DBPEDIA50k  
bash scripts/precode.sh ./data/DBPEDIA50k/ DBPEDIA50k  
```

Step 2, training the model and (optionally) specify the output directory 
```
OUTPUT_DIR=./checkpoint/DBPEDIA50k/ bash scripts/train_db.sh
```

Step 3, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/DBPEDIA50k/model_last.mdl DBPEDIA50k
```

