#!/usr/bin/env bash
set -x
set -e

model_path="bert"
task="WN18RR"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    model_path=$1
    shift
fi
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    task=$1
    shift
fi

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${task}"
fi

neighbor_weight=0.1
rerank_n_hop=2
if [ "${task}" = "WN18RR" ]; then
# WordNet is a sparse graph, use more neighbors for re-rank
  rerank_n_hop=2
fi
if [ "${task}" = "wiki5m_ind" ]; then
# for inductive setting of wiki5m, test nodes never appear in the training set
  neighbor_weight=0.0
fi

python3 -u precode.py \
--task "${task}" \
--is-test \
--precode \
--eval-model-path "${model_path}" \
--pretrained-model /E22101006/bert-base-uncased/ \
--train-path "/E22101006/SimKGC-main/SimKGC-main/data/${task}/train.txt.json" \
--valid-path "/E22101006/SimKGC-main/SimKGC-main/data/${task}/test.txt.json" \
--batch-size 1024 \
--print-freq 100 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--pre-batch 0 \
--finetune-t \
--epochs 50 \
--workers 4 \
--max-to-keep 3  "$@"
