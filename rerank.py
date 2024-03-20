import torch

from typing import List

from config import args
from triplet import EntityDict
from dict_hub import get_link_graph
from doc import Example,train_triplet_dict


def rerank_by_ans(batch_score: torch.tensor,
                    examples: List[Example],
                    entity_dict: EntityDict):


    for idx in range(batch_score.size(0)):
        cur_ex = examples[idx]
        gold_neighbor_ids=set()
        rerank_indice=[]
        ans_ids=train_triplet_dict.get_hr_ans(cur_ex.head_id, cur_ex.relation, cur_ex.tail_id)
        for triple in ans_ids: 
            gold_neighbor_ids.add(triple[0])
            gold_neighbor_ids.add(triple[1])
        if len(gold_neighbor_ids) > 10000:
            logger.debug('{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
        for e_id in gold_neighbor_ids:
            if e_id == cur_ex.tail_id:
                continue
            rerank_indice.append(entity_dict.entity_to_idx(e_id))
                
        rerank_indice = torch.LongTensor(rerank_indice).to(batch_score.device)
        rerank_value=torch.tensor([-0.3 for _ in rerank_indice]).to(batch_score.device)
        batch_score[idx].index_add_(0, rerank_indice, rerank_value)