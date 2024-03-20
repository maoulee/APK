import torch

from typing import List
from config import args
from dict_hub import get_train_triplet_dict, get_entity_dict, EntityDict, TripletDict
from logger_config import logger

entity_dict = get_entity_dict() 
train_triplet_dict = get_train_triplet_dict() 

def construct_mask(row_exs: List, col_exs: List = None,forward=True) -> torch.tensor:
    positive_on_diagonal = col_exs is None
    num_row = len(row_exs)
    col_exs = row_exs if col_exs is None else col_exs
    num_col = len(col_exs)

    # exact match
    row_entity_ids = torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in row_exs])
    col_entity_ids = row_entity_ids if positive_on_diagonal else \
        torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in col_exs])
    # num_row x num_col
    triplet_mask = (row_entity_ids.unsqueeze(1) != col_entity_ids.unsqueeze(0))
    if positive_on_diagonal:
        triplet_mask.fill_diagonal_(True)

    # mask out other possible neighbors
    for i in range(num_row):
        head_id, relation = row_exs[i].head_id, row_exs[i].relation  
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation) 
        ans_ids=train_triplet_dict.get_hr_ans(head_id, relation, row_exs[i].tail_id)
        for triple in ans_ids: 
            neighbor_ids.add(triple[0])
            neighbor_ids.add(triple[1])
        # exact match is enough, no further check needed
        if len(neighbor_ids) <= 1:
            continue

        for j in range(num_col):
            if i == j and positive_on_diagonal:
                continue
            if forward:
                  tail_id = col_exs[j].tail_id
            else:
                  tail_id = col_exs[j].head_id
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False

    return triplet_mask
    
def construct_mask_backward(row_exs: List, col_exs: List = None,forward=True) -> torch.tensor:
    num_row = len(row_exs)
    col_exs = row_exs if col_exs is None else col_exs
    num_col = len(col_exs)

    triplet_mask = torch.ones((num_row,num_col),dtype=torch.bool,requires_grad=False)

    # mask out other possible neighbors
    for i in range(num_row):
        head_id, relation = row_exs[i].head_id, row_exs[i].relation  
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation) 
        ans_ids=train_triplet_dict.get_hr_ans(head_id, relation, row_exs[i].tail_id)
        for triple in ans_ids: 
            neighbor_ids.add(triple[0])
            neighbor_ids.add(triple[1])
            
        # exact match is enough, no further check needed
        if len(neighbor_ids) <= 1:
            continue

        for j in range(num_col):
            if forward:
                tail_id = col_exs[j].head_id
            else:
                tail_id = col_exs[j].tail_id
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False

    return triplet_mask


def construct_mask_neg(row_exs: List, col_exs: List = None) -> torch.tensor:
    num_row = len(row_exs)
    col_exs = row_exs if col_exs is None else col_exs
    num_col = len(col_exs)

    triplet_mask = torch.ones((num_row, num_col), dtype=torch.bool, requires_grad=False)

    # mask out other possible neighbors
    for i in range(num_row):

        head_id, relation = row_exs[i].head_id, row_exs[i].relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        '''
        ans_ids=train_triplet_dict.get_hr_ans(head_id, relation, row_exs[i].tail_id)[:5]
        for triple in ans_ids: 
            neighbor_ids.add(triple[0])
            neighbor_ids.add(triple[1])
        '''
        # exact match is enough, no further check needed
        if len(neighbor_ids) <= 1:
            continue
        for j in range(num_col):
            tail_id = col_exs[j]
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False

    return triplet_mask

def construct_self_negative_mask(exs: List,forward=True) -> torch.tensor:
    mask = torch.ones(len(exs))
    for idx, ex in enumerate(exs):
        if forward:
          head_id, relation = ex.head_id, ex.relation
        else:
          head_id, relation = ex.tail_id, ex.lm_relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        if head_id in neighbor_ids:
            mask[idx] = 0
    return mask.bool()



def construct_self_ans_mask(row_exs: List,col_exs: List) -> torch.tensor:
    num_row = len(row_exs)
    col_exs = row_exs if col_exs is None else col_exs
    num_col = len(col_exs[0])

    triplet_mask = torch.ones((num_row, num_col), dtype=torch.bool, requires_grad=False)

    # mask out other possible neighbors
    for i in range(num_row):

        head_id, relation = row_exs[i].head_id, row_exs[i].relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        # exact match is enough, no further check needed
        if len(neighbor_ids) <= 1:
            continue
        for j in range(num_col):
            tail_id = col_exs[i][j]
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False

    return triplet_mask

def construct_self_ans_mask(row_exs: List,col_exs: List) -> torch.tensor:
    num_row = len(row_exs)
    col_exs = row_exs if col_exs is None else col_exs
    num_col = len(col_exs[0])

    triplet_mask = torch.ones((num_row, num_col), dtype=torch.bool, requires_grad=False)

    # mask out other possible neighbors
    for i in range(num_row):

        head_id, relation = row_exs[i].head_id, row_exs[i].relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        '''
        ans_ids=train_triplet_dict.get_hr_ans(head_id, relation, row_exs[i].tail_id)[:5]
        for triple in ans_ids: 
            neighbor_ids.add(triple[0])
            neighbor_ids.add(triple[1])
        '''
        # exact match is enough, no further check needed
        if len(neighbor_ids) <= 1:
            continue
        for j in range(num_col):
            tail_id = col_exs[i][j]
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False

    return triplet_mask