import json
import os
import shutil
import time
from dataclasses import dataclass, asdict
from typing import List
from transformers import AutoModel, AutoConfig
import torch
import tqdm
import faiss
from config import args
from collections import OrderedDict

from dict_hub import get_entity_dict, get_train_triplet_dict, get_link_graph, get_all_triplet_dict
from logger_config import logger
from predict import BertPredictor
from triplet import EntityDict
from utils import AttrDict, move_to_cuda
from models import build_model, ModelOutput
from doc import collate, Example, Dataset
from  collections import defaultdict
import numpy as np


def _setup_entity_dict() -> EntityDict:
    if args.task == 'wiki5m_ind':
        return EntityDict(entity_dict_dir=os.path.dirname(args.valid_path),
                          inductive_test_path=args.valid_path)
    return get_entity_dict()


entity_dict = _setup_entity_dict()

def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))
    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))
    cnt = len(data)
    relation=open(os.path.join(os.path.dirname(path), 'relations.dict'), encoding='utf-8') 
    for relation_num,relation in enumerate(relation):
        rid, _ = relation.strip().split('\t')
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None
    return examples

@torch.no_grad()
def predict_by_entities(model, entity_exs) -> torch.tensor:
    examples = []
    for entity_ex in entity_exs:
        examples.append(Example(head_id='', relation='', tail_id=entity_ex.entity_id))

    dataset = Dataset(path='', examples=examples, task=args.task)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=max(args.batch_size, 256),
        collate_fn=collate,
        shuffle=False)

    ent_tensor_list = []
    for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
        batch_dict['only_ent_embedding'] = True
        batch_dict = move_to_cuda(batch_dict)
        outputs = model(**batch_dict)
        ent_vec=outputs['ent_vectors']
        ent_tensor_list.append(ent_vec)
    return torch.cat(ent_tensor_list, dim=0)


@torch.no_grad()
def predict_by_relation(model, entity_exs) -> torch.tensor:
    examples = []
    for entity_ex in entity_exs:
        examples.append(Example(head_id='', relation='', tail_id=entity_ex.entity_id))

    dataset = Dataset(path='', examples=examples, task=args.task)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=max(args.batch_size, 256),
        collate_fn=collate,
        shuffle=False)

    ent_tensor_list = []
    for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
        batch_dict['only_ent_embedding'] = True
        batch_dict = move_to_cuda(batch_dict)
        outputs = model(**batch_dict)
        ent_vec=outputs['ent_vectors']
        ent_tensor_list.append(ent_vec)
    return torch.cat(ent_tensor_list, dim=0)



@torch.no_grad()
def predict_by_examples(model, examples: List[Example]):
    dataset = Dataset(path='', examples=examples, task=args.task)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=max(args.batch_size, 256),
        collate_fn=collate,
        shuffle=False)

    hr_tensor_list, tail_tensor_list = [], []
    for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
        batch_dict['only_hr_embedding'] = True
        batch_dict = move_to_cuda(batch_dict)
        outputs = model(**batch_dict)
        hr_tensor_list.append(outputs['ent_vectors'])
        
    return torch.cat(hr_tensor_list, dim=0)

def update_ent_emb(model, args):
    hr_tensor = predict_by_entities(model, entity_dict.entity_exs)
    hr_tensor=hr_tensor.detach().cpu().numpy()
    np.save(os.path.join(os.path.dirname(args.train_path), 'ent'),hr_tensor)
    
def update_rel_emb(model, args):
    hr_tensor = predict_by_entities(model, entity_dict.entity_exs)
    hr_tensor=hr_tensor.detach().cpu().numpy()
    np.save(os.path.join(os.path.dirname(args.train_path), 'ent'),hr_tensor)



if __name__ == '__main__':
    if args.task.lower()=='fb20k':
        args.num_rel=2678
        args.num_ent=19923
        
    if args.task.lower()=='fb15k237owe':
        args.num_rel=470
        args.num_ent=14515 
        
    if args.task.lower()=='wn18rr':
        args.num_rel=11
        args.num_ent=40493 
    if args.task.lower()=='fb15k237':
        args.num_rel=237
        args.num_ent=14514  
    model = build_model(args)
    update_ent_emb(model.cuda(), args)





