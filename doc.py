import os
import json
import torch
import torch.utils.data.dataset
import math
from typing import Optional, List
import random
import pdb
from config import args
from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_ans_mask, construct_mask_backward, entity_dict, train_triplet_dict
from dict_hub import get_link_graph, get_tokenizer
from logger_config import logger

if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True)
    return encoded_inputs


def _custom_tokenize_ans(text_a: str, text_b: Optional[str] = None,text_c: Optional[str] = None,text_d: Optional[str] = None
                         ) -> dict:
    tokenizer = get_tokenizer()
    token_a = tokenizer.tokenize(text=text_a)[0:args.max_num_tokens]
    token_b = None
    token_c = None
    token_d = None
    if text_b != None:
        token_b = tokenizer.tokenize(text=text_b)

    if text_c!=None:
        token_c = tokenizer.tokenize(text=text_c)[0:args.max_num_tokens]
        
    if text_d!=None:
        token_d = tokenizer.tokenize(text=text_d)[0:args.max_num_tokens]
        
    '''
        
    if text_b == None and text_c!=None:
        token = [tokenizer.cls_token] + token_a + [tokenizer.sep_token]+[tokenizer.mask_token]+ [tokenizer.sep_token] +token_c + [tokenizer.sep_token]
        token_type_ids = [0] * (len(token_a) + 2) + [1] * 2+[0]*(len(token_c) +1)
    '''    
    
        
    if text_b == None and text_c!=None:
        token = [tokenizer.cls_token] + token_a + [tokenizer.sep_token] +token_c + [tokenizer.sep_token]
        token_type_ids = [0] * (len(token_a) + 2) + [1]*(len(token_c) +1)


    if text_b != None and text_c==None:
        token = [tokenizer.cls_token] + token_a + [tokenizer.sep_token]+token_b + [tokenizer.sep_token]
        token_type_ids = [0] * (len(token_a) + 2) + [1] * (len(token_b) +1)


    if token_b==None and token_c==None:
        token = [tokenizer.cls_token] + token_a + [tokenizer.sep_token]
        token_type_ids = [0] * (len(token_a) + 2)
        
        
    if token_b==None and token_c==None and token_d!=None:
        token = [tokenizer.cls_token] + token_a + [tokenizer.sep_token]+[tokenizer.mask_token]+[tokenizer.sep_token]
        token_type_ids = [0] * (len(token_a) + 2)+[1]*2

    
    
    input_ids = tokenizer.convert_tokens_to_ids(token)

    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids
    }



def _custom_tokenize_mask(text_a: str,text_b:Optional[str] = None,
                          text_c: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    token = None
    if text_pair != None:
        max_len = args.max_num_tokens - len(text_pair) - 4
        token_entity = tokenizer.tokenize(text=text)[0:max_len]
        token_pair_rel = tokenizer.tokenize(text=text_pair)
        token = [tokenizer.mask_token] + token_entity + [tokenizer.sep_token] + [
            tokenizer.mask_token] + token_pair_rel + [
                    tokenizer.sep_token]
        token_type_ids = [0] * (len(token_entity) + 2) + [1] * (len(token_pair_rel) + 2)
    else:
        max_len = args.max_num_tokens - 2
        token_entity = tokenizer.tokenize(text=text)[0:max_len]
        token = [tokenizer.mask_token] + token_entity + [tokenizer.sep_token]
        token_type_ids = [0] * (len(token_entity) + 2)
    input_ids = tokenizer.convert_tokens_to_ids(token)

    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids
    }


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    return ' '.join(entities)


class Example:

    def __init__(self, head_id, relation, tail_id, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize(self) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        ans_head = []
        ans_tail = []
        ans_triple=[]
        ans_value=[]
        ans_id=[]
        
        if args.use_link_graph:
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)
        

        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)
        tail_word = _parse_entity_name(self.tail)
        tail_text = _concat_name_desc(tail_word, tail_desc)
        hr_encoded_inputs = _custom_tokenize_ans(text_a=head_text, text_b=self.relation)
        head_encoded_inputs = _custom_tokenize_ans(text_a=head_text)
        tail_encoded_inputs = _custom_tokenize_ans(text_a=tail_text)

        if self.relation != '' and args.ans>0:
            queary_ans = train_triplet_dict.get_hr_ans(self.head_id, self.relation, self.tail_id)
            queary_ans = queary_ans[:10]
            if args.is_test:
                queary_ans_triple = queary_ans[:args.ans]
            else:
                #random.shuffle(queary_ans)
                queary_ans_triple = queary_ans[:args.ans]

            
            
            ans_head.append(_custom_tokenize_ans(text_a=head_text,text_d='mask'))

            for idx, answer in enumerate(queary_ans_triple):
                ans_head_des = entity_dict.get_entity_by_id(answer[0]).entity_desc
                ans_tail_des = entity_dict.get_entity_by_id(answer[1]).entity_desc
                if args.use_link_graph:
                    if len(head_desc.split()) < 20:
                        head_desc += ' ' + get_neighbor_desc(head_id=answer[0], tail_id=answer[1])
                    if len(tail_desc.split()) < 20:
                        tail_desc += ' ' + get_neighbor_desc(head_id=answer[1], tail_id=answer[0])
                ans_head_word = _parse_entity_name(entity_dict.get_entity_by_id(answer[0]).entity)
                ans_head_text = _concat_name_desc(ans_head_des, ans_head_word)
                ans_tail_word = _parse_entity_name(entity_dict.get_entity_by_id(answer[1]).entity)
                ans_tail_text = _concat_name_desc(ans_tail_des, ans_tail_word)
                ans_triple.append(_custom_tokenize_ans(text_a=ans_head_text,text_c=ans_tail_text))
                ans_tail.append(_custom_tokenize_ans(text_a=ans_tail_text))
                ans_value.append(answer[2])
                ans_id.append(answer[1])

            ans_value=torch.tensor(ans_value)
            softmax_values =torch.nn.functional.softmax(ans_value, dim=0)


        return {'hr_token_ids': hr_encoded_inputs['input_ids'],
                'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
                'tail_token_ids': tail_encoded_inputs['input_ids'],
                'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
                'head_token_ids': head_encoded_inputs['input_ids'],
                'head_token_type_ids': head_encoded_inputs['token_type_ids'],
                'ans_triple_token_ids': [inputs['input_ids'] for inputs in
                                       ans_triple] if self.relation != '' and args.ans>0 else None,
                'ans_triple_token_type_ids': [inputs['token_type_ids'] for inputs in
                                            ans_triple] if self.relation != '' and args.ans>0  else None,
                'ans_head_token_ids': [inputs['input_ids'] for inputs in
                                       ans_head] if self.relation != '' and args.ans>0  else None,
                'ans_head_token_type_ids': [inputs['token_type_ids'] for inputs in
                                            ans_head] if self.relation != '' and args.ans>0  else None,
                'ans_tail_token_ids': [inputs['input_ids'] for inputs in
                                       ans_tail] if self.relation != '' and args.ans>0 else None,
                'ans_tail_token_type_ids': [inputs['token_type_ids'] for inputs in
                                            ans_tail] if self.relation != '' and args.ans>0  else None,
                'ans_value':softmax_values.reshape(-1,args.ans) if self.relation != '' and args.ans>0 else None,
                'ans_id':ans_id if self.relation != '' and args.ans>0 else None,
                'obj': self}



class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, path, task, examples=None):
        self.path_list = path.split(',')
        self.task = task
        self.num_ent=len(train_triplet_dict.entity_dict)
        self.num_rel=len(train_triplet_dict.relation_dict)
        assert all(os.path.exists(path) for path in self.path_list) or examples

        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))
        print(len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize()


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))
    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))
    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None
    return examples


def collate(batch_data: List[dict]) -> dict:
    hr_token_ids, hr_mask = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
        need_mask=False)

    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False)

    head_token_ids, head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
        need_mask=False)

    if batch_data[0]['ans_head_token_ids'] != None:
        ans_head_token_ids, ans_head_mask = to_indices_and_mask(
            [torch.LongTensor(ans_tail_token_ids) for ex in batch_data
             for ans_tail_token_ids in ex['ans_head_token_ids']],
            pad_token_id=get_tokenizer().pad_token_id)
        ans_head_token_type_ids = to_indices_and_mask(
            [torch.LongTensor(ans_tail_token_type_ids) for ex in batch_data
             for ans_tail_token_type_ids in ex['ans_head_token_type_ids']],
            need_mask=False)

    if batch_data[0]['ans_tail_token_ids'] != None:
        ans_tail_token_ids, ans_tail_mask = to_indices_and_mask(
            [torch.LongTensor(ans_tail_token_ids) for ex in batch_data
             for ans_tail_token_ids in ex['ans_tail_token_ids']],
            pad_token_id=get_tokenizer().pad_token_id)
        ans_tail_token_type_ids = to_indices_and_mask(
            [torch.LongTensor(ans_tail_token_type_ids) for ex in batch_data
             for ans_tail_token_type_ids in ex['ans_tail_token_type_ids']],
            need_mask=False)
        
    if batch_data[0]['ans_triple_token_ids'] != None:
        ans_triple_token_ids, ans_triple_mask = to_indices_and_mask(
            [torch.LongTensor(ans_tail_token_ids) for ex in batch_data
             for ans_tail_token_ids in ex['ans_triple_token_ids']],
            pad_token_id=get_tokenizer().pad_token_id)
        ans_triple_token_type_ids = to_indices_and_mask(
            [torch.LongTensor(ans_tail_token_type_ids) for ex in batch_data
             for ans_tail_token_type_ids in ex['ans_triple_token_type_ids']],
            need_mask=False)

    if batch_data[0]['obj'].relation != '':
        rel_idx=torch.tensor([train_triplet_dict.relation_dict[ex['obj'].relation ] for ex in batch_data]).long()
    
    if batch_data[0]['ans_value'] != None:
        ans_id=[ex['ans_id'] for ex in batch_data]
        
    if batch_data[0]['ans_value'] != None:
        ans_value=torch.cat([ex['ans_value'] for ex in batch_data],dim=0)
        
    batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'ans_triple_token_ids': ans_triple_token_ids if batch_data[0][
                                                        'ans_triple_token_ids'] != None else None,
        'ans_triple_mask': ans_triple_mask if batch_data[0][
                                              'ans_triple_token_ids'] != None else None,
        'ans_triple_token_type_ids': ans_triple_token_type_ids if batch_data[0][
                                                                  'ans_triple_token_ids'] != None else None,
        'ans_head_token_ids': ans_head_token_ids if batch_data[0][
                                                        'ans_head_token_ids'] != None else None,
        'ans_head_mask': ans_head_mask if batch_data[0][
                                              'ans_head_token_ids'] != None else None,
        'ans_head_token_type_ids': ans_head_token_type_ids if batch_data[0][
                                                                  'ans_head_token_ids'] != None else None,
        'ans_tail_token_ids': ans_tail_token_ids if batch_data[0][
                                                        'ans_head_token_ids'] != None else None,
        'ans_tail_mask': ans_tail_mask if batch_data[0][
                                              'ans_tail_token_ids'] != None else None,
        'ans_tail_token_type_ids': ans_tail_token_type_ids if batch_data[0][
                                                                  'ans_tail_token_ids'] != None else None,
        'batch_data': batch_exs,
        'rel_idx':rel_idx if batch_data[0]['obj'].relation != '' else None,
        'ans_value':ans_value if batch_data[0]['ans_value'] != None else None,
        'triplet_mask': construct_mask(row_exs=batch_exs, forward=True) if not args.is_test else None,
        #'ans_mask':construct_self_ans_mask(row_exs=batch_exs,col_exs=ans_id ) if not args.is_test else None,
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices
