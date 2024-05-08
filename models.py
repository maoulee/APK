from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, SequentialSampler
from dataclasses import dataclass
from transformers import AutoModel, AutoConfig
from config import args
from bert_prompt import BertModelForLayerwise
from analogical_prompter import Prompter

def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    hr_logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    loss: torch.tensor


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.config.checkpoint=True
        self.log_inv_t = torch.nn.Parameter(torch.tensor([args.t]), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.hr_bert = BertModelForLayerwise.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)
        self.hr_bert.gradient_checkpointing_enable()
        self.tail_bert.gradient_checkpointing_enable()
        self.prompter=Prompter(self.config,self.config.hidden_size,self.args.ans)


        

  

    def _encode_ori(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        cls_output = _pool_output( mask, last_hidden_state)
        return cls_output
    
    
   
    
    def _encode_rel(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)
        dim_idx, mask_idx = (token_ids == torch.tensor([101]).long().cuda()).nonzero(as_tuple=True)
        last_hidden_state = outputs.last_hidden_state
        slices =[last_hidden_state[i, idx, :] for i, idx in enumerate(mask_idx)]
        mask_output = torch.stack(slices, dim=0)
        output =nn.functional.normalize(mask_output.squeeze(),dim=1)
        cls_output = _pool_output( mask, last_hidden_state)
        return output,cls_output
    
   
    
    def anoalogy_agg_fuc(self,anoalogy_rel,ans_value):

        agg_rel=anoalogy_rel.reshape(ans_value.shape[0],-1,768)
        ans_value=ans_value.unsqueeze(-1).cuda()    
        weighted_tensor = agg_rel * ans_value
        weighted_tensor=weighted_tensor.sum(dim=1)
        return weighted_tensor

    
    def _encode_an(self, encoder, token_ids=None, mask=None, token_type_ids=None, input_emb=None):
        prompt_attention_mask = torch.ones(input_emb.size(0), self.args.ans).type_as(mask)
       # prompt_token_type_ids = torch.ones(input_emb.size(0), 10).type_as(mask)
        mask = torch.cat((mask,prompt_attention_mask), dim=1)
       # token_type_ids= torch.cat((token_type_ids,prompt_token_type_ids), dim=1)
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          layerwise_prompt=input_emb,
                              return_dict=True)
        
            
        last_hidden_state = outputs.last_hidden_state
        cls_output = _pool_output( mask, last_hidden_state)
        return cls_output
    
    
 

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                ans_triple_token_ids,ans_triple_mask,ans_triple_token_type_ids,
                ans_head_token_ids,ans_head_mask,ans_head_token_type_ids,
                ans_tail_token_ids,ans_tail_mask,ans_tail_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        
        if only_ent_embedding:
            tail_vector = self._encode_ori(self.tail_bert,
                                           token_ids=tail_token_ids,
                                           mask=tail_mask,
                                           token_type_ids=tail_token_type_ids)



            return {'ent_vectors': tail_vector.detach()
                    }
        
        
        hr_vector = self._encode_ori(self.hr_bert,
                                        token_ids=hr_token_ids,
                                        mask=hr_mask,
                                        token_type_ids=hr_token_type_ids)

        tail_vector = self._encode_ori(self.tail_bert,
                                       token_ids=tail_token_ids,
                                       mask=tail_mask,
                                       token_type_ids=tail_token_type_ids)
        
        if args.ans>0:
            ans_rel,ans_triple=self._encode_rel(self.hr_bert,
                                        token_ids=ans_triple_token_ids,
                                        mask=ans_triple_mask,
                                        token_type_ids=ans_triple_token_type_ids)
            
            
            ans_rel=ans_rel.reshape([hr_vector.size(0),-1,768])
            ans_rel_agg=self.anoalogy_agg_fuc(ans_rel,kwargs['ans_value'])
            ans_rel_agg=ans_rel_agg.float()
            analogical_prompt=self.prompter(ans_rel_agg)
            ans_emb=self._encode_an(self.hr_bert, token_ids=head_token_ids, mask=head_mask, token_type_ids=head_token_type_ids, input_emb=analogical_prompt)

        return {
            'hr_vector':  hr_vector,
            'tail_vector': tail_vector,
            'ans_rel':ans_rel_agg if args.ans>0 else None,
            'ans_emb':ans_emb if args.ans>0 else None

        }

    
    def compute_logits_ans(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector = output_dict['hr_vector']
        tail_vector = output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)
        if args.ans>0:
            ans_vector=output_dict['ans_emb']
            ans_t=ans_vector @ tail_vector.t()
        
        
        hr_t=hr_vector @ tail_vector.t()

        if self.training:
            hr_t= hr_t - torch.zeros(hr_t.size()).fill_diagonal_(self.add_margin).to(
                hr_vector.device)
            
            if self.args.ans>0:
                ans_t= ans_t - torch.zeros(ans_t.size()).fill_diagonal_(self.add_margin).to(
                hr_vector.device)
                ans_t = ans_t * self.log_inv_t

        hr_t = hr_t * self.log_inv_t
        triplet_mask_hr_t = batch_dict.get('triplet_mask', None).to(hr_vector.device)
        hr_t.masked_fill_(~triplet_mask_hr_t, -1e4)
        loss_fn = nn.CrossEntropyLoss().cuda()
        lm_loss=0
        ans_loss=0
        loss=0
    
        ans_t.masked_fill_(~triplet_mask_hr_t, -1e4)
        ans_loss= self.args.lmd*loss_fn(torch.cat([ans_t],dim=1), labels) + loss_fn(ans_t.t(), labels)
        
        lm_loss= loss_fn(hr_t, labels) + loss_fn(hr_t.t(), labels)
        loss=lm_loss+ans_loss
        
        return {'hr_logits': hr_t,
                'labels': labels.long(),
                'inv_t': self.log_inv_t.detach(),
                'loss': loss }



    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:

        tail_ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': tail_ent_vectors.detach()
               }
    
    
def _pool_output(mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:

    input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
    output_vector = sum_embeddings / sum_mask
    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
