import os
import json
import networkx as nx
import networkit as nk
import numpy as np
import torch
from typing import List
from dataclasses import dataclass
from collections import deque
from config import args
from logger_config import logger
from tqdm import tqdm
import threading 
import math

@dataclass
class EntityExample:
    entity_id: str
    entity: str
    index: int
    entity_desc: str = ''
    
    
class TripletDict_graph:
    def __init__(self, path_list: List[str]):
        self.path_list = path_list
        logger.info('Triplets path: {}'.format(self.path_list))
        self.entity_dict = {}
        self.r_ht = {}

        self.hr2tails = {}
        self.graph = nk.Graph()
        self.relations = set()
        self.relation_dict = {}
        self.examples = None
        self.valid_examples=None
        self.hr_ans = {}
        self.valid_ans={}
        self.ent_emb = {}
        self.distances = None
        self.rel_index_list = []
        self.ent_index_list=[]
        self.graph_dict = {}
        self.entity_json = json.load(
            open(os.path.join(os.path.dirname(args.train_path), 'entities.json'), 'r', encoding='utf-8'))

        self.entity_exs = [EntityExample(**obj) for obj in self.entity_json]
        self.sematic_dis=np.zeros((len(self.entity_json),len(self.entity_json)))

        for path in self.path_list:
            self._load(path)
        logger.info(
            'Triplet statistics: {} relations {} triples'.format(len(self.relations), self.graph.numberOfEdges()))
        
        if args.precode==False:
            self.apsp = nk.distance.APSP(self.graph)
            self.apsp.run()
            self.construc_faiss_index()
            self.compute_shortest_paths(path)

    def _load(self, path: str):
        examples = json.load(open(path, 'r', encoding='utf-8'))
        examples += [reverse_triplet(obj) for obj in examples]
        valid_examples= json.load(open(args.valid_path, 'r', encoding='utf-8'))
        valid_examples += [reverse_triplet(obj) for obj in valid_examples]
        self.examples = examples
        self.valid_examples=valid_examples
        
        for idx,entity in enumerate(self.entity_json):
            self.entity_dict[entity['entity_id']] = entity
            
            
        for idx, ex in enumerate(examples):
            key = (ex['head_id'], ex['relation'])
            key_rel=(ex['head_id'], ex['tail_id'])
            self.relations.add(ex['relation'])
            if key not in self.hr2tails:
                self.hr2tails[key] = set()
                
            if ex['head_id'] not in self.graph_dict:
                self.graph_dict[ex['head_id']] = len(self.graph_dict)
                self.graph.addNode()

            if ex['tail_id'] not in self.graph_dict:
                self.graph_dict[ex['tail_id']] = len(self.graph_dict)
                self.graph.addNode()

            if ex['relation'] not in self.r_ht:
                self.r_ht[ex['relation']] = set()
            

            self.r_ht[ex['relation']].add(key_rel)
            self.graph.addEdge(self.graph_dict[ex['head_id']], self.graph_dict[ex['tail_id']])
            self.hr2tails[key].add(ex['tail_id'])
            self.hr2tails[key].add(ex['head_id'])

            
        for relation in self.r_ht:
            self.r_ht[relation] = list(self.r_ht[relation])

        for idx, relation in enumerate(self.relations):
            self.relation_dict[relation] = idx
            
    @torch.no_grad()
    def construc_faiss_index(self):
        emb_path = os.path.join(os.path.dirname(args.train_path), 'ent.npy')
        self.ent_emb = np.load(emb_path)
        ent_emb_tensor=torch.tensor(self.ent_emb).cuda()
        for i in tqdm(range(0, len(self.entity_json), 256)):
            start_idx = i
            end_idx = min(i + 256, len(self.entity_json))
            batch_embeddings = ent_emb_tensor[start_idx:end_idx]
            similarities = batch_embeddings @ ent_emb_tensor.t()
            self.sematic_dis[start_idx:end_idx] = similarities.cpu().numpy()

        


    def compute_shortest_paths(self, path):
        path = os.path.join(os.path.dirname(path), 'hr_ans.json')
        valid_path= os.path.join(os.path.dirname(path), 'valid_ans.json')
        self.distances = self.apsp.getDistances(asarray=True)
        self.distances[self.distances >args.max_dis] = args.max_dis
        self.distances=1-self.distances/args.max_dis
        if os.path.exists(path):
            self.hr_ans = json.load(open(path, 'r', encoding='utf-8'))        
        else:
            self.find_topk_candidates()
            json.dump(self.hr_ans, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        '''
        if  os.path.exists(valid_path):
            self.valid_ans = json.load(open(valid_path, 'r', encoding='utf-8'))        
        else:
            self.find_topk_candidates_valid()
            json.dump(self.valid_ans, open(valid_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        '''

    def find_topk_candidates(self):
        epoch_iterator = tqdm(self.examples, leave=False, position=0)
        candidates = []  # 用于存储所有候选元素的列表
        for idx, triple in enumerate(epoch_iterator):
            head = triple['head_id']
            tail = triple['tail_id']
            relation = triple['relation']
            head_idx = self.entity_dict[head]['index']
            key = f"{head}-{relation}"
            if key not in self.hr_ans:
                self.hr_ans[key] = []
                head_id = self.graph_dict[head]
                answer_triple = self.r_ht.get(relation, [])
                candidates = [(ex[0],ex[1],self.distances[head_id][self.graph_dict[ex[0]]]+self.sematic_dis[head_idx][self.entity_dict[ex[0]]['index']]) for ex in answer_triple if ex[0]!=head]
                if len(candidates)==0:
                    candidates = [(ex[0],ex[1],self.distances[head_id][self.graph_dict[ex[0]]]+self.sematic_dis[head_idx][self.entity_dict[ex[0]]['index']]) for ex in answer_triple ]
                if len(candidates)<10 and  len(candidates)>0:
                    last=candidates[-1]
                    while len(candidates)<10:
                        candidates.append(last)
                candidates.sort(key=lambda x: x[2], reverse=True)
                self.hr_ans[key] = candidates[:10]
            else:
                continue
                
    def find_topk_candidates_valid(self):
        epoch_iterator = tqdm(self.valid_examples, leave=False, position=0)
        candidates = []  # 用于存储所有候选元素的列表
        for idx, triple in enumerate(epoch_iterator):
            head = triple['head_id']
            tail = triple['tail_id']
            relation = triple['relation']
            head_idx = self.entity_dict[head]['index']
            key = f"{head}-{relation}"
            if key not in self.valid_ans:
                head_idx = self.entity_dict[head]['index']
                answer_triple = self.r_ht.get(relation, [])
                if head in self.graph_dict:
                    head_id = self.graph_dict[head]
                    candidates = [(ex[0],ex[1], self.distances[head_id][self.graph_dict[ex[0]]]+self.sematic_dis[head_idx][self.entity_dict[ex[0]]['index']]) for ex in answer_triple if (ex[0],ex[1]) != (head,tail) ]
                else:
                    candidates = [(ex[0],ex[1],self.sematic_dis[head_idx][self.entity_dict[ex[0]]['index']]) for ex in answer_triple if (ex[0],ex[1]) != (head,tail)]
            
                if len(candidates)==0:
                    candidates.append((head,tail,1,1.0))
                
                if len(candidates)<10 and  len(candidates)>0:
                    last=candidates[-1]
                    while len(candidates)<10:
                        candidates.append(last)
                    
                candidates.sort(key=lambda x: x[2], reverse=True)
                self.valid_ans[key] = candidates[:10]
            else:
                continue

    def get_hr_ans(self, head, relation, tail):
        key = f"{head}-{relation}"
        results= []
        distance=[]
        candidates=[]
        if key not in self.hr_ans and  key not in self.valid_ans:
            head_idx = self.entity_dict[head]['index']
            answer_triple = self.r_ht.get(relation, [])
            if head in self.graph_dict:
                head_id = self.graph_dict[head]
                candidates = [(ex[0],ex[1], self.distances[head_id][self.graph_dict[ex[0]]]+self.sematic_dis[head_idx][self.entity_dict[ex[0]]['index']]) for ex in answer_triple if (ex[0],ex[1]) != (head,tail) ]
            else:
                candidates = [(ex[0],ex[1],self.sematic_dis[head_idx][self.entity_dict[ex[0]]['index']]) for ex in answer_triple if (ex[0],ex[1]) != (head,tail)]
            
            if len(candidates)==0:
                candidates.append((head,tail,1,1.0))
                
            if len(candidates)<10 and  len(candidates)>0:
                last=candidates[-1]
                while len(candidates)<10:
                    candidates.append(last)
                    
            candidates.sort(key=lambda x: x[2], reverse=True)
            self.hr_ans[key] = candidates[:10]
        else: 
            if key in self.hr_ans:
                temp_can=self.hr_ans[key]
            else:
                temp_can=self.valid_ans[key]
            candidates=[ ex for ex in temp_can if (ex[0],ex[1]) != (head,tail)]
            
            if len(candidates)==0:
                candidates.append((head,tail,1,1.0))
                
            if len(candidates)<10:
                last=candidates[-1]
                while len(candidates)<10:
                    candidates.append(last)

        return candidates[:10]
    
    
    
        

    def get_neighbors(self, entity, relation):
        return self.hr2tails.get((entity, relation), set())




class EntityDict:
    def __init__(self, entity_dict_dir: str, inductive_test_path: str = None):
        path = os.path.join(entity_dict_dir, 'entities.json')
        assert os.path.exists(path)
        self.entity_json = json.load(open(path, 'r', encoding='utf-8'))
        for idx, item in enumerate(self.entity_json):
            item['index'] = idx
        self.entity_exs = [EntityExample(**obj) for obj in self.entity_json]

        if inductive_test_path:
            examples = json.load(open(inductive_test_path, 'r', encoding='utf-8'))
            valid_entity_ids = set()
            for ex in examples:
                valid_entity_ids.add(ex['head_id'])
                valid_entity_ids.add(ex['tail_id'])
            self.entity_exs = [ex for ex in self.entity_exs if ex.entity_id in valid_entity_ids]

        self.id2entity = {ex.entity_id: ex for ex in self.entity_exs}
        self.entity2idx = {ex.entity_id: i for i, ex in enumerate(self.entity_exs)}
        logger.info('Load {} entities from {}'.format(len(self.id2entity), path))

    def entity_to_idx(self, entity_id: str) -> int:
        return self.entity2idx[entity_id]

    def get_entity_by_id(self, entity_id: str) -> EntityExample:
        return self.id2entity[entity_id]

    def get_entity_by_idx(self, idx: int) -> EntityExample:
        return self.entity_exs[idx]

    def __len__(self):
        return len(self.entity_exs)
    
class RelationDict:
    def __init__(self, relation_dict_path: str):
        path = os.path.join(entity_dict_dir, 'relations.json')
        assert os.path.exists(path)
        self.rel_json = json.load(open(path, 'r', encoding='utf-8'))
        for idx, item in enumerate(self.rel_json):
            item['index'] = idx
        self.entity_exs = [EntityExample(**obj) for obj in self.entity_json]

        self.id2entity = {ex.entity_id: ex for ex in self.entity_exs}
        self.entity2idx = {ex.entity_id: i for i, ex in enumerate(self.entity_exs)}
        logger.info('Load {} entities from {}'.format(len(self.id2entity), path))
    


class LinkGraph:
    def __init__(self, train_path: str):
        logger.info('Start to build link graph from {}'.format(train_path))
        # id -> set(id)
        self.graph = {}
        examples = json.load(open(train_path, 'r', encoding='utf-8'))
        for ex in examples:
            head_id, tail_id = ex['head_id'], ex['tail_id']
            if head_id not in self.graph:
                self.graph[head_id] = set()
            self.graph[head_id].add(tail_id)
            if tail_id not in self.graph:
                self.graph[tail_id] = set()
            self.graph[tail_id].add(head_id)
        logger.info('Done build link graph with {} nodes'.format(len(self.graph)))

    def get_neighbor_ids(self, entity_id: str, max_to_keep=10) -> List[str]:
        # make sure different calls return the same results
        neighbor_ids = self.graph.get(entity_id, set())
        return sorted(list(neighbor_ids))[:max_to_keep]

    def get_n_hop_entity_indices(self, entity_id: str,
                                 entity_dict: EntityDict,
                                 n_hop: int = 2,
                                 # return empty if exceeds this number
                                 max_nodes: int = 100000) -> set:
        if n_hop < 0:
            return set()

        seen_eids = set()
        seen_eids.add(entity_id)
        queue = deque([entity_id])
        for i in range(n_hop):
            len_q = len(queue)
            for _ in range(len_q):
                tp = queue.popleft()
                for node in self.graph.get(tp, set()):
                    if node not in seen_eids:
                        queue.append(node)
                        seen_eids.add(node)
                        if len(seen_eids) > max_nodes:
                            return set()
        return set([entity_dict.entity_to_idx(e_id) for e_id in seen_eids])

class TripletDict:
    def __init__(self, path_list: List[str]):
        self.path_list = path_list
        logger.info('Triplets path: {}'.format(self.path_list))
        self.relations = set()
        self.hr2tails = {}
        self.triplet_cnt = 0
        self.in_degree = {}
        self.degree = {}
        self.graph = nx.DiGraph()
        for path in self.path_list:
            self._load(path)
        logger.info('Triplet statistics: {} relations, {} triplets'.format(len(self.relations), self.triplet_cnt))

    def _load(self, path: str):
        examples = json.load(open(path, 'r', encoding='utf-8'))
        for ex in examples:
            self.relations.add(ex['relation'])
            key = (ex['head_id'], ex['relation'])
            if key not in self.hr2tails:
                self.hr2tails[key] = set()

            self.graph.add_node(ex['head_id'])
            self.graph.add_node(ex['tail_id'])
            self.graph.add_edge(ex['head_id'], ex['tail_id'], attr=ex['relation'])

            self.hr2tails[key].add(ex['tail_id'])
            self.hr2tails[key].add(ex['head_id'])

            self.degree[ex['head_id']] = self.degree.get(ex['head_id'], 0) + 1
            self.degree[ex['tail_id']] = self.degree.get(ex['tail_id'], 0) + 1

        reverse_examples = [reverse_triplet(obj) for obj in examples]

        for ex in reverse_examples:
            self.relations.add(ex['relation'])
            key = (ex['head_id'], ex['relation'])
            if key not in self.hr2tails:
                self.hr2tails[key] = set()
            self.hr2tails[key].add(ex['tail_id'])
            self.hr2tails[key].add(ex['head_id'])

        self.pagerank_score = nx.pagerank(self.graph)
        self.triplet_cnt = len(examples + reverse_examples)

    def get_neighbors(self, h: str, r: str) -> set:
        return self.hr2tails.get((h, r), set())

    def get_in_degree(self, h: str) -> set:
        return self.in_degree.get((h), 0)

    def get_out_degree(self, h: str) -> set:
        return self.degree.get((h), 0)

    def get_degree(self, h: str) -> set:
        return  self.degree.get(h, 0) 

    def get_pagerank_score(self, h: str) -> set:
        return self.pagerank_score.get((h), 0)


def reverse_triplet(obj):
    return {
        'head_id': obj['tail_id'],
        'head': obj['tail'],
        'relation': 'inverse {}'.format(obj['relation']),
        'tail_id': obj['head_id'],
        'tail': obj['head'],
    }
