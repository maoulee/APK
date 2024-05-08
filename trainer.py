import glob
import json
import torch
import shutil

import torch.nn as nn
import torch.utils.data
import os
from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW
from tqdm import tqdm
from doc import Dataset, collate
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from metric import accuracy
from models import build_model, ModelOutput
from dict_hub import build_tokenizer
from logger_config import logger
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
import numpy as np

class Trainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args)

        # create model
        train_dataset = Dataset(path=args.train_path, task=args.task)
        valid_dataset = Dataset(path=args.valid_path, task=args.task) if args.valid_path else None
        args.num_rel,args.num_ent=train_dataset.num_rel,train_dataset.num_ent
        logger.info("=> creating model")
        self.model = build_model(self.args)
        #logger.info(self.model)
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        #report_num_trainable_parameters(self.model)
        num_training_steps = args.epochs * len(train_dataset) // max(args.batch_size, 1)
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(num_training_steps)
        self.best_metric = None

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=4,
            pin_memory=True,
            drop_last=False)

        self.valid_loader = None
        if valid_dataset:
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate,
                num_workers=4,
                pin_memory=False)
            
        self.fabric=self.set_fabric()
        self.set_model()
        
        
    def set_fabric(self):
        accelerater='gpu'
        device='auto'
        precision="bf16-mixed"
        logger = TensorBoardLogger(root_dir="logs")
        if torch.cuda.device_count()>1:
            strategy='dp'
        else:
            strategy='auto'
        fabric=Fabric(accelerator=accelerater, devices=device, strategy=strategy,precision=precision,loggers=logger)
        
        if torch.cuda.device_count()>1:
            fabric.launch()

        return fabric

    def set_model(self):
        self.model,self.optimizer=self.fabric.setup(self.model,self.optimizer)
        self.train_loader,self.valid_loader=self.fabric.setup_dataloaders(self.train_loader,self.valid_loader)


    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        metric_dict = self.eval_epoch(epoch)
        is_best = self.valid_loader and (self.best_metric is None or metric_dict['Acc@1'] > self.best_metric['Acc@1'])
        if is_best:
            self.best_metric = metric_dict
        

        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

    @torch.no_grad()
    def eval_epoch(self, epoch) -> Dict:
        if not self.valid_loader:
            return {}

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        
        for i, batch_dict in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])
            outputs = self.model(**batch_dict)
           
            outputs = self.model.compute_logits_ans(output_dict=outputs, batch_dict=batch_dict)
            
            
            outputs = ModelOutput(**outputs)
            logits, labels,loss = outputs.hr_logits, outputs.labels,outputs.loss
            losses.update(loss.item(), batch_size)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)
            
            

        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        return metric_dict

    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        R_top1 = AverageMeter('R_Acc@1', ':6.2f')
        R_top3 = AverageMeter('R_Acc@3', ':6.2f')
       
        inv_t = AverageMeter('InvT', ':6.2f')
        
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, inv_t, R_top1, R_top3],
            prefix="Epoch: [{}]".format(epoch))
       
        epoch_iterator = tqdm(self.train_loader,desc="Iteration-{})".format(epoch),leave=False,position=0)
        
        for i, batch_dict in enumerate(epoch_iterator):
            # switch to train mode
            self.model.train()
            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            # compute output
            
            with self.fabric.autocast():
                outputs = self.model(**batch_dict)
                outputs = self.model.compute_logits_ans(output_dict=outputs, batch_dict=batch_dict)
                

            outputs = ModelOutput(**outputs)
            hr_logits,labels,loss = outputs.hr_logits,outputs.labels, outputs.loss
            assert hr_logits.size(0) == batch_size

            acc1, acc3 = accuracy(hr_logits, labels, topk=(1, 3))
            R_top1.update(acc1.item(), batch_size)
            R_top3.update(acc3.item(), batch_size)
            
            
            
            inv_t.update(outputs.inv_t.item(), 1)
            losses.update(loss.item(), batch_size)

            # compute gradient and do SGD step
            
            self.fabric.backward(loss)
            
            self.optimizer.step()
            self.fabric.clip_gradients(self.model,self.optimizer,clip_val=self.args.grad_clip)
            self.optimizer.zero_grad()
            self.scheduler.step()

            if i % self.args.print_freq == 0:
                self.fabric.log("loss", loss.item())
                progress.display(i)
        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))


    def _create_lr_scheduler(self, num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)
