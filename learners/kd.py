from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
import scipy as sp
import scipy.linalg as linalg
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc, loss_fn_kd, Teacher
import copy
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function


class LWF(NormalNN):

    def __init__(self, learner_config):
        super(LWF, self).__init__(learner_config)
        self.previous_teacher = None
        self.replay = False
        self.past_tasks = []
        self.first_task = True
        self.first_block = True
        self.ce_loss = nn.BCELoss()
        self.init_task_param_reg = self.eps > 0

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        # L2 from the start
        if self.init_task_param_reg: self.accumulate_block_memory(train_loader)
        
        # init teacher
        if self.previous_teacher is None:
            teacher = Teacher(solver=self.model)
            self.previous_teacher = copy.deepcopy(teacher)

        # try to load model
        need_train = True
        # if not self.overwrite or self.task_count == 0:
        if True:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if need_train:
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()

            # data weighting
            self.data_weighting(train_dataset)
            
            # Evaluate the performance of current task
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)
        
            losses = [AverageMeter() for l in range(3)]
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()


                for i, (x, y, task)  in enumerate(train_loader):
                    self.step = i

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                    
                    # if KD
                    if self.replay:
                        allowed_predictions = list(range(self.last_valid_out_dim))
                        y_hat, _, _ = self.previous_teacher.generate_scores(x, allowed_predictions=allowed_predictions)
                    else:
                        y_hat = None

                    # model update - training data
                    loss, loss_class, loss_distill, output= self.update_model(x, y, y_hat)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, acc, topk=(self.top_k,))
                    losses[0].update(loss,  y.size(0)) 
                    losses[1].update(loss_class,  y.size(0)) 
                    losses[2].update(loss_distill,  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))
                self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for l in range(3)]
                acc = AverageMeter()

        self.model.eval()

        self.past_tasks.append(np.arange(self.last_valid_out_dim,self.valid_out_dim))
        self.last_last_valid_out_dim = self.last_valid_out_dim
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0 and self.task_count<=1:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        # for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)
        self.replay = True
        if len(self.config['gpuid']) > 1:
            self.previous_linear = copy.deepcopy(self.model.module.last_unbalanced)
        else:
            self.previous_linear = copy.deepcopy(self.model.last_unbalanced)
        
        # prepare dataloaders for block replay methods
        self.accumulate_block_memory(train_loader)

        try:
            return batch_time.avg
        except:
            return None

    def accumulate_block_memory(self, train_loader):
        pass

    def update_model(self, inputs, targets, target_KD = None):
        
        total_loss = torch.zeros((1,), requires_grad=True).cuda()

        if self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        loss_class = self.criterion(logits, targets.long(), dw_cls)
        total_loss += loss_class

        # KD
        if target_KD is not None:
            dw_KD = self.dw_k[-1 * torch.ones(len(target_KD),).long()]
            logits_KD = logits
            loss_distill = loss_fn_kd(logits_KD, target_KD, dw_KD, np.arange(self.last_valid_out_dim).tolist(), self.DTemp)
            total_loss += self.mu * loss_distill
        else:
            loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits

    def combine_data(self, data):
        x, y = [],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(data[i][1])
        x, y = torch.cat(x), torch.cat(y)
        return x, y

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    def combine_data(self, data):
        x, y = [],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(data[i][1])
        x, y = torch.cat(x), torch.cat(y)
        return x, y

class LWF_MC(LWF):

    def __init__(self, learner_config):
        super(LWF_MC, self).__init__(learner_config)
        

    def update_model(self, inputs, targets, target_KD = None):

        # get output
        logits = self.forward(inputs)

        # class loss      
        if target_KD is not None:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
            total_loss = self.ce_loss(torch.sigmoid(logits), target_mod)
        else:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            total_loss = self.ce_loss(torch.sigmoid(logits), target_mod)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), total_loss.detach(), torch.zeros((1,), requires_grad=True).cuda().detach(), logits

class LWF_MC_FEATKD(LWF):

    def __init__(self, learner_config):
        super(LWF_MC_FEATKD, self).__init__(learner_config)
        self.l2_loss = torch.nn.MSELoss(reduction='none')
        

    def update_model(self, inputs, targets, target_KD = None):

        # get output
        logits = self.forward(inputs)

        # class loss
        if target_KD is not None:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
            total_loss = self.ce_loss(torch.sigmoid(logits), target_mod)
        else:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            total_loss = self.ce_loss(torch.sigmoid(logits), target_mod)

        # feature kd loss
        target_feat = self.previous_teacher.generate_scores_pen(inputs)
        feat = self.model.forward(x=inputs, pen=True)
        loss_reg = self.l2_loss(feat, target_feat).mean()

        total_loss = total_loss + self.mu * loss_reg
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), total_loss.detach() - self.mu * loss_reg.detach(), self.mu * loss_reg.detach(), logits

class EWC_MC(LWF):

    def __init__(self, learner_config):
        super(EWC_MC, self).__init__(learner_config)
        self.regularization_terms = {}
        self.l2_loss = torch.nn.MSELoss(reduction='none')
        
    def update_model(self, inputs, targets, target_KD = None):

        # get output
        logits = self.forward(inputs)

        # class loss     
        if target_KD is not None:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
            total_loss = self.ce_loss(torch.sigmoid(logits), target_mod)
        else:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            total_loss = self.ce_loss(torch.sigmoid(logits), target_mod)
        
        # Calculate the reg_loss only when the regularization_terms exists
        reg_loss_l2 = torch.zeros((1,), requires_grad=True).cuda()
        reg_loss_ewc = torch.zeros((1,), requires_grad=True).cuda()
        for i,reg_term in self.regularization_terms.items():
            # print ('yes')
            task_reg_loss_l2 = torch.zeros((1,), requires_grad=True).cuda()
            task_reg_loss_ewc = torch.zeros((1,), requires_grad=True).cuda()
            importance_ewc = reg_term['importance_ewc']
            task_param = reg_term['task_param']
            for n, p in self.params.items():

                # l2
                if self.eps > 0:
                    task_reg_loss_l2 += self.eps * (self.l2_loss(p,task_param[n])).sum()

                # ewc
                if self.mu > 0:
                    # print ('yes2')
                    print (n, ' ', importance_ewc[n].shape)
                    task_reg_loss_ewc += self.mu * (importance_ewc[n] * self.l2_loss(p,task_param[n])).sum()
            
            # update total losses
            reg_loss_l2 += task_reg_loss_l2
            reg_loss_ewc += task_reg_loss_ewc
        
        # total weighted reg loss
        loss_reg = reg_loss_l2 + reg_loss_ewc
        
        # final loss and backprop
        total_loss = total_loss + loss_reg
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), total_loss.detach() - loss_reg.detach(), loss_reg.detach(), logits

    def accumulate_block_memory(self, train_loader):
        dataloader = train_loader
        # Update the diag fisher information
        # There are several ways to estimate the F matrix.
        # We keep the implementation as simple as possible while maintaining a similar performance to the literature.
        if len(self.config['gpuid']) > 1:
            self.params = {n: p for n, p in self.model.module.named_parameters() if p.requires_grad}
        else:
            self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.log('Computing EWC')

        # Initialize the imp matrix
        if self.init_task_param_reg:
            self.init_task_param_reg = False
            self.first_block = False
            task_param = {}
            importance_ewc = {}
            for n, p in self.params.items():
                importance_ewc[n] = p.clone().detach().fill_(0)  # zero initialized
                task_param[n] = p.clone().detach()
            self.regularization_terms['online'] = {'importance_ewc':importance_ewc, 'task_param':task_param}
            return
        elif self.first_block:
            self.first_block = False
            importance_ewc = {}
            for n, p in self.params.items():
                importance_ewc[n] = p.clone().detach().fill_(0)  # zero initialized
        else:
            importance_ewc = self.regularization_terms['online']['importance_ewc']

        mode = self.training
        self.eval()

        # Accumulate the square of gradients
        # task_param = {}
        # for n, p in copy.deepcopy(self.params).items():
        #     p.data.zero_()
        #     task_param[n] = variable(p.data)

        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                input = input.cuda()
                target = target.cuda()

            pred = self.model.forward(input)[:,:self.valid_out_dim]
            ind = pred.max(1)[1].flatten()  # Choose the one with max
            target_mod = get_one_hot(target-self.last_last_valid_out_dim, self.valid_out_dim-self.last_last_valid_out_dim)
            loss = self.ce_loss(torch.sigmoid(pred[:,self.last_last_valid_out_dim:self.last_valid_out_dim]), target_mod)

            self.model.zero_grad()
            loss.backward()

            for n, p in self.params.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    importance_ewc[n] += ((self.params[n].grad ** 2) * len(input) / len(dataloader))

        self.train(mode=mode)
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()
        self.regularization_terms['online'] = { 'importance_ewc':importance_ewc, 'task_param':task_param}

def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot[range(target.shape[0]), target]=1
    return one_hot