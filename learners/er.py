from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T

from torch.utils.data import DataLoader
from utils.pytorchvis import draw_graph
from torchviz import make_dot
from models.resnet import BiasLayer
from .default import NormalNN,  accumulate_acc, loss_fn_kd, loss_fn_class_kd, Teacher

import dataloaders
from utils.metric import AverageMeter, Timer, MMD
from utils.visualisation import plots
from utils.utils import getBack

import numpy as np
import copy
import os
from collections import Counter
import json

class TR(NormalNN):
    """
    consider citing the benchmarking environment this was built on top of
    git url: https://github.com/GT-RIPL/Continual-Learning-Benchmark
    @article{hsu2018re,
        title={Re-evaluating continual learning scenarios: A categorization and case for strong baselines},
        author={Hsu, Yen-Chang and Liu, Yen-Cheng and Ramasamy, Anita and Kira, Zsolt},
        journal={arXiv preprint arXiv:1810.12488},
        year={2018}
    }
    """
    def __init__(self, learner_config):

        super(TR, self).__init__(learner_config)

        # self.batch_size = learner_config['batch_size']

        self.replay = False
        self.replay_size = learner_config['batch_size_replay']
        # self.num_replay_samples = learner_config['num_replay_samples']
        # self.labels_to_names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}  ## to read from config
        # self.class_mapping = {0: 0, 1: 1, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 2, 9: 3} ## to read from config

        self.tasks = learner_config['tasks']

        self.loss_type = learner_config['loss_type']
        
        self.loss_MMD = MMD()
        self.ce_loss = nn.BCELoss()

        self.is_dual_data_loader= learner_config['dual_dataloader']
        self.is_custom_replay_loader= learner_config['custom_replay_loader']
        # self.is_batch_sampler = learner_config['batch_sampler']
        if self.class_weighting_with==2:
            self.class_ratios_manual= json.loads(learner_config['class_ratios'])
        self.num_replay_samples_init = learner_config['num_replay_samples']
        
    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def initialise_replay_loader(self, weights=None):

        if weights is None:
            # print (self.num_replay_samples)
            sampler = dataloaders.WeightedSampler(num_samples=self.num_replay_samples, replay_strategy=self.replay_strategy)
        else:
            sampler = dataloaders.WeightedSampler(num_samples=self.num_replay_samples, weights_init=weights, replay_strategy=self.replay_strategy)

        # if self.is_batch_sampler:
        #     print ('Batch sampler')
        #     batch_sampler = dataloaders.BatchWeightedSampler(sampler=sampler, batch_size=self.batch_size)
        #     self.replay_loader = DataLoader(dataset=self.replay_dataset, batch_sampler=batch_sampler, num_workers=self.workers)
        # else:
            # print ('Sampler')
        self.replay_loader = DataLoader(dataset=self.replay_dataset, sampler=sampler, batch_size=self.batch_size, num_workers=self.workers)
        # self.sampler = sampler
        print ('length of self.replay_loader after init: ', len(self.replay_loader.dataset))


    def batch_data_weighting(self, output, labels, indices, train_data_batch_size):
        
        if not self.replay_strategy:
            self.data_weights = torch.concat([self.data_weights, torch.ones(len(indices[:train_data_batch_size])).cuda() ]) 
            self.data_indices = torch.concat([self.data_indices, indices[:train_data_batch_size]]) 
            return

        if self.epoch==self.epochs_of_interest[-1]:
            batch_weights = dataloaders.compute_weights(self.replay_strategy, output[:train_data_batch_size], labels[:train_data_batch_size])
            # if self.is_custom_replay_loader and hasattr(self, 'replay_loader') and self.is_batch_sampler:
            #     self.replay_loader.batch_sampler.update_weights(batch_new_weights=batch_weights, indices=len(self.replay_loader.batch_sampler)+indices[:train_data_batch_size])
            
            self.data_weights = torch.concat([self.data_weights, batch_weights]) 
            self.data_indices = torch.concat([self.data_indices, indices[:train_data_batch_size]]) 
                
        if self.replay:
            # replay_batch_weights = dataloaders.compute_weights(self.replay_strategy, output[train_data_batch_size:], labels[train_data_batch_size:])
            # if self.is_custom_replay_loader and self.is_batch_sampler:
            #     self.replay_loader.batch_sampler.update_weights(batch_new_weights=replay_batch_weights, indices=indices[train_data_batch_size:])
            if self.epoch==self.epochs_of_interest[-1]: ## and self.is_custom_replay_loader?
                # self.replay_loader.sampler.update_weights(batch_new_weights=replay_batch_weights, indices=indices[train_data_batch_size:])
                replay_batch_weights = dataloaders.compute_weights(self.replay_strategy, output[train_data_batch_size:], labels[train_data_batch_size:]) ##TODO remove this and uncomment at line 100
                self.replay_dataset.update_weights(batch_new_weights=replay_batch_weights, indices=indices[train_data_batch_size:])
        
    def learn_batch(self, train_loader, train_dataset, replay_dataset, model_save_dir, val_target, task, val_loader=None):
        
        # try to load model
        need_train = True

        self.train_loader = train_loader
        self.train_dataset = train_dataset
        self.replay_dataset = replay_dataset
        self.val_target = val_target
        self.model_save_dir = model_save_dir
        self.plot_dir = self.model_save_dir.replace('_outputs', 'plots_and_tables')
        self.epoch=0
        self.init_params_epoch()

        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                if self.replay:
                    self.load_replay_counts(model_save_dir)
                else:
                    self.load_replay_counts(self.model_log_dir)
                need_train = False
            except:
                pass

       
        print ('model_save_dir: ', model_save_dir)

        if not os.path.exists(self.plot_dir): os.makedirs(self.plot_dir)

        # trains
        # self.check_replay_counts = { int(i): set() for i in self.classes}
        if need_train:
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()
                if self.with_class_balance ==1:
                    self.create_new_optimizer(self.model)

            # # data weighting
            self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
            # cuda
            if self.cuda:
                self.dw_k = self.dw_k.cuda()

            
            losses = AverageMeter()
            losses_bl = AverageMeter()
            acc = AverageMeter()
            acc_bl = AverageMeter()
            self.batch_time = AverageMeter()
            batch_timer = Timer()

            self.init_params_task()

            epoch=0

            while epoch < self.config['schedule'][-1]+1:
                # print ('epoch: ', epoch)
                
                
                if epoch > 1: 
                    self.scheduler.step()
                    if hasattr(self, 'new_scheduler'):
                        self.new_scheduler.step()

                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()

                self.init_params_epoch()

                
                for i, data  in enumerate(self.train_loader):
                    
                    epoch += 1
                    self.epoch=epoch

                    if epoch >= self.config['schedule'][-1]+1:
                        break

                    if self.is_dual_data_loader:
                        x, y, indices, x_r, y_r, replay_indices = data
                    else:
                        x, y, indices = data
                        if self.replay:
                            x_r, y_r, replay_indices =  next(self.replay_loader_iter)
                            

                    self.init_params_batch()
                                
                    self.model.train()
                    
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                        
                        try:
                            x_r = x_r.cuda()
                            y_r = y_r.cuda()
                            x_tot = torch.cat([x, x_r])
                            y_tot = torch.cat([y, y_r])
                            indices = torch.cat([indices, replay_indices])
                        except:
                            x_tot = x
                            y_tot = y
                            
                    # for l,ind in zip(y_tot, indices):
                    #     self.check_replay_counts[int(l)].add(int(ind))
                    # print ('check replay_counts: ', {self.labels_to_names[self.class_mapping[cl]]: len(self.check_replay_counts[cl]) for cl in self.check_replay_counts})           
                    
                    output, new_feats = self.forward(x_tot, pen=True)
                    if self.replay:
                        y_hat, _, old_features = self.previous_teacher.generate_scores(x_tot, allowed_predictions=list(range(self.last_valid_out_dim)))
                        loss = self.update_model(output, y_tot, new_feats, old_features, y_hat)
                    else:
                        loss = self.update_model(output, y_tot)
                    
                    self.batch_data_weighting(output, y_tot, indices, len(x))

                    if self.with_class_balance==1:
                        x_bl, y_bl = self.get_class_balanced_data(indices, x_tot, y_tot)
                        output_bl = self.forward(x_bl, balanced=True)
                        loss += self.update_model(output_bl, y_bl, new_optimizer=True)
                    
                    # measure elapsed time
                    self.batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    
                    # measure accuracy and record loss
                    y = y.detach()
                    # accumulate_acc(output[:len(y)], y, acc, topk=(self.top_k,))
                    losses.update(loss, y.size(0)) 
                    batch_timer.tic()
                    self.step_count+=1
                    
                    
                    # print ('check replay_counts: ', {self.labels_to_names[self.class_mapping[cl]]: len(self.check_replay_counts[cl]) for cl in self.check_replay_counts})
                    if epoch in self.epochs_of_interest:
                        self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch,total=self.config['schedule'][-1]))
                        if val_loader is not None:
                            self.val_method_task_acc = self.validation(val_loader, train_val=True)
                        
                    self.analyze()
                    
                    # reset
                    losses = AverageMeter()
                    losses_bl = AverageMeter()
                    acc = AverageMeter()
                    acc_bl = AverageMeter()

            if self.replay:
                plots(self.per_class_accuracy_task, self.per_class_dist_shift, self.labels_to_names, self.class_mapping, self.epochs_of_interest, self.steps_of_interest, self.times_of_interest, self.replay_size,  self.avg_acc, base_path=self.plot_dir+'_after/')
        
        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False
        self.task_count+=1

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)
        
        self.extend_replay()
        self.replay=True

        try:
            return self.batch_time.avg, self.config['schedule'][-1]
        except:
            return None, None

    def extend_replay(self):

        
        if self.is_dual_data_loader and not self.replay_strategy:
            print ('class_ratios_for_replay: ', { self.labels_to_names[self.class_mapping[cl]]: self.class_replay_ratios[cl] for cl in self.class_replay_ratios})
            self.replay_dataset.extend(self.train_dataset, class_replay_ratios= self.class_replay_ratios)
        else:
            print ('class_replay_weights: ', { self.labels_to_names[self.class_mapping[cl]]: self.class_replay_weights[cl] for cl in self.class_replay_weights})
            self.data_indices = torch.argsort(self.data_indices).long()
            self.replay_dataset.extend(self.train_dataset, class_replay_weights= self.class_replay_weights, weights = self.data_weights[self.data_indices],replay_strategy=self.replay_strategy)
            
            self.num_replay_samples = self.num_replay_samples_init if self.num_replay_samples_init!=-1 else len(self.replay_dataset)
            if self.is_custom_replay_loader:
                if not hasattr(self, 'replay_loader'):
                    self.initialise_replay_loader(self.data_weights[self.data_indices])
                # if self.is_batch_sampler:
                #     self.replay_loader.batch_sampler.sampler.num_samples=self.num_replay_samples
                #     self.replay_loader.batch_sampler.sampler.class_weights= torch.Tensor(self.replay_dataset.class_weights).cuda()
                # else:
                # self.replay_loader.sampler.num_samples=self.num_replay_samples
                self.replay_loader.sampler.class_weights= torch.Tensor(self.replay_dataset.class_weights).cuda()
                self.replay_loader.sampler.weights= torch.Tensor(self.replay_dataset.sample_weights).cuda()

            # if self.replay_strategy:
            #     self.replay_dataset.get_weight_distribution(self.replay_strategy)  
        print ('size of replay dataset:',  len(self.replay_dataset))

    def update_model(self, logits, targets, new_features=None, old_features=None, target_KD=None, new_optimizer=False):
        
        
        if self.loss_type!= 'ova':
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
            total_loss = self.criterion(logits, targets.long(), dw_cls)
        else:
            # class loss
            if target_KD is not None:
                target_mod = get_one_hot(targets, self.valid_out_dim)
                target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
                total_loss = self.ce_loss(torch.sigmoid(logits), target_mod)
            else:
                target_mod = get_one_hot(targets, self.valid_out_dim)
                total_loss = self.ce_loss(torch.sigmoid(logits), target_mod)
            
        if self.loss_type == 'pred_kd' and self.replay and not new_optimizer:
            
            labels_end_ind = torch.cumsum(torch.bincount(targets.long()), dim=0).cpu()
            labels_ind = torch.argsort(targets)
            new_features = new_features[labels_ind]
            old_features = old_features[labels_ind]

            distilled = self.loss_MMD.mmd_poly(old_features, new_features, labels_end_ind= labels_end_ind)
            # print ('before norm:', distilled)
            distilled = distilled/distilled.sum(1)
            
            # sim_to_new_cls = self.sim_to_new_cls
            dw_cls = 1-torch.Tensor(self.sim_to_new_cls)
            interference_m = torch.divide(distilled, 1-torch.diag(distilled))
            interference_m *= dw_cls
            # interference_m /= interference_m.sum(1)
            interf_loss =  0.04*(interference_m.mean())
            # print ('total_loss: ', total_loss)
            # print ('interf_loss: ', interf_loss)
            total_loss +=  interf_loss
            

        if not new_optimizer:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        else:
            self.new_optimizer.zero_grad()
            total_loss.backward()
            self.new_optimizer.step()

        return total_loss.detach()

    
def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot=one_hot.scatter(1,target.long().view(-1,1),1.)
    return one_hot