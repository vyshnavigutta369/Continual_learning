from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
import models

import copy
from collections import Counter, OrderedDict

import numpy as np
import os, math

from sklearn.cluster import KMeans

# import dataloaders
from utils.schedulers import CosineSchedule
from utils.metric import accuracy, AverageMeter, Timer, new_vs_old_class_comparison, distribution_shift_comparison
from utils.visualisation import plots
from utils.utils import getBack, near_split, ratio_breakdown

from itertools import cycle

class NormalNN(nn.Module):
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

        super(NormalNN, self).__init__()
        self.log = print
        self.config = learner_config
        self.out_dim = learner_config['out_dim']
        self.model = self.create_model()
        self.reset_optimizer = True
        self.overwrite = learner_config['overwrite']
        self.workers = learner_config['workers']
        self.batch_size = learner_config['batch_size']
        self.previous_teacher = None
        self.tasks = learner_config['tasks']
        self.top_k = learner_config['top_k']
        self.task_count = 0
        self.learner_name = learner_config['learner_name']

        # class balancing
        self.dw = self.config['DW']
        self.replay_type = learner_config['replay_type']
        self.replay_strategy= learner_config['replay_strategy']
        self.memory_size = self.config['memory']

        # distillation
        self.DTemp = learner_config['temp']
        self.mu = learner_config['mu']
        self.beta = learner_config['beta']
        self.eps = learner_config['eps']

        # supervised criterion
        self.criterion_fn = nn.CrossEntropyLoss(reduction="none")

        self.model_log_dir = learner_config['model_log_dir']
        
        # cuda gpu
        if learner_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        
        # highest class index from past task
        self.last_valid_out_dim = 0 

        # highest class index from current task
        self.valid_out_dim = 0

        # set up schedules
        self.schedule_type = self.config['schedule_type']
        self.schedule = self.config['schedule']

        #loss types
        self.loss_type = learner_config['loss_type']

        # initialize optimizer
        self.init_optimizer()

        self.with_class_balance = learner_config['with_class_balance']

        self.replay=False
        self.class_weighting_with = learner_config['class_weighting_with']
        self.weight_reverse = learner_config['weight_reverse']

        self.is_oracle=False if self.learner_name!='NormalNN' else True

        if self.is_oracle:
            # print ('HELLLOOOOO')
            self.replay_size='Oracle'
            self.new_classes = learner_config['new_classes']
            self.old_classes = learner_config['old_classes']

        self.steps = learner_config['steps'][0] 
            
    def init_params_task(self):
        
        self.per_class_accuracy_task = {}
        self.per_class_dist_shift = {}
        self.step_count=0

        self.epochs_of_interest = [i for i in range(0, self.config['schedule'][-1], int(self.config['schedule'][-1]/10))]
        if self.epochs_of_interest[1]!=1:
            self.epochs_of_interest.insert(1,1)
        if self.epochs_of_interest[-1]!=self.config['schedule'][-1]:
            self.epochs_of_interest.append(self.config['schedule'][-1])
        
        self.steps_of_interest = [i for i in range(0, self.steps, int(self.steps/10))] if self.steps!=-1 else []
        if self.steps!=-1 and self.steps_of_interest[1]!=1:
            self.steps_of_interest.insert(1,1)
        if self.steps!=-1 and self.steps_of_interest[-1]!=self.steps:
            self.steps_of_interest.append(self.steps)
       
       
        self.times_of_interest = []

        print ('epochs_of_interest: ', self.epochs_of_interest)
        print ('steps_of_interest: ', self.steps_of_interest)
        self.class_mapping = { v: k for k,v in self.train_dataset.class_mapping.items() if k!=-1}
        self.labels_to_names = { v: k for k,v in self.train_dataset.class_to_idx.items()}

        if self.replay:
            if not self.is_oracle:
                self.old_classes = set([self.replay_dataset.class_mapping[int(y)] for y in self.replay_dataset.targets])
                self.avg_acc = { "Oracle": [], "Method": []}
                self.task_acc = { "Oracle": {epoch: [] for epoch in self.epochs_of_interest}, "Method": {epoch: [] for epoch in self.epochs_of_interest}}

                try:
                    self.class_ratios_manual = { self.train_dataset.class_mapping[self.train_dataset.class_to_idx[cl]]: self.class_ratios_manual[cl] for cl in self.class_ratios_manual if self.class_ratios_manual}
                except:
                    pass
            else:
                self.avg_acc = { "Oracle": []}
                self.task_acc = { "Oracle": {epoch: [] for epoch in self.epochs_of_interest}}

        if not self.is_oracle:
            self.new_classes = set([self.train_dataset.class_mapping[int(y)] for y in self.train_dataset.targets])
        self.classes = self.old_classes.union(self.new_classes) if self.replay else self.new_classes
                
        self.check_replay_counts = { int(i): set() for i in self.class_mapping}
        
        if not self.is_oracle and self.replay and self.is_custom_replay_loader:
            self.replay_loader_iter = cycle(self.replay_loader)

        self.data_weights = OrderedDict()

    def init_params_epoch(self):

        self.per_class_predictions = {}
        self.per_class_correct_predictions = {}
        self.per_class_old_features = {}
        self.per_class_new_features = {}
        
        
            # if self.replay:
            #     self.old_classes = set([self.replay_dataset.class_mapping[int(y)] for y in self.replay_dataset.targets])
                # if self.is_custom_replay_loader:
                #     # self.replay_loader_iter = cycle(self.replay_loader)
                #     if self.epoch==self.epochs_of_interest[-1] and self.is_batch_sampler:
                #         self.replay_loader.batch_sampler.add(len(self.train_dataset)) ## Full replay addition, TODO if u want to limit amount of replay being added.
            # self.new_classes = set([self.train_dataset.class_mapping[int(y)] for y in self.train_dataset.targets])

        # self.check_replay_counts = { int(i): set() for i in self.classes}
            
    
    def init_params_batch(self):

        self.per_class_new_features_train = {}
        self.per_class_old_features_train = {}
        self.per_class_new_data_train = {}
        self.per_class_indices_train = {}

        # self.check_replay_counts = { int(i): set() for i in self.classes}

    def process(self):
        
        # if self.per_class_correct_predictions:
        #     print (self.per_class_correct_predictions[0].keys())
        if not os.path.exists(self.plot_dir+'before/'): os.makedirs(self.plot_dir+'before/')
        if not os.path.exists(self.plot_dir+'_after/'): os.makedirs(self.plot_dir+'_after/')

        self.per_class_correct_predictions = { k: float(sum(self.per_class_predictions[k].values())) for k in self.per_class_predictions if self.per_class_predictions}
        self.processed_per_class_old_features = { k: list(self.per_class_old_features[k].values()) for k in self.per_class_old_features if self.per_class_old_features}
        self.processed_per_class_new_features = { k: list(self.per_class_new_features[k].values()) for k in self.per_class_new_features if self.per_class_new_features}


    def process_batch(self):

        self.per_class_indices_train = { k: list(self.per_class_new_data_train[k].keys()) for k in self.per_class_new_data_train if self.per_class_new_data_train}
        self.per_class_new_data_train = { k: list(self.per_class_new_data_train[k].values()) for k in self.per_class_new_data_train if self.per_class_new_data_train}
        self.per_class_new_features_train = [ list(self.per_class_new_features_train[k].values()) for k in self.per_class_new_features_train if self.per_class_new_features_train]
        self.per_class_old_features_train = [ list(self.per_class_old_features_train[k].values()) for k in self.per_class_old_features_train if self.per_class_old_features_train]

    def split_by_class(self, indices, labels, old_features=None, new_features=None, predictions= None, new_datas_train=None, old_features_train=None, new_features_train=None):
        
        labels_end_ind = torch.cumsum(torch.bincount(labels.long()), dim=0).cpu()
        labels_ind = torch.argsort(labels).tolist()
        
        indices = indices[labels_ind]
        class_wise_indices_split = torch.tensor_split(indices, labels_end_ind[:-1], dim=0)
        class_wise_indices_split = [ cl_ind.tolist() for cl_ind in class_wise_indices_split if cl_ind.shape[0]>0]

        def dict_by_class(data):
            class_wise_split = torch.tensor_split(data[labels_ind], labels_end_ind[:-1], dim=0)
            class_wise_split = [ cl_split for cl_split in class_wise_split if cl_split.shape[0]>0]
            return {int(i): dict(zip(j, x)) for i,j,x in zip(set(labels.tolist()), class_wise_indices_split, class_wise_split)}

        if predictions is not None:
            # print ((predictions==labels).int())
            
            cl_split = dict_by_class((predictions==labels).int())
            
            # print ('1:', self.per_class_correct_predictions)
            # print ('2:', cl_split.keys())
            if self.per_class_predictions:
                
                self.per_class_predictions = { k: {**self.per_class_predictions.get(k,{}), **cl_split.get(k,{})} for k in list(cl_split.keys())+list(self.per_class_predictions.keys())}
            else:
                self.per_class_predictions = cl_split
            
            # print ('3:', self.per_class_predictions)

        if old_features is not None:
            cl_split = dict_by_class(old_features)
            if self.per_class_old_features:
                self.per_class_old_features = { k: {**self.per_class_old_features.get(k,{}), **cl_split.get(k,{})} for k in list(cl_split.keys())+list(self.per_class_old_features.keys())}
            else:
                self.per_class_old_features = cl_split

        if new_features is not None:
            cl_split = dict_by_class(new_features)
            if self.per_class_new_features:
                self.per_class_new_features = { k: {**self.per_class_new_features.get(k,{}), **cl_split.get(k,{})} for k in list(cl_split.keys())+list(self.per_class_new_features.keys())}
            else:
                self.per_class_new_features = cl_split

        if new_datas_train is not None:
            # if self.per_class_new_data_train:
                # print ('1:', self.per_class_new_data_train[0].keys())
            self.per_class_new_data_train.update(dict_by_class(new_datas_train))
            # print ({i: len(x) for i,x in self.per_class_new_data_train.items()})
            # print ('2:', self.per_class_new_data_train[0].keys())

        if old_features_train is not None:
            self.per_class_old_features_train.update(dict_by_class(old_features_train))

        if new_features_train is not None:
            self.per_class_new_features_train.update(dict_by_class(new_features_train))

        # getBack(class_wise_new_features_split[0].grad_fn)
        # class_wise_data_split_p = torch.nn.utils.rnn.pad_sequence(class_wise_new_features_split, batch_first=True)

    def get_class_balanced_data(self, indices, data, labels):

        # self.helper(indices, labels, new_datas_train= data)
        self.split_by_class(indices, labels, new_datas_train=data)
        self.process_batch()
        # print ({i: len(x) for i,x in self.per_class_new_data_train.items()})

        no_of_samples_per_class_balanced = int(min([len(self.per_class_new_data_train[cl]) for cl in self.per_class_new_data_train]))
        
        data_bl = torch.Tensor([]).cuda()
        indices_bl = torch.Tensor([]).cuda()
        labels_bl = torch.Tensor([]).cuda()

    
        for cl in self.per_class_new_data_train:

            # data_bl = torch.cat((data_bl, torch.Tensor(self.per_class_new_data_train[cl][:no_of_samples_per_class_balanced]).cuda() ))
            data_bl = torch.cat((data_bl, torch.stack(self.per_class_new_data_train[cl][:no_of_samples_per_class_balanced]) ))
            indices_bl = torch.cat((indices_bl, torch.tensor(self.per_class_indices_train[cl][:no_of_samples_per_class_balanced]).cuda() ))
            labels_bl = torch.cat((labels_bl, torch.tensor([cl]*no_of_samples_per_class_balanced).cuda() ))
            
        # print (data_bl.shape)
        # print (labels_bl.shape)
        # data_bl = data_bl.split(self.batch_size)
        # labels_bl = labels_bl.split(self.batch_size)
       
        return data_bl, labels_bl

    def update_balanced_head(self, data, labels):
        
        loss=0.0
        # output = torch.Tensor([]).cuda()

        output, features = self.model.forward(data, pen=True) 
        
        # for feat, y in zip(features, labels):

        #     y_pred = self.model.classifier_head_forward(feat)
        #     loss += self.update_model(y_pred, y, new_optimizer=True)
        #     output = torch.cat((output, y_pred))

        y_pred = self.model.classifier_head_forward(features)
        loss += self.update_model(y_pred, labels, new_optimizer=True)
        output = torch.cat((output, y_pred))

        return loss, output
        
    def scale_distribution(self, data, classes, rnge=(1,8)):

        t_max = rnge[0]
        t_min = rnge[1]
        r_max= max(data)
        r_min = min(data)

        def scale_to(x,r_min, r_max, t_min, t_max):
            return ((float(x-r_min)/(r_max-r_min))*(t_max-t_min)) + t_min

        class_replay_ratios =  { cl: math.floor(scale_to(x,r_min, r_max, t_min, t_max)) for cl,x in zip(classes, data) }
        
        if self.weight_reverse:
            class_replay_ratios_sorted = { k: v for k,v in sorted(class_replay_ratios.items(), key=lambda x:x[1])}
            class_replay_ratios_vals = sorted(class_replay_ratios_sorted.values(), reverse=True)
            class_replay_ratios_sorted = { k: class_replay_ratios_vals[i] for i,k in enumerate(class_replay_ratios_sorted)}
            class_replay_ratios = { k: class_replay_ratios_sorted[k] for k in class_replay_ratios}

        if self.epoch==0 and self.is_custom_replay_loader and int(self.class_weighting_with)==13: ##TODO not working epoch 0 update of replay loader weights
            self.class_replay_ratios =  { cl: class_replay_ratios[cl] for cl in class_replay_ratios if cl in self.old_classes}
        else:
            self.class_replay_ratios =   { cl: class_replay_ratios[cl] for cl in class_replay_ratios if cl in self.new_classes}

        # print ('class_ratios_for_replay at epoch: ', { self.labels_to_names[self.class_mapping[cl]]: self.class_replay_ratios[cl] for cl in self.class_replay_ratios})

    def analyze(self):

        self.process()
        
        
        if (not self.replay and ((self.steps==-1 and self.epoch==self.epochs_of_interest[-1]) or (self.steps!=-1 and self.step_count==self.steps_of_interest[-1]))) or (self.replay and ((self.steps==-1 and self.epoch in self.epochs_of_interest) or (self.steps!=-1 and self.step_count in self.steps_of_interest))):

            # print ('helllllooooooooooooooooooo1')
            self.class_accuracy_epoch, self.class_dist_shift_epoch, self.per_class_accuracy_task, self.per_class_dist_shift = distribution_shift_comparison(self.labels_to_names, self.class_mapping, self.processed_per_class_new_features, self.processed_per_class_old_features, self.per_class_correct_predictions, self.per_class_accuracy_task, self.per_class_dist_shift, self.plot_dir)
            
            if not self.is_oracle: 
                if int(self.class_weighting_with)==1: ## regular dual dataloader
                    # print ('helllllooooooooooooooooooo2')
                    self.class_replay_ratios= { i:1 for i in self.new_classes}
                elif int(self.class_weighting_with)==2: ## custom weighting 50%-50%
                    self.class_replay_ratios= { cl: self.class_ratios_manual[cl] for cl in self.class_ratios_manual if cl in self.new_classes}
                elif int(self.class_weighting_with)==11 and self.epoch!=0: ## dist shift
                    classes = list(self.class_dist_shift_epoch.keys())
                    class_dist_shift_epoch = np.array(list(self.class_dist_shift_epoch.values()))
                    self.scale_distribution(class_dist_shift_epoch, classes)
                elif int(self.class_weighting_with)==12: ## acc shift
                    classes = list(self.class_accuracy_epoch.keys()) 
                    class_accuracy_epoch = np.array(list(self.class_accuracy_epoch.values()))
                    # print (self.class_accuracy_epoch)
                    self.scale_distribution(class_accuracy_epoch, classes)
                    # print ({ self.labels_to_names[self.class_mapping[cl]]: self.class_accuracy_epoch[cl] for cl in self.class_accuracy_epoch})
                elif int(self.class_weighting_with)==13: ## sim shift TODO not working, incorptorate in dynamic replay criterion/loss function
                    if self.replay:
                        base_path = self.plot_dir+'_after/new_vs_old/epoch_' + str(self.epoch)+'/' if self.epoch>0 else self.plot_dir+'_before/new_vs_old/'
                        self.sim_table, self.sim_to_new_cls = new_vs_old_class_comparison(self.new_classes, self.old_classes, self.processed_per_class_new_features, self.labels_to_names, self.class_mapping, replay_size = self.replay_size, base_path=base_path, is_oracle=self.is_oracle)
                        # self.sim_to_new_cls = 1-np.array([[ sim_table[k][cl] for cl in sim_table[k]] for k in sim_table]).mean(0) ## for interference
                        classes = list(self.sim_table.keys())
                        self.scale_distribution(self.sim_to_new_cls, classes)
                    else:
                        # self.class_ratios = { k: 1 for k in self.classes}
                        self.class_replay_ratios = {k: 1 for k in self.new_classes}
                
                # self.class_replay_weights = { i: self.class_replay_ratios[i]/self.replay_size for i in self.class_replay_ratios}  ## TODO this will change when using advanced weighting sample-level
                self.class_replay_weights = { i: self.class_replay_ratios[i] for i in self.class_replay_ratios}  ## TODO this will change when using advanced weighting sample-level

                save_dir = self.model_save_dir if self.replay else self.model_log_dir
                # np.save(save_dir+'class_replay_counts.npy', self.class_replay_counts)
                np.save(save_dir+'class_replay_ratios.npy', self.class_replay_ratios)
                np.save(save_dir+'class_replay_weights.npy', self.class_replay_weights)

                
            if self.replay:
                self.compute_val_acc()

            if not self.is_oracle and self.epoch==0 and self.is_custom_replay_loader and int(self.class_weighting_with)==13: ##TODO not working incorpotate this in kd loss
                
                self.replay_dataset.update_all_weights(self.class_replay_weights)
                self.replay_loader.sampler.class_weights= torch.Tensor(self.replay_dataset.class_weights).cuda()
                self.replay_loader_iter = cycle(self.replay_loader)

    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, replay_dataset, model_save_dir, val_target, task, val_loader=None):
        
        # try to load model
        need_train = True
        self.train_dataset = train_dataset
        self.plot_dir = model_save_dir.replace('_outputs', 'plots_and_tables')
        self.val_target = val_target
        self.val_loader= val_loader
        self.epoch=0

        task_index = self.tasks.index(task)+1
        if self.tasks[0]!= task:
            self.replay=True ## TODO need to chage this
            model_save_dir_prev = model_save_dir.replace('task-'+str(task_index), 'task-'+str(task_index-1))
            self.load_prev_model(model_save_dir_prev)
            self.previous_teacher = Teacher(solver=self.prev_model)
            self.last_valid_out_dim = len(self.tasks[task_index-2])
            
        self.init_params_epoch()

        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        if not os.path.exists(self.plot_dir): os.makedirs(self.plot_dir)


        # trains
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


            # epoch=0
            # while epoch < self.config['schedule'][-1]+1:
            for epoch in range(self.config['schedule'][-1]+1):
                self.epoch=epoch
                if self.step_count> self.steps:
                        break

                if epoch > 1: 
                    self.scheduler.step()
                    if hasattr(self, 'new_scheduler'):
                        self.new_scheduler.step()
                        
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()

                self.init_params_epoch()

                for i, (x, y, indices, _, _, _)  in enumerate(train_loader):

                    # epoch += 1
                    # self.epoch=epoch

                    # if epoch >= self.config['schedule'][-1]+1:
                    #     break
                    if epoch==0:
                        break

                    if self.step_count> self.steps:
                        break

                    self.init_params_batch()

                    # send data to gpu
                    if self.gpu:
                        x =x.cuda()
                        y = y.cuda()

                    self.model.train()
                    
                    # model update
                    output, new_feats = self.forward(x, pen=True)    
                    loss = self.update_model(output, y)
                    
                    if self.with_class_balance==1:
                        x_bl, y_bl = self.get_class_balanced_data(indices, x, y)
                        output_bl = self.forward(x_bl, balanced=True)
                        loss += self.update_model(output_bl, y_bl, new_optimizer=True)

                        # output_bl, y_bl = self.get_class_balanced_data(indices, y, new_feats)
                        # loss_bl = self.update_model(output_bl, y_bl, new_optimizer=True)

                    # measure elapsed time
                    self.batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    
                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0)) 
                    batch_timer.tic()
                    self.step_count+=1

                    if self.steps!= -1:
                        losses, losses_b, acc, aacc_bl = self.post_training(acc, losses)
                
                if self.steps == -1:
                    losses, losses_b, acc, aacc_bl = self.post_training(acc, losses)

                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch,total=self.config['schedule'][-1]))

            if self.replay:
                plots(self.per_class_accuracy_task, self.per_class_dist_shift, self.labels_to_names, self.class_mapping, self.epochs_of_interest, self.steps_of_interest, self.times_of_interest, self.replay_size,  self.avg_acc,  base_path=self.plot_dir+'_after/', is_oracle=True)
            
        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False
        self.replay=True ## TODO REMOVE THIS

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)

        try:
            return self.batch_time.avg, self.config['schedule'][-1]
        except:
            return None, None

    def criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised 

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

    def update_model_old(self, logits, targets, new_optimizer=False):
        
        if self.loss_type!= 'ova':
            if self.replay_type == 'random_sample':
                dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
            elif self.replay_type == 'gradient_cb':
                h_m = Counter([int(i) for i in targets]) ## map
                h_m = { i: 1/h_m[i] for i in h_m}
                dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]*torch.Tensor([h_m[int(i)] for i in targets]).cuda()
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
        
        
        if not new_optimizer:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        else:
            self.new_optimizer.zero_grad()
            total_loss.backward()
            self.new_optimizer.step()
        return total_loss.detach()

    @torch.no_grad()
    def post_training(self, acc, losses):

        if (self.steps==-1 and self.epoch in self.epochs_of_interest) or (self.step_count in self.steps_of_interest):

            
            if self.steps==-1:
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch,total=self.config['schedule'][-1]))
            else:
                self.log('Step:{step:.0f}/{total:.0f}'.format(step=self.step_count,total=self.steps))
            self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))
            
            if self.val_loader is not None:
                self.val_method_task_acc = self.validation(self.val_loader, train_val=True)
            
            # print ('hellloooo')
            self.analyze()

        losses = AverageMeter()
        losses_bl = AverageMeter()
        acc = AverageMeter()
        acc_bl = AverageMeter()

        return losses, losses_bl, acc, acc_bl

    @torch.no_grad()
    def validation(self, dataloader, model=None, task_in = None,  verbal = True, train_val=False):

        balanced= False
        if self.with_class_balance==1:
            balanced= True

        
        if model is None:
            model = self.model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()

        for i, (input, target, indices) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            if task_in is None:
                with torch.no_grad():
                    output, new_feats = model.forward(input, pen=True, balanced=balanced)
                acc = accumulate_acc(output, target, acc, topk=(self.top_k,))

                if not train_val:
                    continue

                if (self.steps==-1 and not self.replay and self.epoch<self.epochs_of_interest[-1]) or (self.steps!=-1 and not self.replay and self.step_count<self.steps_of_interest[-1]):
                    if (self.steps==-1 and self.epoch==1) or (self.steps!=-1 and self.step_count==1):
                        if not hasattr(self, 'old_feats'):
                            self.old_feats=[]
                        self.old_feats.append(new_feats)
                    continue
                elif (self.steps==-1 and not self.replay and self.epoch==self.epochs_of_interest[-1]) or (self.steps!=-1 and not self.replay and self.step_count==self.steps_of_interest[-1]):
                    old_feats = self.old_feats[i]
                
                if self.replay:
                    _, _, old_feats = self.previous_teacher.generate_scores(input, allowed_predictions=list(range(self.last_valid_out_dim)), balanced=balanced)
                predictions = torch.argmax(output, dim=1)
                # print (predictions)
                # self.helper(indices.tolist(), target, old_feats, new_feats, predictions)
                self.split_by_class(indices, target, old_features=old_feats, new_features=new_feats, predictions=predictions)
            else:
                mask = target >= task_in[0]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]

                mask = target < task_in[-1]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]
                
                if len(target) > 1:
                    output = model.forward(input, balanced=balanced)[:, task_in]
                    acc = accumulate_acc(output, target-task_in[0], acc, topk=(self.top_k,))

        model.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        
        return acc.avg

    def compute_val_acc(self):
        
        # print (self.per_class_accuracy_task)
        # val_method_avg_acc = float(sum(self.class_accuracy_epoch.values()))/len(self.class_accuracy_epoch)*100
        val_method_avg_acc = self.val_method_task_acc
            
        if not self.is_oracle:
            self.avg_acc["Method"].append(val_method_avg_acc)
            if self.val_target is None:
                self.val_target = 0
            self.avg_acc["Oracle"].append(self.val_target) 
        else:
            self.avg_acc["Oracle"].append(val_method_avg_acc)

        # self.steps_of_interest.append(self.step_count)
        # print (self.steps_of_interest)
        self.times_of_interest.append(round(self.batch_time.sum,3))
        # print (self.times_of_interest)

    def data_weighting(self, dataset, num_seen=None):

        self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
        # cuda
        if self.cuda:
            self.dw_k = self.dw_k.cuda()

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    def save_model(self, filename):
        model_state = self.model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving class model to:', filename)
        torch.save(model_state, filename + 'class.pth')
        self.log('=> Save Done')

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename + 'class.pth'))
        self.log('=> Load Done')
        if self.gpu:
            self.model = self.model.cuda()
        # print (self.model)
        self.model.eval()

    def load_prev_model(self, filename):
        self.prev_model= self.create_model()
        self.prev_model.load_state_dict(torch.load(filename + 'class.pth'))
        self.log('=> Load Done')
        if self.gpu:
            self.prev_model = self.prev_model.cuda()
        # print (self.model)
        self.prev_model.eval()

    def load_replay_counts(self, filename):

        
        if self.is_dual_data_loader:
            if not self.replay_strategy:
                if self.class_weighting_with>=10:
                    self.class_replay_ratios = np.load(filename+'class_replay_ratios.npy', allow_pickle=True)[()]
                elif self.class_weighting_with==1:
                    self.class_replay_ratios =  { cl: 1 for cl in self.classes}
                elif self.class_weighting_with==2:
                    self.class_replay_ratios =  { cl: self.class_ratios_manual[cl] for cl in self.class_ratios_manual if cl in self.classes}
            else:
                try:
                    self.class_replay_weights = np.load(file_name+'class_replay_weights.npy', allow_pickle=True)[()]
                except:
                    self.class_replay_weights =  { cl: 1 for cl in self.class_replay_weights if cl in self.classes}
        
        # else:
            # self.class_replay_counts =  np.load(filename)
        

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedulesif self.schedule_type == 'decay':
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    # returns optimizer for passed model
    def create_new_optimizer(self, model):

        # parse optimizer args
        optimizer_arg = {'params':model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.new_optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)

        # create schedules
        if self.schedule_type == 'cosine':
            self.new_scheduler = CosineSchedule(self.new_optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.new_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.new_optimizer, milestones=self.schedule, gamma=0.1)


    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim)

        return model

    def print_model(self):
        self.log(self.model)
        self.log('#parameter of model:', self.count_parameter())
    
    def reset_model(self):
        self.model.apply(weight_reset)

    def forward(self, x, pen=False, balanced=False):
        if pen:
            output, new_feats = self.model.forward(x, pen=True, balanced=balanced)
            return output[:, :self.valid_out_dim], new_feats
        return self.model.forward(x, balanced=balanced)[:, :self.valid_out_dim]

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        return out
    
    def add_valid_output_dim(self, dim=0):

        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())   

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log("Running on:", device)
        return device

    def pre_steps(self):
        pass

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def accumulate_acc(output, target, meter, topk):
    meter.update(accuracy(output, target, topk), len(target))
    return meter

def loss_fn_kd(scores, target_scores, data_weights, allowed_predictions, T=2., soft_t = False):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].
    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""


    log_scores_norm = F.log_softmax(scores[:, allowed_predictions] / T, dim=1)
    if soft_t:
        targets_norm = target_scores
    else:
        targets_norm = F.softmax(target_scores[:, allowed_predictions] / T, dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm # * T**2

    return KD_loss

def loss_fn_class_kd(scores):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].
    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    
    scores = pad_tensors(scores)
    #print (scores.shape)

##########################################
#            TEACHER CLASS               #
##########################################

class Teacher(nn.Module):

    def __init__(self, solver):

        super().__init__()
        self.solver = solver

    def generate_scores(self, x, allowed_predictions=None, balanced=False, threshold=None):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat_to_compute_scores, y_feat = self.solver.forward(x, pen=True, balanced=balanced)
        
        y_hat = y_hat_to_compute_scores[:, allowed_predictions]
        
        # set model back to its initial mode
        self.train(mode=mode)

        # threshold if desired
        if threshold is not None:
            # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
            y_hat = F.softmax(y_hat, dim=1)
            ymax, y = torch.max(y_hat, dim=1)
            thresh_mask = ymax > (threshold)
            thresh_idx = thresh_mask.nonzero().view(-1)
            y_hat = y_hat[thresh_idx]
            y = y[thresh_idx]
            return y_hat, y, x[thresh_idx]

        else:
            # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
            ymax, y = torch.max(y_hat, dim=1)

        return y_hat, y, y_feat

    def generate_scores_pen(self, x):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)[1]

        # set model back to its initial mode
        self.train(mode=mode)

        return y_hat

