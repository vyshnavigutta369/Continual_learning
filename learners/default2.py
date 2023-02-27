from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
import models
from utils.metric import accuracy, AverageMeter, Timer, new_vs_old_class_comparison, distribution_shift_comparison, per_class_plots
import copy
import numpy as np
from utils.schedulers import CosineSchedule
from collections import Counter
import os

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
        self.batch_size = learner_config['batch_size']
        self.previous_teacher = None
        self.tasks = learner_config['tasks']
        self.top_k = learner_config['top_k']
        self.task_count = 0
        self.learner_name = learner_config['learner_name']

        # class balancing
        self.dw = self.config['DW']
        self.replay_type = learner_config['replay_type']

        # distillation
        self.DTemp = learner_config['temp']
        self.mu = learner_config['mu']

        # supervised criterion
        self.criterion_fn = nn.CrossEntropyLoss(reduction="none")
        
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

        self.plot_dir = learner_config['plot_dir']
        self.with_class_balance = learner_config['with_class_balance']

        self.replay=False ## TODO REMOVE THIS
        self.replay_size='Oracle'
        self.new_classes = learner_config['new_classes']
        self.old_classes = learner_config['old_classes']

        

    def init_plot_params(self):

        is_oracle=False if self.learner_name!='NormalNN' else True

        self.per_class_accuracy = {"MMD_poly": {}}
        self.per_class_dist_shift = {"MMD_poly": {}}
    
        self.epochs_of_interest = [i for i in range(0, self.config['schedule'][-1], int(self.config['schedule'][-1]/10))]
        if self.epochs_of_interest[1]!=1:
            self.epochs_of_interest.insert(1,1)
        if self.epochs_of_interest[-1]!=self.config['schedule'][-1]:
            self.epochs_of_interest.append(self.config['schedule'][-1])
        if self.replay:
            if not is_oracle:
                self.avg_acc = { "Oracle": [], "Method": []}
                self.task_acc = { "Oracle": {epoch: [] for epoch in self.epochs_of_interest}, "Method": {epoch: [] for epoch in self.epochs_of_interest}}
            else:
                self.avg_acc = { "Oracle": []}
                self.task_acc = { "Oracle": {epoch: [] for epoch in self.epochs_of_interest}}

    def init_params(self):

        is_oracle=False if self.learner_name!='NormalNN' else True

        self.class_mapping = { v: k for k,v in self.train_dataset.class_mapping.items() if k!=-1}
        self.labels_to_names = { v: k for k,v in self.train_dataset.class_to_idx.items()}
        self.per_class_correct_predictions = {}
        self.per_class_old_features = {}
        self.per_class_new_features = {}
        self.per_class_new_data_train = {}
        self.check_replay_counts = { int(i): set() for i in self.class_mapping}
        if not is_oracle and not self.replay:
            # self.old_classes = set([int(y) for _,batch_y,_,_,_,_ in train_loader for y in batch_y ])
            self.old_classes = set([self.train_dataset.class_mapping[int(y)] for y in self.train_dataset.targets])
            self.new_classes = set(self.class_mapping.keys())-set(self.old_classes)

        self.feat_q = torch.Tensor([]).cuda()
        self.labels_q = torch.Tensor([]).cuda()
        self.ind_q = torch.Tensor([]).cuda()
        # self.data_ubl = { cl:  torch.Tensor([]).cuda() for cl in self.class_mapping}
        # self.labels_queued = { cl:  torch.Tensor([]).cuda() for cl in self.class_mapping}


    def process_attr(self):
        self.per_class_correct_predictions = { k: sum(self.per_class_correct_predictions[k].values()) for k in self.per_class_correct_predictions if self.per_class_correct_predictions}
        self.per_class_old_features = { k: list(self.per_class_old_features[k].values()) for k in self.per_class_old_features if self.per_class_old_features}
        self.per_class_new_features = { k: list(self.per_class_new_features[k].values()) for k in self.per_class_new_features if self.per_class_new_features}

    def helper(self, indices, labels, old_features=None, new_features=None, predictions= None, new_features_train=None):

        
        for i,(index, label) in enumerate(zip(indices, labels)):
            
            if predictions is not None:
                prediction = predictions[i]
                if int(label) not in self.per_class_correct_predictions.keys():
                    self.per_class_correct_predictions[int(label)] = {}
                    self.per_class_correct_predictions[int(label)][int(index)] = 0
                if int(prediction) == int(label):
                    self.per_class_correct_predictions[int(label)][int(index)] = 1
            if old_features is not None:
                old_feat = old_features[i]
                if int(label) not in self.per_class_old_features.keys():
                    self.per_class_old_features[int(label)] = {}
                self.per_class_old_features[int(label)][int(index)] = old_feat.tolist()
            if new_features is not None:
                new_feat = new_features[i]
                if int(label) not in self.per_class_new_features.keys():
                    self.per_class_new_features[int(label)] = {}
                self.per_class_new_features[int(label)][int(index)] = new_feat.tolist()
            if new_features_train is not None:
                new_feat_train = new_features_train[i]
                if int(label) not in self.per_class_new_data_train.keys():
                    self.per_class_new_data_train[int(label)] = {}
                self.per_class_new_data_train[int(label)][int(index)] = new_feat_train.tolist()
                            
        # print ('per_class_correct_predictions:', self.per_class_correct_predictions)


    def get_class_balanced_data(self, indices, data, labels):

        self.helper(indices, labels, new_data_train=data)
        
        self.per_class_indices_train = { k: list(self.per_class_new_data_train[k].keys()) for k in self.per_class_new_data_train if self.per_class_new_data_train}
        self.per_class_new_data_train = { k: list(self.per_class_new_data_train[k].values()) for k in self.per_class_new_data_train}

        no_of_samples_per_class_balanced = min([len(self.per_class_new_data_train[cl]) for cl in self.per_class_new_data_train])
    
        data_bl = torch.Tensor([]).cuda()
        indices_bl = torch.Tensor([]).cuda()
        labels_bl = torch.Tensor([]).cuda()

        data_ubl = torch.Tensor([]).cuda()
        indices_ubl = torch.Tensor([]).cuda()
        labels_ubl = torch.Tensor([]).cuda()

        for cl in self.per_class_new_data_train:
            data_bl = torch.cat((data_bl, torch.Tensor(self.per_class_new_data_train[cl][:no_of_samples_per_class_balanced]).cuda() ))
            data_ubl = torch.cat((data_ubl, torch.Tensor(self.per_class_new_data_train[cl][no_of_samples_per_class_balanced:]).cuda()  ))

            indices_bl = torch.cat((indices_bl, torch.Tensor(self.per_class_indices_train[cl][:no_of_samples_per_class_balanced]).cuda()  ))
            indices_ubl = torch.cat((indices_ubl, torch.Tensor(self.per_class_indices_train[cl][no_of_samples_per_class_balanced:]).cuda()  ))

            labels_bl = torch.cat((labels_bl, torch.Tensor([cl for i in range(no_of_samples_per_class_balanced)]).cuda() ))
            labels_ubl= torch.cat((labels_ubl, torch.Tensor([cl for i in range(len(self.per_class_new_data_train[cl])-no_of_samples_per_class_balanced)]).cuda() ))

        # print (data_bl.shape)
        # print (labels_bl.shape)
        data_bl = data_bl.split(self.batch_size)
        labels_bl = labels_bl.split(self.batch_size)
        
        self.per_class_indices_train.clear()
        self.per_class_new_data_train.clear()
        return indices_bl, data_bl, labels_bl, indices_ubl, data_ubl, labels_ubl

    def update_balanced_head(self, features, labels):
        
        loss=0.0
        output = torch.Tensor([]).cuda()
        labels_joined = torch.Tensor([]).cuda() ##TODO 

        for feat, y in zip(features, labels):

            y_pred = self.model.classifier_head_forward(feat)
            loss += self.update_model(y_pred, y, new_optimizer=True)
            output = torch.cat((output, y_pred))
            labels_joined = torch.cat((labels_joined, y))

        return loss, output, labels_joined

    def analyze(self):

        self.process_attr()
        # print ('per class correct predictions: ', self.per_class_correct_predictions)

        is_oracle=False if self.learner_name!='NormalNN' else True

        if not is_oracle and not self.replay and self.epoch==self.epochs_of_interest[-1]:
            class_accuracy_epoch, class_dist_shift_epoch = distribution_shift_comparison(self.per_class_new_features, self.per_class_old_features, self.per_class_correct_predictions)
            
            if int(self.weight_with)==0:
                self.class_replay_counts = { i:  self.replay_size for i in self.old_classes}
            else:
                if int(self.weight_with)==1:
                    class_dist_shift_epoch = np.mean([ list(class_dist_shift_epoch[k].values()) for k in class_dist_shift_epoch], axis=0 )
                    class_replay_ratios = self.weight_replay(class_dist_shift_epoch)
                elif int(self.weight_with)==2:
                    class_accuracy_epoch  = np.array(list(class_accuracy_epoch.values()))
                    class_replay_ratios = self.weight_replay(class_accuracy_epoch)
                else:
                    class_replay_ratios =  {'airplane': 3, 'automobile': 1, 'deer': 6, 'dog': 10, 'frog': 6, 'horse': 3, 'ship': 3, 'truck': 1}
                    class_replay_ratios = { self.train_dataset.class_mapping[self.train_dataset.class_to_idx[cl]]: class_replay_ratios[cl] for cl in class_replay_ratios}
                factor = (len(self.old_classes)* self.replay_size)/sum(class_replay_ratios.values())
                self.class_replay_counts={ k: class_replay_ratios[k]*factor for k in class_replay_ratios }
            
            np.save(self.model_save_dir+'class_replay_counts.npy', self.class_replay_counts)

            print ('class_replay_counts:', self.class_replay_counts)

        if self.replay and self.epoch in self.epochs_of_interest:    

            # print ('yeaaahhhh')
            if self.epoch>0:
                base_path=self.plot_dir+'_after/new_vs_old/epoch_' + str(self.epoch)+'/'
            else:
                base_path=self.plot_dir+'_before/new_vs_old/'
            if not os.path.exists(base_path): os.makedirs(base_path)
            sim_table = new_vs_old_class_comparison(self.new_classes, self.old_classes, self.per_class_new_features, self.labels_to_names, self.class_mapping, replay_size = self.replay_size, base_path=base_path, is_oracle=is_oracle)
            # class_replay_weights = self.weight_replay(sim_table)

            self.class_accuracy_epoch, class_dist_shift_epoch = distribution_shift_comparison(self.per_class_new_features, self.per_class_old_features, self.per_class_correct_predictions)
            
            # self.per_class_accuracy = { key: { key2: self.per_class_accuracy[key].get(key2,[])+[class_accuracy_epoch[key].get(key2,[])] for key2 in class_accuracy_epoch[key]} for key in class_accuracy_epoch }
            self.per_class_accuracy = { key: self.per_class_accuracy.get(key,[])+[self.class_accuracy_epoch.get(key,[])] for key in self.class_accuracy_epoch }
            self.per_class_dist_shift = { key: { key2: self.per_class_dist_shift[key].get(key2,[])+[class_dist_shift_epoch[key].get(key2,[])] for key2 in class_dist_shift_epoch[key]} for key in class_dist_shift_epoch }

            if not is_oracle:
                self.task_acc["Method"][self.epoch].append(self.val_method_task_acc)  
                self.task_acc["Oracle"][self.epoch].append(self.val_target)  
            else:
                self.task_acc["Oracle"][self.epoch].append(self.val_method_task_acc)  
            if self.epoch == self.epochs_of_interest[-1]:
                np.save(self.plot_dir+'_after/per_class_accuracy.npy', self.per_class_accuracy)
                np.save(self.plot_dir+'_after/per_class_dist_shift.npy', self.per_class_dist_shift) 
                self.task_acc = { m: [round(sum(self.task_acc[m][k])/len(self.task_acc[m][k]),2) for k in self.task_acc[m] ] for m in self.task_acc }
                # print ('task acc: ', self.task_acc)
        
        # print ('per_class_accuracy: ', self.per_class_accuracy)
        # print ('per_class_dist_shift: ',self.per_class_dist_shift)



    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, replay_dataset, model_save_dir, val_target, task, val_loader=None):
        
        # try to load model
        need_train = True
        self.train_dataset = train_dataset
        self.val_target = val_target

        task_index = self.tasks.index(task)+1
        if self.tasks[0]!= task:
            self.replay=True ## TODO need to chage this
            model_save_dir_prev = model_save_dir.replace('task-'+str(task_index), 'task-'+str(task_index-1))
            # print (model_save_dir_prev)
            self.load_prev_model(model_save_dir_prev)
            self.previous_teacher = Teacher(solver=self.prev_model)
            self.last_valid_out_dim = len(self.tasks[task_index-2])
            # print (self.previous_teacher)

        self.init_params()

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
            
            # # Evaluate the performance of current task
            # self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            # if val_loader is not None:
            #     self.validation(val_loader)
        
            losses = AverageMeter()
            losses_bl = AverageMeter()
            acc = AverageMeter()
            acc_bl = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()

            self.init_plot_params()


            for epoch in range(self.config['schedule'][-1]+1):
                self.epoch=epoch

                if epoch > 1: 
                    self.scheduler.step()
                    if hasattr(self, 'new_scheduler'):
                        self.new_scheduler.step()
                        
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()

                self.init_params()

                for i, (x, y, indices, _, _, _)  in enumerate(train_loader):

                    # verify in train mode
                    if epoch==0:
                        break

                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x =x.cuda()
                        y = y.cuda()
                    
                    # model update
                    output, new_feats = self.model.forward(x, pen=True)    
                    loss = self.update_model(output, y)
                    # loss, output= self.update_model(x, y)

                    if self.with_class_balance==1:
                        output_bl, y_bl = self.get_class_balanced_data(indices, y.detach(), new_feats.detach())
                        loss_bl = self.update_model(output_bl, y_bl, new_optimizer=True)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    
                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0)) 
                    batch_timer.tic()

                    if self.with_class_balance==1:
                        accumulate_acc(output_bl, y_bl, acc_bl, topk=(self.top_k,))
                        losses_bl.update(loss_bl,  y_bl.size(0))

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))
                if self.with_class_balance==1:
                    self.log(' * BL head train Loss {loss.avg:.3f} | BL Train Acc {acc.avg:.3f}'.format(loss=losses_bl,acc=acc_bl))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.val_method_task_acc = self.validation(val_loader, train_val=True)
                    self.analyze()
                    if self.replay and epoch in self.epochs_of_interest:
                        val_method_avg_acc = float(sum(self.class_accuracy_epoch.values()))/len(self.class_accuracy_epoch)*100
                        self.avg_acc["Oracle"].append(val_method_avg_acc)

                # reset
                losses = AverageMeter()
                losses_bl = AverageMeter()
                acc = AverageMeter()
                acc_bl = AverageMeter()
                
        self.model.eval()

        if self.replay:
            per_class_plots(self.per_class_accuracy, self.per_class_dist_shift, self.labels_to_names, self.class_mapping, self.epochs_of_interest, self.replay_size,  self.avg_acc, self.task_acc, base_path=self.plot_dir+'_after/', is_oracle=True)
        

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False
        self.replay=True ## TODO REMOVE THIS

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)

        try:
            return batch_time.avg, self.config['schedule'][-1]
        except:
            return None, None

    def criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised 

    def update_model(self, logits, targets, new_optimizer=False):
        
        if self.replay_type == 'random_sample':
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
            # print (dw_cls.shape)
        elif self.replay_type == 'gradient_cb':
            h_m = Counter([int(i) for i in targets]) ## map
            h_m = { i: 1/h_m[i] for i in h_m}
            # dw_a = self.dw_k[-1 * torch.ones(len(targets)-len(logits_replay)).long()]
            # dw_b = self.dw_k[-1 * torch.ones(len(logits_replay)).long()]
            # dw_cls = torch.cat([dw_a, dw_b])*torch.Tensor([h_m[int(i)] for i in targets]).to("cuda")
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]*torch.Tensor([h_m[int(i)] for i in targets]).cuda()

        total_loss = self.criterion(logits, targets.long(), dw_cls)
        
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
    def validation(self, dataloader, model=None, task_in = None,  verbal = True, train_val=False):

        train= True
        if self.with_class_balance==1:
            train=False

        # print (self.epoch)

        if model is None:
            model = self.model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()

        # all_targets = set()
        for i, (input, target, indices) in enumerate(dataloader):
            # for lab in target:
            #     all_targets.add(int(lab))
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            if task_in is None:
                # output = model.forward(input)[:, :self.valid_out_dim]
                output, new_feats = model.forward(input, pen=True, train=train)
                acc = accumulate_acc(output, target, acc, topk=(self.top_k,))

                if not train_val:
                    continue
                
                # if not self.replay and self.epoch<self.epochs_of_interest[-1]:
                #     if self.epoch==1:
                #         if not hasattr(self, 'old_feats'):
                #             self.old_feats=[]
                #         self.old_feats.append(new_feats)
                #     continue
                # elif not self.replay and self.epoch==self.epochs_of_interest[-1]:
                #     old_feats = self.old_feats[i]
                if self.replay:
                    # print (self.epoch)
                    _, _, old_feats = self.previous_teacher.generate_scores(input, allowed_predictions=list(range(self.last_valid_out_dim)), train=train)
                    predictions = torch.argmax(output, dim=1)
                
                    self.helper(indices.tolist(), target, old_feats, new_feats, predictions)
            else:
                mask = target >= task_in[0]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]

                mask = target < task_in[-1]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]
                
                if len(target) > 1:
                    output = model.forward(input, train=train)[:, task_in]
                    acc = accumulate_acc(output, target-task_in[0], acc, topk=(self.top_k,))

        model.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        return acc.avg



    @torch.no_grad()
    def validation_old(self, dataloader, model=None, task_in = None,  verbal = True, train_val=False):

        train= False
        if self.with_class_balance==0:
            train=True

        if model is None:
            model = self.model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()
        for i, (input, target,_) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            if task_in is None:
                # output = model.forward(input)[:, :self.valid_out_dim]
                output, new_feats = model.forward(input, pen=True, train=train)
                acc = accumulate_acc(output, target, acc, topk=(self.top_k,))
            else:
                mask = target >= task_in[0]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]

                mask = target < task_in[-1]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]
                
                if len(target) > 1:
                    output = model.forward(input, train=train)[:, task_in]
                    acc = accumulate_acc(output, target-task_in[0], acc, topk=(self.top_k,))
            
        model.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        return acc.avg


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

        if self.weight_with==0:
            self.class_replay_counts = { i:  self.replay_size for i in self.old_classes}
        elif self.weight_with==1:
            self.class_replay_counts =  np.load(filename)
        elif self.weight_with==3:
            class_replay_ratios =  {'airplane': 3, 'automobile': 1, 'deer': 6, 'dog': 10, 'frog': 6, 'horse': 3, 'ship': 3, 'truck': 1}
            class_replay_ratios = { self.train_dataset.class_mapping[self.train_dataset.class_to_idx[cl]]: class_replay_ratios[cl] for cl in class_replay_ratios}
            factor = (len(self.old_classes)* self.replay_size)/sum(class_replay_ratios.values())
            self.class_replay_counts={ k: class_replay_ratios[k]*factor for k in class_replay_ratios }

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

    def forward(self, x):
        return self.model.forward(x)[:, :self.valid_out_dim]

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

##########################################
#            TEACHER CLASS               #
##########################################

class Teacher(nn.Module):

    def __init__(self, solver):

        super().__init__()
        self.solver = solver

    def generate_scores(self, x, allowed_predictions=None, train=True, threshold=None):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat_to_compute_scores, y_feat = self.solver.forward(x, pen=True, train=train)
        # print (y_hat_to_compute_scores)
        # print ('y_hat_to_compute_scores: ', y_hat_to_compute_scores.shape)
        # print ('allowed_predictions:', allowed_predictions)
        y_hat = y_hat_to_compute_scores[:, allowed_predictions]
        # print (y_hat)
        # print ('y_hat: ', y_hat.shape)
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
