from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T

from utils.metric import AverageMeter, Timer, new_vs_old_class_comparison, distribution_shift_comparison, per_class_plots
from models.resnet import BiasLayer
from .default import NormalNN,  accumulate_acc, loss_fn_kd, Teacher
from dataloaders.utils import transform_with_pca, make_video_ffmpeg, visualise, create_video, SupConLoss
from sklearn.cluster import KMeans

import numpy as np
import copy
import os, math
from collections import Counter

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
        # self.labels_to_names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}  ## to read from config
        # self.class_mapping = {0: 0, 1: 1, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 2, 9: 3} ## to read from config

        self.tasks = learner_config['tasks']

        self.loss_type = learner_config['loss_type']

        self.weight_with = learner_config['weight_with']
        self.weight_reverse = learner_config['weight_reverse']
        

        # self.weight_with_sim = learner_config['weight_with_sim']

    def weight_replay(self, data, n_clusters=4):

        ## data dict of class: sim value to a target
        classes = self.class_mapping.keys()
        X = data
        
        # print ('data:', X)

        kmeans =  KMeans(n_clusters).fit(X.reshape(-1,1))  ## TODO tune n_clusers
        cluster_centers = kmeans.cluster_centers_.flatten()
        # cluster_centers = [ (x-np.mean(cluster_centers))/np.var(cluster_centers) for x in cluster_centers]
        cluster_centers = [ x/sum(cluster_centers) for x in cluster_centers]
        # print ('cluster centers normed:', cluster_centers)
        cluster_labels = kmeans.labels_
        # print ('cluster_labels:', cluster_labels)
        factor =  float(1/min(cluster_centers))
        # print (cluster_centers*factor)
        cluster_replay_ratio = { i: math.ceil(ratio) for i, ratio in enumerate(([ x*factor for x in cluster_centers]))}
        if self.weight_reverse:
            
            cluster_replay_ratio_sorted = { k: v for k,v in sorted(cluster_replay_ratio.items(), key=lambda x:x[1])}
            cluster_replay_ratio_vals = sorted(cluster_replay_ratio_sorted.values(), reverse=True)
            cluster_replay_ratio_sorted = { k: cluster_replay_ratio_vals[i] for i,k in enumerate(cluster_replay_ratio_sorted)}
            cluster_replay_ratio = { k: cluster_replay_ratio_sorted[k] for k in cluster_replay_ratio}


        class_replay_ratio = { cl: cluster_replay_ratio[label] for cl, label in zip(classes,cluster_labels) if cl not in self.new_classes}
        print ('class_replay_ratio: ', { self.labels_to_names[self.class_mapping[cl]]: class_replay_ratio[cl]for cl in class_replay_ratio})

        # print ('cluster labels:', cluster_labels)
        return class_replay_ratio
        
    
    ##########################################
    #           MODEL TRAINING               #
    ##########################################


    def learn_batch(self, train_loader, train_dataset, replay_dataset, model_save_dir, val_target, task, val_loader=None):
        
        # try to load model
        need_train = True

        self.train_dataset = train_dataset
        self.val_target = val_target
        self.init_params()

        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                self.load_replay_counts(model_save_dir+'class_replay_counts.npy')
                need_train = False
            except:
                pass

        self.model_save_dir = model_save_dir
        print ('model_save_dir: ', model_save_dir)

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
            batch_time = AverageMeter()
            batch_timer = Timer()

            self.init_plot_params()
            

            # print ('Epochs of interest: ', self.epochs_of_interest)
            # print ('Val target:', val_target)

            for epoch in range(0,self.config['schedule'][-1]+1):

                self.epoch=epoch

                if epoch > 1: 
                    self.scheduler.step()
                    if hasattr(self, 'new_scheduler'):
                        self.new_scheduler.step()

                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()

                self.init_params()
                
    
                for i, (x, y, indices, x_r, y_r, replay_indices)  in enumerate(train_loader):

                    
                    if epoch==0:
                        break
                    # verify in train mode

                    self.model.train()
                    
                    if self.gpu:
                        x =x.cuda()
                        y = y.cuda()
                        if self.replay:
                            x_r = x_r.cuda()
                            y_r = y_r.cuda()
                            x_tot = torch.cat([x, x_r])
                            y_tot = torch.cat([y, y_r])
                            indices = indices.tolist() + replay_indices.tolist()
                        else:
                            indices = indices.tolist()
                            x_tot = x
                            y_tot = y
                    
                    
                    if self.replay:
                        # transform = T.ToPILImage()
                        # img = transform(train_dataset.data[indices[0]])
                        # img.save('training_images_see/'+str(i)+'.png')
                        for l,ind in zip(y_r,replay_indices):
                            self.check_replay_counts[int(l)].add(int(ind))

                        # print (len(replay_indices))

                        y_hat, _, _ = self.previous_teacher.generate_scores(x_tot, allowed_predictions= list(range(self.last_valid_out_dim)))
                        output, new_feats = self.model.forward(x_tot, pen=True)    
                        loss = self.update_model(output, y_tot, output[len(x):], y_hat[len(x):])
                        # loss, output= self.update_model(y_pred, y_tot)
                    else:
                        output, new_feats = self.model.forward(x, pen=True)
                        loss= self.update_model(output, y, torch.Tensor([]))

                    if self.with_class_balance==1:

                        feats_bl, labels_bl, ind_q, feat_q, labels_q = self.get_class_balanced_data(indices, new_feats.detach(), y_tot)
                        loss_bl, output_bl, labels_bl = self.update_balanced_head(feats_bl, labels_bl)

                        self.feat_q = torch.cat((self.feat_q, feat_q ))
                        self.labels_q = torch.cat((self.labels_q, labels_q ))
                        self.ind_q = torch.cat((self.ind_q, ind_q ))

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    
                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output[:len(y)], y, acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0)) 
                    batch_timer.tic()
                    
                    if self.with_class_balance==1:
                        accumulate_acc(output_bl, labels_bl, acc_bl, topk=(self.top_k,))
                        losses_bl.update(loss_bl,  labels_bl.size(0)) 

                    # print (self.blah)
                    

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch,total=self.config['schedule'][-1]))
                self.log(' * Train Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))
                if self.with_class_balance==1:
                    self.log(' * BL head train Loss {loss.avg:.3f} | BL Train Acc {acc.avg:.3f}'.format(loss=losses_bl,acc=acc_bl))

                ## PLOTS AND TABLES
                

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.val_method_task_acc = self.validation(val_loader, train_val=True)
                    
                    self.analyze()
                    if self.replay and epoch in self.epochs_of_interest:
                        val_method_avg_acc = float(sum(self.class_accuracy_epoch.values()))/len(self.class_accuracy_epoch)*100
                        # val_target = 0 ##TODO REMOVE THIS 
                        self.avg_acc["Method"].append(val_method_avg_acc)
                        self.avg_acc["Oracle"].append(val_target) 
                        
                # reset
                losses = AverageMeter()
                losses_bl = AverageMeter()
                acc = AverageMeter()
                acc_bl = AverageMeter()

            
            if self.with_class_balance==1:
                feats_bl, labels_bl, _, _, _= self.get_class_balanced_data(self.ind_q, self.feat_q, self.labels_q)
                loss_bl, output_bl, labels_bl = self.update_balanced_head(feats_bl, labels_bl)

                accumulate_acc(output_bl, labels_bl, acc_bl, topk=(self.top_k,))
                losses_bl.update(loss_bl,  labels_bl.size(0)) 
                self.log(' * EOE BL head train Loss {loss.avg:.3f} | BL Train Acc {acc.avg:.3f}'.format(loss=losses_bl,acc=acc_bl))

            # print (self.labels_queued[cl].shape)
            print ('check replay_counts: ', {self.labels_to_names[self.class_mapping[cl]]: len(self.check_replay_counts[cl]) for cl in self.check_replay_counts})

            if self.replay:
                per_class_plots(self.per_class_accuracy, self.per_class_dist_shift, self.labels_to_names, self.class_mapping, self.epochs_of_interest, self.replay_size,  self.avg_acc, self.task_acc, base_path=self.plot_dir+'_after/')
        

        self.model.eval()

        
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)

        if not self.replay:
            # if not hasattr(self, 'class_replay_weights'):
            #     self.class_replay_weights=  {k:1 for k in self.class_mapping}
            replay_dataset.extend(train_dataset, self.class_replay_counts)

        print ('size of replay_dataset:',  len(replay_dataset))
        # print ('plot_save_dir: ', self.plot_dir)

        self.replay=True

        try:
            return batch_time.avg, self.config['schedule'][-1]
        except:
            return None, None

    def update_model(self, logits, targets, logits_replay= [], target_scores = [], dw_scale = 1, new_optimizer=False):
        
        if self.replay_type == 'random_sample' or new_optimizer:
            # if new_optimizer:
            #     for i in targets:
            #         self.blah[int(i)]+=1
            #     print (self.blah)
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
            # print (dw_cls.shape)
        elif self.replay_type == 'gradient_cb':
            h_m = Counter([int(i) for i in targets]) ## map
            b_t = max(h_m.values())
            h_m = { i: b_t/h_m[i] for i in h_m}
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]*torch.Tensor([h_m[int(i)] for i in targets]).cuda()
            # dw_a = self.dw_k[-1 * torch.ones(len(targets)-len(logits_replay)).long()]
            # dw_b = self.dw_k[-1 * torch.ones(len(logits_replay)).long()]
            # print ('dw_a:', dw_a)
            # print ('dw_b:',dw_b)
            # dw_cls = torch.cat([dw_a, dw_b])*torch.Tensor([h_m[int(i)] for i in targets]).to("cuda")
            # dw_cls = torch.cat([dw_a, dw_b])
            # print (dw_cls)
        
        total_loss = self.criterion(logits, targets.long(), dw_cls)
        # print ('total_loss: ', total_loss)
        if self.loss_type == 'pred_kd' and len(target_scores)>0:   # KD
            # print ('pred_kd')
            dw_KD = self.dw_k[-1 * torch.ones(len(target_scores),).long()]
            loss_distill = loss_fn_kd(logits_replay, target_scores, dw_KD, np.arange(self.last_valid_out_dim).tolist(), self.DTemp)
            total_loss += self.mu * loss_distill

        if not new_optimizer:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        else:
            self.new_optimizer.zero_grad()
            total_loss.backward()
            self.new_optimizer.step()
        return total_loss.detach()

    

class Replay(NormalNN):

    def __init__(self, learner_config):
        super(Replay, self).__init__(learner_config)
        self.previous_teacher = None
        self.replay = False
        self.past_tasks = [] # [[0,1],[2,3]] -> current task is [4,5]

        # contributions
        self.loss_type = self.config['loss_type'] # loss done on replay data (e.g., KD loss)
        self.replay_type = self.config['replay_type'] # how do we sample replay data (e.g., random)
        self.ub_rat = self.config['ub_rat'] # ratio of upper bound to train towards as goal (e.g., 0.98)

        self.args = learner_config
        self.num_samples = self.config['batch_size_replay']
        self.replay_strategy = self.config['replay_type']
        self.replay_loader = None
        
    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, replay_dataset, model_save_dir, val_target, val_loader=None):
        
        # update target
        #print ('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', model_save_dir)
        val_target = self.ub_rat * val_target

        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # train
        if need_train:
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()

            # data weighting
            self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
            # cuda
            if self.cuda:
                self.dw_k = self.dw_k.cuda()
            
            # Evaluate the performance of current task
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)
        
            losses = [AverageMeter() for l in range(3)]
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            epoch = -1
            val_epoch = 0
            while epoch < self.config['schedule'][-1] and val_epoch < val_target:
                epoch += 1
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                count= 0
                replay_ixs_selected = []
                for i, (x,y,x_r,y_r)  in enumerate(train_loader): # _r = replay

                    # verify in train mode
                    count += 1
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                        if self.replay:
                            x_r = x_r.cuda()
                            y_r = y_r.cuda()
                        else:
                            x_r = torch.Tensor([]).cuda()
                            y_r = torch.Tensor([]).cuda()

                    # if replay
                    if self.replay:
                        allowed_predictions = list(range(self.last_valid_out_dim))
                        y_hat, _ = self.previous_teacher.generate_scores(x_r, allowed_predictions=allowed_predictions)
                    else:
                        y_hat = None

                    # model update - training data
                    loss, loss_class, loss_distill, output, _ = self.update_model([x, x_r], [y, y_r], y_hat)

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
                #print ('count:', count)
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))
                self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))
                self.log(' * Target Val Acc = ' + str(val_target))

                # Evaluate the performance of current task
                val_epoch = self.validation(val_loader)

                # reset
                losses = [AverageMeter() for l in range(3)]
                acc = AverageMeter()

        self.model.eval()
        self.past_tasks.append(np.arange(self.last_valid_out_dim,self.valid_out_dim))
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False
        
        # Extend memory
        self.task_count += 1
        replay_dataset.extend(train_dataset)

        # new teacher
        teacher = Teacher(solver=self.model)
        
        self.previous_teacher = copy.deepcopy(teacher)

        self.replay = True
        try:
            return batch_time.avg, self.epoch+1
        except:
            return None, None

    def initialize_replay_loader(self, replay_data, replay_ixs, replay_count_dict, proba_list, task_counter):

        print('\nInitializing replay loader...')
        # self.batch_sampler = ReplayBatchSampler(replay_ixs, replay_count_dict, proba_list, task_counter,
        #                                       self.num_samples, replay_strategy=self.replay_type)
        # self.replay_dataset = ReplayDataset(data, return_item_ix=True, return_task_labels=True)
        # self.replay_loader = torch.utils.data.DataLoader(dataset = self.replay_dataset, num_workers=8, batch_sampler=self.batch_sampler)

        # print ('Replay loader beginning size: ', len(set(self.replay_loader.batch_sampler.replay_ixs)))
        self.replay_loader = ReplayBatchloader(replay_ixs, replay_count_dict, proba_list, task_counter, self.num_samples, 
                                                    torch.Tensor(replay_data.data), torch.Tensor(replay_data.targets), replay_strategy=self.replay_type)
        print ('Replay loader beginning size: ', len(set(self.replay_loader.replay_ixs)))

    def fill_replay_loader_with(self, train_loader):

        replay_ixs = []
        proba_list = []
        replay_count_dict = []
        task_counter = []
        for (x, y, batch_item_ixs) in train_loader:
            if self.gpu:
                x = x.cuda()
                y = y.cuda()
            output = self.forward(x)
            
            replay_probabilities = compute_replay_probabilities(self.replay_strategy, output, y, self.dw_k)
            

            replay_ixs.extend([int(ix.item()) for ix in batch_item_ixs])
            proba_list.extend([prob.detach().cpu() for prob in replay_probabilities])
            replay_count_dict.extend([int(self.args['schedule'][0]) for _ in batch_item_ixs])
            task_counter.extend([self.task_count+1 for _ in batch_item_ixs])

        return replay_ixs, torch.Tensor(proba_list), torch.Tensor(replay_count_dict), torch.Tensor(task_counter)

    def update_replay_loader(self, replay_ixs, replay_probabilities, replay_data=None, is_train_data = False):
        
        
        # print (len(replay_ixs), len(replay_probabilities))
        if not is_train_data:
            for ii, (v, replay_probability) in enumerate(zip(replay_ixs, replay_probabilities)):
                try:               
                    curr_ix = int(v.item())
                except:
                    curr_ix = v
                    
                # curr_ix = v
                # self.replay_loader.batch_sampler.replay_ixs_selected.append(curr_ix)
                location = self.replay_loader.replay_ixs.index(curr_ix)
                self.replay_loader.replay_count_dict[location] += 1  # replayed 1 more time
                if 'random' not in self.replay_strategy and 'feature' not in self.replay_strategy:
                    self.replay_loader.proba_list[location] = float(replay_probability.item())
        else:
            replay_loader_size  = len(self.replay_loader.replay_ixs)
            replay_ixs= [replay_loader_size+replay_ix for replay_ix in replay_ixs]
            self.replay_loader.update_buffer( replay_data.data, replay_data.targets, torch.Tensor(replay_ixs), self.task_count+1, replay_probabilities)

    def update_all_replay_data(self):

        ixs = [] 
        proba = []       

        replay_loader = torch.utils.data.DataLoader(self.replay_loader.dataset, batch_size=self.num_samples, num_workers=8)

        for (x,y,ix) in replay_loader:

            x_r = torch.Tensor([])
            y_r = torch.Tensor([])
            if self.gpu:
                x = x.cuda()
                y = y.cuda()
                x_r = x_r.cuda()
                y_r = y_r.cuda()

            _, _, _, _, output = self.update_model([x, x_r], [y,y_r], None)
            replay_probabilities = compute_replay_probabilities(self.replay_strategy, output, y, self.dw_k)

            ixs.extend([ int(i.item()) for i in ix])
            proba.extend(replay_probabilities)
            
        self.update_replay_loader(ixs, proba)

    def learn_incremental_batch(self, train_loader, replay_loader_old, train_dataset, replay_dataset, curr_task, model_save_dir, val_target, val_loader=None, cheat= False):
        
        # update target
        val_target = self.ub_rat * val_target

        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass
        
        # data weighting
        self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
        # cuda
        if self.cuda:
            self.dw_k = self.dw_k.cuda()

        if need_train:
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()

            
            # Evaluate the performance of current task
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)
        
            losses = [AverageMeter() for l in range(3)]
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            epoch = -1
            val_epoch = 0
            
            while epoch < self.config['schedule'][-1] and val_epoch < val_target:
                epoch += 1
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()

                
                train_ixs = []
                train_probabilities = []

                replay_ixs_selected = []        
                replay_ixs_selected_old = [] 

                # if self.replay:
                #     replay_loader_iter = iter(self.replay_loader)
                
                self.replay_loader.replay_ixs_selected = []

                for ((x, y, batch_item_ixs), (x_r_old, y_r_old, replay_ixs_old)) in zip(train_loader, replay_loader_old): 

                    # print (len(set(self.replay_loader.batch_sampler.replay_ixs)))
                    # print ('here :', len(set(self.batch_sampler.replay_ixs_selected)))
                    
                    self.model.train()

                    # send data to gpu
                    if self.replay:
                        replay_ixs_selected_old.extend([int(v.item()) for v in replay_ixs_old])
                        
                        # x_r, y_r, replay_ixs = next(replay_loader_iter)
                        x_r, y_r, replay_ixs = self.replay_loader.get_replay_samples()
                        
                        # print ('here2: ', list( set(self.batch_sampler.replay_ixs_selected) & set([int(ix.item()) for ix in replay_ixs]) ))
                        self.replay_loader.replay_ixs_selected.extend([ix for ix in replay_ixs])
                    else:
                        x_r = torch.Tensor([])
                        y_r = torch.Tensor([])

                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                        x_r = x_r.cuda()
                        y_r = y_r.cuda()                   

                    x_size = len(x)
                    # print ('X shape: ', x.shape)
                    # print ('X_r shape:', x_r.shape)
                    
                    # if replay
                    if self.replay:
                        allowed_predictions = list(range(self.last_valid_out_dim))
                        y_hat, _ = self.previous_teacher.generate_scores(x_r, allowed_predictions=allowed_predictions)
                        loss, loss_class, loss_distill, output, total_output = self.update_model([x, x_r], [y, y_r], y_hat)

                        
                        replay_ixs_selected.extend([v for v in replay_ixs])
                        # print ( len([int(v.item()) for v in replay_ixs]), len(set([int(v.item()) for v in replay_ixs])) )
                        replay_probabilities = compute_replay_probabilities(self.replay_strategy, total_output, torch.cat([y,y_r]), self.dw_k)
                        self.update_replay_loader(replay_ixs, replay_probabilities[x_size:])
                        train_probabilities.extend([prob for prob in replay_probabilities[:x_size]])
                        train_ixs.extend([ix for ix in batch_item_ixs])
                            
                    else:
                        y_hat = None
                        loss, loss_class, loss_distill, output, _ = self.update_model([x, x_r], [y,y_r], y_hat)            
                        
                    # model update - training data
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, acc, topk=(self.top_k,))
                    losses[0].update(loss,  y.size(0)) 
                    losses[1].update(loss_class,  y.size(0)) 
                    losses[2].update(loss_distill,  y.size(0)) 
                    batch_timer.tic()

                    print (len(replay_ixs_selected), " " , len(set(replay_ixs_selected)))
                    print (len(replay_ixs_selected_old), " " , len(set(replay_ixs_selected_old)))

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))
                self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))
                self.log(' * Target Val Acc = ' + str(val_target))

                # Evaluate the performance of current task
                val_epoch = self.validation(val_loader)
                
                
                # reset
                losses = [AverageMeter() for l in range(3)]
                acc = AverageMeter()

        
            #print (len(self.replay_loader.batch_sampler.replay_ixs), len(set(self.replay_loader.batch_sampler.replay_ixs)))
        if not self.replay:
            train_ixs, train_probabilities, replay_count_dict, task_counter = self.fill_replay_loader_with(train_loader)
            self.initialize_replay_loader(train_dataset, train_ixs, replay_count_dict, train_probabilities, task_counter)
        else:
            if not need_train:
                train_ixs, train_probabilities, replay_count_dict, task_counter = self.fill_replay_loader_with(train_loader)
            self.update_replay_loader(train_ixs, train_probabilities, train_dataset, True)

        # cheat 1
        if cheat:
            self.update_all_replay_data()

        self.model.eval()
        
        replay_dataset.extend(train_dataset)
        self.past_tasks.append(np.arange(self.last_valid_out_dim,self.valid_out_dim))
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        
        # Extend memory
        self.task_count += 1

        # new teacher
        teacher = Teacher(solver=self.model)

        self.previous_teacher = copy.deepcopy(teacher)          

        self.replay = True
        try:
            return batch_time.avg, self.epoch+1
        except:
            return None, None


    def update_model(self, inputs, targets, target_KD = None, return_replay_scores=False):
        
        total_loss = torch.zeros((1,), requires_grad=True).cuda()

        # classification loss
        if self.replay_type != 'gradient_cb':
            combined_inputs = torch.cat([inputs[0],inputs[1]])
            combined_targets = torch.cat([targets[0],targets[1]])
            dw_cls = self.dw_k[-1 * torch.ones(combined_targets.size()).long()]
            logits = self.forward(combined_inputs)
            loss_class = self.criterion(logits, combined_targets.long(), dw_cls)
            total_loss += loss_class

        # THIS ASSUMES A CLASS BALANCED DATASET
        else:
            combined_inputs = torch.cat([inputs[0],inputs[1]])
            combined_targets = torch.cat([targets[0],targets[1]])

            # get weighting
            dw_a = self.dw_k[-1 * torch.ones(targets[0].size()).long()]
            dw_b = self.dw_k[-1 * torch.ones(targets[1].size()).long()] * self.task_count
            dw_cls = torch.cat([dw_a, dw_b])

            # loss
            logits = self.forward(combined_inputs)
            loss_class = self.criterion(logits, combined_targets.long(), dw_cls)
            total_loss += loss_class

        #else:
        #    print(oopsies)
        
        # other replay loss
        if self.loss_type == 'pred_kd':

            # KD
            if target_KD is not None:
                dw_KD = self.dw_k[-1 * torch.ones(len(target_KD),).long()]
                logits_KD = logits[len(inputs[0]):]
                loss_distill = loss_fn_kd(logits_KD, target_KD, dw_KD, np.arange(self.last_valid_out_dim).tolist(), self.DTemp)
                total_loss += self.mu * loss_distill
            else:
                loss_distill = torch.zeros((1,), requires_grad=True).cuda()
        
        elif self.loss_type == 'base':
            loss_distill = torch.zeros((1,), requires_grad=True).cuda()
        
        else:
            print(oopsiesdoosies)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits[:len(inputs[0])], logits

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


class Mine(NormalNN):

    def __init__(self, learner_config):
        super(Mine, self).__init__(learner_config)
        self.previous_teacher = None
        self.replay = False
        self.past_tasks = [] # [[0,1],[2,3]] -> current task is [4,5]
        # self.buffer = Buffer(learner_config['memory'])
        # contributions
        self.loss_type = self.config['loss_type'] # loss done on replay data (e.g., KD loss)
        self.replay_type = self.config['replay_type'] # how do we sample replay data (e.g., random)
        self.ub_rat = self.config['ub_rat'] # ratio of upper bound to train towards as goal (e.g., 0.98)

        self.args = learner_config
        self.num_samples = self.config['batch_size_replay']
        self.batch_size = self.config['batch_size']
        self.replay_strategy = self.config['replay_type']
        self.buffer_update = self.config['buffer_update']

        self.mse_loss = nn.MSELoss(reduction='none')

        self.sup_loss = SupConLoss()
        self.cpt = int(self.config['num_classes']/len(self.tasks))
        self.reuse_replay_ixs = self.config['reuse_replay_ixs']

        self.replay_item_ixs_all_tasks = []
        

    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, replay_dataset, model_save_dir, val_target, val_loader=None):
        
        # update target
        #print ('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', model_save_dir)
        val_target = self.ub_rat * val_target

        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # train
        
        if need_train:
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()

            # data weighting
            self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
            # cuda
            if self.cuda:
                self.dw_k = self.dw_k.cuda()
            
            # Evaluate the performance of current task
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)
        
            losses = [AverageMeter() for l in range(3)]
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            epoch = -1
            val_epoch = 0
            while epoch < self.config['schedule'][-1] and val_epoch < val_target:
                epoch += 1
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                count= 0
                replay_ixs_selected = []
                batch_ixs = []
                for i, (x,y,ixs, x_r,y_r)  in enumerate(train_loader): # _r = replay

                    # verify in train mode
                    count += 1
                    self.model.train()
                    batch_ixs.extend(ixs.tolist())
                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                        if self.replay:
                            
                            replay_item_ixs, x_r, y_r = self.get_replay_samples(x, replay_dataset, replay_ixs_to_exclude = replay_ixs_selected)
                            replay_ixs_selected.extend(replay_item_ixs.tolist())
                            #sample_ind = np.random.choice(len(replay_dataset), self.num_samples, replace=False)
                            # x_r, y_r = torch.Tensor(np.array(replay_dataset.data))[sample_ind].permute((0,3,1,2)), torch.Tensor(np.array(replay_dataset.targets))[sample_ind]
                            #x_r, y_r = replay_dataset[sample_ind]

                            # x_r, y_r = self.get_replay_samples(x, replay_dataset, replay_ixs_to_exclude = [])
                            #replay_ixs_selected.extend(replay_item_ixs.tolist())
                            x_r = x_r.cuda()
                            y_r = y_r.cuda()
                        else:
                            x_r = torch.Tensor([]).cuda()
                            y_r = torch.Tensor([]).cuda()

                    # print ('2:', y.shape)
                    # print ('3:', y_r.shape)
                    # if replay
                    if self.replay:
                        allowed_predictions = list(range(self.last_valid_out_dim))
                        y_hat, _ = self.previous_teacher.generate_scores(x_r, allowed_predictions=allowed_predictions)
                    else:
                        y_hat = None

                    # model update - training data
                    loss, loss_class, loss_distill, output = self.update_model([x, x_r], [y, y_r])

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
                #print ('count:', count)
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))
                self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))
                self.log(' * Target Val Acc = ' + str(val_target))

                # Evaluate the performance of current task
                val_epoch = self.validation(val_loader)

                # reset
                losses = [AverageMeter() for l in range(3)]
                acc = AverageMeter()

        self.model.eval()
        self.past_tasks.append(np.arange(self.last_valid_out_dim,self.valid_out_dim))
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False
        
        # Extend memory
        self.task_count += 1
        replay_dataset.extend(train_dataset)
        # replay_dataset.new_extend(train_dataset.data[batch_ixs], train_dataset.targets[batch_ixs])


        # new teacher
        teacher = Teacher(solver=self.model)
        
        self.previous_teacher = copy.deepcopy(teacher)

        self.replay = True
        try:
            return batch_time.avg, self.epoch+1
        except:
            return None, None

    def update_model(self, x, y):
        

        total_loss = torch.zeros((1,), requires_grad=True).cuda()
        combined_inputs = torch.cat([x[0],x[1]])
        combined_targets = torch.cat([y[0],y[1]])
        
        x_logits,_ = self.forward(combined_inputs)
        
        total_loss += self.criterion(x_logits, combined_targets.long(), self.dw_k[-1 * torch.ones(combined_targets.size()).long()])
        loss_class = total_loss
        loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        # print (total_loss)
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), x_logits[:len(x[0])].detach()

    def visualiseold(self, logits_x, logits_r, prob_x, prob_r, model_save_dir):

        output_path = Path(model_save_dir+'/replay_analysis/') 
        output_path.mkdir(exist_ok=True, parents=True)
        output_path_files = os.listdir(output_path)

        for item in output_path_files:
            os.remove(os.path.join(output_path, item))
        
        def plot(embeddings, values, label):
            df_embeddings = pd.DataFrame(embeddings)
            df_embeddings = df_embeddings.rename(columns={0:'x1',1:'x2'})
            df_embeddings = df_embeddings.assign(values=values)
            df_embeddings = df_embeddings.assign(label=label)
            fig = px.scatter(
            df_embeddings, x='x1', y='x2',
            color='label', labels={'color': 'label'},
            size = values,
            title = 'Embedding Visualization')

            return fig


        for i, (batch_x, batch_prob_x) in enumerate(zip(logits_x, prob_x)):
            batch_x_and_r = batch_x
            batch_prob_x_and_r = batch_prob_x
            label = ['train' for i in range(len(batch_x))]
            if self.replay:
                # print (type(batch_prob_x_and_r))
                batch_x_and_r.extend(logits_r[i])
                batch_prob_x_and_r.extend(prob_r[i])
                label.extend(['replay' for i in range(len(logits_r[i]))])
            
            embeddings = transform_with_pca(batch_x_and_r)
            fig = plot(embeddings, values = batch_prob_x_and_r, label = label)
            fig.write_image(output_path/ (f"frame%06d.png" % i), width=1920, height=1080)
        
        vid = make_video_ffmpeg(output_path, "tsne_visualisation.mp4", fps=10, frame_filename=f"frame%06d.png")
        output_path_files = os.listdir(output_path)

        for item in output_path_files:
            if not item.endswith(".mp4"):
                os.remove(os.path.join(output_path, item))

        logits_x_and_r = list(np.concatenate(logits_x, axis=0))
        prob_x_and_r = list(np.concatenate(prob_x, axis=0))
        label =  ['train' for i in range(len(prob_x_and_r))]
        if self.replay:
            logits_r = list(np.concatenate(logits_r, axis=0))
            prob_r = list(np.concatenate(prob_r, axis=0))
            logits_x_and_r.extend(logits_r)
            prob_x_and_r.extend(prob_r)
            label.extend(['replay' for i in range(len(prob_r))])
        embeddings = transform_with_pca(logits_x_and_r)
        fig = plot(embeddings, values = prob_x_and_r, label = label)
        fig.write_image(output_path/"vis.png", width=1920, height=1080)


    def learn_incremental_batch(self, train_loader, replay_loader_old, train_dataset, replay_dataset, curr_task, model_save_dir, val_target, val_loader=None):
        
        # update target
        #print ('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', model_save_dir)
        val_target = self.ub_rat * val_target

        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # train
        # print (self.reuse_replay_ixs)
        if need_train:
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()

            # data weighting
            self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
            # cuda
            if self.cuda:
                self.dw_k = self.dw_k.cuda()
            
            # Evaluate the performance of current task
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)
        
            losses = [AverageMeter() for l in range(3)]
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            epoch = -1
            val_epoch = 0

            task_labels = np.vstack([ list(np.arange(self.last_valid_out_dim,self.valid_out_dim)) for _ in range(len(train_dataset.data))])
            task_counter = np.array([curr_task for _ in range(len(train_dataset.data))])

            while epoch < self.config['schedule'][-1] and val_epoch < val_target:
                epoch += 1
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                
                replay_ixs_selected = []
                batch_ixs = []
                batch_probabilities_x = []
                batch_logits_x =[]
                for i, (x,y,ixs)  in enumerate(train_loader): # _r = replay

                    # verify in train mode
                    
                    self.model.train()
                    
                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                        if self.replay:
                            if self.reuse_replay_ixs==0:
                                replay_item_ixs, x_r, y_r = self.get_replay_samples(x, replay_dataset, replay_ixs_to_exclude = replay_ixs_selected)
                            else:
                                replay_item_ixs, x_r, y_r = self.get_replay_samples(x, replay_dataset, replay_ixs_to_exclude = [])
                            replay_ixs_selected.extend(replay_item_ixs.tolist())
                            x_r = x_r.cuda()
                            y_r = y_r.cuda()
                        else:
                            replay_item_ixs = []
                            x_r = torch.Tensor([]).cuda()
                            y_r = torch.Tensor([]).cuda()

                    # if replay
                    if self.replay:
                        allowed_predictions = list(range(self.last_valid_out_dim))
                        y_hat, _ = self.previous_teacher.generate_scores(x_r, allowed_predictions=allowed_predictions)
                    else:
                        y_hat = None

                   
                    loss, loss_class, loss_distill, output, logits, replay_probabilities = self.update_model_incremental( epoch, [x, x_r], [y, y_r], replay_dataset, replay_item_ixs, y_hat, val_epoch, val_target)       
                    
                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, acc, topk=(self.top_k,))
                    losses[0].update(loss,  y.size(0)) 
                    losses[1].update(loss_class,  y.size(0)) 
                    losses[2].update(loss_distill,  y.size(0)) 
                    batch_timer.tic()

                    #if epoch == self.config['schedule'][-1] or val_epoch+5 >= val_target:
                        # visualise(replay_dataset, batch_logits_x[-1], batch_probabilities_x[-1], replay_item_ixs, model_save_dir, f"frame%06d.png" % i)

                    batch_ixs.extend(ixs.tolist())
                    batch_logits_x.append(logits.tolist())
                    batch_probabilities_x.append(replay_probabilities.tolist())
                    
                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))
                self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))
                self.log(' * Target Val Acc = ' + str(val_target))

                # Evaluate the performance of current task
                val_epoch = self.validation(val_loader)

                # reset
                losses = [AverageMeter() for l in range(3)]
                acc = AverageMeter()

        # create_video(model_save_dir)
        # visualise(replay_dataset, np.concatenate(batch_logits_x, axis=0).tolist(), np.concatenate(batch_probabilities_x, axis=0).tolist(), replay_ixs_selected, model_save_dir, "vis.png")
        if self.replay:
            self.replay_item_ixs_all_tasks.extend(replay_ixs_selected)
            # print (len(set(self.replay_item_ixs_all_tasks)))
        self.model.eval()
        self.past_tasks.append(np.arange(self.last_valid_out_dim,self.valid_out_dim))
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False
        
        # Extend memory
        self.task_count += 1
        
        batch_ixs = np.array(batch_ixs).argsort()
        replay_dataset.extend_replay(train_dataset, proba=np.concatenate(batch_probabilities_x, axis=0)[batch_ixs], logits= np.concatenate(batch_logits_x, axis=0)[batch_ixs].tolist(), task_counter = task_counter, task_labels = task_labels)

        # new teacher
        teacher = Teacher(solver=self.model)
        
        self.previous_teacher = copy.deepcopy(teacher)

        self.replay = True
        try:
            return batch_time.avg, self.epoch+1
        except:
            return None, None

    def update_model_incremental(self, epoch, x, y, replay_dataset=None, replay_item_ixs=None, target_KD=None, val_epoch=None, val_target=None):
        
        total_loss = torch.zeros((1,), requires_grad=True).cuda()
        combined_inputs = torch.cat([x[0],x[1]])
        combined_targets = torch.cat([y[0],y[1]])
        loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        if self.loss_type == 'base' or self.loss_type == 'pred_kd':
               
            x_logits,x_pen_logits = self.forward(combined_inputs)
            total_loss += self.criterion(x_logits, combined_targets.long(), self.dw_k[-1 * torch.ones(combined_targets.size()).long()])
            loss_class = total_loss
            
        elif self.loss_type == 'DERpp':
            # print ('yes1')
            x_logits, x_pen_logits = self.forward(x[0])
            total_loss += self.criterion(x_logits, y[0].long(), self.dw_k[-1 * torch.ones(y[0].size()).long()])
            loss_class = total_loss

            if self.replay:
                x_r = x[1]
                y_r = y[1]

                replay_logits = np.array(replay_dataset.logits)
                replay_task_labels = np.array(replay_dataset.task_labels)
                split_size = int(len(replay_item_ixs)/2)
                x_r1, y_r1, r1_logits, x_r1_task_labels = x_r[:split_size], y_r[:split_size], replay_logits[replay_item_ixs][:split_size], replay_task_labels[replay_item_ixs][:split_size]
                x_r2, y_r2, r2_logits, _ = x_r[split_size:], y_r[split_size:], replay_logits[replay_item_ixs][split_size:], replay_task_labels[replay_item_ixs][split_size:]
                
                idxs = x_r1_task_labels
                r1_new_logits, r1_pen_logits =  self.forward(x_r1)
                r1_new_logits_to_compare = r1_new_logits[:, idxs[0,:]]
                r1_logits_to_compare = torch.Tensor(r1_logits[:, idxs[0,:]]).cuda()
                mse_loss= self.mse_loss(r1_new_logits_to_compare, r1_logits_to_compare).mean()
                total_loss += 0.5 * mse_loss
                
                r2_new_logits, r2_pen_logits =  self.forward(x_r2)
                total_loss += 0.5 * self.criterion(r2_new_logits, y_r2.long(), self.dw_k[-1 * torch.ones(y_r2.size()).long()])
            
                x_logits = torch.cat([x_logits, r1_new_logits, r2_new_logits])
                x_pen_logits = torch.cat([x_pen_logits, r1_pen_logits, r2_pen_logits])
                loss_class = self.criterion(x_logits, combined_targets.long(), self.dw_k[-1 * torch.ones(combined_targets.size()).long()])

        elif self.loss_type == 'SupCon':

            x_logits, x_pen_logits = self.forward(combined_inputs)
            total_loss += self.criterion(x_logits, combined_targets.long(), self.dw_k[-1 * torch.ones(combined_targets.size()).long()])
            loss_class = total_loss

            # print (x_pen_logits.shape)
            bsz =  int(x_pen_logits.shape[0]/2)
            # print ('bsz', bsz)
            f1, f2 = torch.split(x_pen_logits, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=0)
            # print (features.shape)
            # print (combined_targets.shape)
            total_loss = 0.5*self.sup_loss(features, combined_targets, target_labels=list(range(self.task_count*self.cpt, (self.task_count+1)*self.cpt)))

        if self.loss_type == 'pred_kd':
            # KD
            if target_KD is not None:
                
                dw_KD = self.dw_k[-1 * torch.ones(len(target_KD),).long()]
                logits_KD = x_logits[len(x[0]):]
                loss_distill = loss_fn_kd(logits_KD, target_KD, dw_KD, np.arange(self.last_valid_out_dim).tolist(), self.DTemp)
                total_loss += self.mu * loss_distill


        self.optimizer.zero_grad()
        # print (total_loss)
        total_loss.backward()
        self.optimizer.step()

        replay_strategy = self.replay_strategy
        replay_logits = x_logits[len(x[0]):]
        replay_pen_logits = x_pen_logits[len(x[0]):]
        replay_targets =  combined_targets[len(x[0]):]
        replay_ixs = replay_item_ixs
        if 'feature' not in self.replay_strategy:   
            if (self.buffer_update == 'partial_every' or (self.buffer_update == 'partial_last' and (epoch == self.config['schedule'][-1] or val_epoch+5 >= val_target))):
                # print ('-------UPDATING BUFFER------')
                replay_probabilities_buffer = compute_replay_probabilities(self.replay_strategy, replay_logits, replay_targets, self.dw_k)
                self.update_replay_loader(replay_dataset, replay_ixs, replay_probabilities_buffer, replay_pen_logits)
        else:
            self.update_replay_loader(replay_dataset, replay_ixs, replay_logits = replay_pen_logits)
            # elif (self.buffer_update == 'complete_every' or (self.buffer_update == 'complete_last' and epoch == self.config['schedule'][-1])):

            #     if self.replay:
            #         replay_ixs = np.arange(len(replay_dataset)).tolist()
            #         replay_data, replay_targets = replay_dataset[replay_ixs]
            #         replay_data, replay_targets = replay_data.cuda(), replay_targets.cuda()
            #         replay_logits, replay_pen_logits = self.forward(replay_data)

            #     print ('-------UPDATING BUFFER2------')
            #     replay_probabilities_buffer = compute_replay_probabilities(self.replay_strategy, replay_logits, replay_targets, self.dw_k)
            #     self.update_replay_loader(replay_dataset, replay_ixs, replay_probabilities_buffer, replay_pen_logits)
                      
        
        data_logits = x_logits[:len(x[0])]
        data_targets = combined_targets[:len(x[0])]
        # if epoch == self.config['schedule'][-1] or val_epoch+5 >= val_target:   
        if 'feature' in self.replay_strategy and not self.replay:
            replay_strategy = 'random'
        elif 'feature' in self.replay_strategy:
            data_logits = replay_dataset.logits
            data_targets =  x_pen_logits[:len(x[0])]
        replay_probabilities = compute_replay_probabilities(replay_strategy, data_logits, data_targets, self.dw_k)
        # print (sum(replay_probabilities))
        replay_probabilities = replay_probabilities.detach()
        # else:
        #     replay_probabilities = None

        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), x_logits[:len(x[0])].detach(), x_pen_logits[:len(x[0])].detach(), replay_probabilities


    def get_replay_samples(self, train_data, replay_dataset, replay_ixs_to_exclude = []):

        num_replay_samples = min(self.num_samples, len(train_data))
        # allowed_predictions = torch.vstack([ torch.Tensor(np.arange(self.last_valid_out_dim,self.valid_out_dim)) for _ in range(len(train_data))]).long()
        if 'feature' in self.replay_strategy:
            # print ('-------UPDATING BUFFER------')
            
            _, train_logits = self.forward(train_data)
            replay_logits = replay_dataset.logits
            replay_probabilities_buffer =  compute_replay_probabilities(self.replay_strategy, train_logits, replay_logits, self.dw_k)
            self.update_replay_loader(replay_dataset, np.arange(len(replay_dataset)), replay_probabilities_buffer)

        replay_item_ixs, x_r, y_r = replay_dataset.get_data(num_replay_samples, self.replay_strategy, replay_ixs_to_exclude = replay_ixs_to_exclude, return_index=True)
        return replay_item_ixs, x_r, y_r


    def update_replay_loader(self, replay_dataset, replay_ixs, replay_probabilities=None, replay_logits=None):

        # print (replay_ixs)
        # print (replay_probabilities)
        if self.replay:
            if replay_probabilities is not None:
                replay_dataset.proba[replay_ixs] = np.array(replay_probabilities.detach().cpu())
            replay_dataset.replay_count_dict[replay_ixs]+= 1
            if replay_logits is not None:
                replay_dataset.logits = np.array(replay_dataset.logits)
                replay_dataset.logits[replay_ixs] = replay_logits.tolist()
                replay_dataset.logits = list(replay_dataset.logits)
        
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


def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot=one_hot.scatter(1,target.long().view(-1,1),1.)
    return one_hot


def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot=one_hot.scatter(1,target.long().view(-1,1),1.)
    return one_hot
