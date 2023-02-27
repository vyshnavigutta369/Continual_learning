from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T

from utils.metric import AverageMeter, Timer, MMD, per_class_plots
from models.resnet import BiasLayer
from .default import NormalNN,  accumulate_acc, loss_fn_kd, loss_fn_class_kd, pad_tensors, Teacher
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

    def loss_ceigen(self, per_class_features):        

        per_class_features = torch.Tensor(pad_tensors(per_class_features)).cuda()

        per_class_old_features = torch.Tensor(per_class_features[len(self.per_class_new_features_train):]).cuda()
        per_class_features = torch.Tensor(per_class_features[:len(self.per_class_new_features_train)]).cuda()
        MMD_metric = MMD()
        distilled = MMD_metric.mmd_poly(per_class_old_features, per_class_features)
        interference_m = 1-torch.divide(distilled, torch.diag(distilled))
        return interference_m
        
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

                    self.init_params_batch()
                    
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
                   
                    # for l,ind in zip(y_r,replay_indices):
                    #     self.check_replay_counts[int(l)].add(int(ind))

                    output, new_feats = self.model.forward(x_tot, pen=True)  
                    
                    per_class_features= None

                    self.helper(indices, y_tot, new_datas_train= x_tot)
                    self.process_attr_batch()
                        
                    if self.replay:
                        y_hat, _, old_features = self.previous_teacher.generate_scores(x_tot)
                        per_class_new_features, per_class_old_features= self.split_by_class(y_tot, new_feats,old_features)
                        per_class_features= per_class_new_features+per_class_old_features

                    loss = self.update_model(output, y_tot, per_class_features)

                    _, x_bl, y_bl, _, _, _ = self.get_class_balanced_data(indices, x_tot, y_tot)
                    
                    if self.with_class_balance==1:

                        output_bl = self.model.forward(x_bl, balanced=True)   
                        loss += self.update_model(output_bl, y_bl, per_class_features, new_optimizer=True)
                        
                    
                    # measure elapsed time
                    batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    
                    # measure accuracy and record loss
                    y = y.detach()
                    # accumulate_acc(output[:len(y)], y, acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0)) 
                    batch_timer.tic()
                    
                    # if self.with_class_balance==1:
                    #     accumulate_acc(output_bl, y_bl, acc_bl, topk=(self.top_k,))
                    #     losses_bl.update(loss_bl,  y_bl.size(0)) 

                    # print (self.blah)
                    

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch,total=self.config['schedule'][-1]))
                # self.log(' * Train Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))
                # if self.with_class_balance==1:
                #     self.log(' * BL head train Loss {loss.avg:.3f} | BL Train Acc {acc.avg:.3f}'.format(loss=losses_bl,acc=acc_bl))

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

               
            # print ('check replay_counts: ', {self.labels_to_names[self.class_mapping[cl]]: len(self.check_replay_counts[cl]) for cl in self.check_replay_counts})

            if self.replay:
                per_class_plots(self.per_class_accuracy, self.per_class_dist_shift, self.labels_to_names, self.class_mapping, self.epochs_of_interest, self.replay_size,  self.avg_acc, self.task_acc, base_path=self.plot_dir+'_after/')
        

        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)

        if not self.replay:
            # print (self.class_replay_counts)
            replay_dataset.extend(train_dataset, self.class_replay_counts)

        print ('size of replay_dataset:',  len(replay_dataset))
        self.replay=True

        try:
            return batch_time.avg, self.config['schedule'][-1]
        except:
            return None, None

    def update_model(self, logits, targets, per_class_features, logits_replay= [], target_scores = [], dw_scale = 1, new_optimizer=False):
        
        # print (logits.requires_grad)

        

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
        if self.replay:
            print (total_loss)
            total_loss +=  0.8 *self.loss_ceigen(per_class_features).mean()
            print (total_loss)

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

    

def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot=one_hot.scatter(1,target.long().view(-1,1),1.)
    return one_hot


def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot=one_hot.scatter(1,target.long().view(-1,1),1.)
    return one_hot
