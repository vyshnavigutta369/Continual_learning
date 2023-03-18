from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T

from utils.metric import AverageMeter, Timer, MMD, per_class_plots, getBack
from utils.pytorchvis import draw_graph
from torchviz import make_dot
from models.resnet import BiasLayer
from .default import NormalNN,  accumulate_acc, loss_fn_kd, loss_fn_class_kd, Teacher
from dataloaders.utils import transform_with_pca, make_video_ffmpeg, visualise, create_video, SupConLoss

import numpy as np
import copy
import os
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
        
        self.loss_MMD = MMD()
        self.flag=0

        
    
    ##########################################
    #           MODEL TRAINING               #
    ##########################################

        
    def learn_batch(self, train_loader, train_dataset, replay_dataset, model_save_dir, val_target, task, val_loader=None):
        
        # try to load model
        need_train = True

        self.train_loader = train_loader
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
            self.batch_time = AverageMeter()
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

                if epoch>0:
                    
                    # for i, (x, y, indices, x_r, y_r, replay_indices)  in enumerate(train_loader):
                    for i, (x, y, indices)  in enumerate(self.train_loader):

                        self.init_params_batch()
                                 
                        # verify in train mode
                        
                        self.model.train()
                        
                        if self.gpu:
                            x = x.cuda()
                            y = y.cuda()
                            # if self.replay:
                            #     x_r = x_r.cuda()
                            #     y_r = y_r.cuda()
                            #     x_tot = torch.cat([x, x_r])
                            #     y_tot = torch.cat([y, y_r])
                            #     indices = indices.tolist() + replay_indices.tolist()
                            # else:
                            indices = indices.tolist()
                            x_tot = x
                            y_tot = y
                    
                        # for l,ind in zip(y, indices):
                        #     self.check_replay_counts[int(l)].add(int(ind))
                        
                        output, new_feats = self.model.forward(x_tot, pen=True)  
                        self.helper(indices, y_tot, new_datas_train= x_tot)
                        self.process_attr_batch()
                        
                        if self.replay:
                            y_hat, _, old_features = self.previous_teacher.generate_scores(x_tot)
                            # per_class_new_features, per_class_old_features= self.split_by_class(y_tot, new_feats, old_features)
                            loss = self.update_model(output, y_tot, new_feats, old_features)
                        else:
                            loss = self.update_model(output, y_tot)
    
                        if self.with_class_balance==1:
                            x_bl, y_bl = self.get_class_balanced_data(indices, x_tot, y_tot)
                            output_bl = self.model.forward(x_bl, balanced=True)  
                            loss += self.update_model(output_bl, y_bl, new_optimizer=True)
                        
                        # measure elapsed time
                        self.batch_time.update(batch_timer.toc())  
                        batch_timer.tic()
                        
                        # measure accuracy and record loss
                        y = y.detach()
                        # accumulate_acc(output[:len(y)], y, acc, topk=(self.top_k,))
                        losses.update(loss,  y.size(0)) 
                        batch_timer.tic()
                        self.step_count+=1

                        # if self.replay:
                        #     self.validation(val_loader, train_val=True)
                        
                        # if self.with_class_balance==1:
                        #     accumulate_acc(output_bl, y_bl, acc_bl, topk=(self.top_k,))
                        #     losses_bl.update(loss_bl,  y_bl.size(0))
                        # print ('check replay_counts: ', {self.labels_to_names[self.class_mapping[cl]]: len(self.check_replay_counts[cl]) for cl in self.check_replay_counts})                     

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
                        
                # reset
                losses = AverageMeter()
                losses_bl = AverageMeter()
                acc = AverageMeter()
                acc_bl = AverageMeter()


            if self.replay:
                per_class_plots(self.per_class_accuracy, self.per_class_dist_shift, self.labels_to_names, self.class_mapping, self.epochs_of_interest, self.steps_of_interest, self.times_of_interest, self.replay_size,  self.avg_acc, base_path=self.plot_dir+'_after/')
        

        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)

        if not self.replay:
            print ('class_counts: ', self.class_replay_counts)
            replay_dataset.extend(train_dataset, self.class_replay_counts)

        print ('size of replay_dataset:',  len(replay_dataset))
        self.replay=True

        try:
            return self.batch_time.avg, self.config['schedule'][-1]
        except:
            return None, None

    def update_model(self, logits, targets, new_features=None, old_features=None, new_optimizer=False):
        
        
        if (self.replay_type == 'random_sample') or new_optimizer:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        
            
        total_loss = self.criterion(logits, targets.long(), dw_cls)
        
        if self.loss_type == 'pred_kd' and self.replay and not new_optimizer:
            
            labels_end_ind = torch.cumsum(torch.bincount(targets.long()), dim=0).cpu()
            labels_ind = torch.argsort(targets)
            new_features = new_features[labels_ind]
            old_features = old_features[labels_ind]

            distilled = self.loss_MMD.mmd_poly(old_features, new_features, labels_end_ind= labels_end_ind)
            # print ('before norm:', distilled)
            distilled = distilled/distilled.sum(1)
            
            dw_cls = torch.Tensor(self.sim_to_new_cls)
            interference_m = torch.divide(distilled, torch.diag(distilled))
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


def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot=one_hot.scatter(1,target.long().view(-1,1),1.)
    return one_hot
