from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T
import yaml


from utils.metric import AverageMeter, Timer, new_vs_old_class_comparison, distribution_shift_comparison, per_class_plots
from models.resnet import BiasLayer
from .default import NormalNN,  accumulate_acc, loss_fn_kd, Teacher
from dataloaders.utils import transform_with_pca, make_video_ffmpeg, visualise, create_video, SupConLoss

import numpy as np
import copy
import os

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

        self.replay = False
        self.replay_size = learner_config['batch_size_replay']
        # self.labels_to_names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}  ## to read from config
        # self.class_mapping = {0: 0, 1: 1, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 2, 9: 3} ## to read from config
        self.replay_type = self.config['replay_type'] # how do we sample replay data (e.g., random)
        self.loss_type = self.config['loss_type'] # loss done on replay data (e.g., KD loss)
        self.plot_dir = learner_config['plot_dir']
        self.tasks = learner_config['tasks']
        self.task_acc = None
        

    def analyze(self):

        self.per_class_correct_predictions = { k: sum(self.per_class_correct_predictions[k].values()) for k in self.per_class_correct_predictions if self.per_class_correct_predictions}
        self.per_class_total_samples = { k: sum(self.per_class_total_samples[k].values()) for k in self.per_class_total_samples if self.per_class_total_samples}
        self.per_class_old_features = { k: list(self.per_class_old_features[k].values()) for k in self.per_class_old_features if self.per_class_old_features}
        self.per_class_new_features = { k: list(self.per_class_new_features[k].values()) for k in self.per_class_new_features if self.per_class_new_features}

        # print ('per class correct predictions: ', self.per_class_correct_predictions)
        # print ('per class total samples: ',self.per_class_total_samples)

        if not os.path.exists(self.plot_dir+'/before/'): os.makedirs(self.plot_dir+'/before/')
        if not os.path.exists(self.plot_dir+'/after/'): os.makedirs(self.plot_dir+'/after/')

        if self.epoch==0:
            base_path= self.plot_dir+'/before/'  
            new_vs_old_class_comparison(self.new_classes, self.old_classes, self.per_class_old_features, self.labels_to_names, self.class_mapping, replay_size = self.replay_size, base_path=base_path)

        if self.epoch==self.epochs_of_interest[-1]:
            base_path=self.plot_dir+'/after/'
            new_vs_old_class_comparison(self.new_classes, self.old_classes, self.per_class_new_features, self.labels_to_names, self.class_mapping, replay_size = self.replay_size, base_path=base_path)
            
        if self.epoch in self.epochs_of_interest:
            class_accuracy, class_dist_shift = distribution_shift_comparison(self.per_class_new_features, self.per_class_old_features, self.per_class_total_samples, self.per_class_correct_predictions)
            # print ('per_class_accuracy', class_accuracy)
            self.per_class_accuracy = { key: { key2: self.per_class_accuracy[key].get(key2,[])+[class_accuracy[key].get(key2,[])] for key2 in class_accuracy[key]} for key in class_accuracy }
            self.per_class_dist_shift = { key: { key2: self.per_class_dist_shift[key].get(key2,[])+[class_dist_shift[key].get(key2,[])]  for key2 in class_dist_shift[key]} for key in class_dist_shift }

        # print (self.epoch)
        # print ('per_class_dist_shift: ',self.per_class_dist_shift)


    def helper(self, predictions, labels, old_features, new_features, indices):

        for (prediction, label, old_feat, new_feat, index) in zip(predictions, labels, old_features, new_features, indices):
                            
            if int(label) not in self.per_class_correct_predictions.keys():
                self.per_class_correct_predictions[int(label)] = {}
                self.per_class_total_samples[int(label)] = {}
                self.per_class_correct_predictions[int(label)][int(index)] = 0
                self.per_class_old_features[int(label)] = {}
                self.per_class_new_features[int(label)] = {}

            if int(prediction) == int(label):
                self.per_class_correct_predictions[int(label)][int(index)] = 1

            # print (self.per_class_total_samples[int(label)])
            self.per_class_total_samples[int(label)][int(index)] = 1
            
            self.per_class_old_features[int(label)][int(index)] = old_feat.tolist()
            self.per_class_new_features[int(label)][int(index)] = new_feat.tolist()

        # print ('per_class_correct_predictions:', self.per_class_correct_predictions)


    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, replay_dataset, model_save_dir, val_target, task, val_loader=None):
        
        # try to load model
        need_train = True
        if not self.overwrite and not self.replay:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        if not os.path.exists(self.plot_dir): os.makedirs(self.plot_dir)

        self.class_mapping = { v: k for k,v in train_dataset.class_mapping.items() if k!=-1}
        self.labels_to_names = { v: k for k,v in train_dataset.class_to_idx.items()}

        # trains
        # print (val_target)
        if need_train:
            # self.epoch = 0
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()

            # # data weighting
            self.data_weighting(self.estimate_class_distribution(train_loader))

            self.per_class_correct_predictions = {}
            self.per_class_total_samples = {}
            self.per_class_old_features = {}
            self.per_class_new_features = {}
            # per_class_accuracy = {"MSE": {}, "MMD_linear": {}, "MMD_rbf": {}, "MMD_poly": {}, "CKA_linear": {}, "CKA_kernel": {}}
            # per_class_dist_shift = {"MSE": {}, "MMD_linear": {}, "MMD_rbf": {}, "MMD_poly": {}, "CKA_linear": {}, "CKA_kernel": {}}
            self.per_class_accuracy = {"MMD_rbf": {}} ## For testing MMD_poly, change to {"MMD_poly": {}}
            self.per_class_dist_shift = {"MMD_rbf": {}} ## For testing MMD_poly, change to {"MMD_poly": {}}
        
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()

            self.epochs_of_interest = [i for i in range(0, self.config['schedule'][-1], int(self.config['schedule'][-1]/100))] 
            # self.epochs_of_interest = [1,10] 
            self.epochs_of_interest[0] = 1
            self.epochs_of_interest.append(self.config['schedule'][-1])
            print(self.epochs_of_interest)
            # if self.epochs_of_interest[1]!=1:
            #     self.epochs_of_interest.insert(1,1)
            # if self.epochs_of_interest[-1]!=self.config['schedule'][-1]:
            #     self.epochs_of_interest.append(self.config['schedule'][-1])
            # print(self.epochs_of_interest)
            # self.epochs_of_interest = [1,2,3,4,5,10,20,30,40,50,100,200,300,400,500,600,700,800,900,1000]
            # print(apple)
            if self.task_acc is None:
                self.task_acc = { "Oracle": {epoch: [] for epoch in self.epochs_of_interest}, "Method": {epoch: [] for epoch in self.epochs_of_interest}, "FinalAcc": None}

            print ('Epochs of interest: ', self.epochs_of_interest)
            print ('Val target:', val_target)

            # Evaluate the performance of current task
            # self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            # if val_loader is not None:
            #     self.validation(val_loader, train_val=True)
            #     self.task_acc["Method"][0].append(self.validation(val_loader, task_in = task))
            #     self.task_acc["Oracle"][0].append(0) 
            #     print ('task acc: ', self.task_acc)

            epoch = 0
            while epoch < self.config['schedule'][-1]+1:

                

                if epoch > 1: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
    
                for i, (x, y, indices, x_r, y_r, replay_indices)  in enumerate(train_loader):
                    
                    epoch += 1
                    self.epoch=epoch
                    
                    if epoch >= self.config['schedule'][-1]+1:
                        break

                    # verify in train mode
                    self.model.train()
                    
                    if self.gpu:
                        if self.replay:
                            x_tot = torch.cat([x, x_r])
                            y_tot = torch.cat([y, y_r])
                            x_tot = x_tot.cuda()
                            y_tot = y_tot.cuda()
                        else:
                            x = x.cuda()
                            y = y.cuda()
                            # indices_tot = indices.tolist() + replay_indices.tolist()

                    if self.replay:
                        # transform = T.ToPILImage()
                        # img = transform(train_dataset.data[indices[0]])
                        # img.save('training_images_see/'+str(i)+'.png')

                        y_hat, _, old_feats = self.previous_teacher.generate_scores(x_tot, allowed_predictions= list(range(self.last_valid_out_dim)))
                        y_pred, new_feats = self.model.forward(x_tot, pen=True)
                        if self.replay_size==0:
                            loss, output = self.update_model(y_pred[:len(x)], y_tot[:len(x)])
                        else:
                            loss, output = self.update_model(y_pred, y_tot, y_pred[len(x):], y_hat[len(x):])
                            # loss, output= self.update_model(y_pred, y_tot)
                    else:
                        y_pred, y_pred_feat = self.model.forward(x, pen=True)
                        loss, output= self.update_model(y_pred, y)

                    # if (self.replay and epoch!=0):
                    #     predictions = torch.argmax(y_pred, dim=1)
                    #     self.helper(predictions, y_tot, old_feats, new_feats, indices_tot)
                    
                    # measure elapsed time
                    batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    
                    # measure accuracy and record loss
                    if self.replay:
                        y_tot = y_tot.detach()
                        accumulate_acc(output, y_tot, acc, topk=(self.top_k,))
                        losses.update(loss,  y_tot.size(0)) 
                    else:
                        y = y.detach()
                        accumulate_acc(output[:len(y)], y, acc, topk=(self.top_k,))
                        losses.update(loss,  y.size(0)) 
                    batch_timer.tic()

                    # eval update
                    if epoch in self.epochs_of_interest:
                        self.log('Step:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch,total=self.config['schedule'][-1]))
                        self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

                        # Evaluate the performance of current task
                        if val_loader is not None:
                            val_method = self.validation(val_loader, train_val=True)
                            if epoch in self.epochs_of_interest:
                                # self.task_acc["Method"][epoch].append(self.validation(val_loader, task_in = task))
                                self.task_acc["Method"][epoch].append(val_method)
                                self.task_acc["Oracle"][epoch].append(val_target)  
                                self.task_acc["FinalAcc"] = val_method
                                with open(self.plot_dir+'/avg_new-task_acc.yaml', 'w') as yaml_file:
                                    yaml.dump(self.task_acc, yaml_file, default_flow_style=False)

                    ## PLOTS AND TABLES
                    if self.replay:
                        self.analyze()

                    self.per_class_correct_predictions.clear()
                    self.per_class_total_samples.clear()
                    self.per_class_old_features.clear()
                    self.per_class_new_features.clear()

                    # reset
                    losses = AverageMeter()
                    acc = AverageMeter()



        self.model.eval()

        if self.replay:
            np.save(self.plot_dir+'/after/per_class_accuracy.npy', self.per_class_accuracy)
            np.save(self.plot_dir+'/after/per_class_dist_shift.npy', self.per_class_dist_shift) 
            
            del self.task_acc["FinalAcc"]
            self.task_acc = { m: [self.task_acc[m][k][-1] for k in self.task_acc[m] ] for m in self.task_acc }
            print ('task acc: ', self.task_acc)
            per_class_plots(self.per_class_accuracy, self.per_class_dist_shift, self.task_acc, self.labels_to_names, self.class_mapping, self.epochs_of_interest, replay_size = self.replay_size, base_path=self.plot_dir+'/after/')
        
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)

        if not self.replay:
            self.old_classes = set([int(y) for _,batch_y,_,_,_,_ in train_loader for y in batch_y ])
            self.new_classes = set(self.class_mapping.keys())-set(self.old_classes)
            if self.replay_size!=0:
                replay_dataset.extend(train_dataset, self.replay_size, self.mu > 0)
            else:
                replay_dataset.extend(train_dataset, 50)
            

        print ('size of replay_dataset:',  len(replay_dataset))
        print ('plot_save_dir: ', self.plot_dir)

        self.replay=True

        try:
            return batch_time.avg, self.config['schedule'][-1]
        except:
            return None, None

    def update_model(self, logits, targets, logits_replay= None, target_scores = None, dw_force = None, kd_index = None):
        
        if dw_force is not None:
            dw_cls = dw_force
        elif self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]

        # get weighting
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        if self.replay and self.replay_type == 'gradient_cb':
            # const_a = 2.0 * (self.valid_out_dim - self.last_valid_out_dim) / self.valid_out_dim # 0.2 for 9/1 84.6, 85.4 @ 50
            # const_b = 2.0 * self.last_valid_out_dim / self.valid_out_dim # 1.8 for 9/1
            # dw_a = dw_cls[:int(len(targets)/2)] * const_a
            # dw_b =dw_cls[int(len(targets)/2):] * const_b
            # dw_cls = torch.cat([dw_a, dw_b])
            dw_cls = self.dw_k[targets.long()]

        # logits = self.forward(inputs)
        # print(dw_cls)
        total_loss = self.criterion(logits, targets.long(), dw_cls)
        if self.epoch < 2 and self.replay: total_loss = total_loss * 0.05

        # # KD
        # if target_scores is not None and self.loss_type == 'pred_kd':
        #     # if kd_index is None: kd_index = np.arange(len(logits))
        #     # total_loss += self.mu * loss_fn_kd(logits[kd_index], target_scores[kd_index], dw_cls[kd_index], np.arange(self.last_valid_out_dim).tolist(), self.DTemp)

        if self.replay and self.loss_type == 'pred_kd':
            dw_KD = self.dw_k[-1 * torch.ones(len(target_scores),).long()]
            loss_distill = loss_fn_kd(logits_replay, target_scores, dw_KD, np.arange(self.last_valid_out_dim).tolist(), self.DTemp)
            total_loss += loss_distill

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    # data weighting
    def data_weighting(self, num_seen):

        # count number of examples in dataset per class
        self.log('*************************\n\n\n')

        self.log('num seen:' + str(num_seen))
        
        # in case a zero exists in PL...
        num_seen += 1

        # all seen
        seen = np.ones(self.valid_out_dim + 1, dtype=np.float32)
        seen = torch.tensor(seen)
        seen_dw = np.ones(self.valid_out_dim + 1, dtype=np.float32)
        seen_dw[:self.valid_out_dim] = num_seen.sum() / (num_seen * len(num_seen))
        seen_dw = torch.tensor(seen_dw)

        self.dw_k = seen_dw
        print('**********')
        print(self.dw_k)
        print('**********')
        # cuda
        if self.cuda:
            self.dw_k = self.dw_k.cuda()

    # estimate number of classes present for data weighting
    def estimate_class_distribution(self, train_loader):

        labels=[]
        for i in range(10):
            for i, (x, y, indices, x_r, y_r, replay_indices)  in enumerate(train_loader):
                if self.replay:
                    y_com = torch.cat([y, y_r])
                else:
                    y_com = y

                labels.extend(y_com.numpy())
        
        labels = np.asarray(labels, dtype=np.int64)
        return np.asarray([len(labels[labels==k]) for k in range(self.valid_out_dim)], dtype=np.float32)


    @torch.no_grad()
    def validation(self, dataloader, model=None, task_in = None,  verbal = True, train_val=False):

        if model is None:
            model = self.model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()

        # all_targets = set()
        # cm = [[],[]]
        for i, (input, target, indices) in enumerate(dataloader):
            # for lab in target:
            #     all_targets.add(int(lab))
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            if task_in is None:
                # output = model.forward(input)[:, :self.valid_out_dim]
                output, new_feats = model.forward(input, pen=True)
                if (self.replay and train_val):

                    _, _, old_feats = self.previous_teacher.generate_scores(input, allowed_predictions=list(range(self.last_valid_out_dim)))
                    predictions = torch.argmax(output, dim=1)
                    self.helper(predictions, target, old_feats, new_feats, indices.tolist())
                acc = accumulate_acc(output, target, acc, topk=(self.top_k,))
            else:
                mask = target >= task_in[0]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]

                mask = target < task_in[-1]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]
                
                if len(target) > 1:
                    output = model.forward(input)[:, task_in]
                    acc = accumulate_acc(output, target-task_in[0], acc, topk=(self.top_k,))

            # cm[0].extend(target.detach().cpu())
            # cm[1].extend(torch.argmax(output.detach().cpu(),dim=1))


        model.train(orig_mode)

        # from sklearn.metrics import confusion_matrix
        # a = confusion_matrix(cm[0],cm[1])
        # print(a)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        return acc.avg


def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot=one_hot.scatter(1,target.long().view(-1,1),1.)
    return one_hot


def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot=one_hot.scatter(1,target.long().view(-1,1),1.)
    return one_hot
