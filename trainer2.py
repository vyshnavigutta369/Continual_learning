import imp
import os
import torch
import numpy as np
import random
from collections import OrderedDict
import dataloaders
from torch.utils.data import DataLoader
import learners
import yaml
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, args, seed, metric_keys, save_keys):

        # process inputs
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.batch_size_replay = args.batch_size_replay
        self.workers = args.workers
        self.oracle_dir = args.oracle_dir
        
        # model load directory
        if args.load_model_dir is not None:
            self.model_first_dir = args.load_model_dir
        else:
            self.model_first_dir = args.log_dir
        self.model_top_dir = args.log_dir

        # select dataset
        self.top_k = 1
        if args.dataset == 'CIFAR2':
            Dataset = dataloaders.iCIFAR2
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR8':
            Dataset = dataloaders.iCIFAR8
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR10':
            Dataset = dataloaders.iCIFAR10
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'ImageNet':
            Dataset = dataloaders.iIMAGENET
            num_classes = 1000
            self.dataset_size = [224,224,3]
            self.top_k = 5
        elif args.dataset == 'TinyImageNet':
            Dataset = dataloaders.iTinyIMNET
            num_classes = 200
            self.dataset_size = [64,64,3]
        else:
            raise ValueError('Dataset not implemented!')

        # load tasks
        class_order = [0,1,4,5,6,7,8,9,2,3]
        class_order_logits = [0,1,4,5,6,7,8,9,2,3]
        if args.rand_split:
            print('=============================================')
            print('Shuffling....')
            print('pre-shuffle:' + str(class_order))
            if args.dataset == 'ImageNet':
                np.random.seed(1993)
                np.random.shuffle(class_order)
            elif not self.seed == 0:
                random.seed(self.seed)
                random.shuffle(class_order)
            print('post-shuffle:' + str(class_order))
            print('=============================================')
        self.tasks = []
        self.tasks_logits = []
        p = 0
        print ('first split size: ', args.first_split_size)
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p+inc])
            self.tasks_logits.append(class_order_logits[p:p+inc])
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i+1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug)
        test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug)
        self.train_dataset = Dataset(args.dataroot, train=True, tasks=self.tasks,                       
                            download_flag=True, transform=train_transform, 
                            seed=self.seed, validation=args.validation)
        self.replay_dataset = dataloaders.ReplayDataset(transform=train_transform, seed=self.seed, num_samples = self.batch_size)
        self.test_dataset = Dataset(args.dataroot, train=False, tasks=self.tasks,                                 
                                download_flag=False, transform=test_transform, 
                                seed=self.seed, validation=args.validation)

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
                        'lr': args.lr,
                        'momentum': args.momentum,
                        'weight_decay': args.weight_decay,
                        'schedule': args.schedule,
                        'schedule_type': args.schedule_type,
                        'model_type': args.model_type,
                        'model_name': args.model_name,
                        'gen_model_type': args.gen_model_type,
                        'gen_model_name': args.gen_model_name,
                        'optimizer': args.optimizer,
                        'gpuid': args.gpuid,
                        'temp': args.temp,
                        'out_dim': num_classes,
                        'overwrite': args.overwrite == 1,
                        'mu': args.mu,
                        'DW': args.DW,
                        'batch_size': args.batch_size,
                        'tasks': self.tasks_logits,
                        'top_k': self.top_k,
                        'loss_type': args.loss_type,
                        'replay_type': args.replay_type,
                        'batch_size_replay': args.batch_size_replay,
                        'ub_rat': args.ub_rat,
                        'buffer_update': args.buffer_update,
                        'reuse_replay_ixs': args.reuse_replay_ixs,
                        'plot_dir': args.plot_dir
                        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
        self.learner.print_model()

        # self.replay_ixs_count = []


    def train(self, avg_metrics):

        print ('==============TRAINING===============')
    
        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys: temp_table[mkey] = []
        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)

        # for each task
        tb = SummaryWriter(self.log_dir+'/runs') 
        oracle_acc = []

        # print (self.max_task)
        for i in range(self.max_task):

            # save current task index
            self.current_t_index = i

            # set seeds
            random.seed(self.seed + i)
            np.random.seed(self.seed + i)
            torch.manual_seed(self.seed + i)
            torch.cuda.manual_seed(self.seed + i)

            # print name
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            # load dataset for task
            task = self.tasks_logits[i]
            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            # tell learner number of tasks we are doing
            self.learner.max_task = self.max_task

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataloader
            train_dataset_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=int(self.workers))

            if len(self.replay_dataset) > 0:
                #print (self.batch_size_replay)
                replay_loader = DataLoader(self.replay_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=int(self.workers))
            else:
                replay_loader = None

            ## MY CHANGES 
            static_replay_train_loader = dataloaders.DualDataLoader(train_dataset_loader, replay_loader)
            
            # get val target
            if self.oracle_dir is None:
                val_target = None
                # load_file = '/srv/kira-lab/share4/vgutta7/memory-is-cheap-main/_outputs/Sep22/tentask/CIFAR100/oracle' + '/results-acc/global.yaml'
                # with open(load_file, 'r') as yaml_file:
                #     yaml_result = yaml.safe_load(yaml_file)
                # val_target = yaml_result['history'][i][self.seed]
                # self.model_first_dir = '/srv/kira-lab/share4/vgutta7/memory-is-cheap-main/_outputs/Sep22/tentask/CIFAR100/oracle'
                # oracle_acc.append(val_target)
            else:
                load_file = self.oracle_dir + '/results-acc/global.yaml'
                with open(load_file, 'r') as yaml_file:
                    yaml_result = yaml.safe_load(yaml_file)
                val_target = yaml_result['history'][i][self.seed]
                self.model_first_dir = self.oracle_dir
                oracle_acc.append(val_target)

            # learn
            self.test_dataset.load_dataset(i, train=False)
            test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
            if i == 0:
                model_save_dir = self.model_first_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            else:
                model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            
            if self.learner_config['replay_type']!= 'random_sample' and self.learner_config['replay_type']!= 'gradient_cb':
                avg_train_time, epochs_converge = self.learner.learn_incremental_batch(train_dataset_loader, replay_loader, self.train_dataset,  self.replay_dataset, task, model_save_dir, val_target, test_loader)
                # replay_ixs_count = len(set(self.learner.replay_item_ixs_all_tasks))
            else:
                avg_train_time, epochs_converge = self.learner.learn_batch(static_replay_train_loader, self.train_dataset, self.replay_dataset, model_save_dir, val_target, task, test_loader)

            # save model
            self.learner.save_model(model_save_dir)

            # evaluate acc
            acc_table = []
            self.reset_cluster_labels = True
            # print ('i:', i)
            for j in range(i+1):
                task_acc = self.task_eval(j)
                # print ('tas_acc: ', task_acc)
                acc_table.append(task_acc)
                tb.add_scalar('acc-task', task_acc, j) 
            temp_table['acc'].append(np.mean(np.asarray(acc_table)))
            # print ('temp_table:', temp_table)
            tb.add_scalar('acc-mean', np.mean(np.asarray(acc_table)), i)

            # save temporary results
            for mkey in self.metric_keys:
                save_file = temp_dir + mkey + '.csv'
                np.savetxt(save_file, np.asarray(temp_table[mkey]), delimiter=",", fmt='%.2f')  

            if avg_train_time is not None: 
                avg_metrics['time']['global'][i] = avg_train_time
                tb.add_scalar('time-global', avg_train_time, i)
            if epochs_converge is not None: 
                avg_metrics['epochs']['global'][i] = epochs_converge
                tb.add_scalar('epochs-global', epochs_converge, i)
            avg_metrics['mem']['global'][:] = len(self.replay_dataset)
            # print (self.learner_name)
            if self.learner_name== 'Mine':
                avg_metrics['replay_item_count']['global'][i] = len(set(self.learner.replay_item_ixs_all_tasks))

        tb.close()
        return avg_metrics, np.array(oracle_acc)

    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i+1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j,i,self.seed] = acc_table[val_name][train_name]
                avg_acc_pt_local[j,i,self.seed] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)

        # Gather the final avg accuracy
        avg_acc_all[:,self.seed] = avg_acc_history

        # repack dictionary and return
        return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}
    
    def task_eval(self, t_index, local=False):

        val_name = self.task_names[t_index]
        # print('validation split name:', val_name)
        # print ('t_index:', t_index)

        # eval
        self.test_dataset.load_dataset(t_index, train=True)
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)   ## MY CHANGES
        if local:
            return self.learner.validation(test_loader, task_in = self.tasks_logits[t_index])
        else:
            val = self.learner.validation(test_loader)
            # print ('val:', val)
            return val
    
    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}
        r_matrix = np.zeros((self.max_task, self.max_task))

        tb = SummaryWriter(self.log_dir+'/runs')
        for i in range(self.max_task):

            # load model
            if i == 0:
                model_save_dir = self.model_first_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            else:
                model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            self.learner.task_count = i
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # evaluate acc
            metric_table['acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['acc'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(j)
                tb.add_scalar('acc-pt', metric_table['acc'][val_name][self.task_names[i]], j)
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table_local['acc'][val_name][self.task_names[i]] = self.task_eval(j, local=True)
                tb.add_scalar('acc-pt-local',metric_table_local['acc'][val_name][self.task_names[i]], j)
                r_matrix[i][j] = metric_table_local['acc'][val_name][self.task_names[i]]

        # summarize metrics
        tb.close()
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'],  metric_table_local['acc'])

        return avg_metrics, r_matrix

    def compute_metrics(self, Rmat, offline=None):
        # given the R matrix of results and an offline learner's performance, compute all performance metrics
        def fwt_computation(Rmatrix):
            N = Rmatrix.shape[0]
            fwt = 0
            for j in range(N):
                fwt += sum([Rmatrix[i, j] for i in range(j)])
            fwt /= N * (N - 1) / 2
            return fwt

        def bwt_computation(Rmatrix):
            N = Rmatrix.shape[0]
            bwt = 0
            for i in range(1, N):
                bwt += sum([Rmatrix[i, j] - Rmatrix[j, j] for j in range(i - 1)])
            bwt /= N * (N - 1) / 2
            return bwt

        def avg_acc_computation(Rmatrix):
            N = Rmatrix.shape[0]
            acc = 0
            for j in range(N):
                acc += sum([Rmatrix[i, j] for i in range(j, N)])
            acc /= N * (N + 1) / 2
            return acc

        def gamma_t_computation(Rmatrix):
            mean_run = 0
            T = Rmatrix.shape[0]
            for i in range(T):
                acc = sum([Rmatrix[i, k] for k in range(T)]) / T
                mean_run += acc
            return mean_run / T

        def omega_computation(Rmatrix, offline):
            mean_run = 0
            T = Rmatrix.shape[0]
            for i in range(T):
                acc = sum([Rmatrix[i, k] for k in range(T)]) / T
                mean_run += acc / offline[i]
            return mean_run / T

        Rmat = Rmat.T
        fwt_score = fwt_computation(Rmat)
        bwt_score = bwt_computation(Rmat)
        avg_acc_score = avg_acc_computation(Rmat)

        if offline is not None:
            offline = offline / 100
            gamma_t = omega_computation(Rmat, offline)
        else:
            gamma_t = gamma_t_computation(Rmat)

        return avg_acc_score, fwt_score, bwt_score, gamma_t
    