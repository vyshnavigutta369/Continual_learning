import os
import os.path
import numpy as np
import torch
from torch.utils.data import Sampler, BatchSampler
import torch.nn.functional as F
import collections
import copy

def compute_weights(replay_strategy, outputs, targets, start_time=None):
        
    if 'feature' in replay_strategy:
        data = torch.mean(torch.Tensor(outputs),axis=0).cuda()
        replay_data = torch.tensor(targets).cuda()

        
        vals = torch.norm(replay_data - data, dim=1, p=2)
        # knn = dist.topk(3, largest=False)
        # vals = knn.values
        # print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))
        assert vals.shape[0] == replay_data.shape[0]
        return vals.detach()

    # outputs = X1
    labels= targets.long()
    if 'loss' in replay_strategy:
        losses =  (nn.CrossEntropyLoss(reduction="none")(outputs, labels.long())) 
        vals = losses
    elif 'logit_dist' in replay_strategy:
        # print ('hellllloooooooooo')
        mask = torch.arange(labels.shape[0]).cuda()
        conf = outputs[mask, labels]
        vals = abs(conf)
    elif 'confidence' in replay_strategy:
        mask = torch.arange(labels.shape[0]).cuda()
        softmax = torch.nn.Softmax(dim=1)
        p = softmax(outputs)
        vals = p[mask, labels]
    elif 'margin' in replay_strategy:
        Fs = outputs.clone()
        mask = torch.arange(labels.shape[0]).cuda()
        Fs[mask, labels] = -float('inf')
        s_t = torch.argmax(Fs, dim=1)
        vals = outputs[mask, labels] - outputs[mask, s_t]
    elif 'time' in replay_strategy or 'replay_count' in replay_strategy or 'random' in replay_strategy:
        if start_time is None:
            start_time = time.time()
        vals = torch.empty((labels.shape[0]))
        vals.fill_(start_time)  # base init all at same time
    else:
        raise NotImplementedError
    assert vals.shape[0] == outputs.shape[0]
    
    return vals.detach()


class WeightedSampler(Sampler):
   
    def __init__(self, beta=100, weights_init=1, num_samples=-1, momentum=0., replay_strategy=''):
        
        self.num_samples = num_samples
        # Distributionally robustness parameter
        self.beta = beta

        # Momentum used for the update of the loss history.
        self.momentum = momentum

        # Initialization of the per-example loss values with a constant value
        if isinstance(weights_init, float) or isinstance(weights_init, int):
            assert num_samples > 0, \
                "The number of samples should be specified if a constant weights_init is used"
            print('Initialize the weights of the hardness weighted sampler to the value', weights_init)
            
            self.weights = torch.tensor(
                np.random.normal(loc=weights_init, scale=0.001*weights_init, size=num_samples)).cuda()
        # Initialization with a given vector of per-example loss values
        else:
            if isinstance(weights_init, np.ndarray):  # Support for numpy arrays
                weights_init = torch.tensor(weights_init).cuda()
            assert len(weights_init.shape) == 1, "initial weights should be a 1d tensor"
            self.weights = weights_init.float()
            if self.num_samples <= 0:
                self.num_samples = weights_init.shape[0]
            # else:
            #     assert self.num_samples == weights_init.shape[0], \
            #         "weights_init should have a size equal to num_samples"

        self.min_val = True if 'min' in replay_strategy else False 
        
    def get_distribution(self):
        # Apply softmax to the weights vector.
        # This seems to be the most numerically stable way
        # to compute the softmax
        if self.min_val:
            weights_m = 1 / (self.weights + 1e-7)
        else:
            weights_m = self.weights
        
        print ('weights: ', self.weights)
        weights= (weights_m-weights_m.min())/(weights_m.max()-weights_m.min())
        # weights*= self.class_weights[:self.num_samples]
        # weights = weights.exp()


        if not torch.isnan(weights).any():
            
            distribution = F.log_softmax(
                weights, dim=0).data.exp()
        else:
            distribution = copy.deepcopy(weights_m)
        
        # print ('distribution: ', distribution)
        distribution*= self.class_weights
        return distribution

    def draw_samples(self, n):
        """
        Draw n sample indices using the hardness weighting sampling method.
        """
        
        # Get the distribution (softmax)
        distribution = self.get_distribution()
        p = distribution.cpu().numpy()
        # print ('4: ', collections.Counter(p)[0])
        # print ('5: ', distribution)
        # Set min proba to epsilon for stability
        eps = 0.0001 / self.num_samples
        p[p <= eps] = eps
        
        # dic = {i: set() for i in set(p)}
        # p /= set(p).sum()
        # Use numpy implementation of multinomial sampling because it is much faster
        # than the one in PyTorch
        # print ('1:', self.num_samples)
        # print ('2: ', n)
        # print ('3: ', len(p))
        # sample_list = np.random.choice(
        #     self.num_samples,
        #     n,
        #     p=p,
        #     replace=False,
        # ).tolist()
        
        sample_list = torch.multinomial(torch.Tensor(p), self.num_samples, replacement=False).tolist()
        
        return sample_list

    def update_weights(self, batch_new_weights, indices):
        """
        Update the weights for the last batch.
        The indices corresponding the the weights in batch_new_weights
        should be the indices that have been copied into self.batch
        :param batch_new_weights: float or double array; new weights value for the last batch.
        :param indices: int list; indices of the samples to update.
        """
        assert len(indices) == batch_new_weights.size()[0], "number of weights in " \
                                                               "input batch does not " \
                                                               "correspond to the number " \
                                                               "of indices."
        # Update the weights for all the indices in self.batch
        # for idx, new_weight in zip(indices, batch_new_weights):
        #     self.weights[idx] = new_weight
        self.weights[indices] = batch_new_weights

    def extend_weights(self, weights):
        self.weights = torch.concat([self.weights, weights])

    def save_weights(self, save_path):
        torch.save(self.weights, save_path)

    def load_weights(self, weights_path):
        print('Load the sampling weights from %s' % weights_path)
        weights = torch.load(weights_path)
        self.weights = weights
        self.num_samples = self.weights.size()[0]

    def hardest_samples_indices(self, num=100):
        """
        Return the indices of the samples with the highest loss.
        :param num: int; number of indices to return.
        :return: int list; list of indices.
        """
        weights_np = np.array(self.weights)
        hardest_indices = np.argsort((-1) * weights_np)
        return hardest_indices[:num].tolist()

    def __iter__(self):
        sample_list = self.draw_samples(self.num_samples)
        # print (sample_list)
        return iter(sample_list)

    def __len__(self):
        return self.num_samples

class BatchWeightedSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last=False):
        """
        Custom Batch Sampler that calls the sampler once per iteration
        instead of once per epoch.
        An epoch consists in n iterations, where n is equal
        to the number of examples in dataset.
        When the sampler is an instance of WeightedSampler,
        this allows to implement dynamic sampling methods
        that change after each iteration.
        :param sampler: WeightedSampler; a PyTorch sampler
        :param batch_size: int; number of samples per batch.
        :param drop_last: bool; if True, incomplete batch at the end
        of an epoch are dropped.
        """
        assert isinstance(sampler, WeightedSampler), \
            "The sampler used in the BatchWeightedSampler must be a WeightedSampler"
        super(BatchWeightedSampler, self).__init__(
            sampler,
            batch_size,
            drop_last
        )

    @property
    def num_samples(self):
        return self.sampler.num_samples

    @property
    def beta(self):
        return self.sampler.beta

    def add(self, new_n):
        # self.num_samples+= new_n
        self.sampler.weights = torch.concat([self.sampler.weights, torch.ones(new_n).cuda()])


    def update_weights(self, batch_new_weights, indices):
        """
        Update the weights for the last batch.
        The indices corresponding the the weights in batch_new_weights
        should be the indices that have been copied into self.batch
        :param batch_new_weights: float or double array; new weights value for the last batch.
        :param indices: int list; indices of the samples to update.
        """
        assert len(indices) == batch_new_weights.size()[0], "number of weights in " \
                                                               "input batch does not " \
                                                               "correspond to the number " \
                                                               "of indices."
        # Update the weights for all the indices in self.batch
        for idx, new_weight in zip(indices, batch_new_weights):
            self.sampler.weights[idx] = new_weight

    def __iter__(self):
        for _ in range(len(self)):
            batch = self.sampler.draw_samples(self.batch_size)
            self.batch = [x for x in batch]
            yield batch

    def __len__(self):
        """
        :return: int; number of batches per epoch.
        """
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


# For active replay dataloader, need to make a dataloader object which acts like a replay dataloader in rehearsal.py but returns
# active dynamic replay batches not random
class BatchSampler(object): ## TODO OPTIMIZE!! super slow
    def __init__(self, dataset, batch_size, new_classes, old_classes, class_counts=None):
        super(BatchSampler, self).__init__()
        self.dataset = dataset
        self.class_counts = class_counts
        self.batch_size = batch_size
        
        self.old_classes = old_classes
        self.new_classes = new_classes
        self.classes = self.old_classes+self.new_classes

        self.class_mapping = { v: k for k,v in self.dataset.class_mapping.items() if k!=-1}
        self.labels_to_names = { v: k for k,v in self.dataset.class_to_idx.items()}      

        self.init_params(class_counts) 

    def init_params(self, class_ratios=None, class_counts=None):

        if hasattr(self, 'indices'):
            seen_indices = []
            seen_labels = []
            for i,y in enumerate(self.dataset.targets):
                if i not in self.indices:
                    seen_indices.append(i)
                    seen_labels.append(self.dataset.class_mapping[y])
            self.indices.extend(seen_indices)
            self.labels.extend(seen_labels)
        else:
            self.labels = [ self.dataset.class_mapping[y] for y in  self.dataset.targets]
            self.indices = [i for i in range(len(self.labels))]

        self.class_counts = class_counts
        
        if self.class_counts is None:
            self.compute_class_counts(class_ratios)

        self.goal = False
        self.cl_ind = {}
        self.new_cl_filled= []   
        self.batch_no=0          
            
    def compute_class_counts(self, class_ratios=None):

        if len(self.old_classes)==0:
            self.class_counts=  { i: c for i, c in enumerate(near_split(int(self.batch_size/2), len(self.new_classes))) }
        else:
            if class_ratios is not None:
                self.class_ratios = class_ratios
            else:
                # self.class_ratios = { 0: 3, 1: 1, 2: 6, 3: 8, 4: 6, 5: 3, 6: 3, 7: 1, 8: 6, 9: 8} ## default
                raise Exception("Both Class counts & Class ratios cannot be None for using custom batch sampler")
                
            class_ratios_old = {k: self.class_ratios[k] for k in self.class_ratios if k in self.old_classes}
            class_ratios_new = {k: self.class_ratios[k] for k in self.class_ratios if k in self.new_classes}

            factor = float(sum(class_ratios_old.values()))
            class_ratios_old = {k: class_ratios_old[k]/factor for k in class_ratios_old}
            factor = float(sum(class_ratios_new.values()))
            class_ratios_new = {k: class_ratios_new[k]/factor for k in class_ratios_new}

            self.class_counts= {k: x for k,x in zip(class_ratios_old, ratio_breakdown(int(self.batch_size/2), list(class_ratios_old.values())) )} \
                            | {k: x for k,x in zip(class_ratios_new, ratio_breakdown(int(self.batch_size/2), list(class_ratios_new.values())) )}

    def get_data(self):
             
        ind= []
        labels = []
        counts= {}
        filled = set()
        
        for i, y in zip(self.indices, self.labels):

            if len(filled) == len(self.class_counts):
                break
                
            if y not in counts:
                counts[y]=0
            if y not in self.cl_ind:
                self.cl_ind[y]= []
            if counts[y] < self.class_counts[y]:
                self.cl_ind[y].append(i)
                ind.append(i)
                labels.append(y)
                counts[y]+=1
            else:
                filled.add(y)

        for i,y in zip(ind, labels):
            self.labels.remove(y)
            self.indices.remove(i)

        for cl in self.classes:
            if cl not in set(self.labels):
                if cl in self.new_classes:
                    self.new_cl_filled.append(cl)
                self.indices.extend(self.cl_ind[cl])
                self.labels.extend([cl for _ in range(len(self.cl_ind[cl]))])

        if set(self.new_cl_filled) == set(self.new_classes):
            self.goal = True

        if (self.batch_no==0):
            print ('class_counts: ', {self.labels_to_names[self.class_mapping[cl]]: counts[cl] for cl in counts})
        return ind

    def __iter__(self):
        
        batch_inds = []
        while not self.goal:
            batch_ind = self.get_data()
            self.batch_no+=1
            batch_inds.append(batch_ind)
            yield batch_ind
        # return iter(batch_inds)