from PIL import Image
import os
import os.path
import numpy as np
import sys
import pickle
import torch.utils.data as data
from .utils import download_url, check_integrity
import torch
import torch.nn.functional as F
from datasets import concatenate_datasets
import copy

from utils.utils import getBack, near_split, ratio_breakdown
    
class DualDataLoader(object):
    def __init__(self, dset_a, dset_b):
        self.dset_a = dset_a
        self.dset_b = dset_b

        if dset_b is None:
            self.dual_mode = False
        else:
            self.dual_mode = True

        self.iter_a = iter(self.dset_a)
        if self.dual_mode: self.iter_b = iter(self.dset_b)

    def __iter__(self):
        self.iter_a = iter(self.dset_a)
        return self

    def __len__(self):
        return len(self.dset_a)

    def __next__(self):
        
        # main dataset
        x_a, y_a, ix_a = next(self.iter_a)
        # shuffle_idx = torch.randperm(len(y_a), device=y_a.device)
        # x_a, y_a = x_a[shuffle_idx], y_a[shuffle_idx]

        if self.dual_mode:

            # auxillary dataset
            try:
                x_b, y_b, ix_b = next(self.iter_b)
            except:
                self.iter_b = iter(self.dset_b)
                x_b, y_b, ix_b = next(self.iter_b)
            # shuffle_idx = torch.randperm(len(y_b), device=y_b.device)
            # x_b, y_b = x_b[shuffle_idx], y_b[shuffle_idx]

            return x_a, y_a, ix_a, x_b, y_b, ix_b

        else:
            return x_a, y_a, ix_a, torch.Tensor([]), torch.Tensor([]), torch.Tensor([])


class ReplayDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, seed=-1, num_samples=128):
        self.transform = transform
        # self.seed = seed
        
        self.num_samples = num_samples
        # self.num_seen_examples = 0
        self.attributes = ['data', 'labels', 'logits', 'task_labels', 'proba', 'replay_count_dict', 'task_counter']
        self.seed = seed

        for attr_str in self.attributes:
            setattr(self, attr_str, [])
        # self.logits = [[]]
        
        self.data = []
        self.targets= []
        self.weights = []

    def extend(self, dataset, class_replay_counts=None, class_replay_ratios=None, class_replay_weights=None):
        
        to_replay_class_count = {}
        self.class_mapping = dataset.class_mapping

        
        for data, target in zip(dataset.data, dataset.targets):
            if class_replay_ratios is not None:
                self.data.extend([data for _ in range(class_replay_ratios[self.class_mapping[target]])])
                self.targets.extend([target for _ in range(class_replay_ratios[self.class_mapping[target]])])
            elif class_replay_weights is not None:
                self.data.append(data)
                self.targets.append(target)
                if class_replay_weights:
                    self.weights.append(class_replay_weights[self.class_mapping[target]])
            elif class_replay_counts is not None: 
                if class_replay_counts[self.class_mapping[target]] == 5000:
                    self.data.append(data)
                    self.targets.append(target)
                    continue
                if target not in to_replay_class_count.keys():
                    to_replay_class_count[target] = 0
                if to_replay_class_count[target] < class_replay_counts[self.class_mapping[target]]:
                    self.data.append(data)
                    self.targets.append(target)
                    to_replay_class_count[target] +=1
            else:
                raise Exception("Requires one of class_replay_ratios, class_replay_weights or class_replay_counts to add to the Replay dataset!!")
               

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #

        if not isinstance(index, int):

            data = np.array(self.data)
            targets = np.array(self.targets)[index]
            imgs = []
            for i, (ix, target) in enumerate(zip(index, targets)):
            
                img = data[ix]
                img = Image.fromarray(img) 

                if self.transform is not None:
                    if simple:
                        img = self.simple_transform(img)
                    else:
                        img = self.transform(img)
                imgs.append(img)
                targets[i] = self.class_mapping[target]
            return torch.stack(imgs, axis=0), torch.Tensor(targets), index ## CORRECT
        else:
            
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                if simple:
                    img = self.simple_transform(img)
                else:
                    img = self.transform(img)
            return img, self.class_mapping[target], index
        
        # return img, target ## INCORRECT

    def __len__(self):
        return len(self.data)


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

class iDataset(data.Dataset):
    
    def __init__(self, root,
                train=True, transform=None ,download_flag=False,
                tasks=None, seed=-1, validation=False, kfolds=5):

        # process rest of args
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation
        self.seed = seed
        self.t = -1
        self.tasks = tasks
        self.download_flag = download_flag
        self.ic_dict = {}
        self.ic = False
        self.dw = True

        # load dataset
        self.load()
        self.num_classes = len(np.unique(self.targets))

        # remap labels to match task order
        c = 0
        # print ('tasks: ', self.tasks)
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task in self.tasks:
            for k in task:
                self.class_mapping[k] = c
                c += 1
        # print ('class mapping:', self.class_mapping)
        # targets as numpy.array
        self.data = np.asarray(self.data)
        self.targets = np.asarray(self.targets)

        # if validation
        if self.validation:
            
            # shuffle
            state = np.random.get_state()
            np.random.seed(self.seed)
            randomize = np.random.permutation(len(self.targets))
            self.data = self.data[randomize]
            self.targets = self.targets[randomize]
            np.random.set_state(state)

            # sample
            num_data_per_fold = int(len(self.targets) / kfolds)
            start = 0
            stop = num_data_per_fold
            locs_train = []
            locs_val = []
            for f in range(kfolds):
                if self.seed == f:
                    locs_val.extend(np.arange(start,stop))
                else:
                    locs_train.extend(np.arange(start,stop))
                start += num_data_per_fold
                stop += num_data_per_fold

            # train set
            if self.train:
                self.archive = []
                for task in self.tasks:
                    locs = np.isin(self.targets[locs_train], task).nonzero()[0]
                    self.archive.append((self.data[locs_train][locs].copy(), self.targets[locs_train][locs].copy()))

            # val set
            else:
                self.archive = []
                for task in self.tasks:
                    locs = np.isin(self.targets[locs_val], task).nonzero()[0]
                    self.archive.append((self.data[locs_val][locs].copy(), self.targets[locs_val][locs].copy()))

        # else
        else:
            self.archive = []
            # print ('self tasks', self.tasks)
            # print ('self targets', self.targets)
            for task in self.tasks:
                locs = np.isin(self.targets, task).nonzero()[0]
                # print ('locs: ', locs)
                self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))


    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class
        """
        
        # print (index)
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            if simple:
                img = self.simple_transform(img)
            else:
                img = self.transform(img)

        return img, self.class_mapping[target], index   ## MY CHANGES

    def load_dataset(self, t, train=True):
        
        if train:
            self.data, self.targets = self.archive[t] 
        else:
            # print ('i: ', t)
            # print ('self archive', self.archive)
            self.data    = np.concatenate([self.archive[s][0] for s in range(t+1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t+1)], axis=0)
            # print ('len self(data)', self.data)
        self.t = t

    def update_coreset(self, coreset_size, seen):
        num_data_per = coreset_size // len(seen)
        remainder = coreset_size % len(seen)
        data = []
        targets = []
        
        # random coreset management; latest classes take memory remainder
        # coreset selection without affecting RNG state
        state = np.random.get_state()
        np.random.seed(self.seed*10000+self.t)
        for k in reversed(seen):
            mapped_targets = [self.class_mapping[self.targets[i]] for i in range(len(self.targets))]
            locs = (mapped_targets == k).nonzero()[0]
            if (remainder > 0) and (len(locs) > num_data_per):
                num_data_k = num_data_per + 1
                remainder -= 1
            else:
                num_data_k = min(len(locs), num_data_per)
            locs_chosen = locs[np.random.choice(len(locs), num_data_k, replace=False)]
            data.append([self.data[loc] for loc in locs_chosen])
            targets.append([self.targets[loc] for loc in locs_chosen])
            # print (np.array(data).shape)
            # print (np.array(targets).shape)
        self.coreset = (np.concatenate(list(reversed(data)), axis=0), np.concatenate(list(reversed(targets)), axis=0))
        np.random.set_state(state)

    def extend(self, dataset):
        
        if len(dataset)>0:
            self.data = np.concatenate([self.data, dataset.data])
            self.targets = np.concatenate([self.targets, dataset.targets])
            self.class_mapping = dataset.class_mapping

    def load(self):
        pass

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class iCIFAR2(iDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iDataset Dataset.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    im_size=32
    nch=3

    def load(self):

        # download dataset
        if self.download_flag:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train or self.validation:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.course_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])
                
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                if 'coarse_labels' in entry:
                    self.course_targets.extend(entry['coarse_labels'])
                
        
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()
        # print (self.class_to_idx)

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


class iCIFAR10(iDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iDataset Dataset.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    im_size=32
    nch=3

    def load(self):

        # download dataset
        if self.download_flag:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train or self.validation:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.course_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                if 'coarse_labels' in entry:
                    self.course_targets.extend(entry['coarse_labels'])
                
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

class iCIFAR100(iCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iCIFAR10 Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    im_size=32
    nch=3

class iIMAGENET(iDataset):
    
    base_folder = 'ilsvrc'
    im_size=224
    nch=3
    def load(self):
        self.dw = False
        self.data, self.targets = [], []
        images_path = os.path.join(self.root, self.base_folder)
        if self.train or self.validation:
            images_path = os.path.join(images_path, 'train')
            data_dict = get_data(images_path)
        else:
            images_path = os.path.join(images_path, 'val')
            data_dict = get_data(images_path)
        y = 0
        for key in data_dict.keys():
            num_y = len(data_dict[key])
            self.data.extend([data_dict[key][i] for i in np.arange(0,num_y)])
            self.targets.extend([y for i in np.arange(0,num_y)])
            y += 1


    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img_path, target = self.data[index], self.targets[index]
        img = jpg_image_to_array(img_path)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            if simple:
                img = self.simple_transform(img)
            else:
                img = self.transform(img)

        return img, self.class_mapping[target]

    
    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root)
            elif self.split == 'val':
                parse_val_archive(self.root)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

class iTinyIMNET(iDataset):
    
    im_size=64
    nch=3

    def load(self):
        self.dw = False
        self.data, self.targets = [], []

        from os import path
        root = self.root
        FileNameEnd = 'JPEG'
        train_dir = path.join(root, 'tiny-imagenet/tiny-imagenet-200/train')
        self.class_names = sorted(os.listdir(train_dir))
        self.names2index = {v: k for k, v in enumerate(self.class_names)}
        self.data = []
        self.targets = []

        if self.train:
            for label in self.class_names:
                d = path.join(root, 'tiny-imagenet/tiny-imagenet-200/train', label)
                for directory, _, names in os.walk(d):
                    for name in names:
                        filename = path.join(directory, name)
                        if filename.endswith(FileNameEnd):
                            self.data.append(filename)
                            self.targets.append(self.names2index[label])
        else:
            val_dir = path.join(root, 'tiny-imagenet/tiny-imagenet-200/val')
            with open(path.join(val_dir, 'val_annotations.txt'), 'r') as f:
                infos = f.read().strip().split('\n')
                infos = [info.strip().split('\t')[:2] for info in infos]
                self.data = [path.join(val_dir, 'images', info[0])for info in infos]
                self.targets = [self.names2index[info[1]] for info in infos]


    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img_path, target = self.data[index], self.targets[index]
        img = jpg_image_to_array(img_path)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            if simple:
                img = self.simple_transform(img)
            else:
                img = self.transform(img)

        return img, self.class_mapping[target], self.t

def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape 
    (width, height, channels)
    """
    with Image.open(image_path) as image:      
        image = image.convert('RGB')
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
    return im_arr

def get_data(root_images):

    import glob
    files = glob.glob(root_images+'/*/*.JPEG')
    data = {}
    for path in files:
        y = os.path.basename(os.path.dirname(path))
        if y in data:
            data[y].append(path)
        else:
            data[y] = [path]
    return data


