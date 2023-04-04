import time
import torch
import math
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from utils.visualisation import plot_bar
import os

"""
@inproceedings{Hsu18_EvalCL,
  title={Re-evaluating Continual Learning Scenarios: A Categorization and Case for Strong Baselines},
  author={Yen-Chang Hsu and Yen-Cheng Liu and Anita Ramasamy and Zsolt Kira},
  booktitle={NeurIPS Continual learning Workshop },
  year={2018},
  url={https://arxiv.org/abs/1810.12488}
}
"""

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum().item()
            res.append(correct_k*100.0 / batch_size)

        if len(res)==1:
            return res[0]
        else:
            return res


def new_vs_old_class_comparison(new_classes, old_classes, class_features, labels_to_names, class_mapping, replay_size, dim=0, base_path = 'plots_and_tables/replay_of_/', plot_fig=True, is_oracle=False):
    
    # metrics = ["MSE", "MMD_linear", "MMD_rbf", "MMD_poly", "CKA_linear", "CKA_kernel"]
   
    if not os.path.exists(base_path): os.makedirs(base_path)

    metrics = ["MMD_poly"]
    paths_to_save_metric_results = [base_path+'new_vs_old_class_' + metric + '.csv' for metric in metrics]
    classes = new_classes.union(old_classes)
    # print (new_classes)
    # print (old_classes)

    for metric, path in zip(metrics, paths_to_save_metric_results):
        

        new_vs_old_class = {}

        if metric == 'MSE':
            per_class_features = { k : torch.Tensor(class_features[k]).mean(0) for k in class_features}
            for cl in new_classes:
                new_vs_old_class[labels_to_names[class_mapping[cl]]] = { labels_to_names[class_mapping[k]]: float(MSE(per_class_features[k], per_class_features[cl], dim = dim)) for k in classes}  

        elif 'MMD' in metric:
            MMD_kernel  = MMD()
            per_class_features = { k: torch.stack(class_features[k]) for k in class_features}
            # per_class_features = { k: torch.tensor(class_features[k]) for k in class_features}
            if metric == 'MMD_linear':
                for cl in new_classes:
                    new_vs_old_class[labels_to_names[class_mapping[cl]]] = { labels_to_names[class_mapping[k]]: float(MMD_kernel.mmd_linear(per_class_features[k], per_class_features[cl])) for k in classes}  
            elif metric == 'MMD_rbf':
                for cl in new_classes:
                    # print ('yessssss')
                    # print (per_class_features)
                    new_vs_old_class[labels_to_names[class_mapping[cl]]] = { labels_to_names[class_mapping[k]]: float(MMD_kernel.mmd_rbf(per_class_features[k], per_class_features[cl])) for k in classes}  
            else:
                for cl in new_classes:
                    new_vs_old_class[labels_to_names[class_mapping[cl]]] = { labels_to_names[class_mapping[k]]: float(MMD_kernel.mmd_poly(per_class_features[k], per_class_features[cl])) for k in classes} 
                    
        elif 'CKA' in metric:
            CKA_kernel  = CKA()
            per_class_features = { k: torch.Tensor(class_features[k]) for k in class_features}
            if metric == 'CKA_linear':
                for cl in new_classes:
                    new_vs_old_class[labels_to_names[class_mapping[cl]]] = { labels_to_names[class_mapping[k]]: float(CKA_kernel.linear_CKA(per_class_features[k], per_class_features[cl][:len(per_class_features[k])])) for k in classes}  
            elif metric == 'CKA_kernel':
                for cl in new_classes:
                    new_vs_old_class[labels_to_names[class_mapping[cl]]] = { labels_to_names[class_mapping[k]]: float(CKA_kernel.kernel_CKA(per_class_features[k], per_class_features[cl][:len(per_class_features[k])])) for k in classes}  
        
        # print ('new_vs_old_class_unnorm: ', new_vs_old_class)

        for cl in new_vs_old_class.keys():
            factor=1.0/sum(new_vs_old_class[cl].values())
            for k in new_vs_old_class[cl]:
                new_vs_old_class[cl][k] = 1-(new_vs_old_class[cl][k]*factor)

        # print ('new_vs_old_class_norm: ', new_vs_old_class)

        # print (new_vs_old_class)
        df = pd.DataFrame(new_vs_old_class).T
        df.to_csv(path)

        if plot_fig:
            # print (new_vs_old_class.keys())
            colors  = ['r','g', 'b', 'y', 'tomato', 'k', 'c', 'maroon', 'olive', 'm']
            colors = {labels_to_names[class_mapping[k]]:c for k, c in zip(class_mapping.keys(), colors)}

            if not is_oracle:
                title = 'Replay samples per class: '+ str(replay_size)
            else:
                title = 'Oracle'
            for new_cl in new_vs_old_class.keys():
                xticks = new_vs_old_class[new_cl].keys()
                plot_colors = [colors[i] for i in xticks]
                plot_bar(to_plot = new_vs_old_class[new_cl].values(), xticks = xticks, plot_colors = plot_colors, xlabel = 'classes', 
                ylabel = 'Similarity w.r.t. '+ new_cl + ' class', title =title, fig_name = base_path+new_cl+'_vs_old_classes_' + metric + '.png')

        class_mapping = { v: k for k,v in class_mapping.items() }
        labels_to_names = { v: k for k,v in labels_to_names.items() }

        # { class_mapping[labels_to_names[k]]: new_vs_old_class[k] for k in new_vs_old_class}
        new_vs_old_class = { class_mapping[labels_to_names[k]]: { class_mapping[labels_to_names[k2]]: new_vs_old_class[k][k2] for k2 in new_vs_old_class[k]} for k in new_vs_old_class}
        sim_to_new_cls = np.array([[ new_vs_old_class[k][cl] for cl in new_vs_old_class[k]] for k in new_vs_old_class]).mean(0)
        return new_vs_old_class, sim_to_new_cls
        

def distribution_shift_comparison(labels_to_names, class_mapping, class_new_features, class_old_features, per_class_correct_predictions, per_class_accuracy, per_class_dist_shift, save_dir):

    # metrics = ["MSE", "MMD_linear", "MMD_rbf", "MMD_poly", "CKA_linear", "CKA_kernel"]
    
    # print ('per_class_total_samples:', per_class_total_samples)
    per_class_accuracy_epoch = {k: float(per_class_correct_predictions[k])/len(class_old_features[k]) for k in per_class_correct_predictions}
    per_class_dist_shift_epoch = {}

    MMD_kernel  = MMD()
    per_class_new_features = { k: torch.stack(class_new_features[k]) for k in class_new_features}
    per_class_old_features = { k: torch.stack(class_old_features[k]) for k in class_old_features}
    # per_class_new_features = { k: torch.Tensor(class_new_features[k]) for k in class_new_features}
    # per_class_old_features = { k: torch.Tensor(class_old_features[k]) for k in class_old_features}

    per_class_dist_shift_epoch = {k: float(MMD_kernel.mmd_poly(per_class_new_features[k], per_class_old_features[k], return_diag=True)) for k in per_class_correct_predictions}

    if sum(per_class_dist_shift_epoch.values())!=0:
        factor=1.0/sum(per_class_dist_shift_epoch.values())
        for k in per_class_dist_shift_epoch:
            per_class_dist_shift_epoch[k] = (per_class_dist_shift_epoch[k]*factor)
        
    # per_class_accuracy_epoch = {k: per_class_accuracy_epoch for k in metrics}
    
    per_class_accuracy = { key: per_class_accuracy.get(key,[])+[per_class_accuracy_epoch.get(key,[])] for key in per_class_accuracy_epoch.keys()}
    # per_class_dist_shift = { key: { key2: per_class_dist_shift[key].get(key2,[])+[per_class_dist_shift_epoch[key].get(key2,[])] for key2 in per_class_dist_shift_epoch[key]} for key in per_class_dist_shift_epoch }
    per_class_dist_shift = { key: per_class_dist_shift.get(key,[])+[per_class_dist_shift_epoch.get(key,[])] for key in per_class_dist_shift_epoch.keys()}

    # print (per_class_accuracy_epoch)
    # print (per_class_accuracy)
    # print (per_class_dist_shift)
    
    # print ({ labels_to_names[class_mapping[cl]]: per_class_dist_shift_epoch['MMD_poly'][cl] for cl in per_class_dist_shift_epoch['MMD_poly']})
    # print ({ labels_to_names[class_mapping[cl]]: per_class_dist_shift['MMD_poly'][cl] for cl in per_class_dist_shift['MMD_poly']})
    np.save(save_dir+'_after/per_class_accuracy.npy', per_class_accuracy)
    np.save(save_dir+'_after/per_class_dist_shift.npy', per_class_dist_shift) 

    return per_class_accuracy_epoch, per_class_dist_shift_epoch, per_class_accuracy, per_class_dist_shift
                    

def MSE(input1, input2, dim, p=2):
    return torch.norm(input1 - input2, dim=dim, p=p)

class MMD(object):
    def __init__(self, device="cuda"):
        self.device = device

    def polynomial_kernel(self, X, Y, degree=2, gamma=1, coef0=0):
        """
            Compute the polynomial kernel between two matrices X and Y::
                K(x, y) = (<x, y> + c)^p
            for each pair of rows x in X and y in Y.

            Args:
                X - (n, d) NumPy array (n datapoints each with d features)
                Y - (m, d) NumPy array (m datapoints each with d features)
                c - a coefficient to trade off high-order and low-order terms (scalar)
                p - the degree of the polynomial kernel

            Returns:
                kernel_matrix - (n, m) Numpy array containing the kernel matrix
        """
        # return ((gamma*(X.type(torch.float32) @ Y.transpose(0,1).type(torch.float32)) + coef0) ** degree)

        # print ('X: ', X.shape)
        # print ('Y: ', Y.shape)
        if len(X.shape)==3:
            return ((gamma*(torch.einsum("abc, dec -> adbe", X, Y)) + coef0) ** degree)
        return ((gamma*(X @ Y.transpose(0,1)) + coef0) ** degree)

    def mmd_poly(self, X, Y, degree=2, gamma=1, coef0=0, labels_end_ind=None, return_diag=False):
        """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            degree {int} -- [degree] (default: {2})
            gamma {int} -- [gamma] (default: {1})
            coef0 {int} -- [constant item] (default: {0})
        Returns:
            [scalar] -- [MMD value]
        """
        XX = self.polynomial_kernel(X, X, degree, gamma, coef0)
        YY = self.polynomial_kernel(Y, Y, degree, gamma, coef0)
        XY = self.polynomial_kernel(X, Y, degree, gamma, coef0)
        
        
        if len(X.shape)==3:
            no_of_classes = X.shape[0]
            Dis = torch.zeros(no_of_classes, no_of_classes)
            for i in range(no_of_classes):
                for j in range(no_of_classes):
                    Dis[i][j] = XX[i][i].mean()+YY[j][j].mean()-2*XY[i][j].mean()
            return Dis

        
        if labels_end_ind is None:
            # print (X.shape)
            # print (Y.shape)
            # print ((XX + YY- 2 * XY).shape)
            if return_diag:
                # print ('yess')
                return XX.diag().mean()+YY.diag().mean()-2*XY.diag().mean()
            else:
                return XX.mean() + YY.mean() - 2 * XY.mean()
       
        no_of_classes = labels_end_ind.shape[0]
        Dis = torch.zeros(no_of_classes, no_of_classes)
        for i in range(no_of_classes):
            start_i = labels_end_ind[i-1] if i!=0 else 0
            end_i = labels_end_ind[i]
            for j in range(no_of_classes):
                start_j = labels_end_ind[j-1] if j!=0 else 0
                end_j = labels_end_ind[j]
                Dis[i][j] = XX[start_i:end_i, start_i:end_i].mean()+YY[start_j:end_j, start_j:end_j].mean()-2*XY[start_i:end_i, start_j:end_j].mean()

        # print ('4:', Dis)
        return Dis

class CKA(object):
    def __init__(self):
        pass 
    
    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H) 

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX
 
    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = float(self.sum) / self.count

    def update_count(self, multiplier):
        self.count = self.count * multiplier
        self.avg = float(self.sum) / self.count


class Timer(object):
    """
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval