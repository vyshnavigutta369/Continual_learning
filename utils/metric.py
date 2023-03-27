import time
import torch
import math
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

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

def new_vs_old_class_comparison(new_classes, old_classes, class_features, labels_to_names, class_mapping, replay_size, dim=0, base_path = 'plots_and_tables/replay_of_/', plot_fig=True):
    
    # metrics = ["MSE", "MMD_linear", "MMD_rbf", "MMD_poly", "CKA_linear", "CKA_kernel"]
    metrics = ["MMD_poly"]
    # paths_to_save_metric_results = [base_path+'new_vs_old_class_MSE.csv', base_path+'new_vs_old_class_MMD_linear.csv', base_path+'new_vs_old_class_MMD_rbf.csv', base_path+'new_vs_old_class_MMD_poly.csv', 
    #                                 base_path+'new_vs_old_class_CKA_linear.csv', base_path+'new_vs_old_class_CKA_kernel.csv']
    paths_to_save_metric_results = [base_path+'new_vs_old_class_MMD_rbf.csv']

    for metric, path in zip(metrics, paths_to_save_metric_results):

        new_vs_old_class = {}

        if metric == 'MSE':
            per_class_features = { k : torch.Tensor(class_features[k]).mean(0) for k in class_features}
            for cl in new_classes:
                new_vs_old_class[labels_to_names[class_mapping[cl]]] = { labels_to_names[class_mapping[k]]: float(MSE(per_class_features[k], per_class_features[cl], dim = dim)) for k in class_mapping}  

        elif 'MMD' in metric:
            MMD_kernel  = MMD()
            per_class_features = { k: torch.Tensor(class_features[k]) for k in class_features}
            if metric == 'MMD_linear':
                for cl in new_classes:
                    new_vs_old_class[labels_to_names[class_mapping[cl]]] = { labels_to_names[class_mapping[k]]: float(MMD_kernel.mmd_linear(per_class_features[k], per_class_features[cl])) for k in class_mapping}  
            elif metric == 'MMD_rbf':
                for cl in new_classes:
                    # print ('yessssss')
                    # print (per_class_features)
                    new_vs_old_class[labels_to_names[class_mapping[cl]]] = { labels_to_names[class_mapping[k]]: float(MMD_kernel.mmd_rbf(per_class_features[k], per_class_features[cl])) for k in class_mapping}  
            else:
                for cl in new_classes:
                    new_vs_old_class[labels_to_names[class_mapping[cl]]] = { labels_to_names[class_mapping[k]]: float(MMD_kernel.mmd_poly(per_class_features[k], per_class_features[cl])) for k in class_mapping} 
        elif 'CKA' in metric:
            CKA_kernel  = CKA()
            per_class_features = { k: torch.Tensor(class_features[k]) for k in class_features}
            if metric == 'CKA_linear':
                for cl in new_classes:
                    new_vs_old_class[labels_to_names[class_mapping[cl]]] = { labels_to_names[class_mapping[k]]: float(CKA_kernel.linear_CKA(per_class_features[k], per_class_features[cl][:len(per_class_features[k])])) for k in class_mapping}  
            elif metric == 'CKA_kernel':
                for cl in new_classes:
                    new_vs_old_class[labels_to_names[class_mapping[cl]]] = { labels_to_names[class_mapping[k]]: float(CKA_kernel.kernel_CKA(per_class_features[k], per_class_features[cl][:len(per_class_features[k])])) for k in class_mapping}  
        
        for cl in new_vs_old_class.keys():
            factor=1.0/sum(new_vs_old_class[cl].values())
            for k in new_vs_old_class[cl]:
                new_vs_old_class[cl][k] = 1-(new_vs_old_class[cl][k]*factor)

        
        df = pd.DataFrame(new_vs_old_class).T
        df.to_csv(path)

        if plot_fig:

            colors  = ['r','g', 'b', 'y', 'tomato', 'k', 'c', 'maroon', 'olive', 'm']
            colors = {labels_to_names[class_mapping[k]]:c for k, c in zip(class_mapping.keys(), colors)}

            for new_cl in new_vs_old_class.keys():

                plt.clf()
                x = new_vs_old_class[new_cl].keys()
                y = new_vs_old_class[new_cl].values()

                ax = plt.gca()
                ax.tick_params(axis='x', which='major', labelsize=8.5)
                plt.bar(x, y, color = [colors[i] for i in x], edgecolor='k', width = 0.5, align='edge')
                xlocs, xlabs = plt.xticks()

                for i, v in enumerate(y):
                    v =  float("{:.2f}".format(v))
                    plt.text(xlocs[i] - 0.25, v + 0.01, str(v), size=8.5)

                plt.subplots_adjust(hspace=1)
                plt.xlabel('classes')
                plt.ylabel('Similarity w.r.t. '+ new_cl + ' class')
                plt.title('Replay samples per class: '+ str(replay_size))
                #plt.show()
                plt.savefig(base_path+'new('+new_cl+')_vs_old_class_MMD_rbf.png')


def distribution_shift_comparison(class_new_features, class_old_features, per_class_total_samples, per_class_correct_predictions):

    # metrics = ["MSE", "MMD_linear", "MMD_rbf", "MMD_poly", "CKA_linear", "CKA_kernel"]
    metrics = ["MMD_rbf"]

    # print ('per_class_correct_predictions:', per_class_correct_predictions)
    # print ('per_class_total_samples:', per_class_total_samples)
    per_class_accuracy = {k: float(per_class_correct_predictions[k])/per_class_total_samples[k] for k in per_class_correct_predictions}
    per_class_dist_shift = {}

    for metric in metrics:
        if metric == "MSE":
            per_class_dist_shift[metric] = {k: float(MSE(torch.Tensor(class_new_features[k]), torch.Tensor(class_old_features[k]), dim=1).sum()) for k in per_class_correct_predictions}
        elif 'MMD' in metric:
            MMD_kernel  = MMD()
            per_class_new_features = { k: torch.Tensor(class_new_features[k]) for k in class_new_features}
            per_class_old_features = { k: torch.Tensor(class_old_features[k]) for k in class_old_features}
            if metric == 'MMD_linear':
                per_class_dist_shift[metric] = {k: float(MMD_kernel.mmd_linear(per_class_new_features[k], per_class_old_features[k]).sum()) for k in per_class_correct_predictions}
            elif metric == 'MMD_linear':
                per_class_dist_shift[metric] = {k: float(MMD_kernel.mmd_rbf(per_class_new_features[k], per_class_old_features[k]).sum()) for k in per_class_correct_predictions}
            else:
                per_class_dist_shift[metric] = {k: float(MMD_kernel.mmd_poly(per_class_new_features[k], per_class_old_features[k]).sum()) for k in per_class_correct_predictions}
        elif 'CKA' in metric:
            CKA_kernel  = CKA()
            per_class_new_features = { k: torch.Tensor(class_new_features[k]) for k in class_new_features}
            per_class_old_features = { k: torch.Tensor(class_old_features[k]) for k in class_old_features}
            if metric == 'CKA_linear':
                per_class_dist_shift[metric] = {k: float(CKA_kernel.linear_CKA(per_class_new_features[k], per_class_old_features[k]).sum()) for k in per_class_correct_predictions}
            elif metric == 'CKA_kernel':
                per_class_dist_shift[metric] = {k: float(CKA_kernel.kernel_CKA(per_class_new_features[k], per_class_old_features[k]).sum()) for k in per_class_correct_predictions}

        if sum(per_class_dist_shift[metric].values())!=0:
            factor=1.0/sum(per_class_dist_shift[metric].values())
            for k in per_class_dist_shift[metric]:
                per_class_dist_shift[metric][k] = (per_class_dist_shift[metric][k]*factor)
        
    per_class_accuracy = {k: per_class_accuracy for k in metrics}
    return per_class_accuracy, per_class_dist_shift
                    
def per_class_plots(per_class_accuracy, per_class_dist_shift, task_acc, labels_to_names, class_mapping, epochs_of_interest, replay_size, base_path = 'plots_and_tables/'):

    colors  = ['r','g', 'b', 'y', 'tomato', 'k', 'c', 'maroon', 'olive', 'm']
    colors_a = {labels_to_names[class_mapping[k]]:c for k, c in zip(class_mapping.keys(), colors)}
    colors_b = { k: c for k, c in zip(task_acc.keys(), colors)}
    colors = {**colors_a, **colors_b}

    def plot_utils(data):
        
        to_plot = []
        labels = []
        plot_colors = []

        # print ('hellllloooooooooooooo', data)
        for i, (k, v) in enumerate(data.items()):
            try:
                labels.append(labels_to_names[class_mapping[k]])
            except:
                labels.append(k)
            to_plot.append(v)
            plot_colors.append(colors[labels[-1]])
        
        return to_plot, labels, plot_colors

    def plot(to_plot, labels, plot_colors, xlabel, ylabel, y_lim_bottom, title, fig_name, sort_handles=False):

        import matplotlib.pyplot as plt

        plt.clf()
        max_v = -10000
        for i, v in enumerate(to_plot):
            label = labels[i]
            plt.plot(range(0, len(v)), v, '.-', label=label, color = plot_colors[i])
            max_v = max(max_v, max(v))
             
        if sort_handles:
            handles,labels = plt.gca().get_legend_handles_labels()
            legend_order = np.array(to_plot)[:, -1].argsort()[::-1]
            handle_order, label_order = np.array(handles)[legend_order], np.array(labels)[legend_order]
            plt.legend(handle_order, label_order)  # To draw legend
        
        x = epochs_of_interest
        xi = list(range(len(x)))
        plt.xlabel(xlabel)
        plt.xticks(xi, x)

        plt.gca().set_ylim(bottom=y_lim_bottom, top=max_v*1.05)
        
        plt.ylabel(ylabel) 
        plt.title(title)
        plt.savefig(fig_name)

        plt.clf()

    for metric in per_class_accuracy:

        to_plot, labels, plot_colors = plot_utils(per_class_accuracy[metric])     
        plot(to_plot, labels, plot_colors, xlabel='epochs', ylabel='accuracy', y_lim_bottom=0.3, title='Replay samples per class: '+ str(replay_size), fig_name = base_path+'per_class_accuracy_' + metric + '.png', sort_handles=True)
        
        to_plot, labels, plot_colors = plot_utils(per_class_dist_shift[metric])
        plot(to_plot, labels, plot_colors, xlabel='epochs', ylabel='distribution shift between task one and task two model', y_lim_bottom=0, title='Replay samples per class: '+ str(replay_size), fig_name = base_path+'per_class_dist_shift_' + metric + '.png', sort_handles=True)
        
    to_plot, labels, plot_colors = plot_utils(task_acc)     
    print ('helloooooo', to_plot)
    plot(to_plot, labels, plot_colors, xlabel='epochs', ylabel='average task accuracy', y_lim_bottom=0, title='Replay samples per class: '+ str(replay_size), fig_name = base_path+'avg_task_accuracy.png')

def MSE(input1, input2, dim, p=2):
    return torch.norm(input1 - input2, dim=dim, p=p)

class MMD(object):
    def __init__(self, device="cuda"):
        self.device = device

    def mmd_linear(self, X, Y):
        """MMD using linear kernel (i.e., k(x,y) = <x,y>)
        Note that this is not the original linear MMD, only the reformulated and faster version.
        The original version is:
            def mmd_linear(X, Y):
                XX = np.dot(X, X.T)
                YY = np.dot(Y, Y.T)
                XY = np.dot(X, Y.T)
                return XX.mean() + YY.mean() - 2 * XY.mean()
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Returns:
            [scalar] -- [MMD value]
        """
        delta = X.mean(0) - Y.mean(0)
        return delta.dot(delta.T)


    def mmd_rbf(self, X, Y, gamma=1.0):
        """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            gamma {float} -- [kernel parameter] (default: {1.0})
        Returns:
            [scalar] -- [MMD value]
        """
        XX = metrics.pairwise.rbf_kernel(X, X, gamma)
        YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
        XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()


    def mmd_poly(self, X, Y, degree=2, gamma=1, coef0=0):
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
        XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
        YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
        XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
        return XX.mean() + YY.mean() - 2 * XY.mean()


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