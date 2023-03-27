import os
import os.path
import hashlib
import errno
import torch
from torchvision import transforms
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from torchvision import transforms as T
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
import torch
import torch.nn as nn

dataset_stats = {
    'CIFAR2' : {'mean': (0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
                 'std' : (0.2470322324632819, 0.24348512800005573, 0.26158784172796434),
                 'size' : 32},
    'CIFAR8' : {'mean': (0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
                 'std' : (0.2470322324632819, 0.24348512800005573, 0.26158784172796434),
                 'size' : 32},
    'CIFAR10' : {'mean': (0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
                 'std' : (0.2470322324632819, 0.24348512800005573, 0.26158784172796434),
                 'size' : 32},
    'CIFAR100': {'mean': (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                 'std' : (0.2673342858792409, 0.25643846291708816, 0.2761504713256834),
                 'size' : 32},
    'ImageNet32': {'mean': (0.4037, 0.3823, 0.3432),
                 'std' : (0.2417, 0.2340, 0.2235),
                 'size' : 32},
    'ImageNet84': {'mean': (0.4399, 0.4184, 0.3772),
                 'std' : (0.2250, 0.2199, 0.2139),
                 'size' : 84},
    'ImageNet': {'mean': (0.485, 0.456, 0.406),
                 'std' : (0.229, 0.224, 0.225),
                 'size' : 224},   
    'TinyImageNet': {'mean': (0.4389, 0.4114, 0.3682),
                 'std' : (0.2402, 0.2350, 0.2268),
                 'size' : 64},   
    'ImageNet_R': {
                 'size' : 224}, 
    'ImageNet_D': {
                 'size' : 224},
    'DomainNet': {
                 'size' : 224},  
                }

# transformations
def get_transform(dataset='cifar100', phase='test', aug=True, resize_imnet=False):
    transform_list = []
    # get out size
    crop_size = dataset_stats[dataset]['size']

    # get mean and std
    dset_mean = (0.0,0.0,0.0) # dataset_stats[dataset]['mean']
    dset_std = (1.0,1.0,1.0) # dataset_stats[dataset]['std']

    if dataset == 'ImageNet32' or dataset == 'ImageNet84':
        transform_list.extend([
            transforms.Resize((crop_size,crop_size))
        ])

    if phase == 'train':
        transform_list.extend([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dset_mean, dset_std),
                            ])
    else:
        if dataset.startswith('ImageNet') or dataset == 'DomainNet':
            transform_list.extend([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std),
                                ])
        else:
            transform_list.extend([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std),
                                ])


    return transforms.Compose(transform_list)

def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True

def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)

def make_video_ffmpeg(frame_dir, output_file_name='vis.mp4', frame_filename="frame%06d.png", fps=30):

    frame_ref_path = str(frame_dir / frame_filename)
    video_path = str(frame_dir / output_file_name)
    subprocess.call(
        f"ffmpeg -r {fps} -i {frame_ref_path} -vcodec libx264 -crf 10 -pix_fmt yuv420p"
        f" {video_path}".split()
    )
    return video_path

def transform_with_pca(embeddings, n_components=2):
    # text_prompts = np.load(train_on)
    # text_embeddings = get_text_conditioned_embeddings(pipeline, text_prompts)
    # nx, ny, nz = embeddings[0].shape
    # print (nx, ny, nz)
    # embeddings = [embedding.cpu().tolist() for embedding in embeddings]

    nsamples = len(embeddings)
    X = np.array(embeddings).reshape((nsamples,-1))
    X = (X - np.min(X))/(np.max(X) - np.min(X))
    # pca_plots(X)

    pcamodel = PCA(n_components=0.95)
    embeddings = pcamodel.fit_transform(X)
    return embeddings


def create_video(output_path):    

    output_path = Path(output_path+'/replay_analysis/') 
    vid = make_video_ffmpeg(output_path, "tsne_visualisation.mp4", fps=10, frame_filename=f"frame%06d.png")
    output_path_files = os.listdir(output_path)

    for item in output_path_files:
        if not item.endswith(".mp4"):
            os.remove(os.path.join(output_path, item))


def visualise( replay_dataset, logits, proba, replay_ixs, model_save_dir, filename):


    output_path = Path(model_save_dir+'/replay_analysis/') 
    output_path.mkdir(exist_ok=True, parents=True)

    # if filename=='frame000000.png':
    #     # print ('YESSSSSSSS')
    #     output_path_files = os.listdir(output_path)
    #     for item in output_path_files:
    #         path = model_save_dir+'/replay_analysis/'+item
    #         if os.path.exists(path):
    #             os.remove(path)
    
    
    def plot(embeddings, values, label, symbol):
        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings = df_embeddings.rename(columns={0:'x1',1:'x2'})
        df_embeddings = df_embeddings.assign(values=values)
        df_embeddings = df_embeddings.assign(label=label)
        # df_embeddings = df_embeddings.assign(symbol= symbol)
        fig = px.scatter(
        df_embeddings, x='x1', y='x2',
        color='label', labels={'color': 'label'},
        size = values,
        symbol = symbol, 
        symbol_sequence= ['circle', 'circle-open-dot', 'square-open-dot'],
        color_discrete_map= {'train': 'red', 'replay': 'goldenrod', 'non-replay': 'powderblue'})

        return fig

    label =  ['train' for i in range(len(proba))]
    symbol = [1 for i in range(len(proba))]
    if replay_ixs is not None:
        # replay_ixs = replay_ixs.tolist()
        logits.extend(np.array(replay_dataset.logits)[replay_ixs].tolist())
        proba.extend(np.array(replay_dataset.proba)[replay_ixs].tolist())
        label.extend(['replay' for i in range(len(replay_ixs))])
        symbol.extend([2 for i in range(len(replay_ixs))])
        
        non_replay_ixs = [i for i in range(len(replay_dataset)) if i not in replay_ixs]
        logits.extend(np.array(replay_dataset.logits)[non_replay_ixs].tolist())
        proba.extend(np.array(replay_dataset.proba)[non_replay_ixs].tolist())
        label.extend(['non-replay' for i in range(len(non_replay_ixs))])
        symbol.extend([3 for i in range(len(non_replay_ixs))])

    embeddings = transform_with_pca(logits)
    proba -= proba.min(1)[0]
    proba /= proba.max(1)[0]
    # print (proba)
    fig = plot(embeddings, values = proba, label = label, symbol = symbol)
    fig.write_image(output_path/filename, width=1920, height=1080)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature

    def forward(self, features, labels=None, mask=None, target_labels=None, reduction='mean'):
        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        curr_class_mask = torch.zeros_like(labels)
        for tc in target_labels:
            curr_class_mask += (labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(device)
        loss = curr_class_mask * loss.view(anchor_count, batch_size)

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            loss = loss.mean(0)
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))

        return loss
