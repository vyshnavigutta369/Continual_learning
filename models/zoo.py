import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np

def tensor_prompt(a, b, c=None):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    nn.init.uniform_(p)
    return p

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048]
}


class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count_f = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.expand_and_freeze = False
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # init frequency table
        for e in self.e_layers:
            setattr(self, f'freq_curr_{e}',torch.nn.Parameter(torch.zeros(self.e_pool_size,), requires_grad=False))
            setattr(self, f'freq_past_{e}',torch.nn.Parameter(torch.zeros(self.e_pool_size,), requires_grad=False))

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)

    def _init_smart(self, emb_d, prompt_param):
        
        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0,1]
        self.e_layers = [3,4,5]

        # prompt pool size
        self.g_p_length = prompt_param[2]
        self.e_p_length = prompt_param[1]
        self.e_pool_size = prompt_param[0]

    def process_frequency(self):
        self.task_count_f += 1
        if not self.task_id_bootstrap:
            for e in self.e_layers:
                f_ = getattr(self, f'freq_curr_{e}')
                f_ = f_ / torch.sum(f_)
                setattr(self, f'freq_past_{e}',torch.nn.Parameter(f_, requires_grad=False))


    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            if self.expand_and_freeze:
                K = getattr(self,f'e_k_{l}')
                p = getattr(self,f'e_p_{l}')

                # freeze/control past tasks
                pt = self.e_pool_size / self.n_tasks
                s = int(self.task_count_f * pt)
                f = int((self.task_count_f + 1) * pt)
                
                if train:
                    if self.task_count_f > 0:
                        K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                        p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                    else:
                        K = K[s:f]
                        p = p[s:f]
                else:
                    K = K[0:f]
                    p = p[0:f]
                
            else:
                K = getattr(self,f'e_k_{l}') # 0 based indexing here
                p = getattr(self,f'e_p_{l}') # 0 based indexing here
            

            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            if train:

                # prompting
                if self.task_id_bootstrap:
                    loss = 1.0 - cos_sim[:,task_id].mean()  # the cosine similarity is always le 1
                    P_ = p[task_id].expand(len(x_querry),-1,-1)
                else:
                    if self.task_count_f > 0:
                        f_ = getattr(self, f'freq_past_{l}')
                        f_tensor = f_.expand(B,-1)
                        # cos_sim_scaled = 1.0 - (f_tensor * (1.0 - cos_sim))
                        cos_sim_scaled = cos_sim
                    else:
                        cos_sim_scaled = cos_sim
                    top_k = torch.topk(cos_sim_scaled, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = 1.0 - cos_sim[:,k_idx].mean()  # the cosine similarity is always le 1
                    P_ = p[k_idx][:,0]

                    # update frequency
                    f_ = getattr(self, f'freq_curr_{l}')
                    f_to_add = torch.bincount(k_idx.flatten().detach(),minlength=self.e_pool_size)
                    f_ += f_to_add
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices

                P_ = p[k_idx][:,0]
                
            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]
        
        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            Gk = P_[:,:j,:]
            Gv = P_[:,j:,:]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block

class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0,1,3,4,5]
        else:
            self.e_layers = [0]

        if prompt_param[3] == 3:
            self.expand_and_freeze = True


        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = prompt_param[1]
        self.e_pool_size = prompt_param[0]


class ResNetZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, mode=1, prompt_flag=False, prompt_param=None):
        super(ResNetZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None

        # get feature encoder
        if mode == 0:
            if pt:
                zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, 
                                           num_heads=12, use_grad_checkpointing=False, ckpt_layer=0,
                                           drop_path_rate=0
                                          )   
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu", check_hash=True)
                state_dict = checkpoint["model"]     
                msg = zoo_model.load_state_dict(state_dict,strict=False)
                self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])

        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])

        else:
            self.prompt = None
        
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
        

    def forward(self, x, pen=False, train=False):

        
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feat(x)
                q = q[:,0,:]
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            out = out[:,0,:]
        else:
            out, _ = self.feat(x)
            out = out[:,0,:]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            
            return out, prompt_loss
        else:
            return out
            

def vit_pt_imnet(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None):
    return ResNetZoo(num_classes=out_dim, pt=True, mode=0, prompt_flag=prompt_flag, prompt_param=prompt_param)

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet18', head='linear', feat_dim=128, out_dim=None):
        super(SupConResNet, self).__init__()
        
        zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, 
                                           num_heads=12, use_grad_checkpointing=False, ckpt_layer=0,
                                           drop_path_rate=0
                                          )   
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True)
        state_dict = checkpoint["model"]     
        msg = zoo_model.load_state_dict(state_dict,strict=False)

        dim_in= 768
        
        # model_fun, dim_in = model_dict[name]
        # zoo_model = model_fun().cuda()

        self.encoder = zoo_model
        

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def reinit_head(self):
        for layers in self.head.children():
            if hasattr(layers, 'reset_parameters'):
                layers.reset_parameters()


    def forward(self, x, return_feat=False, norm=False):
        encoded,_ = self.encoder(x)
        encoded = encoded[:,0,:]
        encoded = encoded.view(encoded.size(0), -1)
        # encoded = self.encoder(x)
        if norm:
            feat = F.normalize(self.head(encoded), dim=1)
        else:
            feat = self.head(encoded)
        if return_feat:
            return feat, encoded
        else:
            return feat


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes=10):
        super(LinearClassifier, self).__init__()
        feat_dim = 768
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        features = features[:,0,:]
        features = features.view(features.size(0), -1)
        # feat = F.normalize(self.head(encoded), dim=1)
        return self.fc(features)
        # return F.normalize(self.fc(features), dim=1)