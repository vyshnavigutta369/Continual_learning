import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np


class ResNetZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, mode=1, prompt_flag=False, prompt_param=None):
        super(ResNetZoo, self).__init__()

        # get last layer
        self.last_balanced = nn.Linear(768, num_classes)
        self.last_unbalanced = nn.Linear(768, num_classes)
        self.task_id = None

        # get feature encoder
        if mode == 0:
            if pt:
                zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                           num_heads=12, use_grad_checkpointing=False, ckpt_layer=0,
                                           drop_path_rate=0
                                          )
                from timm.models import vit_base_patch16_224
                load_dict = vit_base_patch16_224(pretrained=True).state_dict()
                del load_dict['head.weight']; del load_dict['head.bias']
                zoo_model.load_state_dict(load_dict)
        
        # feature encoder changes if transformer vs resnet
        self.encoder = zoo_model
        
    def classifier_head_forward(self, out, balanced = False):
        
        if not balanced:
            outer = self.last_unbalanced(out)
        else:
            outer = self.last_balanced(out)

        return outer    

    def encode(self,x):

        out, _ = self.encoder(x)
        # print (out)
        out = out[:,0,:]
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x, pen=False, balanced=False):

        out = self.encode(x)
        # print (out)
        if not balanced: 
            # print ('yeaaaahhhhh')
            outer = self.last_unbalanced(out)
        else:
            # print ('noooooooooo')
            outer = self.last_balanced(out.detach())
        if pen: 
            return outer, out
        else:
            return outer
            

def vit_pt_imnet(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None):
    return ResNetZoo(num_classes=out_dim, pt=True, mode=0)

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