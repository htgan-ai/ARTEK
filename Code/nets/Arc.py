import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter

from nets.iresnet import (iresnet18, iresnet34, iresnet50, iresnet64, iresnet100, iresnet200)
from nets.mobilefacenet import get_mbf
from nets.artifingerNet import get_our_model, swap_conv2d, swap_conv2d_new, DEFAULT_DEN, count_and_log_wconv, replace_prelu
from nets.mobilenet import get_mobilenet
from nets.EdgeFace import get_edgeface_model
from nets.parameternet import parameternet_600m
from nets.ghostnetv3 import ghostnetv3
from nets.faster_vit_any_res import (
    faster_vit_0_any_res, faster_vit_1_any_res, faster_vit_2_any_res,
    faster_vit_3_any_res, faster_vit_4_any_res, faster_vit_5_any_res,
    faster_vit_6_any_res,
    
    faster_vit_4_21k_224_any_res, faster_vit_4_21k_384_any_res,
    faster_vit_4_21k_512_any_res, faster_vit_4_21k_768_any_res,
)
FASTER_VIT_FACTORIES = {
    "faster_vit_0_any_res": faster_vit_0_any_res,
    "faster_vit_1_any_res": faster_vit_1_any_res,
    "faster_vit_2_any_res": faster_vit_2_any_res,
    "faster_vit_3_any_res": faster_vit_3_any_res,
    "faster_vit_4_any_res": faster_vit_4_any_res,
    "faster_vit_5_any_res": faster_vit_5_any_res,
    "faster_vit_6_any_res": faster_vit_6_any_res,
    
    "faster_vit_4_21k_224_any_res": faster_vit_4_21k_224_any_res,
    "faster_vit_4_21k_384_any_res": faster_vit_4_21k_384_any_res,
    "faster_vit_4_21k_512_any_res": faster_vit_4_21k_512_any_res,
    "faster_vit_4_21k_768_any_res": faster_vit_4_21k_768_any_res,
}   
from nets.shvit import shvit_s1, shvit_s2, shvit_s3, shvit_s4
SHVIT_FACTORIES = {
    "shvit_s1": shvit_s1,
    "shvit_s2": shvit_s2,
    "shvit_s3": shvit_s3,
    "shvit_s4": shvit_s4,
}
from nets.EfficientViM import EfficientViM_M1, EfficientViM_M2, EfficientViM_M3, EfficientViM_M4
EfficientViM_FACTORIES = {
    "efficientvim_m1": EfficientViM_M1,
    "efficientvim_m2": EfficientViM_M2,
    "efficientvim_m3": EfficientViM_M3,
    "efficientvim_m4": EfficientViM_M4,
}
from nets.groupmamba.groupmamba import groupmamba_tiny, groupmamba_small, groupmamba_base
Group_Mamba_FACTORIES = {
    "groupmamba_tiny": groupmamba_tiny,
    "groupmamba_small": groupmamba_small,
    "groupmamba_base": groupmamba_base,
}
from nets.tinyvim.tinyvim import TinyViM_S, TinyViM_B, TinyViM_L
TinyViM_FACTORIES = {
    "tinyvim_s": TinyViM_S,
    "tinyvim_b": TinyViM_B,
    "tinyvim_l": TinyViM_L,
}

class Arcface_Head(Module):
    def __init__(self, embedding_size=128, num_classes=6000, s=64., m=0.5):
        super(Arcface_Head, self).__init__()
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine  = F.linear(input, F.normalize(self.weight))
        sine    = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = torch.where(cosine.float() > self.th, phi.float(), cosine.float() - self.mm)

        one_hot = torch.zeros(cosine.size()).type_as(phi).long()
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output  *= self.s
        return output

class Arcface(nn.Module):
    def __init__(self, num_classes, backbone="mobilefacenet", pretrained=False, 
                 head_type="arcface", mode="predict", head_kwargs=None,
                 embedding_size=128):
        super(Arcface, self).__init__()
        if head_kwargs is None:
            head_kwargs = {}
        s = head_kwargs.get("s", 64)
        self.use_backbone_classifier = False
        if backbone=="mobilefacenet":
            embedding_size  = embedding_size
            s               = s
            self.arcface    = get_mbf(embedding_size=embedding_size, pretrained=pretrained)
        elif backbone=="artifingerNet":
            embedding_size  = embedding_size
            s               = s
            self.arcface    = get_our_model(embedding_size=embedding_size, pretrained=pretrained)
            
            
            den_map = {**DEFAULT_DEN, 3:[0.75], 7:[0.3,0.6,0.9]}  
            
            swap_conv2d_new(self.arcface, to="wconv", den_map=den_map, learnable_phi=True, normalize_phi=True)
            
            
            
        elif backbone=="mobilenetv1":
            embedding_size  = embedding_size
            s               = s
            self.arcface    = get_mobilenet(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="iresnet18":
            embedding_size  = embedding_size
            s               = s
            self.arcface    = iresnet18(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="iresnet34":
            embedding_size  = embedding_size
            s               = s
            self.arcface    = iresnet34(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="iresnet50":
            embedding_size  = embedding_size
            s               = s
            self.arcface    = iresnet50(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="iresnet64":
            embedding_size  = embedding_size
            s               = s
            self.arcface    = iresnet64(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="iresnet100":
            embedding_size  = embedding_size
            s               = s
            self.arcface    = iresnet100(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="iresnet200":
            embedding_size  = embedding_size
            s               = s
            self.arcface    = iresnet200(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)
        elif backbone=='edgeface_xs_gamma_06':
            embedding_size  = embedding_size
            s               = s
            self.arcface    = get_edgeface_model('edgeface_xs_gamma_06')
            head_type       = 'cosface'
        elif backbone=='parameternet_600m':
            self.arcface = parameternet_600m(pretrained=pretrained, num_experts=4)
            self.arcface.reset_classifier(num_classes) 
            self.use_backbone_classifier = True
        elif backbone=='ghostnetv3':
            self.arcface = ghostnetv3(num_classes=num_classes)
            self.use_backbone_classifier = True
        elif backbone in FASTER_VIT_FACTORIES:
            self.arcface = FASTER_VIT_FACTORIES[backbone](pretrained=pretrained, num_classes=num_classes,
                                                          resolution=[256,256])
            self.use_backbone_classifier = True
        elif backbone in SHVIT_FACTORIES:
            
            
            self.arcface = SHVIT_FACTORIES[backbone](
                num_classes=num_classes,
                pretrained=pretrained,
                distillation=False,
                fuse=False,
            )
            self.use_backbone_classifier = True
        elif backbone in EfficientViM_FACTORIES:
            self.arcface = EfficientViM_FACTORIES[backbone](pretrained=pretrained, num_classes=num_classes)
            self.use_backbone_classifier = True
        elif backbone in Group_Mamba_FACTORIES:
            self.arcface = Group_Mamba_FACTORIES[backbone](pretrained=pretrained, num_classes=num_classes)
            self.use_backbone_classifier = True
        elif backbone in TinyViM_FACTORIES:
            self.arcface = TinyViM_FACTORIES[backbone](pretrained=pretrained, num_classes=num_classes)
            self.use_backbone_classifier = True
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilefacenet, mobilenetv1.'.format(backbone))

        self.mode = mode
        
        
        if self.use_backbone_classifier:
            self.head = nn.Identity()
            self.use_norms = False
        else:
            default_head_kwargs = dict(embedding_size=embedding_size,
                                    class_num=num_classes,
                                    m=0.5,
                                    s=s,
                                    t_alpha=0.8,
                                    h=0.4)
            default_head_kwargs.update(head_kwargs or {})
            self.head = build_head(head_type=head_type, **default_head_kwargs)
            self.use_norms = False
            if head_type == "adaface":
                self.use_norms = True

    def forward(self, x, y = None, mode = "predict"):
        
        if self.use_backbone_classifier:
            logits = self.arcface(x)              
            if mode == "predict":
                
                return F.softmax(logits, dim=-1)
            else:
                return logits                     
        feat = self.arcface(x)
        feat = feat.view(feat.size(0), -1)
        if mode == "predict":
            return F.normalize(feat, dim=1)
        else:
            if self.use_norms:
                norms = torch.norm(feat, p=2, dim=1, keepdim=True)
                feat_normed = feat / (norms + 1e-6)
                return self.head(feat_normed, norms, y)
            else:
                feat_normed = F.normalize(feat, dim=1)
            return self.head(feat_normed, y)


def build_head(head_type,
               embedding_size=512,
               class_num=10,
               m=0.5,
               t_alpha=1.0,
               h=0.333,
               s=64.0,
               ):

    if head_type == 'adaface':
        head = AdaFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       h=h,
                       s=s,
                       t_alpha=t_alpha,
                       )
    elif head_type == 'arcface':
        head = ArcFace_Head_2(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       s=s,
                       )
    elif head_type == 'cosface':
        head = CosFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       s=s,
                       )
    elif head_type == 'curricularface':
        head = CurricularFace(in_features=embedding_size,
                              out_features=class_num,
                              m=m,
                              s=s,
                              )
    else:
        raise ValueError('not a correct head type', head_type)
    return head

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output


class AdaFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))

        
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) 

        safe_norms = torch.clip(norms, min=0.001, max=100) 
        safe_norms = safe_norms.clone().detach()

        
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) 
        margin_scaler = margin_scaler * self.h 
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m

class CosFace(nn.Module):

    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.4):
        super(CosFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m  
        self.s = s  
        self.eps = 1e-4

        print('init CosFace with ')
        print('self.m', self.m)
        print('self.s', self.s)

    def forward(self, embbedings, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) 

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        cosine = cosine - m_hot
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m


class ArcFace_Head_2(Module):

    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.5):
        super(ArcFace_Head_2, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.s = s 

        self.eps = 1e-4

    
    def forward(self, embbedings, label):
        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) 

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        theta = cosine.acos()

        theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi-self.eps)
        cosine_m = theta_m.cos()
        scaled_cosine_m = cosine_m * self.s

        return scaled_cosine_m
    

class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, m = 0.5, s = 64.):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis = 1)
        kernel_norm = l2_norm(self.kernel, axis = 0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m 
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        
        return output