import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from .backbone import resnet18
# from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion
import einops
import copy
# from models.VAVL_git.VAVL.conformer.model import Conformer
from models.fusion_gates import *
from torch.distributions import Categorical, Normal
from typing import Dict


class VClassifier_CREMAD_linearcls(nn.Module):
    def __init__(self, args, encs):
        super(VClassifier_CREMAD_linearcls, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1
        modality = args.get("modality", "visual")
        self.visual_net = resnet18(modality=modality)
        # self.vcaster = nn.Conv2d(9,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, weights='ResNet18_Weights.DEFAULT') # , weights='ResNet18_Weights.DEFAULT'

        # self.vclassifier = nn.Linear(512, num_classes)
        # self.vclassifier = nn.Linear(1000, num_classes)

        self.vclassifier =  nn.Sequential(
        #     nn.Linear(d_model, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(fc_inner, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, **kwargs):



        #
        # v = self.visual_net(self.vcaster(x[1].flatten(start_dim=1, end_dim=2)))
        # pred_v = self.vclassifier(v)
        # print(x[1].shape)
        v = self.visual_net(x[1])
        B = x[1].shape[0]
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        video_feat = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(video_feat, 1)
        v = torch.flatten(v, 1)

        if "detach_enc1" in kwargs and kwargs["detach_enc1"]:
            v = v.detach()
            video_feat = video_feat.detach()

        if "detach_pred" in kwargs and kwargs["detach_pred"]:
            pred_v = self.vclassifier(v.detach())
        else:
            pred_v = self.vclassifier(v)

        return {"preds":{"combined":pred_v}, "features":{"combined":v}, "nonaggr_features":{"combined": video_feat.flatten(start_dim=2)}}
class VClassifier_CREMAD_IB_linearcls(nn.Module):
    def __init__(self, args, encs):
        super(VClassifier_CREMAD_IB_linearcls, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        modality = args.get("modality", "visual")
        self.visual_net = resnet18(modality=modality)

        self.fc_mu = nn.Linear(d_model, d_model, bias=False)
        self.fc_logvar = nn.Linear(d_model, d_model, bias=False)

        self.vclassifier =  nn.Sequential(
            nn.Linear(d_model, num_classes)
        )
    def IB_loss(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD * self.args.bias_infusion.lib

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, **kwargs):


        v = self.visual_net(x[1])
        B = x[1].shape[0]
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        video_feat = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(video_feat, 1)
        v = torch.flatten(v, 1)

        # v = torch.nn.functional.relu(v)
        feat_mu = self.fc_mu(v)
        feat_logvar = self.fc_logvar(v)
        v = self.reparameterize(feat_mu, feat_logvar)

        if "detach_enc1" in kwargs and kwargs["detach_enc1"]:
            v = v.detach()
            video_feat = video_feat.detach()

        if "detach_pred" in kwargs and kwargs["detach_pred"]:
            pred_v = self.vclassifier(v.detach())
        else:
            pred_v = self.vclassifier(v)

        return {"preds":{"combined":pred_v}, "features":{"combined":v}, "nonaggr_features":{"combined": video_feat.flatten(start_dim=2)}}
class VClassifier_CREMAD_linearcls_stopgrad(nn.Module):
    def __init__(self, args, encs):
        super(VClassifier_CREMAD_linearcls_stopgrad, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1
        modality = args.get("modality", "visual")
        self.visual_net = resnet18(modality=modality)
        # self.vcaster = nn.Conv2d(9,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, weights='ResNet18_Weights.DEFAULT') # , weights='ResNet18_Weights.DEFAULT'

        # self.vclassifier = nn.Linear(512, num_classes)
        # self.vclassifier = nn.Linear(1000, num_classes)

        self.vclassifier =  nn.Sequential(
        #     nn.Linear(d_model, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(fc_inner, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, **kwargs):

        #
        # v = self.visual_net(self.vcaster(x[1].flatten(start_dim=1, end_dim=2)))
        # pred_v = self.vclassifier(v)
        # print(x[1].shape)
        v = self.visual_net(x[1])
        B = x[1].shape[0]
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        pred_v = self.vclassifier(v.detach())

        return {"preds":{"combined":pred_v}, "features":{"combined":v}}
class VClassifier_CREMAD(nn.Module):
    def __init__(self, args, encs):
        super(VClassifier_CREMAD, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)
        modality = args.get("modality", "visual")
        self.visual_net = resnet18(modality=modality)
        # self.vcaster = nn.Conv2d(9,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False) # , weights='ResNet18_Weights.DEFAULT'
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, weights='ResNet18_Weights.DEFAULT') # , weights='ResNet18_Weights.DEFAULT'

        # self.vclassifier = nn.Linear(512, num_classes)
        # self.vclassifier = nn.Linear(1000, num_classes)

        self.vclassifier =  nn.Sequential(
            nn.Linear(d_model, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, num_classes)
        )

    def forward(self, x, **kwargs):

        #
        # v = self.visual_net(self.vcaster(x[1].flatten(start_dim=1, end_dim=2)))
        # pred_v = self.vclassifier(v)

        # v = self.visual_net(einops.rearrange(x[1],"b i c h w -> b (i c) h w"))
        # v = F.adaptive_avg_pool2d(v, 1)

        v = self.visual_net(x[1])
        B = x[1].shape[0]
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        pred_v = self.vclassifier(v)

        return {"preds":{"combined":pred_v}, "features":{"combined":v}}
class AClassifier_CREMAD_linearcls(nn.Module):
    def __init__(self, args, encs):
        super(AClassifier_CREMAD_linearcls, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1

        # self.fusion_module = ConcatFusion(output_dim=n_classes)
        # self.visual_net = resnet18(modality='visual')
        self.audio_net = resnet18(modality='audio')
        # self.vcaster = nn.Conv2d(9,3,1)
        # self.acaster = nn.Conv2d(1,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, pretrained=False)
        # self.audio_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, pretrained=False)
        self.aclassifier = nn.Linear(512, num_classes)


        # self.common_fc = nn.Sequential(
        #     nn.Linear(d_model, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(fc_inner, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(fc_inner, num_classes)
        # )

    def forward(self, x, **kwargs):

        # a = self.audio_net(self.acaster(x[0].unsqueeze(dim=1)))
        # pred_a = self.common_fc(a)

        audio_feat = self.audio_net(x[0].unsqueeze(dim=1))
        a = F.adaptive_avg_pool2d(audio_feat, 1)
        a = torch.flatten(a, 1)
        if "detach_enc0" in kwargs and kwargs["detach_enc0"]:
            a = a.detach()
            audio_feat = audio_feat.detach()
        if "detach_pred" in kwargs and kwargs["detach_pred"]:
            pred_a = self.aclassifier(a.detach())
        else:
            pred_a = self.aclassifier(a)

        return {"preds": {"combined": pred_a}, "features": {"combined": a}, "nonaggr_features":{"combined": audio_feat.flatten(start_dim=2)}}
class AClassifier_CREMAD_IB_linearcls(nn.Module):
    def __init__(self, args, encs):
        super(AClassifier_CREMAD_IB_linearcls, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1

        # self.fusion_module = ConcatFusion(output_dim=n_classes)
        # self.visual_net = resnet18(modality='visual')
        self.audio_net = resnet18(modality='audio')
        # self.vcaster = nn.Conv2d(9,3,1)
        # self.acaster = nn.Conv2d(1,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, pretrained=False)
        # self.audio_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, pretrained=False)
        self.aclassifier = nn.Linear(512, num_classes)

        self.fc_mu = nn.Linear(d_model, d_model, bias=False)
        self.fc_logvar = nn.Linear(d_model, d_model, bias=False)


        # self.common_fc = nn.Sequential(
        #     nn.Linear(d_model, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(fc_inner, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(fc_inner, num_classes)
        # )
    def IB_loss(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD * self.args.bias_infusion.lib

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if self.training:
            return mu + eps * std
        else:
            return mu

    def forward(self, x, **kwargs):

        # a = self.audio_net(self.acaster(x[0].unsqueeze(dim=1)))
        # pred_a = self.common_fc(a)

        audio_feat = self.audio_net(x[0].unsqueeze(dim=1))
        a = F.adaptive_avg_pool2d(audio_feat, 1)
        a = torch.flatten(a, 1)

        # v = torch.nn.functional.relu(v)
        feat_mu = self.fc_mu(a)
        feat_logvar = self.fc_logvar(a)
        a = self.reparameterize(feat_mu, feat_logvar)

        IB_loss = self.IB_loss(feat_mu, feat_logvar)

        if "detach_enc0" in kwargs and kwargs["detach_enc0"]:
            a = a.detach()
            audio_feat = audio_feat.detach()


        if "detach_pred" in kwargs and kwargs["detach_pred"]:
            pred_a = self.aclassifier(a.detach())
        else:
            pred_a = self.aclassifier(a)

        output =  {"preds": {"combined": pred_a}, "features": {"combined": a}, "nonaggr_features":{"combined": audio_feat.flatten(start_dim=2)}}
        output["losses"] = {"IB_loss": IB_loss}

        return output

class AClassifier_CREMAD_linearcls_stopgrad(nn.Module):
    def __init__(self, args, encs):
        super(AClassifier_CREMAD_linearcls_stopgrad, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1

        # self.fusion_module = ConcatFusion(output_dim=n_classes)
        # self.visual_net = resnet18(modality='visual')
        self.audio_net = resnet18(modality='audio')
        # self.vcaster = nn.Conv2d(9,3,1)
        # self.acaster = nn.Conv2d(1,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, pretrained=False)
        # self.audio_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, pretrained=False)
        self.aclassifier = nn.Linear(512, num_classes)


        # self.common_fc = nn.Sequential(
        #     nn.Linear(d_model, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(fc_inner, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(fc_inner, num_classes)
        # )

    def forward(self, x, **kwargs):

        # a = self.audio_net(self.acaster(x[0].unsqueeze(dim=1)))
        # pred_a = self.common_fc(a)

        a = self.audio_net(x[0].unsqueeze(dim=1))
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)
        pred_a = self.aclassifier(a.detach())

        return {"preds": {"combined": pred_a}, "features": {"combined": a}}
class AClassifier_CREMAD(nn.Module):
    def __init__(self, args, encs):
        super(AClassifier_CREMAD, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1

        # self.fusion_module = ConcatFusion(output_dim=n_classes)
        # self.visual_net = resnet18(modality='visual')
        self.audio_net = resnet18(modality='audio')
        # self.vcaster = nn.Conv2d(9,3,1)
        # self.acaster = nn.Conv2d(1,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, pretrained=False)
        # self.audio_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, pretrained=False)
        # self.aclassifier = nn.Linear(512, num_classes)


        self.aclassifier = nn.Sequential(
            nn.Linear(d_model, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, num_classes)
        )

    def forward(self, x, **kwargs):

        # a = self.audio_net(self.acaster(x[0].unsqueeze(dim=1)))
        # pred_a = self.common_fc(a)

        a = self.audio_net(x[0].unsqueeze(dim=1))
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)
        pred_a = self.aclassifier(a)

        return {"preds": {"combined": pred_a}, "features": {"combined": a}}

class ConcatClassifier_CREMAD(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        d_model = args.d_model
        self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1
        # self.fusion_module = ConcatFusion(output_dim=n_classes)

        # self.vcaster = nn.Conv2d(9,3,1)
        # self.acaster = nn.Conv2d(1,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False) #, weights='ResNet18_Weights.DEFAULT'
        # self.audio_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False)

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')
        # self.vclassifier = nn.Linear(1000, n_classes)
        # self.aclassifier = nn.Linear(1000, n_classes)
        # self.classifier = nn.Linear(2000, n_classes)
        # self.classifier = nn.Linear(2000, n_classes)
        # self.classifier = nn.Sequential(
        #                 nn.Dropout(0.4),
                        # nn.Linear(2000, 64),
                        # nn.ReLU(),
                        # nn.Dropout(0.2),
                        # nn.Linear(64, n_classes)
                    # )

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.inst_norm = nn.InstanceNorm1d(num_classes, track_running_stats=False)

        elif self.cls_type != "linear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.inst_norm = nn.InstanceNorm1d(fc_inner, track_running_stats=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
            if not self.shared_pred:
                self.fc_0 = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, num_classes)
                )

                self.fc_1 = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, num_classes)
                )



    def _get_features(self, x):


        # print(v.shape)

        # vout = self.vclassifier(v)

        # B = x["spec"].shape[0]

        # a = self.audio_net(self.acaster(x[0].unsqueeze(dim=1)))
        # print(x[0].shape)
        a = self.audio_net(x[0].unsqueeze(dim=1))
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)

        v = self.visual_net(x[1])
        #
        # print(v.shape)
        B = x[1].shape[0]
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)

        return a, v

    def forward(self, x, **kwargs):

        a, v = self._get_features(x)

        pred_a = torch.matmul(a, self.fc_0_lin.weight.T) + self.fc_0_lin.bias/2
        pred_v = torch.matmul(v, self.fc_1_lin.weight.T) + self.fc_0_lin.bias/2

        if self.args.bias_infusion.get("inst_norm", False):
            pred_a = self.inst_norm(pred_a)
            pred_v = self.inst_norm(pred_v)

        pred = pred_a + pred_v
        if self.cls_type != "linear":
            pred = self.common_fc(pred)
            if self.shared_pred:
                pred_a = self.common_fc(pred_a)
                pred_v = self.common_fc(pred_v)
            else:
                pred_a = self.fc_0(pred_a)
                pred_v = self.fc_1(pred_v)

        return {"preds":{"combined":pred, "c":pred_a, "g":pred_v}, "features": {"c": a, "g": v, "combined": (a + v)/2}}
class ConcatClassifier_CREMAD_OGM(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        d_model = args.d_model
        self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)
        modalities = args.get("modalities", ["audio", "visual"])

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        # self.fusion_module = ConcatFusion(output_dim=n_classes)

        # self.vcaster = nn.Conv2d(9,3,1)
        # self.acaster = nn.Conv2d(1,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False) #, weights='ResNet18_Weights.DEFAULT'
        # self.audio_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False)

        self.net_mod0 = resnet18(modality=modalities[0])
        self.net_mod1 = resnet18(modality=modalities[1])
        # self.vclassifier = nn.Linear(1000, n_classes)
        # self.aclassifier = nn.Linear(1000, n_classes)
        # self.classifier = nn.Linear(2000, n_classes)
        # self.classifier = nn.Linear(2000, n_classes)
        # self.classifier = nn.Sequential(
        #                 nn.Dropout(0.4),
                        # nn.Linear(2000, 64),
                        # nn.ReLU(),
                        # nn.Dropout(0.2),
                        # nn.Linear(64, n_classes)
                    # )


        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.inst_norm = nn.InstanceNorm1d(num_classes, track_running_stats=False)

        if self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type != "linear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.inst_norm = nn.InstanceNorm1d(fc_inner, track_running_stats=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )

    def _get_features(self, x):


        # print(v.shape)

        # vout = self.vclassifier(v)

        # B = x["spec"].shape[0]

        # a = self.audio_net(self.acaster(x[0].unsqueeze(dim=1)))
        # print(x[0].shape)
        a = self.net_mod0(x[0].unsqueeze(dim=1))
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)

        v = self.net_mod1(x[1])
        #
        # print(v.shape)
        B = x[1].shape[0]
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)

        return a, v

    def forward(self, x, **kwargs):

        a, v = self._get_features(x)

        if self.mmcosine:
            pred_a = torch.mm(F.normalize(a, dim=1),F.normalize(torch.transpose(self.fc_0_lin.weight, 0, 1), dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            pred_v = torch.mm(F.normalize(v, dim=1), F.normalize(torch.transpose(self.fc_1_lin.weight, 0, 1), dim=0))
            pred_a = pred_a * self.mmcosine_scaling
            pred_v = pred_v * self.mmcosine_scaling
            pred = pred_a + pred_v

            if self.args.bias_infusion.get("inst_norm", False):
                pred_a = self.inst_norm(pred_a)
                pred_v = self.inst_norm(pred_v)
        else:
            pred_a = torch.matmul(a, self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v, self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2

            if self.args.bias_infusion.get("inst_norm", False):
                pred_a = self.inst_norm(pred_a)
                pred_v = self.inst_norm(pred_v)

            pred = pred_a + pred_v
        if self.cls_type != "linear":
            pred = self.common_fc(pred)
            pred_a = self.common_fc(pred_a)
            pred_v = self.common_fc(pred_v)


        return {"preds":{"combined":pred, "c":pred_a, "g":pred_v}, "features": {"c": a, "g": v, "combined": (a + v)/2}}

class ConcatClassifier_CREMAD_OGM_plus(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_plus, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        d_model = args.d_model
        self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)
        modalities = args.get("modalities", ["audio", "visual"])

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.net_mod0 = resnet18(modality=modalities[0])
        self.net_mod1 = resnet18(modality=modalities[1])

        if self.cls_type == "linear":
            self.fc_mod0_lin = nn.Linear(d_model, num_classes)
            self.fc_mod1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.inst_norm = nn.InstanceNorm1d(num_classes, track_running_stats=False)

        if self.cls_type == "highlynonlinear":
            self.fc_mod0_lin = nn.Linear(d_model, 4096)
            self.fc_mod1_lin = nn.Linear(d_model, 4096, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type != "linear":
            self.fc_mod0_lin = nn.Linear(d_model, fc_inner)
            self.fc_mod1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.inst_norm = nn.InstanceNorm1d(fc_inner, track_running_stats=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
            if not self.shared_pred:
                self.fc_0 = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, num_classes)
                )

                self.fc_1 = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, num_classes)
                )

    def _get_features(self, x):

        a = self.net_mod0(x[0].unsqueeze(dim=1))
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)

        v = self.net_mod1(x[1])
        #
        B = x[1].shape[0]
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)

        return a, v

    def forward(self, x, **kwargs):

        a, v = self._get_features(x)

        if self.mmcosine:
            pred_a = torch.mm(F.normalize(a, dim=1),F.normalize(torch.transpose(self.fc_0_lin.weight, 0, 1), dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            pred_v = torch.mm(F.normalize(v, dim=1), F.normalize(torch.transpose(self.fc_1_lin.weight, 0, 1), dim=0))
            pred_a = pred_a * self.mmcosine_scaling
            pred_v = pred_v * self.mmcosine_scaling
            pred = pred_a + pred_v

            if self.args.bias_infusion.get("inst_norm", False):
                pred_a = self.inst_norm(pred_a)
                pred_v = self.inst_norm(pred_v)
        else:
            pred_a = torch.matmul(a, self.fc_mod0_lin.weight.T) + self.fc_mod0_lin.bias / 2
            pred_v = torch.matmul(v, self.fc_mod1_lin.weight.T) + self.fc_mod0_lin.bias / 2

            if self.args.bias_infusion.get("inst_norm", False):
                pred_a = self.inst_norm(pred_a)
                pred_v = self.inst_norm(pred_v)

            pred = pred_a + pred_v
        if self.cls_type != "linear":
            pred = self.common_fc(pred)
            if self.shared_pred:
                pred_a = self.common_fc(pred_a)
                pred_v = self.common_fc(pred_v)
            else:
                pred_a = self.fc_0(pred_a)
                pred_v = self.fc_1(pred_v)

        return {"preds":{"combined":pred, "c":pred_a, "g":pred_v}, "features": {"c": a, "g": v, "combined": (a + v)/2}}

class ConcatClassifier_CREMAD_Faster(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_Faster, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        d_model = args.d_model
        self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1
        # self.fusion_module = ConcatFusion(output_dim=n_classes)

        # self.vcaster = nn.Conv2d(9,3,1)
        # self.acaster = nn.Conv2d(1,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False) #, weights='ResNet18_Weights.DEFAULT'
        # self.audio_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False)

        self.net_mod0 = resnet18(modality='audio')
        self.net_mod1 = resnet18(modality='visual')
        # self.vclassifier = nn.Linear(1000, n_classes)
        # self.aclassifier = nn.Linear(1000, n_classes)
        # self.classifier = nn.Linear(2000, n_classes)
        # self.classifier = nn.Linear(2000, n_classes)
        # self.classifier = nn.Sequential(
        #                 nn.Dropout(0.4),
                        # nn.Linear(2000, 64),
                        # nn.ReLU(),
                        # nn.Dropout(0.2),
                        # nn.Linear(64, n_classes)
                    # )


        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.inst_norm = nn.InstanceNorm1d(num_classes, track_running_stats=False)

        if self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
            import copy
            self.common_fc_0 = copy.deepcopy(self.common_fc)
            self.common_fc_1 = copy.deepcopy(self.common_fc)

        elif self.cls_type != "linear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.inst_norm = nn.InstanceNorm1d(fc_inner, track_running_stats=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
            import copy
            self.common_fc_0 = copy.deepcopy(self.common_fc)
            self.common_fc_1 = copy.deepcopy(self.common_fc)
            if not self.shared_pred:
                self.fc_mod0 = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, num_classes)
                )

                self.fc_mod1 = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, num_classes)
                )



    def _get_features(self, x):


        # print(v.shape)

        # vout = self.vclassifier(v)

        # B = x["spec"].shape[0]

        # a = self.audio_net(self.acaster(x[0].unsqueeze(dim=1)))
        # print(x[0].shape)
        a = self.net_mod0(x[0].unsqueeze(dim=1))
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)

        v = self.net_mod1(x[1])
        #
        # print(v.shape)
        B = x[1].shape[0]
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)

        pred_a = torch.matmul(a, self.fc_0_lin.weight.T) + self.fc_0_lin.bias/2
        pred_v = torch.matmul(v, self.fc_1_lin.weight.T) + self.fc_0_lin.bias/2

        return pred_a, pred_v, a, v

    def forward(self, x, **kwargs):

        pred_a, pred_v, a, v = self._get_features(x)

        if self.args.bias_infusion.get("inst_norm", False):
            pred_a = self.inst_norm(pred_a)
            pred_v = self.inst_norm(pred_v)

        if self.cls_type != "linear":
            if self.shared_pred:
                pred = self.common_fc(pred_a+pred_v)

                self.common_fc_0.load_state_dict(self.common_fc.state_dict())
                self.common_fc_1.load_state_dict(self.common_fc.state_dict())
                pred_a = self.common_fc_0(pred_a)
                pred_v = self.common_fc_1(pred_v)




        return {"preds":{"combined":pred, "c":pred_a, "g":pred_v}, "features": {"c": a, "g": v, "combined": (a + v)/2}}

class ConcatClassifier_CREMAD_OGM_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_pre, self).__init__()

        self.args = args
        # self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.cls_type = args.cls_type

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
        elif self.cls_type == "dec":
            pass

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )

        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, num_classes)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=512, dim=512, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")
    def _get_features(self, x):

        a = self.enc_0(x)
        v = self.enc_1(x)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x)

        if self.mmcosine:
            pred_a = torch.mm(F.normalize(a["features"]["combined"], dim=1),F.normalize(torch.transpose(self.fc_0_lin.weight, 0, 1), dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            pred_v = torch.mm(F.normalize(v["features"]["combined"], dim=1), F.normalize(torch.transpose(self.fc_1_lin.weight, 0, 1), dim=0))
            pred_a = pred_a * self.mmcosine_scaling
            pred_v = pred_v * self.mmcosine_scaling
            pred = pred_a + pred_v

        elif self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2

            pred = pred_a + pred_v
        else:
            pred = pred_aa + pred_vv

        if self.cls_type == "film" or self.cls_type == "gated":
            pred = self.common_fc([a["features"]["combined"], v["features"]["combined"]])
        elif self.cls_type == "tf":
            pred = self.common_fc([a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]])
        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)
        if (self.args.bias_infusion.method == "OGM" or self.args.bias_infusion.method == "OGM_GE" or self.args.bias_infusion.method == "MSLR") and self.cls_type!="dec":
            pred_aa = pred_a
            pred_vv = pred_v

        return {"preds":{"combined":pred,
                         "c":pred_aa,
                         "g":pred_vv
                         },
                "features": {"c": a["features"]["combined"],
                             "g": v["features"]["combined"]}}
class ConcatClassifier_CREMAD_OGM_Des_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_Des_pre, self).__init__()

        self.args = args
        # self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.cls_type = args.cls_type

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(512, num_classes)
            self.fc_1_lin = nn.Linear(512, num_classes, bias=False)
        elif self.cls_type == "dec":
            pass

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )

        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, num_classes)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=512, dim=512, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")

        self.des_0 = nn.Sequential(
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        self.des_1 = nn.Sequential(
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        # self.des_1 =  nn.Linear(512, num_classes)

    def _get_features(self, x):

        a = self.enc_0(x)
        v = self.enc_1(x)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):

        a_features = a["features"]["combined"]
        v_features = v["features"]["combined"]
        a_features_nonaggr = a["nonaggr_features"]["combined"]
        v_features_nonaggr = v["nonaggr_features"]["combined"]

        if "detach_a" in kwargs and kwargs["detach_a"]:
            a_features = a_features.detach()
            a_features_nonaggr = a_features_nonaggr.detach()
        if "detach_v" in kwargs and kwargs["detach_v"]:
            v_features = v_features.detach()
            v_features_nonaggr = v_features_nonaggr.detach()
        if self.mmcosine:
            pred_a = torch.mm(F.normalize(a_features, dim=1),F.normalize(torch.transpose(self.fc_0_lin.weight, 0, 1), dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            pred_v = torch.mm(F.normalize(v_features, dim=1), F.normalize(torch.transpose(self.fc_1_lin.weight, 0, 1), dim=0))
            pred_a = pred_a * self.mmcosine_scaling
            pred_v = pred_v * self.mmcosine_scaling
            pred = pred_a + pred_v

        elif self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a_features, self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v_features, self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2

            pred = pred_a + pred_v
        else:
            pred = pred_aa + pred_vv

        if self.cls_type == "film" or self.cls_type == "gated":
            pred = self.common_fc([a_features, v_features])
        elif self.cls_type == "tf":
            pred = self.common_fc([a_features_nonaggr, v_features_nonaggr])
        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)
        if (self.args.bias_infusion.method == "OGM" or self.args.bias_infusion.method == "OGM_GE" or self.args.bias_infusion.method == "MSLR") and self.cls_type!="dec":
            pred_aa = pred_a
            pred_vv = pred_v
        return pred, pred_aa, pred_vv

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x)

        pred, pred_aa, pred_vv = self._forward_main(a, v, pred_aa, pred_vv)
        # pred_deta, _, _ = self._forward_main(a, v, pred_aa, pred_vv, detach_a=True)
        # pred_detv, _, _ = self._forward_main(a, v, pred_aa, pred_vv, detach_v=True)

        pred_des_a = self.des_0(a["features"]["combined"].detach())
        pred_des_v = self.des_1(v["features"]["combined"].detach())

        return {"preds":{"combined":pred,
                         "c":pred_aa,
                         "g":pred_vv,
                        # "deta":pred_deta,
                        # "detv":pred_detv,
                        "des_a":pred_des_a,
                        "des_v":pred_des_v
                         },
                "features": {"c": a["features"]["combined"],
                             "g": v["features"]["combined"]}}

class Modality_out(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x

class ConcatClassifier_CREMAD_AGM_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_AGM_pre, self).__init__()

        self.args = args
        # self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
        elif self.cls_type == "dec":
            pass
        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )

        elif self.cls_type == "film":
            self.common_fc = FiLM(d_model, 512, num_classes)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=d_model, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=d_model, dim=512, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")

        self.m_v_o = Modality_out()
        self.m_a_o = Modality_out()

        self.scale_a = 1.0
        self.scale_v = 1.0

        self.m_a_o.register_full_backward_hook(self.hooka)
        self.m_v_o.register_full_backward_hook(self.hookv)

    def hooka(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_a,

    def hookv(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_v,

    def update_scale(self, coeff_a, coeff_v):
        self.scale_a = coeff_a
        self.scale_v = coeff_v

    def _get_features(self, x):

        a = self.enc_0(x)
        v = self.enc_1(x)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def _get_preds_padded(self, x, feat_a, feat_v, pred_aa, pred_vv, pad_audio = False, pad_visual = False):

        data = copy.deepcopy(x)
        if pad_audio:
            if 0 in data:
                data[0] = torch.zeros_like(data[0], device=x[0].device)
            elif 2 in data:
                data[2] = torch.zeros_like(data[2], device=x[2].device)
            a = self.enc_0(data)
            if self.cls_type == "dec":
                pred = a["preds"]["combined"] + pred_vv
            else:
                pred = self._forward_main(a, feat_v)

        if pad_visual:
            if 1 in data:
                data[1] = torch.zeros_like(data[1], device=x[1].device)
            elif 3 in data:
                data[3] = torch.zeros_like(data[3], device=x[3].device)
            v = self.enc_1(data)
            if self.cls_type == "dec":
                pred = pred_aa + v["preds"]["combined"]
            else:
                pred = self._forward_main(feat_a, v)

        return pred

    def _forward_main(self, a, v):

        if self.mmcosine:
            pred_a = torch.mm(F.normalize(a["features"]["combined"], dim=1),F.normalize(torch.transpose(self.fc_0_lin.weight, 0, 1), dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            pred_v = torch.mm(F.normalize(v["features"]["combined"], dim=1), F.normalize(torch.transpose(self.fc_1_lin.weight, 0, 1), dim=0))
            pred_a = pred_a * self.mmcosine_scaling
            pred_v = pred_v * self.mmcosine_scaling
            pred = pred_a + pred_v
        elif self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":
            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2
            pred = pred_a + pred_v

        if self.cls_type == "film" or self.cls_type == "gated":
            pred = self.common_fc([a["features"]["combined"], v["features"]["combined"]])
        elif self.cls_type == "tf":
            pred = self.common_fc([a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]])

        # if self.cls_type == "film" or self.cls_type == "gated" or self.cls_type == "tf":
        #     pred = self.common_fc([a, v])
        #     return pred
        elif self.cls_type != "linear":
            pred = self.common_fc(pred)

        return pred


    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x)

        training_mode = True if self.training else False
        self.eval()
        pred_za = self._get_preds_padded(x, feat_a=a, feat_v=v, pred_aa=pred_aa, pred_vv=pred_vv,pad_audio=True, pad_visual=False)
        pred_zv = self._get_preds_padded(x, feat_a=a, feat_v=v, pred_aa=pred_aa, pred_vv=pred_vv,pad_audio=False, pad_visual=True)
        if training_mode:
            self.train()

        if self.cls_type == "dec":
            pred = pred_aa + pred_vv
        else:
            pred = self._forward_main(a, v)

        pred_a = self.m_a_o(0.5*(pred - pred_za + pred_zv))
        pred_v = self.m_v_o(0.5*(pred - pred_zv + pred_za))


        return {"preds":{"combined":pred_a + pred_v,
                         "both": pred,
                         "c":pred_a,
                         "g":pred_v
                         },
                "features": {"c": a["features"]["combined"],
                             "g": v["features"]["combined"]}}


class ConditionalGMVAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_components, n_classes):
        super(ConditionalGMVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_components = n_components
        self.n_classes = n_classes

        # GMVAE Encoder
        self.gmvae_encoder_fc1 = nn.Linear(latent_dim + n_classes, hidden_dim)
        self.gmvae_encoder_means = nn.Linear(hidden_dim, n_components * latent_dim)
        self.gmvae_encoder_logvars = nn.Linear(hidden_dim, n_components * latent_dim)
        self.gmvae_encoder_weights = nn.Linear(hidden_dim, n_components)

        # GMVAE Decoder
        self.gmvae_decoder_fc1 = nn.Linear(latent_dim + n_classes, hidden_dim)
        self.gmvae_decoder_fc2 = nn.Linear(hidden_dim, latent_dim)

    def gmvae_encoder(self, z_initial, y):
        # Combine latent with class labels
        z_initial = torch.cat([z_initial, y], dim=1)
        h = F.relu(self.gmvae_encoder_fc1(z_initial))
        means = self.gmvae_encoder_means(h).view(-1, self.n_components, self.latent_dim)
        logvars = self.gmvae_encoder_logvars(h).view(-1, self.n_components, self.latent_dim)
        weights = F.softmax(self.gmvae_encoder_weights(h), dim=-1)
        return means, logvars, weights

    def gmvae_decoder(self, z_gmvae, y):
        # Combine latent with class labels
        z_gmvae = torch.cat([z_gmvae, y], dim=1)
        h = F.relu(self.gmvae_decoder_fc1(z_gmvae))
        z_initial_recon = self.gmvae_decoder_fc2(h)
        return z_initial_recon

    def sample_from_gmm(self, means, logvars, weights):
        categorical = Categorical(weights)
        component = categorical.sample()
        chosen_mean = means[range(len(means)), component]
        chosen_logvar = logvars[range(len(logvars)), component]
        std = torch.exp(0.5 * chosen_logvar)
        normal = Normal(chosen_mean, std)
        z = normal.rsample()
        return z

    def gmvae_loss(self, z_initial, z_initial_recon, means, logvars, weights):
        recon_loss = F.mse_loss(z_initial_recon, z_initial, reduction='sum')
        kl_div = 0.5 * torch.sum(weights * (torch.sum(logvars.exp() + means.pow(2) - 1 - logvars, dim=2)), dim=1).mean()
        loss = recon_loss + kl_div
        return loss, recon_loss, kl_div

    def forward(self, z_initial, y):
        means, logvars, weights = self.gmvae_encoder(z_initial, y)
        z_gmvae = self.sample_from_gmm(means, logvars, weights)
        z_initial_recon = self.gmvae_decoder(z_gmvae, y)
        return z_initial_recon, means, logvars, weights

    def sample_from_class(self, class_index, n_samples=1):
        # Ensure the class_index is within the range of components
        # assert 0 <= class_index < self.n_classes, "Invalid class index"

        # Create one-hot encoding for the class
        # class_one_hot = torch.zeros(n_samples, self.n_classes).to(next(self.parameters()).device)
        # class_one_hot[:, class_index] = 1

        # Sample from the Gaussian mixture in the latent space
        means = torch.zeros(n_samples*len(class_index), self.n_components, self.latent_dim).to(next(self.parameters()).device)
        logvars = torch.zeros(n_samples*len(class_index), self.n_components, self.latent_dim).to(next(self.parameters()).device)
        weights = torch.ones(n_samples*len(class_index), self.n_components).to(next(self.parameters()).device) / self.n_components

        z_gmvae = self.sample_from_gmm(means, logvars, weights)

        # Decode the sampled latent variables to generate new data points
        new_samples = self.gmvae_decoder(z_gmvae, class_index.repeat(n_samples,1))
        return new_samples


class TF_Fusion(nn.Module):
    def __init__(self, input_dim, dim, layers, output_dim):
        super(TF_Fusion, self).__init__()
        self.common_net = Conformer(
                            input_dim=input_dim,
                            encoder_dim=dim,
                            num_encoder_layers=layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)
        self.mod_0_token = nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)
        self.mod_1_token = nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)
        self.mod_2_token = nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)

        self.common_fc = nn.Linear(dim, output_dim)


    def forward(self, x, **kwargs):
        x_0 = x[0].permute(0,2,1)
        x_1 = x[1].permute(0,2,1)

        x_0 = self.mod_0_token.repeat(x_0.shape[0], x_0.shape[1], 1) + x_0
        x_1 = self.mod_1_token.repeat(x_1.shape[0], x_1.shape[1], 1) + x_1
        xlist = [x_0, x_1]
        if len(x)>2:
            x_2 = x[2].permute(0,2,1)
            x_2 = self.mod_2_token.repeat(x_2.shape[0], x_2.shape[1], 1) + x_2
            xlist.append(x_2)
        if "detach_a" in kwargs and kwargs["detach_a"]:
            xlist[0] = xlist[0].detach()
        if "detach_v" in kwargs and kwargs["detach_v"]:
            xlist[1] = xlist[1].detach()

        feat_mm = torch.concatenate([xi for xi in xlist], dim=1)
        feat_mm = torch.concatenate([self.cls_token.repeat(feat_mm.shape[0], 1, 1), feat_mm], dim=1)
        feat_mm = self.common_net(feat_mm)[:,0]
        pred = self.common_fc(feat_mm)
        return pred
class ConcatClassifier_CREMAD_OGM_ShuffleGrad_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_ShuffleGrad_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)



        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.batchnorm_features = args.get("batchnorm_features", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)


        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.count_trainingsteps = 0

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(d_model, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(d_model, track_running_stats=True)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(4096), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(4096, track_running_stats=True)


            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(fc_inner), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(fc_inner, track_running_stats=True)



            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, num_classes)
        elif self.cls_type == "filmv":
            self.common_fc = FiLM(512, 512, num_classes, x_film=False)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=512, dim=512, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")

    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, detach_pred=not self.shufflegradmulti, **kwargs)
        v = self.enc_1(x, detach_pred=not self.shufflegradmulti, **kwargs)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):


        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) #+ self.fc_0_lin.bias / 2
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_a = pred_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_v = pred_v.detach()

            if "skip_bias" in kwargs and kwargs["skip_bias"]:
                pass
            else:
                pred_v = pred_v + self.bias_lin/2
                pred_a = pred_a + self.bias_lin/2

            pred = pred_a + pred_v

            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "w_a": self.fc_0_lin.weight.norm(),
                           "w_v": self.fc_1_lin.weight.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

        else:
            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_aa = pred_aa.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_vv = pred_vv.detach()

            pred = pred_aa + pred_vv


        if self.cls_type == "film" or self.cls_type == "filmv" or self.cls_type == "gated":
            this_feat_a, this_feat_v = a["features"]["combined"], v["features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)
        elif self.cls_type == "tf":
            this_feat_a, this_feat_v = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)

        return pred, pred_aa, pred_vv

    def shuffle_ids(self, label):
        batch_size = label.size(0)
        shuffle_data = []
        if "rand" in self.args.bias_infusion.shuffle_type:
            while len(shuffle_data) < self.args.bias_infusion.num_samples:

                if self.args.bias_infusion.shuffle:
                    shuffle_idx = torch.randperm(batch_size)
                    if "rsl" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(label[shuffle_idx] == label)
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    elif "rsi" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(shuffle_idx == torch.arange(batch_size))
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    else:
                        nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                else:
                    nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                    shuffle_idx = torch.arange(batch_size)

                if nonequal_label.sum() <= 1:
                    continue
                shuffle_data.append({"shuffle_idx": shuffle_idx, "data": nonequal_label})
        elif "samelabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li == lj and i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "difflabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li != lj:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "alllabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        return shuffle_data

    def shuffle_data_fn(self, a, v, pred_aa, pred_vv, pred, label):

        shuffle_data = self.shuffle_ids(label)

        if len(shuffle_data) == 0 or shuffle_data[0]["shuffle_idx"].size(0) <= 1:
            return False, False, False, False, False, False, False, False, False, False

        sa = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        sv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        s_pred_aa = torch.concatenate([pred_aa[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_vv = torch.concatenate([pred_vv[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)

        na = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        nv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        n_pred_aa = torch.concatenate([pred_aa[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_vv = torch.concatenate([pred_vv[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_pred = torch.concatenate([pred[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_label = torch.concatenate([label[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        return sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x, **kwargs)

        pred, pred_aa, pred_vv = self._forward_main(a, v, pred_aa, pred_vv, **kwargs)

        output = {"preds":{"combined":pred,
                            "c":pred_aa,
                            "g":pred_vv
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"]}}

        if self.training:

            sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label = self.shuffle_data_fn( a, v, pred_aa, pred_vv, pred, kwargs["label"])
            if not sa:
                return output
            pred_dtv_sa, _, _ = self._forward_main(sa, nv, s_pred_aa, n_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
            pred_dta_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv, detach_a=True, notwandb=True, **kwargs)
            output["preds"]["sa_detv"] = pred_dtv_sa
            output["preds"]["sa_deta"] = pred_dta_sa

            pred_dtv_sv, _, _ = self._forward_main(na, sv, n_pred_aa, s_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
            pred_dta_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv, detach_a=True, notwandb=True, **kwargs)
            output["preds"]["sv_detv"] = pred_dtv_sv
            output["preds"]["sv_deta"] = pred_dta_sv

            pred_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv.detach(), notwandb=True, **kwargs)
            pred_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv.detach(), notwandb=True, **kwargs)
            output["preds"]["sv"] = pred_sv
            output["preds"]["sa"] = pred_sa

            output["preds"]["ncombined"] = n_pred

            output["preds"]["n_label"] = n_label

        return output
class ConcatClassifier_CREMAD_OGM_ShuffleGrad_IB_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_ShuffleGrad_IB_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)



        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.batchnorm_features = args.get("batchnorm_features", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)
        self.latent_dim = latent_dim


        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.fc_0_mu = nn.Linear(self.latent_dim, d_model, bias=False)
        self.fc_1_mu = nn.Linear(self.latent_dim, d_model, bias=False)
        self.fc_0_logvar = nn.Linear(self.latent_dim, d_model, bias=False)
        self.fc_1_logvar = nn.Linear(self.latent_dim, d_model, bias=False)

        self.count_trainingsteps = 0

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(d_model, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(d_model, track_running_stats=True)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(4096), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(4096, track_running_stats=True)


            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(fc_inner), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(fc_inner, track_running_stats=True)



            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, num_classes)
        elif self.cls_type == "filmv":
            self.common_fc = FiLM(512, 512, num_classes, x_film=False)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=512, dim=512, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")

    def IB_loss(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD
    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, detach_pred=not self.shufflegradmulti, **kwargs)
        v = self.enc_1(x, detach_pred=not self.shufflegradmulti, **kwargs)

        feat_a_mu = self.fc_0_mu(a["features"]["combined"])
        feat_a_logvar = self.fc_0_logvar(a["features"]["combined"])
        feat_v_mu = self.fc_1_mu(v["features"]["combined"])
        feat_v_logvar = self.fc_1_logvar(v["features"]["combined"])

        std = torch.exp(0.5 * feat_a_logvar)
        eps = torch.randn_like(std)
        z_a = feat_a_mu + eps * std

        std = torch.exp(0.5 * feat_v_logvar)
        eps = torch.randn_like(std)
        z_v = feat_v_mu + eps * std

        a["features"]["combined"] = z_a
        v["features"]["combined"] = z_v

        a["IB_loss"] = self.IB_loss(feat_a_mu, feat_a_logvar)
        v["IB_loss"] = self.IB_loss(feat_v_mu, feat_v_logvar)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):


        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) #+ self.fc_0_lin.bias / 2
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_a = pred_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_v = pred_v.detach()

            if "skip_bias" in kwargs and kwargs["skip_bias"]:
                pass
            else:
                pred_v = pred_v + self.bias_lin/2
                pred_a = pred_a + self.bias_lin/2

            pred = pred_a + pred_v

            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "w_a": self.fc_0_lin.weight.norm(),
                           "w_v": self.fc_1_lin.weight.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

        else:
            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_aa = pred_aa.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_vv = pred_vv.detach()

            pred = pred_aa + pred_vv


        if self.cls_type == "film" or self.cls_type == "filmv" or self.cls_type == "gated":
            this_feat_a, this_feat_v = a["features"]["combined"], v["features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)
        elif self.cls_type == "tf":
            this_feat_a, this_feat_v = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)

        return pred, pred_aa, pred_vv

    def shuffle_ids(self, label):
        batch_size = label.size(0)
        shuffle_data = []
        if "rand" in self.args.bias_infusion.shuffle_type:
            while len(shuffle_data) < self.args.bias_infusion.num_samples:

                if self.args.bias_infusion.shuffle:
                    shuffle_idx = torch.randperm(batch_size)
                    if "rsl" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(label[shuffle_idx] == label)
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    elif "rsi" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(shuffle_idx == torch.arange(batch_size))
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    else:
                        nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                else:
                    nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                    shuffle_idx = torch.arange(batch_size)

                if nonequal_label.sum() <= 1:
                    continue
                shuffle_data.append({"shuffle_idx": shuffle_idx, "data": nonequal_label})
        elif "samelabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li == lj and i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "difflabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li != lj:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "alllabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        return shuffle_data

    def shuffle_data_fn(self, a, v, pred_aa, pred_vv, pred, label):

        shuffle_data = self.shuffle_ids(label)

        if len(shuffle_data) == 0 or shuffle_data[0]["shuffle_idx"].size(0) <= 1:
            return False, False, False, False, False, False, False, False, False, False

        sa = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        sv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        s_pred_aa = torch.concatenate([pred_aa[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_vv = torch.concatenate([pred_vv[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)

        na = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        nv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        n_pred_aa = torch.concatenate([pred_aa[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_vv = torch.concatenate([pred_vv[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_pred = torch.concatenate([pred[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_label = torch.concatenate([label[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        return sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x, **kwargs)

        pred, pred_aa, pred_vv = self._forward_main(a, v, pred_aa, pred_vv, **kwargs)

        output = {"preds":{"combined":pred,
                            "c":pred_aa,
                            "g":pred_vv
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"]},
                    "losses": {"IB_0": a["IB_loss"], "IB_1": v["IB_loss"]}
                  }

        if self.training:

            sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label = self.shuffle_data_fn( a, v, pred_aa, pred_vv, pred, kwargs["label"])
            if not sa:
                return output
            pred_dtv_sa, _, _ = self._forward_main(sa, nv, s_pred_aa, n_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
            pred_dta_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv, detach_a=True, notwandb=True, **kwargs)
            output["preds"]["sa_detv"] = pred_dtv_sa
            output["preds"]["sa_deta"] = pred_dta_sa

            pred_dtv_sv, _, _ = self._forward_main(na, sv, n_pred_aa, s_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
            pred_dta_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv, detach_a=True, notwandb=True, **kwargs)
            output["preds"]["sv_detv"] = pred_dtv_sv
            output["preds"]["sv_deta"] = pred_dta_sv

            pred_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv.detach(), notwandb=True, **kwargs)
            pred_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv.detach(), notwandb=True, **kwargs)
            output["preds"]["sv"] = pred_sv
            output["preds"]["sa"] = pred_sa

            output["preds"]["ncombined"] = n_pred

            output["preds"]["n_label"] = n_label

        return output
class ConcatClassifier_CREMAD_OGM_ShuffleGradMP_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_ShuffleGradMP_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)



        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.batchnorm_features = args.get("batchnorm_features", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)


        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.count_trainingsteps = 0

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(d_model, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(d_model, track_running_stats=True)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(4096), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(4096, track_running_stats=True)


            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(fc_inner), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(fc_inner, track_running_stats=True)



            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, num_classes)
        elif self.cls_type == "filmv":
            self.common_fc = FiLM(512, 512, num_classes, x_film=False)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=512, dim=512, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")

    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, detach_pred=not self.shufflegradmulti, **kwargs)
        v = self.enc_1(x, detach_pred=not self.shufflegradmulti, **kwargs)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):


        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) #+ self.fc_0_lin.bias / 2
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_a = pred_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_v = pred_v.detach()

            if "skip_bias" in kwargs and kwargs["skip_bias"]:
                pass
            else:
                pred_v = pred_v + self.bias_lin/2
                pred_a = pred_a + self.bias_lin/2

            pred = pred_a + pred_v

            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "w_a": self.fc_0_lin.weight.norm(),
                           "w_v": self.fc_1_lin.weight.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

        else:
            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_aa = pred_aa.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_vv = pred_vv.detach()

            pred = pred_aa + pred_vv


        if self.cls_type == "film" or self.cls_type == "filmv" or self.cls_type == "gated":
            this_feat_a, this_feat_v = a["features"]["combined"], v["features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)
        elif self.cls_type == "tf":
            this_feat_a, this_feat_v = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)

        return pred, pred_aa, pred_vv

    def shuffle_ids(self, label):
        batch_size = label.size(0)
        shuffle_data = []
        if "rand" in self.args.bias_infusion.shuffle_type:
            while len(shuffle_data) < self.args.bias_infusion.num_samples:

                if self.args.bias_infusion.shuffle:
                    shuffle_idx = torch.randperm(batch_size)
                    if "rsl" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(label[shuffle_idx] == label)
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    elif "rsi" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(shuffle_idx == torch.arange(batch_size))
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    else:
                        nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                else:
                    nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                    shuffle_idx = torch.arange(batch_size)

                if nonequal_label.sum() <= 1:
                    continue
                shuffle_data.append({"shuffle_idx": shuffle_idx, "data": nonequal_label})
        elif "samelabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li == lj and i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "difflabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li != lj:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "alllabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        return shuffle_data

    def shuffle_data(self, x, pred, label):

        shuffle_data = self.shuffle_ids(label)

        x_sa = copy.deepcopy(x)
        x_sv = copy.deepcopy(x)
        ai = 0 if 0 in x_sa else 2
        vi = 1 if 1 in x_sa else 3
        x_sa[ai] = torch.concatenate([x_sa[ai][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        x_sa[vi] = torch.concatenate([x_sa[vi][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        x_sv[vi] = torch.concatenate([x_sv[vi][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        x_sv[ai] = torch.concatenate([x_sv[ai][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        sa, nv, s_pred_aa, n_pred_vv = self._get_features(x_sa)
        na, sv, n_pred_aa, s_pred_vv = self._get_features(x_sv)

        n_pred = torch.concatenate([pred[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_label = torch.concatenate([label[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        return sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x, **kwargs)

        pred, pred_aa, pred_vv = self._forward_main(a, v, pred_aa, pred_vv, **kwargs)

        output = {"preds":{"combined":pred,
                            "c":pred_aa,
                            "g":pred_vv
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"]}}

        if self.training:

            sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label = self.shuffle_data( x, pred, kwargs["label"])

            pred_dtv_sa, _, _ = self._forward_main(sa, nv, s_pred_aa, n_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
            pred_dta_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv, detach_a=True, notwandb=True, **kwargs)
            output["preds"]["sa_detv"] = pred_dtv_sa
            output["preds"]["sa_deta"] = pred_dta_sa

            pred_dtv_sv, _, _ = self._forward_main(na, sv, n_pred_aa, s_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
            pred_dta_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv, detach_a=True, notwandb=True, **kwargs)
            output["preds"]["sv_detv"] = pred_dtv_sv
            output["preds"]["sv_deta"] = pred_dta_sv

            pred_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv.detach(), notwandb=True, **kwargs)
            pred_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv.detach(), notwandb=True, **kwargs)
            output["preds"]["sv"] = pred_sv
            output["preds"]["sa"] = pred_sa

            output["preds"]["ncombined"] = n_pred

            output["preds"]["n_label"] = n_label

        return output
class ConcatClassifier_CREMAD_OGM_ShuffleGradEP_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_ShuffleGradEP_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)



        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.batchnorm_features = args.get("batchnorm_features", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)


        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.count_trainingsteps = 0

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(d_model, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(d_model, track_running_stats=True)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(4096), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(4096, track_running_stats=True)


            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(fc_inner), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(fc_inner, track_running_stats=True)



            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, num_classes)
        elif self.cls_type == "filmv":
            self.common_fc = FiLM(512, 512, num_classes, x_film=False)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=512, dim=512, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")

    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, detach_pred=not self.shufflegradmulti, **kwargs)
        v = self.enc_1(x, detach_pred=not self.shufflegradmulti, **kwargs)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):


        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) #+ self.fc_0_lin.bias / 2
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_a = pred_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_v = pred_v.detach()

            if "skip_bias" in kwargs and kwargs["skip_bias"]:
                pass
            else:
                pred_v = pred_v + self.bias_lin/2
                pred_a = pred_a + self.bias_lin/2

            pred = pred_a + pred_v

            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "w_a": self.fc_0_lin.weight.norm(),
                           "w_v": self.fc_1_lin.weight.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

        else:
            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_aa = pred_aa.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_vv = pred_vv.detach()

            pred = pred_aa + pred_vv


        if self.cls_type == "film" or self.cls_type == "filmv" or self.cls_type == "gated":
            this_feat_a, this_feat_v = a["features"]["combined"], v["features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)
        elif self.cls_type == "tf":
            this_feat_a, this_feat_v = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)

        return pred, pred_aa, pred_vv

    def shuffle_ids(self, label):
        batch_size = label.size(0)
        shuffle_data = []
        random_shuffling = True
        if "rand" in self.args.bias_infusion.shuffle_type:
            while len(shuffle_data) < self.args.bias_infusion.num_samples:

                if self.args.bias_infusion.shuffle:
                    shuffle_idx = torch.randperm(batch_size)
                    if "rsl" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(label[shuffle_idx] == label)
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    elif "rsi" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(shuffle_idx == torch.arange(batch_size))
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    else:
                        nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                else:
                    nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                    shuffle_idx = torch.arange(batch_size)

                if nonequal_label.sum() <= 1:
                    continue
                shuffle_data.append({"shuffle_idx": shuffle_idx, "data": nonequal_label})
        elif "samelabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li == lj and i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "difflabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li != lj:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "alllabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        return shuffle_data

    def shuffle_data(self, x, pred, label):
        if not self.args.bias_infusion.get("training_mode", False):
            self.eval()

        a, v, pred_aa, pred_vv = self._get_features(x)

        shuffle_data = self.shuffle_ids(label)

        sa = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        sv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        s_pred_aa = torch.concatenate([pred_aa[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_vv = torch.concatenate([pred_vv[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)

        na = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        nv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        n_pred_aa = torch.concatenate([pred_aa[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_vv = torch.concatenate([pred_vv[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_pred = torch.concatenate([pred[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_label_shuffled = torch.concatenate([label[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        n_label = torch.concatenate([label[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        if not self.args.bias_infusion.get("training_mode", False):
            self.train()

        return sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label, n_label_shuffled

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x, **kwargs)

        pred, pred_aa, pred_vv = self._forward_main(a, v, pred_aa, pred_vv, **kwargs)

        output = {"preds":{"combined":pred,
                            "c":pred_aa,
                            "g":pred_vv
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"]}}

        if self.training:

            sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label, n_label_shuffled = self.shuffle_data( x, pred, kwargs["label"])

            pred_dtv_sa, _, _ = self._forward_main(sa, nv, s_pred_aa, n_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
            pred_dta_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv, detach_a=True, notwandb=True, **kwargs)
            output["preds"]["sa_detv"] = pred_dtv_sa
            output["preds"]["sa_deta"] = pred_dta_sa

            pred_dtv_sv, _, _ = self._forward_main(na, sv, n_pred_aa, s_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
            pred_dta_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv, detach_a=True, notwandb=True, **kwargs)
            output["preds"]["sv_detv"] = pred_dtv_sv
            output["preds"]["sv_deta"] = pred_dta_sv

            pred_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv.detach(), notwandb=True, **kwargs)
            pred_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv.detach(), notwandb=True, **kwargs)
            output["preds"]["sv"] = pred_sv
            output["preds"]["sa"] = pred_sa

            output["preds"]["ncombined"] = n_pred

            output["preds"]["n_label"] = n_label
            output["preds"]["n_label_shuffled"] = n_label_shuffled

        return output
class ConcatClassifier_CREMAD_OGM_ShuffleGradEPIB_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_ShuffleGradEPIB_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)



        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.batchnorm_features = args.get("batchnorm_features", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)


        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.count_trainingsteps = 0

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(d_model, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(d_model, track_running_stats=True)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(4096), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(4096, track_running_stats=True)


            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(fc_inner), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(fc_inner, track_running_stats=True)



            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, num_classes)
        elif self.cls_type == "filmv":
            self.common_fc = FiLM(512, 512, num_classes, x_film=False)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=512, dim=512, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")

        self.fc_yz = nn.Sequential(
            nn.Linear(num_classes, d_model, bias=False),
            nn.ReLU(),
            nn.Linear(d_model, d_model*2, bias=False),
        )

    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, detach_pred=not self.shufflegradmulti, **kwargs)
        v = self.enc_1(x, detach_pred=not self.shufflegradmulti, **kwargs)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):


        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) #+ self.fc_0_lin.bias / 2
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_a = pred_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_v = pred_v.detach()

            if "skip_bias" in kwargs and kwargs["skip_bias"]:
                pass
            else:
                pred_v = pred_v + self.bias_lin/2
                pred_a = pred_a + self.bias_lin/2

            pred = pred_a + pred_v

            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "w_a": self.fc_0_lin.weight.norm(),
                           "w_v": self.fc_1_lin.weight.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

        else:
            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_aa = pred_aa.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_vv = pred_vv.detach()

            pred = pred_aa + pred_vv


        if self.cls_type == "film" or self.cls_type == "filmv" or self.cls_type == "gated":
            this_feat_a, this_feat_v = a["features"]["combined"], v["features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)
        elif self.cls_type == "tf":
            this_feat_a, this_feat_v = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)

        return pred, pred_aa, pred_vv

    def shuffle_ids(self, label):
        batch_size = label.size(0)
        shuffle_data = []
        random_shuffling = True
        if "rand" in self.args.bias_infusion.shuffle_type:
            while len(shuffle_data) < self.args.bias_infusion.num_samples:

                if self.args.bias_infusion.shuffle:
                    shuffle_idx = torch.randperm(batch_size)
                    if "rsl" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(label[shuffle_idx] == label)
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    elif "rsi" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(shuffle_idx == torch.arange(batch_size))
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    else:
                        nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                else:
                    nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                    shuffle_idx = torch.arange(batch_size)

                if nonequal_label.sum() <= 1:
                    continue
                shuffle_data.append({"shuffle_idx": shuffle_idx, "data": nonequal_label})
        elif "samelabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li == lj and i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "difflabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li != lj:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "alllabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        return shuffle_data

    def shuffle_data(self, x, pred, label):
        if not self.args.bias_infusion.get("training_mode", False):
            self.eval()

        a, v, pred_aa, pred_vv = self._get_features(x)

        shuffle_data = self.shuffle_ids(label)

        sa = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        sv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        s_pred_aa = torch.concatenate([pred_aa[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_vv = torch.concatenate([pred_vv[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)

        na = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        nv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        n_pred_aa = torch.concatenate([pred_aa[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_vv = torch.concatenate([pred_vv[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_pred = torch.concatenate([pred[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_label_shuffled = torch.concatenate([label[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        n_label = torch.concatenate([label[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        if not self.args.bias_infusion.get("training_mode", False):
            self.train()

        return sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label, n_label_shuffled

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x, **kwargs)

        pred, pred_aa, pred_vv = self._forward_main(a, v, pred_aa, pred_vv, **kwargs)


        output = {"preds":{"combined":pred,
                            "c":pred_aa,
                            "g":pred_vv
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"]}}

        if self.training:
            if self.training:
                # one hot encoding of the labels
                # one_hot_labels = torch.nn.functional.one_hot(kwargs["label"], self.args.num_classes).float()
                pred_feat = self.fc_yz(pred.detach())
                combined_features = torch.cat([a["features"]["combined"], v["features"]["combined"]], dim=1)
                CMI_yz_Loss = torch.nn.MSELoss()(combined_features, pred_feat) * self.args.bias_infusion.get("lib", 0)

                output["losses"] = {"CMI_yz_Loss": CMI_yz_Loss}#, "CMI_yz_reg_Loss": a["CMI_yz_reg_Loss"]}

            sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label, n_label_shuffled = self.shuffle_data( x, pred, kwargs["label"])

            pred_dtv_sa, _, _ = self._forward_main(sa, nv, s_pred_aa, n_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
            pred_dta_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv, detach_a=True, notwandb=True, **kwargs)
            output["preds"]["sa_detv"] = pred_dtv_sa
            output["preds"]["sa_deta"] = pred_dta_sa

            pred_dtv_sv, _, _ = self._forward_main(na, sv, n_pred_aa, s_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
            pred_dta_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv, detach_a=True, notwandb=True, **kwargs)
            output["preds"]["sv_detv"] = pred_dtv_sv
            output["preds"]["sv_deta"] = pred_dta_sv

            pred_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv.detach(), notwandb=True, **kwargs)
            pred_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv.detach(), notwandb=True, **kwargs)
            output["preds"]["sv"] = pred_sv
            output["preds"]["sa"] = pred_sa

            output["preds"]["ncombined"] = n_pred

            output["preds"]["n_label"] = n_label
            output["preds"]["n_label_shuffled"] = n_label_shuffled

        return output
class ConcatClassifier_CREMAD_OGM_ShuffleGradEP_IB_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_ShuffleGradEP_IB_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)



        self.num_classes = args.num_classes
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.batchnorm_features = args.get("batchnorm_features", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)
        # self.latent_dim = args.get("latent_dim", 128)
        # d_model = self.latent_dim

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.count_trainingsteps = 0

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(d_model, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(d_model, track_running_stats=True)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(4096), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(4096, track_running_stats=True)


            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(fc_inner), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(fc_inner, track_running_stats=True)



            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, num_classes)
        elif self.cls_type == "filmv":
            self.common_fc = FiLM(512, 512, num_classes, x_film=False)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=512, dim=512, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")

        # self.fc_yz = nn.Linear(num_classes, d_model*2, bias=False)
        # self.fc_yz = nn.Sequential(
        #     nn.Linear(num_classes, d_model, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(d_model, d_model*2, bias=False),
        # )
        # self.fc_1_mu = nn.Linear(d_model,  self.latent_dim, bias=False)
        # self.fc_0_logvar = nn.Linear(d_model, self.latent_dim, bias=False)
        # self.fc_1_logvar = nn.Linear(d_model, self.latent_dim, bias=False)


    # def IB_loss(self, mu, logvar):
    #     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #     return KLD * self.args.bias_infusion.lib
    #
    # def mutual_info_penalty(self, mu1, logvar1, mu2, logvar2, y, num_classes, beta=0.0):
    #     # Approximate the conditional mutual information penalty
    #     logvar1 = torch.clamp(logvar1, -10, 10)
    #     logvar2 = torch.clamp(logvar2, -10, 10)
    #
    #     y_one_hot = nn.functional.one_hot(y, num_classes).float()
    #
    #     # Conditional mean of mu1 and mu2 given y
    #     cond_mean_mu1 = torch.matmul(y_one_hot.t(), mu1) / (y_one_hot.sum(dim=0).unsqueeze(1) + 1e-10)
    #     cond_mean_mu2 = torch.matmul(y_one_hot.t(), mu2) / (y_one_hot.sum(dim=0).unsqueeze(1) + 1e-10)
    #
    #     # Conditional variance of z1 and z2 given y (exp(logvar))
    #     cond_var1 = torch.matmul(y_one_hot.t(), torch.exp(logvar1)) / (y_one_hot.sum(dim=0).unsqueeze(1) + 1e-10)
    #     cond_var2 = torch.matmul(y_one_hot.t(), torch.exp(logvar2)) / (y_one_hot.sum(dim=0).unsqueeze(1) + 1e-10)
    #
    #     # Compute the conditional mutual information penalty
    #     mean_penalty = torch.mean((cond_mean_mu1 - cond_mean_mu2).pow(2))
    #     var_penalty = torch.mean((cond_var1 - cond_var2).pow(2))
    #
    #     return (mean_penalty + beta * var_penalty) * self.args.bias_infusion.lib
    #
    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, detach_pred=not self.shufflegradmulti, **kwargs)
        v = self.enc_1(x, detach_pred=not self.shufflegradmulti, **kwargs)

        # feat_a = torch.nn.functional.relu(a["features"]["combined"])
        # feat_v = torch.nn.functional.relu(v["features"]["combined"])
        #
        # feat_a_mu = self.fc_0_mu(feat_a)
        # feat_a_logvar = self.fc_0_logvar(feat_a)
        # feat_v_mu = self.fc_1_mu(feat_v)
        # feat_v_logvar = self.fc_1_logvar(feat_v)
        #
        # a["features"]["combined"] = self.reparameterize(feat_a_mu, feat_a_logvar)
        # v["features"]["combined"] = self.reparameterize(feat_v_mu, feat_v_logvar)
        #
        # # a["IB_loss"] = self.IB_loss(feat_a_mu, feat_a_logvar)
        # # v["IB_loss"] = self.IB_loss(feat_v_mu, feat_v_logvar)
        #
        # if "label" in kwargs:
        #     a["CMI_Loss"] = self.mutual_info_penalty(feat_a_mu, feat_a_logvar, feat_v_mu, feat_v_logvar, kwargs["label"], self.num_classes)
        # if self.training:
        #     #one hot encoding of the labels
        #     one_hot_labels = torch.nn.functional.one_hot(kwargs["label"], self.num_classes).float()
        #     pred_feat = self.fc_yz(one_hot_labels.detach())
        #     combined_features = torch.cat([a["features"]["combined"], v["features"]["combined"]], dim=1)
        #     a["CMI_yz_Loss"] = torch.nn.MSELoss()(combined_features, pred_feat) * self.args.bias_infusion.get("lib",0)
        #     # a["CMI_yz_reg_Loss"] = torch.nn.MSELoss()(combined_features, pred_feat.detach()) * self.args.bias_infusion.get("lib",0)


        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):


        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) #+ self.fc_0_lin.bias / 2
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_a = pred_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_v = pred_v.detach()

            if "skip_bias" in kwargs and kwargs["skip_bias"]:
                pass
            else:
                pred_v = pred_v + self.bias_lin/2
                pred_a = pred_a + self.bias_lin/2

            pred = pred_a + pred_v

            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "w_a": self.fc_0_lin.weight.norm(),
                           "w_v": self.fc_1_lin.weight.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

        else:
            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_aa = pred_aa.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_vv = pred_vv.detach()

            pred = pred_aa + pred_vv


        if self.cls_type == "film" or self.cls_type == "filmv" or self.cls_type == "gated":
            this_feat_a, this_feat_v = a["features"]["combined"], v["features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)
        elif self.cls_type == "tf":
            this_feat_a, this_feat_v = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)

        return pred, pred_aa, pred_vv

    def shuffle_ids(self, label):
        batch_size = label.size(0)
        shuffle_data = []
        random_shuffling = True
        if "rand" in self.args.bias_infusion.shuffle_type:
            while len(shuffle_data) < self.args.bias_infusion.num_samples:

                if self.args.bias_infusion.shuffle:
                    shuffle_idx = torch.randperm(batch_size)
                    if "rsl" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(label[shuffle_idx] == label)
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    elif "rsi" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(shuffle_idx == torch.arange(batch_size))
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    else:
                        nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                else:
                    nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                    shuffle_idx = torch.arange(batch_size)

                if nonequal_label.sum() <= 1:
                    continue
                shuffle_data.append({"shuffle_idx": shuffle_idx, "data": nonequal_label})
        elif "samelabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li == lj and i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "difflabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li != lj:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "alllabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        return shuffle_data

    def shuffle_data(self, x, pred, label):
        if not self.args.bias_infusion.get("training_mode", False):
            self.eval()

        a, v, pred_aa, pred_vv = self._get_features(x)

        shuffle_data = self.shuffle_ids(label)

        sa = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        sv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        s_pred_aa = torch.concatenate([pred_aa[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_vv = torch.concatenate([pred_vv[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)

        na = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        nv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        n_pred_aa = torch.concatenate([pred_aa[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_vv = torch.concatenate([pred_vv[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_pred = torch.concatenate([pred[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_label_shuffled = torch.concatenate([label[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        n_label = torch.concatenate([label[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        if not self.args.bias_infusion.get("training_mode", False):
            self.train()

        return sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label, n_label_shuffled

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x, **kwargs)

        pred, pred_aa, pred_vv = self._forward_main(a, v, pred_aa, pred_vv, **kwargs)

        output = {  "preds":{"combined":pred,
                            "c":pred_aa,
                            "g":pred_vv
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"]}
                  }

        if self.training:

            # output["losses"] = {"CMI_yz_Loss": a["CMI_yz_Loss"]}#, "CMI_yz_reg_Loss": a["CMI_yz_reg_Loss"]}

            sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label, n_label_shuffled = self.shuffle_data( x, pred, kwargs["label"])

            pred_dtv_sa, _, _ = self._forward_main(sa, nv, s_pred_aa, n_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
            pred_dta_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv, detach_a=True, notwandb=True, **kwargs)
            output["preds"]["sa_detv"] = pred_dtv_sa
            output["preds"]["sa_deta"] = pred_dta_sa

            pred_dtv_sv, _, _ = self._forward_main(na, sv, n_pred_aa, s_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
            pred_dta_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv, detach_a=True, notwandb=True, **kwargs)
            output["preds"]["sv_detv"] = pred_dtv_sv
            output["preds"]["sv_deta"] = pred_dta_sv

            pred_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv.detach(), notwandb=True, **kwargs)
            pred_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv.detach(), notwandb=True, **kwargs)
            output["preds"]["sv"] = pred_sv
            output["preds"]["sa"] = pred_sa

            output["preds"]["ncombined"] = n_pred

            output["preds"]["n_label"] = n_label
            output["preds"]["n_label_shuffled"] = n_label_shuffled

        return output

class ConcatClassifier_CREMAD_OGM_ShuffleGrad3d_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_ShuffleGrad3d_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.batchnorm_features = args.get("batchnorm_features", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)


        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.enc_2 = encs[2]

        self.count_trainingsteps = 0

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_2_lin = nn.Linear(d_model, num_classes, bias=False)
            self.cls_bias = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

            # if self.batchnorm_features:
            #     self.bn_0 = nn.BatchNorm1d(num_classes, track_running_stats=True)
            #     self.bn_1 = nn.BatchNorm1d(num_classes, track_running_stats=True)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_2_lin = nn.Linear(d_model, 4096, bias=False)
            self.cls_bias = nn.Parameter(torch.zeros(4096), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_2 = nn.BatchNorm1d(4096, track_running_stats=True)


            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_2_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.cls_bias = nn.Parameter(torch.zeros(fc_inner), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_2 = nn.BatchNorm1d(fc_inner, track_running_stats=True)



            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=d_model, dim=d_model, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")

    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, detach_pred=not self.shufflegradmulti, **kwargs)
        v = self.enc_1(x, detach_pred=not self.shufflegradmulti, **kwargs)
        f = self.enc_2(x, detach_pred=not self.shufflegradmulti, **kwargs)

        return a, v, f, a["preds"]["combined"], v["preds"]["combined"], f["preds"]["combined"]

    def shuffle_ids(self, label):
        batch_size = label.size(0)
        shuffle_data = []
        while len(shuffle_data) < self.args.bias_infusion.num_samples:

            if self.args.bias_infusion.shuffle:
                shuffle_idx_0 = torch.randperm(batch_size)
                shuffle_idx_1 = torch.randperm(batch_size)
                shuffle_idx_2 = torch.randperm(batch_size)
                nonequal_label = ~((label[shuffle_idx_0] == label[shuffle_idx_1]) & (label[shuffle_idx_1] == label[shuffle_idx_2]))
                if nonequal_label.sum() <= 1:
                    continue
                shuffle_idx_0 = shuffle_idx_0[nonequal_label.cpu()]
                shuffle_idx_1 = shuffle_idx_1[nonequal_label.cpu()]
                shuffle_idx_2 = shuffle_idx_2[nonequal_label.cpu()]
            else:
                nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                shuffle_idx_0 = torch.arange(batch_size)
                shuffle_idx_1 = torch.arange(batch_size)
                shuffle_idx_2 = torch.arange(batch_size)

            if nonequal_label.sum() <= 1:
                continue
            shuffle_data.append({"shuffle_idx_0": shuffle_idx_0,
                                 "shuffle_idx_1": shuffle_idx_1,
                                 "shuffle_idx_2": shuffle_idx_2,
                                 "data": nonequal_label})

        return shuffle_data

    def shuffle_data(self, c, g, f, pred_c, pred_g, pred_f, pred_mm, label):

        shuffle_data = self.shuffle_ids(label)

        sc = {feat: {"combined": torch.concatenate([c[feat]["combined"][sh_data_i["shuffle_idx_0"]] for sh_data_i in shuffle_data], dim=0)} for feat in c}
        sg = {feat: {"combined": torch.concatenate([g[feat]["combined"][sh_data_i["shuffle_idx_1"]] for sh_data_i in shuffle_data], dim=0)} for feat in g}
        sf = {feat: {"combined": torch.concatenate([f[feat]["combined"][sh_data_i["shuffle_idx_2"]] for sh_data_i in shuffle_data], dim=0)} for feat in f}

        nc = {feat: {"combined": torch.concatenate([c[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)} for feat in c}
        ng = {feat: {"combined": torch.concatenate([g[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)} for feat in g}
        nf = {feat: {"combined": torch.concatenate([f[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)} for feat in f}

        s_pred_c = torch.concatenate([pred_c[sh_data_i["shuffle_idx_0"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_g = torch.concatenate([pred_g[sh_data_i["shuffle_idx_1"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_f = torch.concatenate([pred_f[sh_data_i["shuffle_idx_2"]] for sh_data_i in shuffle_data], dim=0)

        n_pred_c = torch.concatenate([pred_c[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_g = torch.concatenate([pred_g[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_f = torch.concatenate([pred_f[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_pred = torch.concatenate([pred_mm[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_label = torch.concatenate([label[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        return sc, sg, sf, s_pred_c, s_pred_g, s_pred_f, nc, ng, nf, n_pred_c, n_pred_g, n_pred_f, n_pred, n_label
    def _forward_main(self, a, v, f, pred_aa, pred_vv, pred_ff, **kwargs):


        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_f = torch.matmul(f["features"]["combined"], self.fc_2_lin.weight.T) #+ self.fc_0_lin.bias / 2
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_a = pred_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_v = pred_v.detach()
            if "detach_f" in kwargs and kwargs["detach_f"]:
                pred_f = pred_f.detach()

            if "skip_bias" in kwargs and kwargs["skip_bias"]:
                pass
            else:
                pred_v = pred_v
                pred_a = pred_a
                pred_f = pred_f

            pred = pred_a + pred_v + pred_f + self.cls_bias

            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                            "wf_f": pred_f.norm(),
                           "w_a": self.fc_0_lin.weight.norm(),
                           "w_v": self.fc_1_lin.weight.norm(),
                            "w_f": self.fc_2_lin.weight.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm(),
                            "f_f": f["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

        else:
            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                            "wf_f": pred_ff.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm(),
                            "f_f": f["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
                pred_ff = (pred_ff - pred_ff.mean()) / pred_ff.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
                pred_ff = F.softmax(pred_ff, dim=1)
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_aa = pred_aa.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_vv = pred_vv.detach()
            if "detach_f" in kwargs and kwargs["detach_f"]:
                pred_ff = pred_ff.detach()

            pred = pred_aa + pred_vv + pred_ff


        if self.cls_type == "film" or self.cls_type == "filmv" or self.cls_type == "gated":
            this_feat_a, this_feat_v, this_feat_f = a["features"]["combined"], v["features"]["combined"], f["features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            if "detach_f" in kwargs and kwargs["detach_f"]:
                this_feat_f = this_feat_f.detach()
            pred = self.common_fc([this_feat_a, this_feat_v, this_feat_f], **kwargs)
        elif self.cls_type == "tf":
            this_feat_a, this_feat_v, this_feat_f = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"], f["nonaggr_features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            if "detach_f" in kwargs and kwargs["detach_f"]:
                this_feat_f = this_feat_f.detach()
            pred = self.common_fc([this_feat_a, this_feat_v, this_feat_f], **kwargs)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)

        return pred, pred_aa, pred_vv, pred_ff

    def forward(self, x, **kwargs):

        c, g, f, pred_c, pred_g, pred_f = self._get_features(x, **kwargs)

        pred, pred_aa, pred_vv, pred_ff = self._forward_main(c, g, f, pred_c, pred_g, pred_f, **kwargs)

        output = {"preds":{"combined":pred,
                            "c":pred_aa,
                            "g":pred_vv,
                           "f":pred_ff
                            },
                    "features": {"c": c["features"]["combined"],
                                "g": g["features"]["combined"],
                                 "f": f["features"]["combined"]}}

        if self.training:
            sc, sg, sf, s_pred_c, s_pred_g, s_pred_f, nc, ng, nf, n_pred_c, n_pred_g, n_pred_f, n_pred, n_label = self.shuffle_data( c, g, f, pred_c, pred_g, pred_f, pred, kwargs["label"])

            kwargs["notwandb"] = True
            output["preds"]["sc_detc"], _, _, _ = self._forward_main(sc, ng, nf, s_pred_c, n_pred_g, n_pred_f, detach_c=True, **kwargs)
            output["preds"]["sg_detc"], _, _, _ = self._forward_main(nc, sg, nf, n_pred_c, s_pred_g, n_pred_f, detach_c=True, **kwargs)
            output["preds"]["sf_detc"], _, _, _ = self._forward_main(nc, ng, sf, n_pred_c, n_pred_g, s_pred_f, detach_c=True, **kwargs)

            output["preds"]["sc_detg"], _, _, _ = self._forward_main(sc, ng, nf, s_pred_c, n_pred_g, n_pred_f, detach_g=True, **kwargs)
            output["preds"]["sg_detg"], _, _, _ = self._forward_main(nc, sg, nf, n_pred_c, s_pred_g, n_pred_f, detach_g=True, **kwargs)
            output["preds"]["sf_detg"], _, _, _ = self._forward_main(nc, ng, sf, n_pred_c, n_pred_g, s_pred_f, detach_g=True, **kwargs)

            output["preds"]["sc_detf"], _, _, _ = self._forward_main(sc, ng, nf, s_pred_c, n_pred_g, n_pred_f, detach_f=True, **kwargs)
            output["preds"]["sg_detf"], _, _, _ = self._forward_main(nc, sg, nf, n_pred_c, s_pred_g, n_pred_f, detach_f=True, **kwargs)
            output["preds"]["sf_detf"], _, _, _ = self._forward_main(nc, ng, sf, n_pred_c, n_pred_g, s_pred_f, detach_f=True, **kwargs)

            output["preds"]["sc_detgf"], _, _, _ = self._forward_main(sc, ng, nf, s_pred_c, n_pred_g, n_pred_f, detach_g=True, detach_f=True, **kwargs)
            output["preds"]["sg_detgf"], _, _, _ = self._forward_main(nc, sg, nf, n_pred_c, s_pred_g, n_pred_f, detach_g=True, detach_f=True, **kwargs)
            output["preds"]["sf_detgf"], _, _, _ = self._forward_main(nc, ng, sf, n_pred_c, n_pred_g, s_pred_f, detach_g=True, detach_f=True, **kwargs)

            output["preds"]["sc_detcf"], _, _, _ = self._forward_main(sc, ng, nf, s_pred_c, n_pred_g, n_pred_f, detach_c=True, detach_f=True, **kwargs)
            output["preds"]["sg_detcf"], _, _, _ = self._forward_main(nc, sg, nf, n_pred_c, s_pred_g, n_pred_f, detach_c=True, detach_f=True, **kwargs)
            output["preds"]["sf_detcf"], _, _, _ = self._forward_main(nc, ng, sf, n_pred_c, n_pred_g, s_pred_f, detach_c=True, detach_f=True, **kwargs)

            output["preds"]["sc_detcg"], _, _, _ = self._forward_main(sc, ng, nf, s_pred_c, n_pred_g, n_pred_f, detach_c=True, detach_g=True, **kwargs)
            output["preds"]["sg_detcg"], _, _, _ = self._forward_main(nc, sg, nf, n_pred_c, s_pred_g, n_pred_f, detach_c=True, detach_g=True, **kwargs)
            output["preds"]["sf_detcg"], _, _, _ = self._forward_main(nc, ng, sf, n_pred_c, n_pred_g, s_pred_f, detach_c=True, detach_g=True, **kwargs)

            output["preds"]["scf_detc"], _, _, _ = self._forward_main(sc, ng, sf, s_pred_c, n_pred_g, s_pred_f, detach_c=True, **kwargs)
            output["preds"]["sgf_detc"], _, _, _ = self._forward_main(nc, sg, sf, n_pred_c, s_pred_g, s_pred_f, detach_c=True, **kwargs)
            output["preds"]["scg_detc"], _, _, _ = self._forward_main(sc, sg, nf, s_pred_c, s_pred_g, n_pred_f, detach_c=True, **kwargs)

            output["preds"]["scf_detg"], _, _, _ = self._forward_main(sc, ng, sf, s_pred_c, n_pred_g, s_pred_f, detach_g=True, **kwargs)
            output["preds"]["sgf_detg"], _, _, _ = self._forward_main(nc, sg, sf, n_pred_c, s_pred_g, s_pred_f, detach_g=True, **kwargs)
            output["preds"]["scg_detg"], _, _, _ = self._forward_main(sc, sg, nf, s_pred_c, s_pred_g, n_pred_f, detach_g=True, **kwargs)

            output["preds"]["scf_detf"], _, _, _ = self._forward_main(sc, ng, sf, s_pred_c, n_pred_g, s_pred_f, detach_f=True, **kwargs)
            output["preds"]["sgf_detf"], _, _, _ = self._forward_main(nc, sg, sf, n_pred_c, s_pred_g, s_pred_f, detach_f=True, **kwargs)
            output["preds"]["scg_detf"], _, _, _ = self._forward_main(sc, sg, nf, s_pred_c, s_pred_g, n_pred_f, detach_f=True, **kwargs)

            output["preds"]["scf_detgf"], _, _, _ = self._forward_main(sc, ng, sf, s_pred_c, n_pred_g, s_pred_f, detach_g=True, detach_f=True, **kwargs)
            output["preds"]["sgf_detgf"], _, _, _ = self._forward_main(nc, sg, sf, n_pred_c, s_pred_g, s_pred_f, detach_g=True, detach_f=True, **kwargs)
            output["preds"]["scg_detgf"], _, _, _ = self._forward_main(sc, sg, nf, s_pred_c, s_pred_g, n_pred_f, detach_g=True, detach_f=True, **kwargs)

            output["preds"]["scf_detcf"], _, _, _ = self._forward_main(sc, ng, sf, s_pred_c, n_pred_g, s_pred_f, detach_c=True, detach_f=True, **kwargs)
            output["preds"]["sgf_detcf"], _, _, _ = self._forward_main(nc, sg, sf, n_pred_c, s_pred_g, s_pred_f, detach_c=True, detach_f=True, **kwargs)
            output["preds"]["scg_detcf"], _, _, _ = self._forward_main(sc, sg, nf, s_pred_c, s_pred_g, n_pred_f, detach_c=True, detach_f=True, **kwargs)

            output["preds"]["scf_detcg"], _, _, _ = self._forward_main(sc, ng, sf, s_pred_c, n_pred_g, s_pred_f, detach_c=True, detach_g=True, **kwargs)
            output["preds"]["sgf_detcg"], _, _, _ = self._forward_main(nc, sg, sf, n_pred_c, s_pred_g, s_pred_f, detach_c=True, detach_g=True, **kwargs)
            output["preds"]["scg_detcg"], _, _, _ = self._forward_main(sc, sg, nf, s_pred_c, s_pred_g, n_pred_f, detach_c=True, detach_g=True, **kwargs)

            output["preds"]["sc"], _, _, _ = self._forward_main(sc, ng, nf, s_pred_c, n_pred_g, n_pred_f, **kwargs)
            output["preds"]["sg"], _, _, _ = self._forward_main(nc, sg, nf, n_pred_c, s_pred_g, n_pred_f, **kwargs)
            output["preds"]["sf"], _, _, _ = self._forward_main(nc, ng, sf, n_pred_c, n_pred_g, s_pred_f, **kwargs)

            output["preds"]["scf"], _, _, _ = self._forward_main(sc, ng, sf, s_pred_c, n_pred_g, s_pred_f, **kwargs)
            output["preds"]["sgf"], _, _, _ = self._forward_main(nc, sg, sf, n_pred_c, s_pred_g, s_pred_f, **kwargs)
            output["preds"]["scg"], _, _, _ = self._forward_main(sc, sg, nf, s_pred_c, s_pred_g, n_pred_f, **kwargs)

            output["preds"]["ncombined"] = n_pred
            output["preds"]["n_label"] = n_label


        return output
class ConcatClassifier_CREMAD_OGM_ShuffleGrad3dIB_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_ShuffleGrad3dIB_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.batchnorm_features = args.get("batchnorm_features", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)


        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.enc_2 = encs[2]

        self.count_trainingsteps = 0

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_2_lin = nn.Linear(d_model, num_classes, bias=False)
            self.cls_bias = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

            # if self.batchnorm_features:
            #     self.bn_0 = nn.BatchNorm1d(num_classes, track_running_stats=True)
            #     self.bn_1 = nn.BatchNorm1d(num_classes, track_running_stats=True)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_2_lin = nn.Linear(d_model, 4096, bias=False)
            self.cls_bias = nn.Parameter(torch.zeros(4096), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_2 = nn.BatchNorm1d(4096, track_running_stats=True)


            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_2_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.cls_bias = nn.Parameter(torch.zeros(fc_inner), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_2 = nn.BatchNorm1d(fc_inner, track_running_stats=True)



            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=d_model, dim=d_model, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")

        self.fc_yz = nn.Sequential(
            nn.Linear(num_classes, d_model, bias=False),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 3, bias=False),
        )

    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, detach_pred=not self.shufflegradmulti, **kwargs)
        v = self.enc_1(x, detach_pred=not self.shufflegradmulti, **kwargs)
        f = self.enc_2(x, detach_pred=not self.shufflegradmulti, **kwargs)

        return a, v, f, a["preds"]["combined"], v["preds"]["combined"], f["preds"]["combined"]

    def shuffle_ids(self, label):
        batch_size = label.size(0)
        shuffle_data = []
        while len(shuffle_data) < self.args.bias_infusion.num_samples:

            if self.args.bias_infusion.shuffle:
                shuffle_idx_0 = torch.randperm(batch_size)
                shuffle_idx_1 = torch.randperm(batch_size)
                shuffle_idx_2 = torch.randperm(batch_size)
                nonequal_label = ~((label[shuffle_idx_0] == label[shuffle_idx_1]) & (label[shuffle_idx_1] == label[shuffle_idx_2]))
                if nonequal_label.sum() <= 1:
                    continue
                shuffle_idx_0 = shuffle_idx_0[nonequal_label.cpu()]
                shuffle_idx_1 = shuffle_idx_1[nonequal_label.cpu()]
                shuffle_idx_2 = shuffle_idx_2[nonequal_label.cpu()]
            else:
                nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                shuffle_idx_0 = torch.arange(batch_size)
                shuffle_idx_1 = torch.arange(batch_size)
                shuffle_idx_2 = torch.arange(batch_size)

            if nonequal_label.sum() <= 1:
                continue
            shuffle_data.append({"shuffle_idx_0": shuffle_idx_0,
                                 "shuffle_idx_1": shuffle_idx_1,
                                 "shuffle_idx_2": shuffle_idx_2,
                                 "data": nonequal_label})

        return shuffle_data

    def shuffle_data(self, c, g, f, pred_c, pred_g, pred_f, pred_mm, label):

        shuffle_data = self.shuffle_ids(label)

        sc = {feat: {"combined": torch.concatenate([c[feat]["combined"][sh_data_i["shuffle_idx_0"]] for sh_data_i in shuffle_data], dim=0)} for feat in c}
        sg = {feat: {"combined": torch.concatenate([g[feat]["combined"][sh_data_i["shuffle_idx_1"]] for sh_data_i in shuffle_data], dim=0)} for feat in g}
        sf = {feat: {"combined": torch.concatenate([f[feat]["combined"][sh_data_i["shuffle_idx_2"]] for sh_data_i in shuffle_data], dim=0)} for feat in f}

        nc = {feat: {"combined": torch.concatenate([c[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)} for feat in c}
        ng = {feat: {"combined": torch.concatenate([g[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)} for feat in g}
        nf = {feat: {"combined": torch.concatenate([f[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)} for feat in f}

        s_pred_c = torch.concatenate([pred_c[sh_data_i["shuffle_idx_0"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_g = torch.concatenate([pred_g[sh_data_i["shuffle_idx_1"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_f = torch.concatenate([pred_f[sh_data_i["shuffle_idx_2"]] for sh_data_i in shuffle_data], dim=0)

        n_pred_c = torch.concatenate([pred_c[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_g = torch.concatenate([pred_g[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_f = torch.concatenate([pred_f[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_pred = torch.concatenate([pred_mm[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_label = torch.concatenate([label[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        return sc, sg, sf, s_pred_c, s_pred_g, s_pred_f, nc, ng, nf, n_pred_c, n_pred_g, n_pred_f, n_pred, n_label
    def _forward_main(self, a, v, f, pred_aa, pred_vv, pred_ff, **kwargs):


        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_f = torch.matmul(f["features"]["combined"], self.fc_2_lin.weight.T) #+ self.fc_0_lin.bias / 2
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_a = pred_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_v = pred_v.detach()
            if "detach_f" in kwargs and kwargs["detach_f"]:
                pred_f = pred_f.detach()

            if "skip_bias" in kwargs and kwargs["skip_bias"]:
                pass
            else:
                pred_v = pred_v
                pred_a = pred_a
                pred_f = pred_f

            pred = pred_a + pred_v + pred_f + self.cls_bias

            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                            "wf_f": pred_f.norm(),
                           "w_a": self.fc_0_lin.weight.norm(),
                           "w_v": self.fc_1_lin.weight.norm(),
                            "w_f": self.fc_2_lin.weight.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm(),
                            "f_f": f["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

        else:
            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                            "wf_f": pred_ff.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm(),
                            "f_f": f["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
                pred_ff = (pred_ff - pred_ff.mean()) / pred_ff.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
                pred_ff = F.softmax(pred_ff, dim=1)
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_aa = pred_aa.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_vv = pred_vv.detach()
            if "detach_f" in kwargs and kwargs["detach_f"]:
                pred_ff = pred_ff.detach()

            pred = pred_aa + pred_vv + pred_ff


        if self.cls_type == "film" or self.cls_type == "filmv" or self.cls_type == "gated":
            this_feat_a, this_feat_v, this_feat_f = a["features"]["combined"], v["features"]["combined"], f["features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            if "detach_f" in kwargs and kwargs["detach_f"]:
                this_feat_f = this_feat_f.detach()
            pred = self.common_fc([this_feat_a, this_feat_v, this_feat_f], **kwargs)
        elif self.cls_type == "tf":
            this_feat_a, this_feat_v, this_feat_f = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"], f["nonaggr_features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            if "detach_f" in kwargs and kwargs["detach_f"]:
                this_feat_f = this_feat_f.detach()
            pred = self.common_fc([this_feat_a, this_feat_v, this_feat_f], **kwargs)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)

        return pred, pred_aa, pred_vv, pred_ff

    def forward(self, x, **kwargs):

        c, g, f, pred_c, pred_g, pred_f = self._get_features(x, **kwargs)

        pred, pred_aa, pred_vv, pred_ff = self._forward_main(c, g, f, pred_c, pred_g, pred_f, **kwargs)

        output = {"preds":{"combined":pred,
                            "c":pred_aa,
                            "g":pred_vv,
                           "f":pred_ff
                            },
                    "features": {"c": c["features"]["combined"],
                                "g": g["features"]["combined"],
                                 "f": f["features"]["combined"]}}

        if self.training:
            pred_feat = self.fc_yz(pred.detach())
            combined_features = torch.cat([c["features"]["combined"], g["features"]["combined"], f["features"]["combined"]], dim=1)
            CMI_yz_Loss = torch.nn.MSELoss()(combined_features, pred_feat) * self.args.bias_infusion.get("lib", 0)

            output["losses"] = {"CMI_yz_Loss": CMI_yz_Loss}  # ,

            sc, sg, sf, s_pred_c, s_pred_g, s_pred_f, nc, ng, nf, n_pred_c, n_pred_g, n_pred_f, n_pred, n_label = self.shuffle_data( c, g, f, pred_c, pred_g, pred_f, pred, kwargs["label"])

            kwargs["notwandb"] = True
            output["preds"]["sc_detc"], _, _, _ = self._forward_main(sc, ng, nf, s_pred_c, n_pred_g, n_pred_f, detach_c=True, **kwargs)
            output["preds"]["sg_detc"], _, _, _ = self._forward_main(nc, sg, nf, n_pred_c, s_pred_g, n_pred_f, detach_c=True, **kwargs)
            output["preds"]["sf_detc"], _, _, _ = self._forward_main(nc, ng, sf, n_pred_c, n_pred_g, s_pred_f, detach_c=True, **kwargs)

            output["preds"]["sc_detg"], _, _, _ = self._forward_main(sc, ng, nf, s_pred_c, n_pred_g, n_pred_f, detach_g=True, **kwargs)
            output["preds"]["sg_detg"], _, _, _ = self._forward_main(nc, sg, nf, n_pred_c, s_pred_g, n_pred_f, detach_g=True, **kwargs)
            output["preds"]["sf_detg"], _, _, _ = self._forward_main(nc, ng, sf, n_pred_c, n_pred_g, s_pred_f, detach_g=True, **kwargs)

            output["preds"]["sc_detf"], _, _, _ = self._forward_main(sc, ng, nf, s_pred_c, n_pred_g, n_pred_f, detach_f=True, **kwargs)
            output["preds"]["sg_detf"], _, _, _ = self._forward_main(nc, sg, nf, n_pred_c, s_pred_g, n_pred_f, detach_f=True, **kwargs)
            output["preds"]["sf_detf"], _, _, _ = self._forward_main(nc, ng, sf, n_pred_c, n_pred_g, s_pred_f, detach_f=True, **kwargs)

            output["preds"]["sc_detgf"], _, _, _ = self._forward_main(sc, ng, nf, s_pred_c, n_pred_g, n_pred_f, detach_g=True, detach_f=True, **kwargs)
            output["preds"]["sg_detgf"], _, _, _ = self._forward_main(nc, sg, nf, n_pred_c, s_pred_g, n_pred_f, detach_g=True, detach_f=True, **kwargs)
            output["preds"]["sf_detgf"], _, _, _ = self._forward_main(nc, ng, sf, n_pred_c, n_pred_g, s_pred_f, detach_g=True, detach_f=True, **kwargs)

            output["preds"]["sc_detcf"], _, _, _ = self._forward_main(sc, ng, nf, s_pred_c, n_pred_g, n_pred_f, detach_c=True, detach_f=True, **kwargs)
            output["preds"]["sg_detcf"], _, _, _ = self._forward_main(nc, sg, nf, n_pred_c, s_pred_g, n_pred_f, detach_c=True, detach_f=True, **kwargs)
            output["preds"]["sf_detcf"], _, _, _ = self._forward_main(nc, ng, sf, n_pred_c, n_pred_g, s_pred_f, detach_c=True, detach_f=True, **kwargs)

            output["preds"]["sc_detcg"], _, _, _ = self._forward_main(sc, ng, nf, s_pred_c, n_pred_g, n_pred_f, detach_c=True, detach_g=True, **kwargs)
            output["preds"]["sg_detcg"], _, _, _ = self._forward_main(nc, sg, nf, n_pred_c, s_pred_g, n_pred_f, detach_c=True, detach_g=True, **kwargs)
            output["preds"]["sf_detcg"], _, _, _ = self._forward_main(nc, ng, sf, n_pred_c, n_pred_g, s_pred_f, detach_c=True, detach_g=True, **kwargs)

            output["preds"]["scf_detc"], _, _, _ = self._forward_main(sc, ng, sf, s_pred_c, n_pred_g, s_pred_f, detach_c=True, **kwargs)
            output["preds"]["sgf_detc"], _, _, _ = self._forward_main(nc, sg, sf, n_pred_c, s_pred_g, s_pred_f, detach_c=True, **kwargs)
            output["preds"]["scg_detc"], _, _, _ = self._forward_main(sc, sg, nf, s_pred_c, s_pred_g, n_pred_f, detach_c=True, **kwargs)

            output["preds"]["scf_detg"], _, _, _ = self._forward_main(sc, ng, sf, s_pred_c, n_pred_g, s_pred_f, detach_g=True, **kwargs)
            output["preds"]["sgf_detg"], _, _, _ = self._forward_main(nc, sg, sf, n_pred_c, s_pred_g, s_pred_f, detach_g=True, **kwargs)
            output["preds"]["scg_detg"], _, _, _ = self._forward_main(sc, sg, nf, s_pred_c, s_pred_g, n_pred_f, detach_g=True, **kwargs)

            output["preds"]["scf_detf"], _, _, _ = self._forward_main(sc, ng, sf, s_pred_c, n_pred_g, s_pred_f, detach_f=True, **kwargs)
            output["preds"]["sgf_detf"], _, _, _ = self._forward_main(nc, sg, sf, n_pred_c, s_pred_g, s_pred_f, detach_f=True, **kwargs)
            output["preds"]["scg_detf"], _, _, _ = self._forward_main(sc, sg, nf, s_pred_c, s_pred_g, n_pred_f, detach_f=True, **kwargs)

            output["preds"]["scf_detgf"], _, _, _ = self._forward_main(sc, ng, sf, s_pred_c, n_pred_g, s_pred_f, detach_g=True, detach_f=True, **kwargs)
            output["preds"]["sgf_detgf"], _, _, _ = self._forward_main(nc, sg, sf, n_pred_c, s_pred_g, s_pred_f, detach_g=True, detach_f=True, **kwargs)
            output["preds"]["scg_detgf"], _, _, _ = self._forward_main(sc, sg, nf, s_pred_c, s_pred_g, n_pred_f, detach_g=True, detach_f=True, **kwargs)

            output["preds"]["scf_detcf"], _, _, _ = self._forward_main(sc, ng, sf, s_pred_c, n_pred_g, s_pred_f, detach_c=True, detach_f=True, **kwargs)
            output["preds"]["sgf_detcf"], _, _, _ = self._forward_main(nc, sg, sf, n_pred_c, s_pred_g, s_pred_f, detach_c=True, detach_f=True, **kwargs)
            output["preds"]["scg_detcf"], _, _, _ = self._forward_main(sc, sg, nf, s_pred_c, s_pred_g, n_pred_f, detach_c=True, detach_f=True, **kwargs)

            output["preds"]["scf_detcg"], _, _, _ = self._forward_main(sc, ng, sf, s_pred_c, n_pred_g, s_pred_f, detach_c=True, detach_g=True, **kwargs)
            output["preds"]["sgf_detcg"], _, _, _ = self._forward_main(nc, sg, sf, n_pred_c, s_pred_g, s_pred_f, detach_c=True, detach_g=True, **kwargs)
            output["preds"]["scg_detcg"], _, _, _ = self._forward_main(sc, sg, nf, s_pred_c, s_pred_g, n_pred_f, detach_c=True, detach_g=True, **kwargs)

            output["preds"]["sc"], _, _, _ = self._forward_main(sc, ng, nf, s_pred_c, n_pred_g, n_pred_f, **kwargs)
            output["preds"]["sg"], _, _, _ = self._forward_main(nc, sg, nf, n_pred_c, s_pred_g, n_pred_f, **kwargs)
            output["preds"]["sf"], _, _, _ = self._forward_main(nc, ng, sf, n_pred_c, n_pred_g, s_pred_f, **kwargs)

            output["preds"]["scf"], _, _, _ = self._forward_main(sc, ng, sf, s_pred_c, n_pred_g, s_pred_f, **kwargs)
            output["preds"]["sgf"], _, _, _ = self._forward_main(nc, sg, sf, n_pred_c, s_pred_g, s_pred_f, **kwargs)
            output["preds"]["scg"], _, _, _ = self._forward_main(sc, sg, nf, s_pred_c, s_pred_g, n_pred_f, **kwargs)

            output["preds"]["ncombined"] = n_pred
            output["preds"]["n_label"] = n_label


        return output
class ConcatClassifier_CREMAD_OGM_VAEGrad_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_VAEGrad_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)

        self.num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.batchnorm_features = args.get("batchnorm_features", False)


        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.count_trainingsteps = 0

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, self.num_classes, bias=False)
            self.fc_1_lin = nn.Linear(d_model, self.num_classes, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(self.num_classes), requires_grad=True)

            # if self.batchnorm_features:
            #     self.bn_0 = nn.BatchNorm1d(num_classes, track_running_stats=True)
            #     self.bn_1 = nn.BatchNorm1d(num_classes, track_running_stats=True)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(4096), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(4096, track_running_stats=True)


            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(fc_inner), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(fc_inner, track_running_stats=True)



            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, self.num_classes)
            )
        elif self.cls_type == "film":
            self.common_fc = FiLM(d_model, d_model, self.num_classes)
        elif self.cls_type == "filmv":
            self.common_fc = FiLM(d_model, d_model, self.num_classes, x_film=False)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=d_model, dim=d_model, output_dim=self.num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=d_model, dim=d_model, layers=2, output_dim=self.num_classes)
        else:
            raise ValueError("Unknown cls_type")


        self.CGMVAE_0 = ConditionalGMVAE(d_model, int(d_model/4), 30, self.num_classes)
        self.CGMVAE_1 = ConditionalGMVAE(d_model, int(d_model/4), 30, self.num_classes)



    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, detach_pred=True, **kwargs)
        v = self.enc_1(x, detach_pred=True, **kwargs)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def _forward_main(self, a, v, **kwargs):

        pred_aa = kwargs.get("pred_aa", None)
        pred_vv = kwargs.get("pred_vv", None)

        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) #+ self.fc_0_lin.bias / 2
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_a = pred_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_v = pred_v.detach()

            if "skip_bias" in kwargs and kwargs["skip_bias"]:
                pass
            else:
                pred_v = pred_v + self.bias_lin/2
                pred_a = pred_a + self.bias_lin/2

            pred = pred_a + pred_v

            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "w_a": self.fc_0_lin.weight.norm(),
                           "w_v": self.fc_1_lin.weight.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

        if self.cls_type == "film" or self.cls_type == "filmv" or self.cls_type == "gated":
            this_feat_a, this_feat_v = a["features"]["combined"], v["features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)
        elif self.cls_type == "tf":
            this_feat_a, this_feat_v = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)

        return pred, pred_aa, pred_vv

    def draw_data(self, num_samples, labels, a, v):



        sa = self.CGMVAE_0.sample_from_class( labels, num_samples)
        sv = self.CGMVAE_0.sample_from_class( labels, num_samples)

        sa = {"features": {"combined": sa}}
        sv = {"features": {"combined": sv}}

        features_to_aggregate = ["features", "nonaggr_features"]
        if self.cls_type != "tf":
            features_to_aggregate = ["features"]
        na = {feat: {"combined": torch.concatenate([a[feat]["combined"] for i in range(num_samples)], dim=0) } for feat in features_to_aggregate}
        nv = {feat: {"combined": torch.concatenate([v[feat]["combined"] for i in range(num_samples)], dim=0) } for feat in features_to_aggregate}

        return sa, sv, na, nv

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x, **kwargs)

        one_hot_label = F.one_hot(kwargs["label"], num_classes=self.num_classes)

        z_initial_recon, means, logvars, weights = self.CGMVAE_0(a["features"]["combined"].detach(), one_hot_label)
        loss_0, recon_loss, kl_div = self.CGMVAE_0.gmvae_loss(a["features"]["combined"].detach(), z_initial_recon, means, logvars, weights)

        z_initial_recon, means, logvars, weights = self.CGMVAE_1(v["features"]["combined"].detach(), one_hot_label)
        loss_1, recon_loss, kl_div = self.CGMVAE_1.gmvae_loss(v["features"]["combined"].detach(), z_initial_recon, means, logvars, weights)


        pred, pred_aa, pred_vv = self._forward_main(a, v, pred_aa = pred_aa, pred_vv = pred_vv, **kwargs)

        output = {"preds":{"combined":pred,
                            "c":pred_aa,
                            "g":pred_vv
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"]},
                  "losses":{"vae_loss": loss_0+loss_1}}

        if "detach" in kwargs and kwargs["detach"]:

            sa, sv, na, nv = self.draw_data(kwargs["num_samples"], one_hot_label, a, v)

            pred_dtv_sa, _, _ = self._forward_main(sa, nv, detach_v=True, **kwargs)
            pred_dta_sa, _, _ = self._forward_main(sa, nv, detach_a=True, **kwargs)
            output["preds"]["sa_detv"] = pred_dtv_sa
            output["preds"]["sa_deta"] = pred_dta_sa

            pred_dtv_sv, _, _ = self._forward_main(na, sv, detach_v=True, **kwargs)
            pred_dta_sv, _, _ = self._forward_main(na, sv, detach_a=True, **kwargs)
            output["preds"]["sv_detv"] = pred_dtv_sv
            output["preds"]["sv_deta"] = pred_dta_sv

            pred_dtav_sa, _, _ = self._forward_main(sa, nv, detach_a=True, detach_v=True, **kwargs)
            pred_dtav_sv, _, _ = self._forward_main(na, sv, detach_a=True, detach_v=True, **kwargs)
            output["preds"]["sv_detav"] = pred_dtav_sv
            output["preds"]["sa_detav"] = pred_dtav_sa

            pred_sa, _, _ = self._forward_main(sa, nv, **kwargs)
            pred_sv, _, _ = self._forward_main(na, sv, **kwargs)
            output["preds"]["sv"] = pred_sv
            output["preds"]["sa"] = pred_sa

            pred_nav, _, _ = self._forward_main(na, nv, **kwargs)
            output["preds"]["ncombined"] = pred_nav


        return output
class ConcatClassifier_CREMAD_OGM_NoiseGrad_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_NoiseGrad_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)

        self.num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.batchnorm_features = args.get("batchnorm_features", False)


        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.count_trainingsteps = 0

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(512, self.num_classes, bias=False)
            self.fc_1_lin = nn.Linear(512, self.num_classes, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(self.num_classes), requires_grad=True)

            # if self.batchnorm_features:
            #     self.bn_0 = nn.BatchNorm1d(num_classes, track_running_stats=True)
            #     self.bn_1 = nn.BatchNorm1d(num_classes, track_running_stats=True)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(4096), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(4096, track_running_stats=True)


            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(fc_inner), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(fc_inner, track_running_stats=True)



            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, self.num_classes)
            )
        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, self.num_classes)
        elif self.cls_type == "filmv":
            self.common_fc = FiLM(512, 512, self.num_classes, x_film=False)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=self.num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=512, dim=512, layers=2, output_dim=self.num_classes)
        else:
            raise ValueError("Unknown cls_type")



    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, detach_pred=True, **kwargs)
        v = self.enc_1(x, detach_pred=True, **kwargs)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def _forward_main(self, a, v, **kwargs):

        pred_aa = kwargs.get("pred_aa", None)
        pred_vv = kwargs.get("pred_vv", None)

        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) #+ self.fc_0_lin.bias / 2
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_a = pred_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_v = pred_v.detach()

            if "skip_bias" in kwargs and kwargs["skip_bias"]:
                pass
            else:
                pred_v = pred_v + self.bias_lin/2
                pred_a = pred_a + self.bias_lin/2

            pred = pred_a + pred_v

            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "w_a": self.fc_0_lin.weight.norm(),
                           "w_v": self.fc_1_lin.weight.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

        if self.cls_type == "film" or self.cls_type == "filmv" or self.cls_type == "gated":
            this_feat_a, this_feat_v = a["features"]["combined"], v["features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)
        elif self.cls_type == "tf":
            this_feat_a, this_feat_v = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)

        return pred, pred_aa, pred_vv

    def draw_data(self, num_samples, labels, a, v):

        def _add_noise_to_tensor(tens: torch.Tensor, over_dim: int = 0) -> torch.Tensor:
            return tens + torch.randn_like(tens) * tens.std(dim=over_dim)

        features_to_aggregate = ["features", "nonaggr_features"]
        if self.cls_type != "tf":
            features_to_aggregate = ["features"]

        sa = {feat: {"combined": torch.concatenate([_add_noise_to_tensor(a[feat]["combined"]) for i in range(num_samples)], dim=0) } for feat in features_to_aggregate}
        sv = {feat: {"combined": torch.concatenate([_add_noise_to_tensor(v[feat]["combined"]) for i in range(num_samples)], dim=0) } for feat in features_to_aggregate}

        na = {feat: {"combined": torch.concatenate([a[feat]["combined"] for i in range(num_samples)], dim=0) } for feat in features_to_aggregate}
        nv = {feat: {"combined": torch.concatenate([v[feat]["combined"] for i in range(num_samples)], dim=0) } for feat in features_to_aggregate}

        return sa, sv, na, nv

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x, **kwargs)

        one_hot_label = F.one_hot(kwargs["label"], num_classes=self.num_classes)

        pred, pred_aa, pred_vv = self._forward_main(a, v, pred_aa = pred_aa, pred_vv = pred_vv, **kwargs)

        output = {"preds":{"combined":pred,
                            "c":pred_aa,
                            "g":pred_vv
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"]}
                  }

        if "detach" in kwargs and kwargs["detach"]:

            sa, sv, na, nv = self.draw_data(kwargs["num_samples"], one_hot_label, a, v)

            pred_dtv_sa, _, _ = self._forward_main(sa, nv, detach_v=True, **kwargs)
            pred_dta_sa, _, _ = self._forward_main(sa, nv, detach_a=True, **kwargs)
            output["preds"]["sa_detv"] = pred_dtv_sa
            output["preds"]["sa_deta"] = pred_dta_sa

            pred_dtv_sv, _, _ = self._forward_main(na, sv, detach_v=True, **kwargs)
            pred_dta_sv, _, _ = self._forward_main(na, sv, detach_a=True, **kwargs)
            output["preds"]["sv_detv"] = pred_dtv_sv
            output["preds"]["sv_deta"] = pred_dta_sv

            pred_dtav_sa, _, _ = self._forward_main(sa, nv, detach_a=True, detach_v=True, **kwargs)
            pred_dtav_sv, _, _ = self._forward_main(na, sv, detach_a=True, detach_v=True, **kwargs)
            output["preds"]["sv_detav"] = pred_dtav_sv
            output["preds"]["sa_detav"] = pred_dtav_sa

            pred_sa, _, _ = self._forward_main(sa, nv, **kwargs)
            pred_sv, _, _ = self._forward_main(na, sv, **kwargs)
            output["preds"]["sv"] = pred_sv
            output["preds"]["sa"] = pred_sa

            pred_nav, _, _ = self._forward_main(na, nv, **kwargs)
            output["preds"]["ncombined"] = pred_nav


        return output
class ConcatClassifier_CREMAD_OGM_NoiseGradG_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_NoiseGradG_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)

        self.num_classes = args.num_classes
        self.samples = args.bias_infusion.num_samples
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.batchnorm_features = args.get("batchnorm_features", False)


        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.count_trainingsteps = 0

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(512, self.num_classes, bias=False)
            self.fc_1_lin = nn.Linear(512, self.num_classes, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(self.num_classes), requires_grad=True)

            # if self.batchnorm_features:
            #     self.bn_0 = nn.BatchNorm1d(num_classes, track_running_stats=True)
            #     self.bn_1 = nn.BatchNorm1d(num_classes, track_running_stats=True)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(4096), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(4096, track_running_stats=True)


            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(fc_inner), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(fc_inner, track_running_stats=True)



            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, self.num_classes)
            )
        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, self.num_classes)
        elif self.cls_type == "filmv":
            self.common_fc = FiLM(512, 512, self.num_classes, x_film=False)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=self.num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=512, dim=512, layers=2, output_dim=self.num_classes)
        else:
            raise ValueError("Unknown cls_type")

    def _add_noise_to_tensor(self, tens: torch.Tensor, over_dim: int = 0) -> torch.Tensor:
        return tens + torch.randn_like(tens) * tens.std(dim=over_dim)

    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, detach_pred=True, **kwargs)
        v = self.enc_1(x, detach_pred=True, **kwargs)

        return a["features"]["combined"], v["features"]["combined"], a["preds"]["combined"], v["preds"]["combined"]

    def _get_features_noisy(self, x, **kwargs):

        for i in x:
            x[i] = self._add_noise_to_tensor(x[i].repeat(self.samples, *([1] * (x[i].dim() - 1))), over_dim=0)

        a = self.enc_0(x, detach_pred=True, **kwargs)
        v = self.enc_1(x, detach_pred=True, **kwargs)

        return a["features"]["combined"], v["features"]["combined"], a["preds"]["combined"], v["preds"]["combined"]

    def _forward_main(self, a, v, **kwargs):

        pred_aa = kwargs.get("pred_aa", None)
        pred_vv = kwargs.get("pred_vv", None)

        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a, self.fc_0_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v, self.fc_1_lin.weight.T) #+ self.fc_0_lin.bias / 2
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_a = pred_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_v = pred_v.detach()

            if "skip_bias" in kwargs and kwargs["skip_bias"]:
                pass
            else:
                pred_v = pred_v + self.bias_lin/2
                pred_a = pred_a + self.bias_lin/2

            pred = pred_a + pred_v

            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "w_a": self.fc_0_lin.weight.norm(),
                           "w_v": self.fc_1_lin.weight.norm(),
                           "f_a": a.norm(),
                           "f_v": v.norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

        if self.cls_type == "film" or self.cls_type == "filmv" or self.cls_type == "gated":
            this_feat_a, this_feat_v = a, v
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)
        elif self.cls_type == "tf":
            this_feat_a, this_feat_v = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)

        return pred, pred_aa, pred_vv

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x, **kwargs)
        a_noisy, v_noisy, pred_noisy_aa, pred_noisy_vv = self._get_features_noisy(x, **kwargs)

        pred, pred_aa, pred_vv = self._forward_main(a, v, pred_aa = pred_aa, pred_vv = pred_vv, **kwargs)
        output = {"preds":{"combined":pred,
                            "c":pred_aa,
                            "g":pred_vv
                            },
                    "features": {"c": a,
                                "g": v}
                  }
        if self.training:
            pred_noisy_0, _, _ = self._forward_main(a_noisy, v.repeat(self.samples, *([1] * (v.dim() - 1))), pred_aa = pred_noisy_aa, pred_vv = pred_vv.repeat(self.samples, 1), notwandb=True, **kwargs)
            pred_noisy_1, _, _ = self._forward_main(a.repeat(self.samples, *([1] * (a.dim() - 1))), v_noisy, pred_aa = pred_aa.repeat(self.samples, 1), pred_vv = pred_noisy_vv, notwandb=True, **kwargs)

            pred_noisy_0_sa, _, _ = self._forward_main(a_noisy, v.repeat(self.samples, *([1] * (v.dim() - 1))), pred_aa = pred_noisy_aa, pred_vv = pred_vv.repeat(self.samples, 1), notwandb=True, detach_a=True, **kwargs)
            pred_noisy_1_sa, _, _ = self._forward_main(a.repeat(self.samples, *([1] * (a.dim() - 1))), v_noisy, pred_aa = pred_aa.repeat(self.samples, 1), pred_vv = pred_noisy_vv, notwandb=True, detach_a=True, **kwargs)

            pred_noisy_0_sv, _, _ = self._forward_main(a_noisy, v.repeat(self.samples, *([1] * (v.dim() - 1))), pred_aa = pred_noisy_aa, pred_vv = pred_vv.repeat(self.samples, 1), notwandb=True, detach_v=True, **kwargs)
            pred_noisy_1_sv, _, _ = self._forward_main(a.repeat(self.samples, *([1] * (a.dim() - 1))), v_noisy, pred_aa = pred_aa.repeat(self.samples, 1), pred_vv = pred_noisy_vv, notwandb=True, detach_v=True, **kwargs)

            output["preds"].update({
                "noisy_0": pred_noisy_0,
                "noisy_1": pred_noisy_1,
                "noisy_0_sa": pred_noisy_0_sa,
                "noisy_1_sa": pred_noisy_1_sa,
                "noisy_0_sv": pred_noisy_0_sv,
                "noisy_1_sv": pred_noisy_1_sv })

        return output

class ConcatClassifier_CREMAD_OGM_MultiShuffleGrad_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_MultiShuffleGrad_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)



        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.batchnorm_features = args.get("batchnorm_features", False)


        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.count_trainingsteps = 0

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(512, num_classes, bias=False)
            self.fc_1_lin = nn.Linear(512, num_classes, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

            # if self.batchnorm_features:
            #     self.bn_0 = nn.BatchNorm1d(num_classes, track_running_stats=True)
            #     self.bn_1 = nn.BatchNorm1d(num_classes, track_running_stats=True)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(4096), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(4096, track_running_stats=True)


            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(fc_inner), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(fc_inner, track_running_stats=True)



            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, num_classes)
        elif self.cls_type == "filmv":
            self.common_fc = FiLM(512, 512, num_classes, x_film=False)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=512, dim=512, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")

    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, detach_pred=False, **kwargs)
        v = self.enc_1(x, detach_pred=False, **kwargs)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):


        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) #+ self.fc_0_lin.bias / 2
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_a = pred_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_v = pred_v.detach()

            if "skip_bias" in kwargs and kwargs["skip_bias"]:
                pass
            else:
                pred_v = pred_v + self.bias_lin/2
                pred_a = pred_a + self.bias_lin/2

            pred = pred_a + pred_v

            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "w_a": self.fc_0_lin.weight.norm(),
                           "w_v": self.fc_1_lin.weight.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

        else:
            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_aa = pred_aa.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_vv = pred_vv.detach()

            pred = pred_aa + pred_vv


        if self.cls_type == "film" or self.cls_type == "filmv" or self.cls_type == "gated":
            this_feat_a, this_feat_v = a["features"]["combined"], v["features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)
        elif self.cls_type == "tf":
            this_feat_a, this_feat_v = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)

        return pred, pred_aa, pred_vv

    def shuffle_data(self, shuffle_data, a, v, pred_aa, pred_vv, pred):

        features_to_aggregate = ["features", "nonaggr_features"]
        if self.cls_type != "tf":
            features_to_aggregate = ["features"]
        sa = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in features_to_aggregate}
        sv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in features_to_aggregate}
        s_pred_aa = torch.concatenate([pred_aa[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_vv = torch.concatenate([pred_vv[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)

        na = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in features_to_aggregate}
        nv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in features_to_aggregate}
        n_pred_aa = torch.concatenate([pred_aa[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_vv = torch.concatenate([pred_vv[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_pred = torch.concatenate([pred[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        return sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x, **kwargs)

        pred, pred_aa, pred_vv = self._forward_main(a, v, pred_aa, pred_vv, **kwargs)

        output = {"preds":{"combined":pred,
                            "c":pred_aa,
                            "g":pred_vv
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"]}}

        if "detach" in kwargs and kwargs["detach"]:

            sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred = self.shuffle_data(kwargs["shuffle_data"], a, v, pred_aa, pred_vv, pred)

            pred_dtv_sa, _, _ = self._forward_main(sa, nv, s_pred_aa, n_pred_vv.detach(), detach_v=True, **kwargs)
            pred_dta_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv, detach_a=True, **kwargs)
            output["preds"]["sa_detv"] = pred_dtv_sa
            output["preds"]["sa_deta"] = pred_dta_sa

            pred_dtv_sv, _, _ = self._forward_main(na, sv, n_pred_aa, s_pred_vv.detach(), detach_v=True, **kwargs)
            pred_dta_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv, detach_a=True, **kwargs)
            output["preds"]["sv_detv"] = pred_dtv_sv
            output["preds"]["sv_deta"] = pred_dta_sv

            pred_dtav_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv.detach(), detach_a=True, detach_v=True, **kwargs)
            pred_dtav_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv.detach(), detach_a=True, detach_v=True, **kwargs)
            output["preds"]["sv_detav"] = pred_dtav_sv
            output["preds"]["sa_detav"] = pred_dtav_sa

            pred_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv.detach(), **kwargs)
            pred_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv.detach(), **kwargs)
            output["preds"]["sv"] = pred_sv
            output["preds"]["sa"] = pred_sa

            output["preds"]["ncombined"] = n_pred

        return output

class ConcatClassifier_CREMAD_OGM_ShuffleGrad_TF_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_ShuffleGrad_TF_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)



        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.batchnorm_features = args.get("batchnorm_features", False)


        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.count_trainingsteps = 0

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(512, num_classes, bias=False)
            self.fc_1_lin = nn.Linear(512, num_classes, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

            # if self.batchnorm_features:
            #     self.bn_0 = nn.BatchNorm1d(num_classes, track_running_stats=True)
            #     self.bn_1 = nn.BatchNorm1d(num_classes, track_running_stats=True)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(4096), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(4096, track_running_stats=True)


            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(fc_inner), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(fc_inner, track_running_stats=True)



            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, num_classes)
        elif self.cls_type == "filmv":
            self.common_fc = FiLM(512, 512, num_classes, x_film=False)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=512, dim=512, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")

    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, detach_pred=True, **kwargs)
        v = self.enc_1(x, detach_pred=True, **kwargs)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):


        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) #+ self.fc_0_lin.bias / 2
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_a = pred_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_v = pred_v.detach()

            if "skip_bias" in kwargs and kwargs["skip_bias"]:
                pass
            else:
                pred_v = pred_v + self.bias_lin/2
                pred_a = pred_a + self.bias_lin/2

            pred = pred_a + pred_v

            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "w_a": self.fc_0_lin.weight.norm(),
                           "w_v": self.fc_1_lin.weight.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

        else:
            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_aa = pred_aa.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_vv = pred_vv.detach()

            pred = pred_aa + pred_vv


        if self.cls_type == "film" or self.cls_type == "filmv" or self.cls_type == "gated":
            this_feat_a, this_feat_v = a["features"]["combined"], v["features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)
        elif self.cls_type == "tf":
            this_feat_a, this_feat_v = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)

        return pred, pred_aa, pred_vv

    def shuffle_ids(self, batch_size, label):

        shuffle_data = []
        while len(shuffle_data) < self.args.bias_infusion.num_samples:

            if self.args.bias_infusion.shuffle:
                shuffle_idx = torch.randperm(batch_size)
                nonequal_label = ~(label[shuffle_idx] == label)
                if nonequal_label.sum() <= 1:
                    continue
                shuffle_idx = shuffle_idx[nonequal_label.cpu()]
            else:
                nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                shuffle_idx = torch.arange(batch_size)

            if nonequal_label.sum() <= 1:
                continue
            shuffle_data.append({"shuffle_idx": shuffle_idx, "data": nonequal_label})

        return shuffle_data

    def shuffle_data(self, a, v, pred_aa, pred_vv, pred, label):

        shuffle_data = self.shuffle_ids(label)

        sa = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        sv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        s_pred_aa = torch.concatenate([pred_aa[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_vv = torch.concatenate([pred_vv[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)

        na = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        nv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in ["features", "nonaggr_features"]}
        n_pred_aa = torch.concatenate([pred_aa[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_vv = torch.concatenate([pred_vv[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_pred = torch.concatenate([pred[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        return sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x, **kwargs)

        pred, pred_aa, pred_vv = self._forward_main(a, v, pred_aa, pred_vv, **kwargs)

        output = {"preds":{"combined":pred,
                            "c":pred_aa,
                            "g":pred_vv
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"]}}

        if self.training:

            sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred = self.shuffle_data( a, v, pred_aa, pred_vv, pred, kwargs["label"])

            pred_dtv_sa, _, _ = self._forward_main(sa, nv, s_pred_aa, n_pred_vv.detach(), detach_v=True, **kwargs)
            pred_dta_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv, detach_a=True, **kwargs)
            output["preds"]["sa_detv"] = pred_dtv_sa
            output["preds"]["sa_deta"] = pred_dta_sa

            pred_dtv_sv, _, _ = self._forward_main(na, sv, n_pred_aa, s_pred_vv.detach(), detach_v=True, **kwargs)
            pred_dta_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv, detach_a=True, **kwargs)
            output["preds"]["sv_detv"] = pred_dtv_sv
            output["preds"]["sv_deta"] = pred_dta_sv

            pred_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv.detach(), **kwargs)
            pred_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv.detach(), **kwargs)
            output["preds"]["sv"] = pred_sv
            output["preds"]["sa"] = pred_sa

            output["preds"]["ncombined"] = n_pred

        return output


class ConcatClassifier_CREMAD_OGM_PeGrad_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_PeGrad_pre, self).__init__()

        self.args = args
        # self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.cross_att_va = torch.nn.MultiheadAttention(1, 1,  batch_first=True)
        self.cross_att_av = torch.nn.MultiheadAttention(1, 1,  batch_first=True)

        self.dropout = nn.Dropout(0.3)

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(512, num_classes)
            self.fc_1_lin = nn.Linear(512, num_classes, bias=False)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type != "linear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )

    def _get_features(self, x):

        a = self.enc_0(x)
        v = self.enc_1(x)

        return a["features"]["combined"], v["features"]["combined"], a["preds"]["combined"], v["preds"]["combined"]

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x)

        pred_a = torch.matmul(a, self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
        pred_v = torch.matmul(v, self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2

        pred = pred_a + pred_v


        if self.cls_type != "linear":
            pred = self.common_fc(pred)
            pred_aa = self.common_fc(pred_a)
            pred_vv = self.common_fc(pred_v)


        return {"preds":{"combined":pred,
                         "c":pred_aa,
                         "g":pred_vv
                         },
                "features": {"c": a,
                             "g": v}}
class ConcatClassifier_CREMAD_OGM_FusionGates_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_FusionGates_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)
        self.fusion_gates = self.args.get("fusion_gates", False)
        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.fusion_gates == "film":
            self.fusion = FiLM(512, 512, num_classes)
        elif self.fusion_gates == "gated":
            self.fusion = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)


    def _get_features(self, x):

        a = self.enc_0(x)
        v = self.enc_1(x)

        return a["features"]["combined"], v["features"]["combined"], a["preds"]["combined"], v["preds"]["combined"]

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x)

        pred = self.fusion([a, v])


        return {"preds":{"combined":pred,
                         "c":pred_aa,
                         "g":pred_vv
                         },
                "features": {"c": a,
                             "g": v}}
class ConcatClassifier_CREMAD_AGM_FusionGates_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_AGM_FusionGates_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)
        self.fusion_gates = self.args.get("fusion_gates", False)
        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.fusion_gates == "film":
            self.fusion = FiLM(512, 512, num_classes)
        elif self.fusion_gates == "gated":
            self.fusion = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)

        self.m_v_o = Modality_out()
        self.m_a_o = Modality_out()

        self.scale_a = 1.0
        self.scale_v = 1.0

        self.m_a_o.register_full_backward_hook(self.hooka)
        self.m_v_o.register_full_backward_hook(self.hookv)

    def hooka(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_a,

    def hookv(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_v,

    def update_scale(self, coeff_a, coeff_v):
        self.scale_a = coeff_a
        self.scale_v = coeff_v

    def _get_features(self, x):

        a = self.enc_0(x)
        v = self.enc_1(x)

        return a["features"]["combined"], v["features"]["combined"], a["preds"]["combined"], v["preds"]["combined"]

    def _get_preds_padded(self, x, feat_a, feat_v, pad_audio = False, pad_visual = False):

        data = copy.deepcopy(x)
        if pad_audio:
            if 0 in data:
                data[0] = torch.zeros_like(data[0], device=x[0].device)
            elif 2 in data:
                data[2] = torch.zeros_like(data[2], device=x[2].device)
            a = self.enc_0(data)
            pred = self._forward_main(a["features"]["combined"], feat_v)

        if pad_visual:
            if 1 in data:
                data[1] = torch.zeros_like(data[1], device=x[1].device)
            elif 3 in data:
                data[3] = torch.zeros_like(data[3], device=x[3].device)
            v = self.enc_1(data)
            pred = self._forward_main(feat_a, v["features"]["combined"])

        return pred

    def _forward_main(self, a, v):

        pred = self.fusion([a, v])

        return pred


    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x)

        training_mode = True if self.training else False
        self.eval()
        pred_za = self._get_preds_padded(x, feat_a=a, feat_v=v, pad_audio=True, pad_visual=False)
        pred_zv = self._get_preds_padded(x, feat_a=a, feat_v=v, pad_audio=False, pad_visual=True)
        if training_mode:
            self.train()
        pred = self._forward_main(a, v)

        pred_a = self.m_a_o(0.5*(pred - pred_za + pred_zv))
        pred_v = self.m_v_o(0.5*(pred - pred_zv + pred_za))

        return {"preds":{"combined":pred_a + pred_v,
                         "both": pred,
                         "c":pred_a,
                         "g":pred_v
                         },
                "features": {"c": a,
                             "g": v}}
    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x)

        pred = self.fusion([a, v])


        return {"preds":{"combined":pred,
                         "c":pred_aa,
                         "g":pred_vv
                         },
                "features": {"c": a,
                             "g": v}}

class ConcatClassifier_CREMAD_VAVL_UN_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_VAVL_UN_pre, self).__init__()

        self.args = args
        # self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type
        self.precondition = args.get("precondition", False)

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.hidden_2 = 512

        self.video_tokens = torch.nn.Parameter(torch.randn(1, 1, self.hidden_2), requires_grad=True)
        self.audio_tokens = torch.nn.Parameter(torch.randn(1, 1, self.hidden_2), requires_grad=True)
        self.class_tokens = torch.nn.Parameter(torch.randn(1, 1, self.hidden_2), requires_grad=True )


        self.common_net = Conformer(
                            input_dim=self.hidden_2,
                            encoder_dim=self.hidden_2,
                            num_encoder_layers=args.get("common_layer", 1))

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.cls_type == "linear":
            self.common_fc = nn.Linear(512, num_classes)

        elif self.cls_type == "highlynonlinear":
            self.common_fc = nn.Sequential(
                nn.Linear(d_model, 4096),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type != "linear":
            self.common_fc = nn.Sequential(
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )

    def forward(self, x, **kwargs):

        a = self.enc_0(x)
        v = self.enc_1(x)
        pred_aa, pred_vv = a["preds"]["combined"], v["preds"]["combined"]
        a_shape = a["nonaggr_features"]["combined"].shape
        # v_shape = v["nonaggr_features"]["combined"].shape

        feat_a = a["nonaggr_features"]["combined"] # + self.audio_tokens.repeat(a_shape[0],a_shape[1],1)
        feat_v = v["nonaggr_features"]["combined"] # + self.video_tokens.repeat(v_shape[0],v_shape[1],1)

        # feat_mm = torch.concatenate([self.class_tokens.repeat(1,a_shape[1],1), feat_a, feat_v], dim=0)
        feat_mm = torch.concatenate([feat_a, feat_v], dim=0)
        feat_mm = self.common_net(feat_mm)
        feat_mm = nn.AdaptiveAvgPool1d(1)(feat_mm.permute(1, 2, 0)).squeeze(2)
        pred = self.common_fc(feat_mm)
        # pred = self.common_fc(feat_mm[0])

        return {"preds":{"combined":pred,
                         "c":pred_aa,
                         "g":pred_vv
                         },
                "features": {"c": a,
                             "g": v}}

class ConcatClassifier_CREMAD_OGM_Ens(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_Ens, self).__init__()

        self.args = args

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.num_encs = len(encs)

    def forward(self, x, **kwargs):

        preds = []
        for i in range(self.num_encs):
            enc = getattr(self, f"enc_{i}")
            res = enc(x)
            preds.append(res["preds"]["combined"])

        pred = torch.mean(torch.stack(preds), dim=0)

        return {"preds":{"combined":pred,
                         "c":preds[0],
                         "g":preds[1]
                         }}


class Modality_Text(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,total_out,
                pad_visual_out,pad_audio_out,pad_text_out,
                pad_visual_audio_out,pad_visual_text_out,pad_audio_text_out,
                zero_padding_out):
        return (total_out-pad_text_out+pad_visual_audio_out)/3 + (pad_visual_out - pad_audio_text_out+pad_audio_out-pad_visual_text_out)/6

class Modality_Audio(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,total_out,
                pad_visual_out,pad_audio_out,pad_text_out,
                pad_visual_audio_out,pad_visual_text_out,pad_audio_text_out,
                zero_padding_out):
        return (total_out-pad_audio_out+pad_visual_text_out) / 3 + (pad_visual_out - pad_audio_text_out + pad_text_out - pad_visual_audio_out) / 6

class Modality_Visual(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,total_out,
                pad_visual_out,pad_audio_out,pad_text_out,
                pad_visual_audio_out,pad_visual_text_out,pad_audio_text_out,
                zero_padding_out):
        return (total_out-pad_visual_out+pad_audio_text_out)/3 + (pad_audio_out-pad_visual_text_out + pad_text_out - pad_visual_audio_out)/6

class ConcatClassifier_MOSEI_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_MOSEI_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.precondition = args.get("precondition", False)
        self.norm_decision = args.get("norm_decision", False)

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.enc_2 = encs[2]

        self.dropout = nn.Dropout(0.3)

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_2_lin = nn.Linear(d_model, num_classes, bias=False)

            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(num_classes, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(num_classes, track_running_stats=False)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_2_lin = nn.Linear(d_model, 4096, bias=False)

            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(d_model, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(d_model, track_running_stats=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_2_lin =  nn.Linear(d_model, fc_inner, bias=False)

            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(d_model, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(d_model, track_running_stats=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=40, dim=40, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")


    def forward(self, x, **kwargs):


        a = self.enc_0(x)
        v = self.enc_1(x)
        z = self.enc_2(x)
        feat_z  = z["features"]["combined"]
        feat_a = a["features"]["combined"]
        feat_v = v["features"]["combined"]
        pred_aa = a["preds"]["combined"]
        pred_vv = v["preds"]["combined"]
        pred_zz = z["preds"]["combined"]

        if self.mmcosine:
            pred_a = torch.mm(F.normalize(feat_a, dim=1),F.normalize(torch.transpose(self.fc_0_lin.weight, 0, 1), dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            pred_v = torch.mm(F.normalize(feat_v, dim=1), F.normalize(torch.transpose(self.fc_1_lin.weight, 0, 1), dim=0))
            pred_z = torch.mm(F.normalize(feat_z, dim=1), F.normalize(torch.transpose(self.fc_2_lin.weight, 0, 1), dim=0))
            pred_a = pred_a * self.mmcosine_scaling
            pred_v = pred_v * self.mmcosine_scaling
            pred_z = pred_z * self.mmcosine_scaling

            pred = pred_a + pred_v + pred_z

        elif self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(feat_a, self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 3
            pred_v = torch.matmul(feat_v, self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 3
            pred_z = torch.matmul(feat_z, self.fc_2_lin.weight.T) + self.fc_0_lin.bias / 3
            pred = pred_a + pred_v + pred_z
        else:
            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
                pred_zz = (pred_zz - pred_zz.mean()) / pred_zz.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
                pred_zz = F.softmax(pred_zz, dim=1)

            pred = pred_aa + pred_vv + pred_zz

        if self.cls_type == "film" or self.cls_type == "gated":
            pred = self.common_fc([a["features"]["combined"], v["features"]["combined"], z["features"]["combined"]])
        elif self.cls_type == "tf":
            tf_input = [a["nonaggr_features"]["combined"].permute(1,2,0),
                        v["nonaggr_features"]["combined"].permute(1,2,0),
                        z["nonaggr_features"]["combined"].permute(1,2,0)]
            pred = self.common_fc(tf_input)
        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)
        bias_method = self.args.get("bias_infusion", {"method": False}).get("method", False)
        if (bias_method == "OGM" or bias_method == "OGM_GE" or bias_method == "MSLR") and self.cls_type!="dec":
            pred_aa = pred_a
            pred_vv = pred_v
            pred_zz = pred_z


        output = {"preds":{
            "combined":pred,
            "c":pred_aa,
            "g":pred_vv,
            "f":pred_zz},
                  "features":{"c":feat_a,
                              "g":feat_v,
                              "f":feat_z}}
        return output
class ConcatClassifier_MOSEI_AGM_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_MOSEI_AGM_pre, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.precondition = args.get("precondition", False)
        self.norm_decision = args.get("norm_decision", False)

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.enc_2 = encs[2]

        self.dropout = nn.Dropout(0.3)

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_2_lin = nn.Linear(d_model, num_classes, bias=False)

            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(num_classes, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(num_classes, track_running_stats=False)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_2_lin = nn.Linear(d_model, 4096, bias=False)

            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(d_model, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(d_model, track_running_stats=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_2_lin =  nn.Linear(d_model, fc_inner, bias=False)

            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(d_model, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(d_model, track_running_stats=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )

        elif self.cls_type == "tf":

            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_2_lin = nn.Linear(d_model, num_classes, bias=False)

            self.common_fc = TF_Fusion(input_dim=40, dim=40, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")
        self.m_v_o = Modality_out()
        self.m_f_o = Modality_out()
        self.m_l_o = Modality_out()

        self.m_f = Modality_Text()
        self.m_l = Modality_Audio()
        self.m_v = Modality_Visual()
        self.m_v_o = Modality_out()
        self.m_f_o = Modality_out()
        self.m_l_o = Modality_out()

        self.scale_f = 1.0
        self.scale_v = 1.0
        self.scale_l = 1.0

        self.m_f_o.register_full_backward_hook(self.hookf)
        self.m_v_o.register_full_backward_hook(self.hookv)
        self.m_l_o.register_full_backward_hook(self.hookl)

        # if cfg.CHECKPOINT_PATH:
        #     print("We are loading from {}".format(cfg.CHECKPOINT_PATH))
        #     self.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location="cpu"))

    def hookl(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_l,

    def hookv(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_v,

    def hookf(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_f,

    def update_scale(self,coeff_c,coeff_g,coeff_f):
        self.scale_v = coeff_c
        self.scale_l = coeff_g
        self.scale_f = coeff_f


    def make_zero_batch(self, batch: Dict[str, torch.Tensor]):
        zero_input = {}
        for key in batch:
            zero_input[key] = torch.zeros_like(batch[key])
        return zero_input

    def classifier(self, features, noaggr_features, pred_aa, pred_vv, pred_zz, return_all=False, **kwargs):
        feat_a = features["c"]
        feat_v = features["g"]
        feat_z = features["flow"]
        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(feat_a, self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 3
            pred_v = torch.matmul(feat_v, self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 3
            pred_z = torch.matmul(feat_z, self.fc_2_lin.weight.T) + self.fc_0_lin.bias / 3
            preds = pred_a + pred_v + pred_z

        elif self.cls_type == "tf":
            tf_input = [noaggr_features["c"].permute(1, 2, 0),
                        noaggr_features["g"].permute(1, 2, 0),
                        noaggr_features["flow"].permute(1, 2, 0)]

            pred_aa = torch.matmul(feat_a, self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 3
            pred_vv = torch.matmul(feat_v, self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 3
            pred_zz = torch.matmul(feat_z, self.fc_2_lin.weight.T) + self.fc_0_lin.bias / 3

            preds = self.common_fc(tf_input)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            preds = self.common_fc(torch.concatenate([feat_a, feat_v, feat_z], dim=1))
        else:
            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
                pred_zz = (pred_zz - pred_zz.mean()) / pred_zz.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
                pred_zz = F.softmax(pred_zz, dim=1)

            preds = pred_aa + pred_vv + pred_zz
        if return_all:
            return preds, pred_aa, pred_vv, pred_zz
        return preds

    def forward(self, x, **kwargs):


        a = self.enc_0(x)
        v = self.enc_1(x)
        z = self.enc_2(x)
        feat_z  = z["features"]["combined"]
        feat_a = a["features"]["combined"]
        feat_v = v["features"]["combined"]
        nonaggr_feat_z  = z["nonaggr_features"]["combined"]
        nonaggr_feat_a = a["nonaggr_features"]["combined"]
        nonaggr_feat_v = v["nonaggr_features"]["combined"]
        pred_aa = a["preds"]["combined"]
        pred_vv = v["preds"]["combined"]
        pred_zz = z["preds"]["combined"]

        features = {"c": feat_a, "g": feat_v, "flow": feat_z}
        nonaggr_features = {"c": nonaggr_feat_a, "g": nonaggr_feat_v, "flow": nonaggr_feat_z}

        preds, pred_aa, pred_vv, pred_zz = self.classifier(features, nonaggr_features, pred_aa, pred_vv, pred_zz, return_all = True)


        train_flag = self.training == 'train'

        self.eval()
        with torch.no_grad():
            zero_input = self.make_zero_batch(x)

            v = self.enc_0(zero_input, return_features=True)
            l = self.enc_1(zero_input, return_features=True)
            f = self.enc_2(zero_input, return_features=True)
            video_zero_features = v["features"]["combined"]
            layout_zero_features = l["features"]["combined"]
            flow_zero_features = f["features"]["combined"]
            video_zero_noaggr_features = v["nonaggr_features"]["combined"]
            layout_zero_noaggr_features = l["nonaggr_features"]["combined"]
            flow_zero_noaggr_features = f["nonaggr_features"]["combined"]

            features = {"c": video_zero_features, "g": feat_v, "flow": feat_z}
            nonaggr_features = {"c": video_zero_noaggr_features, "g": nonaggr_feat_v, "flow": nonaggr_feat_z}
            preds_zv = self.classifier(features, nonaggr_features, pred_aa, pred_vv, pred_zz)

            features = {"c": feat_a, "g": layout_zero_features, "flow": feat_z}
            nonaggr_features = {"c": nonaggr_feat_a, "g": layout_zero_noaggr_features, "flow": nonaggr_feat_z}
            preds_zl = self.classifier(features, nonaggr_features, pred_aa, pred_vv, pred_zz)

            features = {"c": feat_a, "g": feat_v, "flow": flow_zero_features}
            nonaggr_features = {"c": nonaggr_feat_a, "g": nonaggr_feat_v, "flow": flow_zero_noaggr_features}
            preds_zf = self.classifier(features, nonaggr_features, pred_aa, pred_vv, pred_zz)

            features = {"c": video_zero_features, "g": layout_zero_features, "flow": feat_z}
            nonaggr_features = {"c": video_zero_noaggr_features, "g": layout_zero_noaggr_features, "flow": nonaggr_feat_z}
            preds_zvl = self.classifier(features, nonaggr_features, pred_aa, pred_vv, pred_zz)

            features = {"c": video_zero_features, "g": feat_v, "flow": flow_zero_features}
            nonaggr_features = {"c": video_zero_noaggr_features, "g": nonaggr_feat_v, "flow": flow_zero_noaggr_features}
            preds_zvf = self.classifier(features, nonaggr_features, pred_aa, pred_vv, pred_zz)

            features = {"c": feat_a, "g": layout_zero_features, "flow": flow_zero_features}
            nonaggr_features = {"c": nonaggr_feat_a, "g": layout_zero_noaggr_features, "flow": flow_zero_noaggr_features}
            preds_zlf = self.classifier(features, nonaggr_features,pred_aa, pred_vv, pred_zz)

            features = {"c": video_zero_features, "g": layout_zero_features, "flow": feat_z}
            nonaggr_features = {"c": video_zero_noaggr_features, "g": layout_zero_noaggr_features, "flow": nonaggr_feat_z}
            preds_zvlf = self.classifier(features, nonaggr_features, pred_aa, pred_vv, pred_zz)


        if train_flag: self.train()
        m_v_out = self.m_v_o(self.m_v(preds,
                                      preds_zv, preds_zl, preds_zf,
                                      preds_zvl, preds_zvf, preds_zlf,
                                      preds_zvlf))
        m_l_out = self.m_l_o(self.m_l(preds,
                                      preds_zv, preds_zl, preds_zf,
                                      preds_zvl, preds_zvf, preds_zlf,
                                      preds_zvlf))
        m_f_out = self.m_f_o(self.m_f(preds,
                                      preds_zv, preds_zl, preds_zf,
                                      preds_zvl, preds_zvf, preds_zlf,
                                      preds_zvlf))

        # individual marginal contribution (contain zero padding)
        m_l_mc = m_l_out - preds_zvlf / 3
        m_v_mc = m_v_out - preds_zvlf / 3
        m_f_mc = m_f_out - preds_zvlf / 3
        pred = {}
        pred.update({"both": preds})
        pred.update({"combined": m_v_out + m_l_out + m_f_out})
        pred.update({"c_mc": m_v_mc})
        pred.update({"g_mc": m_l_mc})
        pred.update({"f_mc": m_f_mc})
        pred.update({"c": m_v_out})
        pred.update({"g": m_l_out})
        pred.update({"f": m_f_out})

        output = {"preds":pred,
                  "features":{"c":feat_a,
                              "g":feat_v,
                              "f":feat_z}}
        return output

class ConcatClassifier_CREMAD_DE_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_DE_pre, self).__init__()

        self.args = args
        self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type
        self.precondition = args.get("precondition", False)

        num_classes = args.num_classes
        d_model = args.d_model

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0_0 = encs[0]
        self.enc_0_1 = copy.deepcopy(encs[0])
        self.enc_0_2 = copy.deepcopy(encs[0])
        self.enc_0_3 = copy.deepcopy(encs[0])
        self.enc_0_4 = copy.deepcopy(encs[0])

        self.enc_1_0 = encs[1]
        self.enc_1_1 = copy.deepcopy(encs[1])
        self.enc_1_2 = copy.deepcopy(encs[1])
        self.enc_1_3 = copy.deepcopy(encs[1])
        self.enc_1_4 = copy.deepcopy(encs[1])

        self.lin = nn.Sequential(nn.Dropout1d(0.2),nn.Linear(d_model*10, num_classes))

        self.fc_0_0_lin = nn.Linear(d_model, num_classes)
        self.fc_0_1_lin = nn.Linear(d_model, num_classes)
        self.fc_0_2_lin = nn.Linear(d_model, num_classes)
        self.fc_0_3_lin = nn.Linear(d_model, num_classes)
        self.fc_0_4_lin = nn.Linear(d_model, num_classes)

        self.fc_1_0_lin = nn.Linear(d_model, num_classes)
        self.fc_1_1_lin = nn.Linear(d_model, num_classes)
        self.fc_1_2_lin = nn.Linear(d_model, num_classes)
        self.fc_1_3_lin = nn.Linear(d_model, num_classes)
        self.fc_1_4_lin = nn.Linear(d_model, num_classes)

        # self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
        if self.precondition:
            self.inst_norm = nn.InstanceNorm1d(num_classes, track_running_stats=False)
        self.precond_norm_0 = nn.BatchNorm1d(512, track_running_stats=False)
        self.precond_norm_1 = nn.BatchNorm1d(512, track_running_stats=False)

    def forward(self, x, **kwargs):

        a, v, pred_a, pred_v = [], [], {}, {}
        for i in range(5):
            a.append(self.precond_norm_0(self.__getattr__("enc_0_{}".format(i))(x)["features"]["combined"]))
            v.append(self.precond_norm_1(self.__getattr__("enc_1_{}".format(i))(x)["features"]["combined"]))

        aa = torch.cat(a, dim=1)
        vv = torch.cat(v, dim=1)
        pred = self.lin(torch.cat([aa, vv], dim=1))

        return {"preds":{"combined":pred,
                         "a_0": self.fc_0_0_lin(a[0]),
                         "a_1": self.fc_0_1_lin(a[1]),
                         "a_2": self.fc_0_2_lin(a[2]),
                         "a_3": self.fc_0_3_lin(a[3]),
                         "a_4": self.fc_0_4_lin(a[4]),
                         "v_0": self.fc_1_0_lin(v[0]),
                         "v_1": self.fc_1_1_lin(v[1]),
                         "v_2": self.fc_1_2_lin(v[2]),
                         "v_3": self.fc_1_3_lin(v[3]),
                         "v_4": self.fc_1_4_lin(v[4])
                         },
                "features": {
                    "a_0": a[0],
                    "a_1": a[1],
                    "a_2": a[2],
                    "a_3": a[3],
                    "a_4": a[4],
                    "v_0": v[0],
                    "v_1": v[1],
                    "v_2": v[2],
                    "v_3": v[3],
                    "v_4": v[4]
                             }}


class ConcatClassifier_CREMAD_OGM_Decision_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_Decision_pre, self).__init__()

        self.args = args
        # self.shared_pred = args.shared_pred
        self.num_classes = args.num_classes
        self.norm_decision = args.get("norm_decision", False)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.norm_decision == "batch_norm":
            self.norm_0 = nn.BatchNorm1d(self.num_classes , track_running_stats=False)
            self.norm_1 = nn.BatchNorm1d(self.num_classes , track_running_stats=False)
        elif self.norm_decision == "instance_norm":
            self.norm_0 = nn.InstanceNorm1d(self.num_classes , track_running_stats=False)
            self.norm_1 = nn.InstanceNorm1d(self.num_classes , track_running_stats=False)
        elif self.norm_decision == "softmax":
            self.norm_0 = nn.Softmax(dim=1)
            self.norm_1 = nn.Softmax(dim=1)


    def _get_features(self, x):
        if self.enc_0.args.get("freeze_encoder", False):
            self.enc_0.eval()
        if self.enc_1.args.get("freeze_encoder", False):
            self.enc_1.eval()

        a = self.enc_0(x)
        v = self.enc_1(x)

        return a["preds"]["combined"], v["preds"]["combined"], a["features"]["combined"], v["features"]["combined"]

    def forward(self, x, **kwargs):

        pred_a, pred_v, a, v = self._get_features(x)

        if self.norm_decision == "standardization":
            pred_a = (pred_a - pred_a.mean())/pred_a.std()
            pred_v = (pred_v - pred_v.mean())/pred_v.std()
            pred = pred_a + pred_v

        elif self.norm_decision == "batch_norm" or self.norm_decision == "instance_norm":

            pred_a = self.norm_0(pred_a)
            pred_v = self.norm_1(pred_v)
            pred = pred_a + pred_v

        elif self.norm_decision == "softmax":

            pred_a = self.norm_0(pred_a)
            pred_v = self.norm_1(pred_v)
            pred = torch.nn.functional.softmax(pred_a, dim=1) + torch.nn.functional.softmax(pred_v, dim=1)
        else:
            pred = pred_a + pred_v

        if a.shape != v.shape:
            return {"preds":{"combined":pred, "c":pred_a, "g":pred_v}, "features": {"c": a, "g": v}}
        return {"preds":{"combined":pred, "c":pred_a, "g":pred_v}, "features": {"c": a, "g": v, "combined": (a + v)/2}}
class ConcatClassifier_CREMAD_PeGGrad(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_PeGGrad, self).__init__()

        self.args = args
        self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type
        self.precondition = args.get("precondition", False)

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.cls_type == "linear":
            # self.enc_0_lin = nn.Linear(d_model, num_classes, bias=False)
            # self.enc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            # self.bias_lin = nn.Parameter(torch.rand(num_classes))
            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(num_classes, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(num_classes, track_running_stats=False)

    def _get_features(self, x):

        # if self.enc_0.module.args.get("freeze_encoder", False):
        #     self.enc_0.eval()
        # if self.enc_1.module.args.get("freeze_encoder", False):
        #     self.enc_1.eval()

        a = self.enc_0(x)["features"]["combined"]
        v = self.enc_1(x)["features"]["combined"]

        return a, v

    def forward(self, x, **kwargs):

        a, v = self._get_features(x)

        if self.precondition:
            a = self.precond_norm(a)
            v = self.precond_norm(v)

        # pred_a = torch.matmul(a, self.enc_0_lin.weight.T) + self.bias_lin/2
        # pred_a = torch.matmul(a, self.enc_0_lin.weight.T)
        # pred_zero_a = torch.matmul(zero_a, self.enc_0_lin.weight.T)
        # # pred_v = torch.matmul(v, self.enc_1_lin.weight.T) + self.bias_lin/2
        # pred_v = torch.matmul(v, self.enc_1_lin.weight.T)
        # pred_zero_v = torch.matmul(zero_v, self.enc_1_lin.weight.T)

        pred_a = torch.matmul(a, self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
        pred_v = torch.matmul(v, self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2

        pred = pred_a + pred_v


        x_z = {i: torch.zeros_like(x[i]) for i in x}
        zero_a, zero_v = self._get_features(x_z)

        pred_zero_a = torch.matmul(zero_a, self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
        pred_zero_v = torch.matmul(zero_v, self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2

        pred_za = pred_zero_a + pred_v
        pred_zv = pred_a + pred_zero_v
        pred_zav = pred_zero_a + pred_zero_v


        return {"preds":{
                        "combined":pred,
                        # "combined_za": pred_za,
                        # "combined_zv": pred_zv,
                        # "combined_zav": pred_zav,
                        "c":pred_a,
                        "g":pred_v
                        },
                "features": {
                        "c": a,
                        "g": v,
                        "combined": (a + v)/2
                        }
                }


class ConcatClassifier_CREMAD_OGM_Faster_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_OGM_Faster_pre, self).__init__()

        self.args = args
        self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type
        self.precondition = args.get("precondition", False)

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(num_classes, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(num_classes, track_running_stats=False)

        elif self.cls_type == "highlynonlinear":

            self.fc_0_lin = nn.Linear(d_model, 4096)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)

            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(d_model, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(d_model, track_running_stats=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
            self.common_fc_0 = copy.deepcopy(self.common_fc)
            self.common_fc_1 = copy.deepcopy(self.common_fc)

            if not self.shared_pred:
                self.fc_0 = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Linear(fc_inner, num_classes)
                )

                self.fc_1 = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Linear(fc_inner, num_classes)
                )

        elif self.cls_type != "linear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(fc_inner, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(fc_inner, track_running_stats=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
            self.common_fc_0 = copy.deepcopy(self.common_fc)
            self.common_fc_1 = copy.deepcopy(self.common_fc)

            if not self.shared_pred:
                self.fc_0 = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, num_classes)
                )

                self.fc_1 = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, num_classes)
                )

    def _get_features(self, x):

        if self.enc_0.module.args.get("freeze_encoder", False):
            self.enc_0.eval()
        if self.enc_1.module.args.get("freeze_encoder", False):
            self.enc_1.eval()

        a = self.enc_0(x)["features"]["combined"]
        v = self.enc_1(x)["features"]["combined"]

        return a, v

    def forward(self, x, **kwargs):

        a, v = self._get_features(x)

        if self.mmcosine:
            pred_a = torch.mm(F.normalize(a, dim=1),F.normalize(torch.transpose(self.fc_0_lin.weight, 0, 1), dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            pred_v = torch.mm(F.normalize(v, dim=1), F.normalize(torch.transpose(self.fc_1_lin.weight, 0, 1), dim=0))
            pred_a = pred_a * self.mmcosine_scaling
            pred_v = pred_v * self.mmcosine_scaling
            pred = pred_a + pred_v

            if self.precondition:
                pred_a = self.precond_norm(pred_a)
                pred_v = self.precond_norm(pred_v)
        else:
            if self.precondition:
                a = self.precond_norm(a)
                v = self.precond_norm(v)

            pred_a = torch.matmul(a, self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v, self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2

            pred = pred_a + pred_v
        if self.cls_type != "linear":
            # print(pred.shape)
            pred = self.common_fc(pred)
            if self.shared_pred:
                self.common_fc_0.load_state_dict(self.common_fc.state_dict())
                self.common_fc_1.load_state_dict(self.common_fc.state_dict())
                pred_a = self.common_fc_0(pred_a)
                pred_v = self.common_fc_1(pred_v)
            else:
                pred_a = self.fc_0(pred_a)
                pred_v = self.fc_1(pred_v)
        elif self.cls_type == "tf":
            tf_input = [a["nonaggr_features"]["combined"].permute(1,2,0),
                        v["nonaggr_features"]["combined"].permute(1,2,0),
                        z["nonaggr_features"]["combined"].permute(1,2,0)]
            pred = self.common_fc(tf_input)

        return {"preds":{"combined":pred, "c":pred_a, "g":pred_v}, "features": {"c": a, "g": v, "combined": (a + v)/2}}

class ConcatClassifier_CREMAD_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_pre, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        d_model = args.d_model
        self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1
        # self.fusion_module = ConcatFusion(output_dim=n_classes)

        # self.vcaster = nn.Conv2d(9,3,1)
        # self.acaster = nn.Conv2d(1,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False) #, weights='ResNet18_Weights.DEFAULT'
        # self.audio_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False)

        # self.visual_net = resnet18(modality='visual')
        # self.audio_net = resnet18(modality='audio')


        self.enc_0 = encs[0]
        self.enc_1 = encs[1]


        if self.cls_type == "linear":
            self.fc_mod0_lin = nn.Linear(d_model, num_classes)
            self.fc_mod1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.inst_norm = nn.InstanceNorm1d(num_classes, track_running_stats=False)

        elif self.cls_type != "linear":
            self.fc_mod0_lin = nn.Linear(d_model, fc_inner)
            self.fc_mod1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.inst_norm = nn.InstanceNorm1d(fc_inner, track_running_stats=False)
            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
            if not self.shared_pred:
                self.fc_0 = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, num_classes)
                )

                self.fc_1 = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, fc_inner),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_inner, num_classes)
                )



    def _get_features(self, x):

        # a = self.audio_net(x[0].unsqueeze(dim=1))
        # a = F.adaptive_avg_pool2d(a, 1)
        # a = torch.flatten(a, 1)
        #
        # v = self.visual_net(x[1])
        # #
        # # print(v.shape)
        # B = x[1].shape[0]
        # (_, C, H, W) = v.size()
        # v = v.view(B, -1, C, H, W)
        # v = v.permute(0, 2, 1, 3, 4)
        # v = F.adaptive_avg_pool3d(v, 1)
        # v = torch.flatten(v, 1)
        #
        if self.enc_0.module.args.get("freeze_encoder", False):
            self.enc_0.eval()
        if self.enc_1.module.args.get("freeze_encoder", False):
            self.enc_1.eval()

        aud = self.enc_0(x)
        vis = self.enc_1(x)

        a = aud["features"]["combined"]
        v = vis["features"]["combined"]

        return a, v

    def forward(self, x, **kwargs):

        a, v = self._get_features(x)

        pred_a = torch.matmul(a, self.fc_mod0_lin.weight.T) + self.fc_mod0_lin.bias/2
        pred_v = torch.matmul(v, self.fc_mod1_lin.weight.T) + self.fc_mod0_lin.bias/2

        if self.args.bias_infusion.get("inst_norm", False):
            pred_a = self.inst_norm(pred_a)
            pred_v = self.inst_norm(pred_v)

        pred = pred_a + pred_v
        if self.cls_type != "linear":
            pred = self.common_fc(pred)
            if self.shared_pred:
                pred_a = self.common_fc(pred_a)
                pred_v = self.common_fc(pred_v)
            else:
                pred_a = self.fc_0(pred_a)
                pred_v = self.fc_1(pred_v)

        return {"preds":{"combined":pred, "c":pred_a, "g":pred_v}, "features": {"c": a, "g": v, "combined": (a + v)/2}}

class SumClassifier_CREMAD_linearcls(nn.Module):
    def __init__(self, args, encs):
        super(SumClassifier_CREMAD_linearcls, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1
        # self.fusion_module = ConcatFusion(output_dim=n_classes)

        # self.vcaster = nn.Conv2d(9,3,1)
        # self.acaster = nn.Conv2d(1,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False) #, weights='ResNet18_Weights.DEFAULT'
        # self.audio_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False)

        self.visual_net = resnet18(modality='visual')
        self.audio_net = resnet18(modality='audio')
        # self.vclassifier = nn.Linear(1000, n_classes)
        # self.aclassifier = nn.Linear(1000, n_classes)
        # self.classifier = nn.Linear(2000, n_classes)
        # self.classifier = nn.Linear(2000, n_classes)
        # self.classifier = nn.Sequential(
        #                 nn.Dropout(0.4),
                        # nn.Linear(2000, 64),
                        # nn.ReLU(),
                        # nn.Dropout(0.2),
                        # nn.Linear(64, n_classes)
                    # )
        if not self.args.shared_pred:
            self.fc_0 = nn.Sequential(
                nn.Linear(d_model, num_classes)
            )

            self.fc_1 = nn.Sequential(
                nn.Linear(d_model, num_classes)
            )

        self.common_fc = nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, **kwargs):

        #
        # v = self.visual_net(self.vcaster(x[1].flatten(start_dim=1, end_dim=2)))
        v = self.visual_net(x[1])
        #
        # print(v.shape)
        B = x[1].shape[0]
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        # print(v.shape)

        # vout = self.vclassifier(v)

        # B = x["spec"].shape[0]

        # a = self.audio_net(self.acaster(x[0].unsqueeze(dim=1)))
        # print(x[0].shape)
        a = self.audio_net(x[0].unsqueeze(dim=1))
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)
        # print(a.shape)
        # aout = self.aclassifier(a)
        #
        # out = self.classifier(torch.cat([v, a],dim=1))

        if self.args.shared_pred:
            pred_g = self.common_fc(v)
            pred_c = self.common_fc(a)
        else:
            pred_g = self.fc_0(a)
            pred_c = self.fc_1(v)


        pred = self.common_fc((a + v)/2)

        # out = self.classifier(v)
        return {"preds":{"combined":pred, "c":pred_c, "g":pred_g}, "features": {"c": a, "g": v, "combined": (a + v)/2}}
class SumClassifier_CREMAD(nn.Module):
    def __init__(self, args, encs):
        super(SumClassifier_CREMAD, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1
        # self.fusion_module = ConcatFusion(output_dim=n_classes)

        # self.vcaster = nn.Conv2d(9,3,1)
        # self.acaster = nn.Conv2d(1,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False) #, weights='ResNet18_Weights.DEFAULT'
        # self.audio_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False)

        self.visual_net = resnet18(modality='visual')
        self.audio_net = resnet18(modality='audio')
        # self.vclassifier = nn.Linear(1000, n_classes)
        # self.aclassifier = nn.Linear(1000, n_classes)
        # self.classifier = nn.Linear(2000, n_classes)
        # self.classifier = nn.Linear(2000, n_classes)
        # self.classifier = nn.Sequential(
        #                 nn.Dropout(0.4),
                        # nn.Linear(2000, 64),
                        # nn.ReLU(),
                        # nn.Dropout(0.2),
                        # nn.Linear(64, n_classes)
                    # )
        if not self.args.shared_pred:
            self.fc_0 = nn.Sequential(
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)

            # nn.Linear(d_model, num_classes)

            )

            self.fc_1 = nn.Sequential(
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            # nn.Linear(d_model, num_classes)

            )

        self.common_fc = nn.Sequential(
            nn.Linear(d_model, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, num_classes)
        # nn.Linear(d_model, num_classes)

        )

    def forward(self, x, **kwargs):

        #
        # v = self.visual_net(self.vcaster(x[1].flatten(start_dim=1, end_dim=2)))
        v = self.visual_net(x[1])
        #
        # print(v.shape)
        B = x[1].shape[0]
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        # print(v.shape)

        # vout = self.vclassifier(v)

        # B = x["spec"].shape[0]

        # a = self.audio_net(self.acaster(x[0].unsqueeze(dim=1)))
        # print(x[0].shape)
        a = self.audio_net(x[0].unsqueeze(dim=1))
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)
        # print(a.shape)
        # aout = self.aclassifier(a)
        #
        # out = self.classifier(torch.cat([v, a],dim=1))

        if self.args.shared_pred:
            pred_g = self.common_fc(v)
            pred_c = self.common_fc(a)
        else:
            pred_g = self.fc_0(a)
            pred_c = self.fc_1(v)


        pred = self.common_fc((a + v)/2)

        # out = self.classifier(v)
        return {"preds":{"combined":pred, "c":pred_c, "g":pred_g}, "features": {"c": a, "g": v, "combined": (a + v)/2}}
class SumClassifier_CREMAD_mmcosine(nn.Module):
    def __init__(self, args, encs):
        super(SumClassifier_CREMAD_mmcosine, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1
        # self.fusion_module = ConcatFusion(output_dim=n_classes)

        self.visual_net = resnet18(modality='visual')
        self.audio_net = resnet18(modality='audio')

        self.fc_0_lin = nn.Linear(d_model, fc_inner)
        self.fc_1_lin = nn.Linear(d_model, fc_inner)

        if not self.args.shared_pred:
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_0 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
            self.fc_1 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )

        self.common_fc = nn.Sequential(
            # nn.Linear(d_model, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, num_classes)
        )

    def forward(self, x, **kwargs):

        v = self.visual_net(x[1])
        B = x[1].shape[0]
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)

        a = self.audio_net(x[0].unsqueeze(dim=1))
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)

        pred_a = torch.mm(F.normalize(a,dim=1), F.normalize(torch.transpose(self.fc_0_lin.weight, 0, 1),dim=0)) * 10
        pred_v = torch.mm(F.normalize(v,dim=1), F.normalize(torch.transpose(self.fc_1_lin.weight, 0, 1),dim=0)) * 10

        if self.args.shared_pred:
            pred_g = self.common_fc(pred_a)
            pred_c = self.common_fc(pred_v)
        else:
            pred_g = self.fc_0(pred_a)
            pred_c = self.fc_1(pred_v)


        pred = self.common_fc((pred_a + pred_v))

        # out = self.classifier(v)
        return {"preds":{"combined":pred, "c":pred_c, "g":pred_g}, "features": {"c": a, "g": v, "combined": (a + v)/2}}
class SumClassifier_CREMAD_instancenorm(nn.Module):
    def __init__(self, args, encs):
        super(SumClassifier_CREMAD_instancenorm, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1
        # self.fusion_module = ConcatFusion(output_dim=n_classes)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        # self.visual_net = resnet18(modality='visual')
        # self.audio_net = resnet18(modality='audio')

        # self.fc_0_lin = nn.Sequential(nn.Linear(d_model, fc_inner),
                                      # nn.InstanceNorm1d(fc_inner, track_running_stats=False)
                                      # )
        # self.fc_1_lin = nn.Sequential(nn.Linear(d_model, fc_inner),
                                      # nn.InstanceNorm1d(fc_inner, track_running_stats=False)
                                      # )

        if not self.args.shared_pred:
            # self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_0 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
            self.fc_1 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )

        self.common_fc = nn.Sequential(
            nn.Linear(d_model, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, num_classes)
        )

    def forward(self, x, **kwargs):

        # v = self.enc_0(x[1])
        # B = x[1].shape[0]
        # (_, C, H, W) = v.size()
        # v = v.view(B, -1, C, H, W)
        # v = v.permute(0, 2, 1, 3, 4)
        # v = F.adaptive_avg_pool3d(v, 1)
        # v = torch.flatten(v, 1)
        #
        # a = self.enc_1(x[0].unsqueeze(dim=1))
        # a = F.adaptive_avg_pool2d(a, 1)
        # a = torch.flatten(a, 1)

        aud = self.enc_0(x)
        vis = self.enc_1(x)
        a = aud["features"]["combined"]
        v = vis["features"]["combined"]


        # v = self.fc_0_lin(v)
        # a = self.fc_1_lin(a)

        # pred_a = torch.mm(F.normalize(a,dim=1), F.normalize(torch.transpose(self.fc_0_lin.weight, 0, 1),dim=0)) * 10
        # pred_v = torch.mm(F.normalize(v,dim=1), F.normalize(torch.transpose(self.fc_1_lin.weight, 0, 1),dim=0)) * 10

        if self.args.shared_pred:
            pred_c = self.common_fc(v)
            pred_g = self.common_fc(a)
        else:
            pred_c = self.fc_0(v)
            pred_g = self.fc_1(a)


        pred = self.common_fc((a + v)/2)

        # out = self.classifier(v)
        return {"preds":{"combined":pred, "c":pred_c, "g":pred_g}, "features": {"c": a, "g": v, "combined": (a + v)/2}}
class SumClassifier_Pre_CREMAD(nn.Module):
    def __init__(self, args, encs):
        super(SumClassifier_Pre_CREMAD, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if not self.args.shared_pred:
            self.fc_0 = nn.Sequential(
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)

            # nn.Linear(d_model, num_classes)

            )

            self.fc_1 = nn.Sequential(
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            # nn.Linear(d_model, num_classes)

            )

        self.common_fc = nn.Sequential(
            nn.Linear(d_model, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, num_classes)
        # nn.Linear(d_model, num_classes)
        )

    def forward(self, x, **kwargs):

        #
        # v = self.visual_net(self.vcaster(x[1].flatten(start_dim=1, end_dim=2)))
        aud = self.enc_0(x)
        vis = self.enc_1(x)
        a = aud["features"]["combined"]
        v = vis["features"]["combined"]

        if self.args.shared_pred:
            pred_g = self.common_fc(v)
            pred_c = self.common_fc(a)
        else:
            pred_g = self.fc_0(v)
            pred_c = self.fc_1(a)


        pred = self.common_fc((a + v)/2)

        # out = self.classifier(v)
        return {"preds":{"combined":pred, "c":pred_c, "g":pred_g, "a": aud["preds"]["combined"],"v": vis["preds"]["combined"]}, "features": {"c": a, "g": v, "combined": (a + v)/2}}

