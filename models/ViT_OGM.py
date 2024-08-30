from transformers import  ViTConfig, ViTModel
import torch
import torch.nn as nn
import copy
from .backbone import resnet18
import einops
import torch.nn.functional as F

from transformers import AutoProcessor, ASTModel, ASTConfig, HubertForSequenceClassification, HubertModel, HubertConfig, AutoModelForCTC, Wav2Vec2FeatureExtractor, AutoFeatureExtractor, Wav2Vec2Config, Wav2Vec2Model
from transformers import Wav2Vec2ConformerConfig, Wav2Vec2ConformerModel
# from models.VAVL_git.VAVL.conformer.model import Conformer
from transformers import VivitModel, VivitConfig, VivitImageProcessor


class VClassifier_vit_linearcls(nn.Module):
    def __init__(self, args, encs):
        super(VClassifier_vit_linearcls, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model

        configuration = ViTConfig()

        self.visual_net = ViTModel(configuration)

        if args.get("pretrained_encoder", True):
            self.visual_net = self.visual_net.from_pretrained("google/vit-base-patch16-224-in21k", config=configuration, ignore_mismatched_sizes=True, cache_dir="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/huggingface")

        self.vclassifier =  nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, **kwargs):
        v = self.visual_net(pixel_values=einops.rearrange(x[1], "b c i h w -> (b i) c h w"))
        v_feature = einops.rearrange(v["pooler_output"], "(b i) f -> b f i", b = x[1].shape[0], i = x[1].shape[2])
        v_feature = F.adaptive_avg_pool1d(v_feature, 1)
        v_feature = torch.flatten(v_feature, 1)
        pred_v = self.vclassifier(v_feature)


        return {"preds":{"combined":pred_v}, "features":{"combined":v_feature}}

class VClassifier_VAVL_linearcls(nn.Module):
    def __init__(self, args, encs):
        super(VClassifier_VAVL_linearcls, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model

        configuration = ViTConfig()

        self.visual_net = ViTModel(configuration)

        if args.get("pretrained_encoder", True):
            self.visual_net = self.visual_net.from_pretrained("google/vit-base-patch16-224-in21k", config=configuration, ignore_mismatched_sizes=True)

        self.vclassifier =  nn.Sequential(
            nn.Linear(d_model, num_classes)
        )
        self.a_dim, self.v_dim = 1024, 1408
        self.d_v = 50
        self.hidden_2 = 512

        self.conv_1d_v = nn.Conv1d(self.v_dim, self.d_v, kernel_size=1, padding=0, bias=False)

        self.x_visual = Conformer(input_dim=self.d_v,
                                encoder_dim=self.hidden_2,
                                num_encoder_layers=3)

    def forward(self, x, **kwargs):

        # # 1-D Convolution visual/audio features
        # visual = x_vid if self.v_dim == self.d_v else self.conv_1d_v(x_vid)
        #
        # proj_x_v = visual.permute(2, 0, 1)
        # visual_feats = self.x_visual(proj_x_v)

        v = self.visual_net(pixel_values=einops.rearrange(x[1], "b c i h w -> (b i) c h w"))
        v_feature = einops.rearrange(v["pooler_output"], "(b i) f -> b f i", b = x[1].shape[0], i = x[1].shape[2])
        v_feature = F.adaptive_avg_pool1d(v_feature, 1)
        v_feature = torch.flatten(v_feature, 1)

        if "detach_enc1" in kwargs and kwargs["detach_enc1"]:
            v["pooler_output"] = v["pooler_output"].detach()
            v_feature = v_feature.detach()
        if "detach_pred" in kwargs and kwargs["detach_pred"]:
            pred_v = self.vclassifier(v_feature.detach())
        else:
            pred_v = self.vclassifier(v_feature)

        return {"preds":{"combined":pred_v}, "features":{"combined":v_feature}, "nonaggr_features": {"combined": v["pooler_output"]}}

class AClassifier_AudioSets_linearcls(nn.Module):
    def __init__(self, args, encs):
        super(AClassifier_AudioSets_linearcls, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model

        configuration = ASTConfig()
        configuration.num_mel_bins = 257
        # configuration.max_length = 1024

        # Load model directly
        # Load model directly
        from transformers import AutoProcessor, AutoModel, pipeline
        # self.pipe = pipeline("audio-classification", model="ALM/hubert-base-audioset")
        # self.processor = AutoProcessor.from_pretrained("ALM/hubert-base-audioset")
        self.audio_net = AutoModel.from_pretrained("ALM/hubert-base-audioset")

        # from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        #
        # # self.extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        # self.audio_net = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", config=configuration, ignore_mismatched_sizes=True)
        #
        # from models.LanguageBind.languagebind import LanguageBind, LanguageBindImageTokenizer, LanguageBindAudio, LanguageBindAudioProcessor, LanguageBindAudioTokenizer
        # # self.audio_net = ASTModel(configuration)
        #
        # # if args.get("pretrained", False):
        # #     self.audio_net = self.audio_net.from_pretrained(pretrained_model_name_or_path="MIT/ast-finetuned-audioset-10-10-0.4593",
        # #                                                     config=configuration, ignore_mismatched_sizes=True, cache_dir="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/huggingface")
        #
        # clip_type = {
        #     # 'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
        #     'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
        #     # 'thermal': 'LanguageBind_Thermal',
        #     # 'image': 'LanguageBind_Image',
        #     # 'depth': 'LanguageBind_Depth',
        # }
        #
        # pretrained_ckpt = 'LanguageBind/LanguageBind_Audio_FT'  # also 'LanguageBind/LanguageBind_Audio'
        # self.audio_net = LanguageBindAudio.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
        # self.tokenizer = LanguageBindAudioTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
        # # self.audio_process = LanguageBindAudioProcessor(self.audio_net.config, self.tokenizer)
        #
        # # self.audio_net = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
        # # pretrained_ckpt = f'lb203/LanguageBind_Image'
        # # self.tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt,
        # #                                                        cache_dir='./cache_dir/tokenizer_cache_dir')

        self.vclassifier =  nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, **kwargs):

        # print(x[0].shape)
        # data = self.pipe(x[0])
        # # data = self.processor(x[0], return_tensors='pt')
        # print(data)
        #
        #
        #
        # a = self.tokenizer(x[0], return_tensors="pt", padding=True, truncation=True)

        a = self.audio_net(x[2])
        a_feature = einops.rearrange(a["last_hidden_state"], "b i f -> b f i")
        a_feature = F.adaptive_avg_pool1d(a_feature, 1)
        a_feature = torch.flatten(a_feature, 1)
        pred_a = self.vclassifier(a_feature)

        return {"preds":{"combined":pred_a}, "features":{"combined":a_feature}, "nonaggr_features": {"combined": a["last_hidden_state"]}}

class AClassifier_AST_linearcls(nn.Module):
    def __init__(self, args, encs):
        super(AClassifier_AST_linearcls, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model

        configuration = ASTConfig()
        configuration.num_mel_bins = 257
        # configuration.max_length = 1024

        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        # self.extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.audio_net = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", config=configuration, ignore_mismatched_sizes=True)

        # self.audio_net = ASTModel(configuration)

        # if args.get("pretrained", False):
        #     self.audio_net = self.audio_net.from_pretrained(pretrained_model_name_or_path="MIT/ast-finetuned-audioset-10-10-0.4593",
        #                                                     config=configuration, ignore_mismatched_sizes=True, cache_dir="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/huggingface")


        self.vclassifier =  nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, **kwargs):

        a = self.audio_net(einops.rearrange(x[0], "b f t -> b t f"))
        pred_a = self.vclassifier(a["pooler_output"])

        return {"preds":{"combined":pred_a}, "features":{"combined":a["pooler_output"]}}

class AClassifier_VaVL_linearcls(nn.Module):
    def __init__(self, args, encs):
        super(AClassifier_VaVL_linearcls, self).__init__()


        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model

        real_model_name = "wav2vec2-large-robust"
        # self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/" + real_model_name, cache_dir="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/huggingface")
        if self.args.get("pretrained_encoder", True):
            self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/" + real_model_name)
            self.wav2vec_model.freeze_feature_encoder()
        else:
            #get non pretrained model
            wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/" + real_model_name)
            config = wav2vec_model.config
            del wav2vec_model
            print(config)
            self.wav2vec_model = Wav2Vec2Model(config)
        if real_model_name == "wav2vec2-large-robust":
            del self.wav2vec_model.encoder.layers[12:]
        # self.wav2vec_model.requires_grad_(False)

        self.a_dim, self.v_dim = 1024, 1408
        self.d_v = 50
        self.hidden_2 = 512

        self.conv_1d_a = nn.Conv1d(self.a_dim, self.d_v, kernel_size=1, padding=0, bias=False)


        self.audio_net = Conformer(
                            input_dim=self.d_v,
                            encoder_dim=self.hidden_2,
                            num_encoder_layers=5)

        # feature_extractor = AutoFeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")

        # c = HubertConfig()
        # c.classifier_proj_size = 512
        # c.num_labels = 28
        # self.audio_net = HubertForSequenceClassification(config=c)
        # self.audio_net = self.audio_net.from_pretrained("superb/hubert-base-superb-ks")
        # self.audio_net = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-ks")
        self.vclassifier =  nn.Sequential(
            nn.Linear(d_model, num_classes)
            # nn.Linear(self.d_v, num_classes)
        )


    def forward(self, x, **kwargs):

        # print(x[2].shape)
        # self.wav2vec_model.eval()
        # with torch.no_grad():
        #     x_in = self.wav2vec_model(x[2], attention_mask=None).last_hidden_state
        if "attention_mask_audio" in x:
            x_in = self.wav2vec_model(x[2], attention_mask=x["attention_mask_audio"]).last_hidden_state
        else:
            x_in = self.wav2vec_model(x[2]).last_hidden_state

        x_in = x_in.transpose(1, 2)

        # # 1-D Convolution visual/audio features
        audio = x_in if self.a_dim == self.d_v else self.conv_1d_a(x_in)
        #
        feat_a = audio.permute(2, 0, 1)
        #
        audio_feat = self.audio_net(feat_a)
        # # print(feat_a.shape)
        #
        feat_a = nn.AdaptiveAvgPool1d(1)(audio_feat.permute(1, 2, 0)).squeeze(2)
        # feat_a = nn.AdaptiveAvgPool1d(1)(x_in).squeeze(2)
        #

        if "detach_enc0" in kwargs and kwargs["detach_enc0"]:
            feat_a = feat_a.detach()
            audio_feat = audio_feat.detach()
        if "detach_pred" in kwargs and kwargs["detach_pred"]:
            pred_a = self.vclassifier(feat_a.detach())
        else:
            pred_a = self.vclassifier(feat_a)

        # return {"preds": {"combined": pred_a}}
        return {"preds": {"combined": pred_a}, "features": {"combined": feat_a}, "nonaggr_features": {"combined": audio_feat.permute(1,2,0)}}

class VClassifier_FacesVaVL_linearcls(nn.Module):
    def __init__(self, args, encs):
        super(VClassifier_FacesVaVL_linearcls, self).__init__()


        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model

        self.a_dim, self.v_dim = 1024, 1408
        self.d_v = 50
        self.hidden_2 = 512

        # 1D convolutional projection layers
        self.conv_1d_v = nn.Conv1d(self.v_dim, self.d_v, kernel_size=1, padding=0, bias=False)


        self.faces_net = Conformer(
                            input_dim=self.d_v,
                            encoder_dim=self.hidden_2,
                            num_encoder_layers=5)


        self.vclassifier =  nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, **kwargs):


        x_vid = x[3].transpose(1, 2)

        # 1-D Convolution visual/audio features
        visual = x_vid if self.v_dim == self.d_v else self.conv_1d_v(x_vid)

        proj_x_v = visual.permute(2, 0, 1)
        visual_feats = self.faces_net(proj_x_v)

        feat_v = nn.AdaptiveAvgPool1d(1)(visual_feats.permute(1, 2, 0)).squeeze(2)
        # feat_a = nn.AdaptiveAvgPool1d(1)(x_in).squeeze(2)
        #
        pred_v = self.vclassifier(feat_v)



        # return {"preds": {"combined": pred_a}}
        return {"preds": {"combined": pred_v}, "features": {"combined": feat_v}, "nonaggr_features": {"combined": visual_feats.permute(1,2,0)}}

class ConcatClassifier_CREMAD_VAVL_pre(nn.Module):
    def __init__(self, args, encs):
        super(ConcatClassifier_CREMAD_VAVL_pre, self).__init__()

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

        self.cross_att_va = torch.nn.MultiheadAttention(1, 1,  batch_first=True)
        self.cross_att_av = torch.nn.MultiheadAttention(1, 1,  batch_first=True)

        self.dropout = nn.Dropout(0.3)

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(512, num_classes)
            self.fc_1_lin = nn.Linear(768, num_classes, bias=False)
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

        elif self.cls_type != "linear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
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

        a = self.enc_0(x)
        v = self.enc_1(x)

        return a["features"]["combined"], v["features"]["combined"], a["preds"]["combined"], v["preds"]["combined"]

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x)

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

            # av = self.cross_att_av(self.dropout(v).unsqueeze(dim=-1), a.unsqueeze(dim=-1), a.unsqueeze(dim=-1))
            # av = av[0].squeeze()
            # va = self.cross_att_va(self.dropout(a).unsqueeze(dim=-1), v.unsqueeze(dim=-1), v.unsqueeze(dim=-1))
            # va = va[0].squeeze()



            # pred_av = torch.matmul(av, self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
            # pred_va = torch.matmul(va, self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2

            pred_a = torch.matmul(a, self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v, self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2

            pred = pred_a + pred_v

            # x_z = {i: torch.zeros_like(copy.deepcopy(x[i])).detach() for i in x}
            # zero_a, zero_v, pred_za, pred_zv = self._get_features(x_z)
            #
            # pred_zero_a = torch.matmul(zero_a, self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
            # pred_zero_v = torch.matmul(zero_v, self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2
            #
            # pred_za = pred_zero_a + pred_v
            # pred_zv = pred_a + pred_zero_v
            # pred_zav = pred_zero_a + pred_zero_v

        if self.cls_type != "linear":
            pred = self.common_fc(pred)
            if self.shared_pred:
                pred_a = self.common_fc(pred_a)
                pred_v = self.common_fc(pred_v)
            else:
                pred_a = self.fc_0(pred_a)
                pred_v = self.fc_1(pred_v)

        # return {"preds":{"combined":pred_av + pred_va,
        return {"preds":{"combined":pred,
                         # "combined_za": pred_za,
                         # "combined_zv": pred_zv,
                         # "combined_zav": pred_zav,
                         "c":pred_a,
                         # "av":pred_av,
                         # "va":pred_va,
                         "g":pred_v
                         },
                "features": {"c": a,
                             "g": v}}

class VClassifier_vivit_linearcls(nn.Module):
    def __init__(self, args, encs):
        super(VClassifier_vivit_linearcls, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model

        configuration = VivitConfig()
        configuration.num_frames=args.num_frame
        self.visual_net = VivitModel(configuration)

        if args.get("pretrained", True):
            self.visual_net = self.visual_net.from_pretrained("google/vivit-b-16x2-kinetics400", num_frames=args.num_frame, ignore_mismatched_sizes=True)

        # self.projector =  nn.Linear(d_model, 512)
        self.vclassifier =  nn.Linear(d_model, num_classes)

    def forward(self, x, **kwargs):
        # print(x[1].shape)
        # print(x["attention_mask_video"].shape)
        # einops.rearrange(x[1], "b c i h w -> b i c h w")
        v = self.visual_net(pixel_values=x[1])
        pred_v = self.vclassifier(v["pooler_output"])

        # return {"preds":{"combined":pred_v}, "features":{"combined":self.projector(v["pooler_output"])}}

        return {"preds":{"combined":pred_v}, "features":{"combined":v["pooler_output"]}, "nonaggr_features":{"combined":v["last_hidden_state"]}}
