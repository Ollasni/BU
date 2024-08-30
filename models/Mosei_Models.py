from transformers import  ViTConfig, ViTModel
import torch
import torch.nn as nn
import copy
from .backbone import resnet18
import einops
import torch.nn.functional as F

from transformers import AutoProcessor, ASTModel, ASTConfig, HubertForSequenceClassification, HubertModel, HubertConfig, AutoModelForCTC, Wav2Vec2FeatureExtractor, AutoFeatureExtractor, Wav2Vec2Config, Wav2Vec2Model
from transformers import Wav2Vec2ConformerConfig, Wav2Vec2ConformerModel
from models.VAVL_git.VAVL.conformer.model import Conformer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Mosei_A_CLS(nn.Module):
    def __init__(self, args, encs):
        super(Mosei_A_CLS, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        #write me a transformer in pytorch with batch first

        # self.tf = nn.Transformer(d_model= 74, nhead=2, num_encoder_layers=12, batch_first=True)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=74, nhead=2, batch_first=True)
        # self.tf = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # visual_enc = RNNEncoder(
        #     in_size=hp.d_vin,
        #     hidden_size=32,
        #     out_size=32,
        #     num_layers=4,
        #     dropout=0.3,
        #     bidirectional=True
        # )
        # acoustic_enc = RNNEncoder(
        #     in_size=hp.d_ain,
        #     hidden_size=32,
        #     out_size=32,
        #     num_layers=4,
        #     dropout=0.3,
        #     bidirectional=True
        # )
        self.audio_enc = RNNEncoder(
            in_size=74,
            hidden_size=64,
            out_size=64,
            num_layers=4,
            dropout=0.3,
            bidirectional=True
        )
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 74))

        # configuration = BertConfig()
        #
        # # Initializing a model (with random weights) from the bert-base-uncased style configuration
        #
        # model = BertModel(configuration)

        self.cls =  nn.Sequential(
            nn.Linear(64, num_classes)
        )

    def forward(self, x, **kwargs):

        feat = x[0]

        # feat = torch.cat([self.cls_token.repeat(feat.size(0), 1, 1), feat], dim=1)

        # feat = self.tf(feat)
        feat = einops.rearrange(feat, 'b s f -> s b f')
        feat, _ = self.audio_enc(feat, None)

        pred = self.cls(feat)

        return {"preds":{"combined":pred}, "features":{"combined":feat}}

class Mosei_V_CLS(nn.Module):
    def __init__(self, args, encs):
        super(Mosei_V_CLS, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        #write me a transformer in pytorch with batch first

        # self.tf = nn.Transformer(d_model= 74, nhead=2, num_encoder_layers=12, batch_first=True)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=74, nhead=2, batch_first=True)
        # self.tf = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # visual_enc = RNNEncoder(
        #     in_size=hp.d_vin,
        #     hidden_size=32,
        #     out_size=32,
        #     num_layers=4,
        #     dropout=0.3,
        #     bidirectional=True
        # )
        # acoustic_enc = RNNEncoder(
        #     in_size=hp.d_ain,
        #     hidden_size=32,
        #     out_size=32,
        #     num_layers=4,
        #     dropout=0.3,
        #     bidirectional=True
        # )
        self.video_enc = RNNEncoder(
            in_size=35,
            hidden_size=64,
            out_size=64,
            num_layers=4,
            dropout=0.3,
            bidirectional=True
        )
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 74))

        # configuration = BertConfig()
        #
        # # Initializing a model (with random weights) from the bert-base-uncased style configuration
        #
        # model = BertModel(configuration)

        self.cls =  nn.Sequential(
            nn.Linear(64, num_classes)
        )

    def forward(self, x, **kwargs):

        feat = x[1]

        # feat = torch.cat([self.cls_token.repeat(feat.size(0), 1, 1), feat], dim=1)

        # feat = self.tf(feat)
        feat = einops.rearrange(feat, 'b s f -> s b f')
        feat, _ = self.video_enc(feat, None)

        pred = self.cls(feat)

        return {"preds":{"combined":pred}, "features":{"combined":feat}}

class Mosei_T_CLS(nn.Module):
    def __init__(self, args, encs):
        super(Mosei_T_CLS, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        #write me a transformer in pytorch with batch first

        # self.tf = nn.Transformer(d_model= 74, nhead=2, num_encoder_layers=12, batch_first=True)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=74, nhead=2, batch_first=True)
        # self.tf = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # visual_enc = RNNEncoder(
        #     in_size=hp.d_vin,
        #     hidden_size=32,
        #     out_size=32,
        #     num_layers=4,
        #     dropout=0.3,
        #     bidirectional=True
        # )
        # acoustic_enc = RNNEncoder(
        #     in_size=hp.d_ain,
        #     hidden_size=32,
        #     out_size=32,
        #     num_layers=4,
        #     dropout=0.3,
        #     bidirectional=True
        # )
        self.text_enc = RNNEncoder(
            in_size=300,
            hidden_size=64,
            out_size=64,
            num_layers=4,
            dropout=0.3,
            bidirectional=True
        )
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 74))

        # configuration = BertConfig()
        #
        # # Initializing a model (with random weights) from the bert-base-uncased style configuration
        #
        # model = BertModel(configuration)

        self.cls =  nn.Sequential(
            nn.Linear(64, num_classes)
        )

    def forward(self, x, **kwargs):

        feat = x[2]

        # feat = torch.cat([self.cls_token.repeat(feat.size(0), 1, 1), feat], dim=1)

        # feat = self.tf(feat)
        feat = einops.rearrange(feat, 'b s f -> s b f')
        feat, _ = self.text_enc(feat, None)

        pred = self.cls(feat)

        return {"preds":{"combined":pred}, "features":{"combined":feat}}


class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear((2 if bidirectional else 1)*hidden_size, out_size)

    def forward(self, x, lengths, use_seq=False):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        # lengths = lengths.to(torch.int64)
        bs = x.size(0)
        # print('x_shape:{}'.format(x.shape))
        # print('lengths_shape:{}'.format(lengths.shape))

        # packed_sequence = pack_padded_sequence(x, lengths, enforce_sorted=False)
        # print('x shape:{}'.format(x.shape))
        # print('length shape:{}'.format(lengths.shape))
        out_pack, final_states = self.rnn(x)
        # print('out_pack_data_shape:{}'.format(out_pack.data.shape))

        if self.bidirectional:
            h = self.dropout(torch.cat((final_states[0][0],final_states[0][1]),dim=-1))
        else:
            h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        # print('h_shape:{}'.format(h.shape))

        if use_seq:
            x_sort_idx = torch.argsort(-lengths)
            x_unsort_idx = torch.argsort(x_sort_idx).long()
            # print('out_pack_shape:{}'.format(out_pack.shape))
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=True)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            return y_1, out
        else:
            return y_1, None


import torch
from torch import nn
import torch.nn.functional as F

from models.mosei_transformer import TransformerEncoder


class MULTModel(nn.Module):
    def __init__(self, args, encs):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.args = args

        # self.orig_d_l, self.orig_d_a, self.orig_d_v = 74, 35, 300
        # self.orig_d_l, self.orig_d_a, self.orig_d_v = 300, 5, 20
        self.orig_d_l, self.orig_d_a, self.orig_d_v = 300, 74, 35
        self.d_l, self.d_a, self.d_v = 40, 40, 40
        self.vonly = self.args.activate["video"] and not (self.args.activate["audio"] or self.args.activate["text"])
        self.aonly = self.args.activate["audio"] and not (self.args.activate["video"] or self.args.activate["text"])
        self.lonly = self.args.activate["text"] and not (self.args.activate["video"] or self.args.activate["audio"])

        self.vonly = True
        self.aonly = True
        self.lonly = True

        self.num_heads = 8
        self.layers = 4

        self.attn_dropout = 0
        self.attn_dropout_a = 0
        self.attn_dropout_v = 0
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.0
        self.embed_dropout = 0.25

        self.attn_mask = True

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l  # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)

        # output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=4)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=4)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=4)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, self.args.num_classes)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x, return_features=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l, x_a, x_v = x[2], x[0], x[1]




        # Project the textual/visual/audio features
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        x_a = x_a.transpose(1, 2)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_a = proj_x_a.permute(2, 0, 1)

        x_v = x_v.transpose(1, 2)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            # print(self.trans_l_mem)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return {"preds": {"combined": output}, "features": last_hs_proj}
#
class MULTModel_Uni(nn.Module):
    def __init__(self, args, encs):
        """
        Construct a MulT model.
        """
        super(MULTModel_Uni, self).__init__()
        self.args = args
        if self.args.get("dataset", False) == 'MOSI':
            self.orig_d_l, self.orig_d_a, self.orig_d_v = 300, 5, 20

        else:
            self.orig_d_l, self.orig_d_a, self.orig_d_v = 300, 74, 35
        self.d_l, self.d_a, self.d_v = 40, 40, 40
        self.vonly = self.args.activate["video"] and not (self.args.activate["audio"] or self.args.activate["text"])
        self.aonly = self.args.activate["audio"] and not (self.args.activate["video"] or self.args.activate["text"])
        self.lonly = self.args.activate["text"] and not (self.args.activate["video"] or self.args.activate["audio"])

        # self.vonly = True
        # self.aonly = True
        # self.lonly = True

        self.num_heads = 8
        self.layers = 4

        self.attn_dropout = 0.1
        self.attn_dropout_a = 0
        self.attn_dropout_v = 0
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.1
        self.embed_dropout = 0.2

        self.attn_mask = True

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l  # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)

        # output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers

        if self.vonly:
            self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
            self.trans_v_mem = self.get_network(self_type='v_mem', layers=self.layers)
        if self.lonly:
            self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
            self.trans_l_mem = self.get_network(self_type='l_mem', layers=self.layers)

        if self.aonly:
            self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
            self.trans_a_mem = self.get_network(self_type='a_mem', layers=self.layers)

        self.proj1 = nn.Linear(self.d_l, self.d_l)
        self.proj2 = nn.Linear(self.d_l, self.d_l)
        self.out_layer = nn.Linear(self.d_l, 1)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x, return_features=False, detach_pred=False, **kwargs):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """

        if self.vonly:
            x_v = x[1]
            x_v = x_v.transpose(1, 2)
            proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
            proj_x_v = proj_x_v.permute(2, 0, 1)
            h = self.trans_v_mem(proj_x_v)
            if type(h) == tuple:
                h = h[0]
            last_h = last_hs = h[-1]

        if self.aonly:
            x_a = x[0]
            x_a = x_a.transpose(1, 2)
            proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
            proj_x_a = proj_x_a.permute(2, 0, 1)

            h = self.trans_a_mem(proj_x_a)
            if type(h) == tuple:
                h = h[0]
            last_h = last_hs = h[-1]

        if self.lonly:
            x_l = x[2]
            x_l = x_l.transpose(1, 2)
            proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
            proj_x_l = proj_x_l.permute(2, 0, 1)

            h = self.trans_l_mem(proj_x_l)
            if type(h) == tuple:
                h = h[0]
            last_h = last_hs = h[-1]

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_h)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_h

        output = self.out_layer(last_hs_proj) if not detach_pred else self.out_layer(last_hs_proj.detach())
        return {"preds": {"combined": output}, "features": {"combined": last_hs_proj}, "nonaggr_features": {"combined": h}}

class All3Model_Mosei(nn.Module):
        def __init__(self, args, encs):
            super(All3Model_Mosei, self).__init__()
            self.args = copy.deepcopy(args)
            # self.pretraining_paths = self.args.get("pretraining_paths", {})
            # self.frozen_encoders = self.args.get("frozen_encoders", False)
            self.fusion = self.args.get("fusion", "ensemble")
            self.norm_preds = self.args.get("norm_preds", False)

            if self.fusion == "late":
                self.classifier = nn.Sequential(
                    nn.BatchNorm1d(self.args.d_model),
                    nn.Linear(self.args.d_model, self.args.num_classes))
            elif self.fusion == "midlate":
                self.classifier = nn.Linear(self.args.d_model, self.args.num_classes)
            elif self.fusion == "mid":
                self.classifier = nn.Linear(self.args.d_model, self.args.num_classes)

            for i in range(len(encs)):
                self.__setattr__("enc_{}".format(i), encs[i])
                # if frozen_encoders:
                #     enc = self.__getattr__("mod{}_enc".format(i))
                #
                #     for param in enc.parameters():
                #         param.requires_grad = False
        def forward(self, x, return_features=False):
            preds = {}
            features = {}

            out = self.enc_0(x, return_features=True)
            features.update({"c":out["features"]})
            preds.update({"c":out["preds"]["combined"]})

            out = self.enc_1(x, return_features=True)
            features.update({"g":out["features"]})
            preds.update({"g":out["preds"]["combined"]})

            out = self.enc_2(x, return_features=True)
            features.update({"f":out["features"]})
            preds.update({"f":out["preds"]["combined"]})

            if self.fusion =="ensemble":
                preds["combined"] = torch.sum(torch.concat([preds[i].unsqueeze(dim=0) for i in preds]), dim=0)
            else:
                preds["combined"] = self.classifier(torch.concat([features[i] for i in features], dim=1))

            return {"preds": preds, "features": features}




"""Implements the MultimodalTransformer Model. See https://github.com/yaohungt/Multimodal-Transformer for more."""
import math
import torch
import torch.nn.functional as F
from torch import nn


class MULTModel_Bench(nn.Module):
    """
    Implements the MultimodalTransformer Model.

    See https://github.com/yaohungt/Multimodal-Transformer for more.
    """

    class DefaultHyperParams():
        """Set default hyperparameters for the model."""

        # num_heads = 3
        # layers = 3
        # attn_dropout = 0.1
        # attn_dropout_modalities = [0.0] * 1000
        # relu_dropout = 0.1
        # res_dropout = 0.1
        # out_dropout = 0.0
        # embed_dropout = 0.25
        # embed_dim = 9
        # attn_mask = True
        # output_dim = 1
        # all_steps = False
        num_heads = 8
        layers = 4
        attn_dropout = 0.1
        attn_dropout_modalities = [0, 0, 0.1]
        relu_dropout = 0.1
        res_dropout = 0.1
        out_dropout = 0.1
        embed_dropout = 0.2
        embed_dim = 40
        attn_mask = True
        output_dim = 1
        all_steps = False

    def __init__(self, args, encs, hyp_params=DefaultHyperParams):
        """Construct a MulT model."""
        super().__init__()
        n_modalities = 3
        n_features = [74, 35, 300]
        # n_features = [5, 20, 300]
        self.n_modalities = n_modalities
        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_modalities = hyp_params.attn_dropout_modalities
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        self.all_steps = hyp_params.all_steps

        combined_dim = self.embed_dim * n_modalities * n_modalities

        # This is actually not a hyperparameter :-)
        output_dim = hyp_params.output_dim

        # 1. Temporal convolutional layers
        self.proj = [nn.Conv1d(n_features[i], self.embed_dim, kernel_size=1,
                               padding=0, bias=False) for i in range(n_modalities)]
        self.proj = nn.ModuleList(self.proj)

        # 2. Crossmodal Attentions
        self.trans = [nn.ModuleList([self.get_network(i, j, mem=False) for j in range(
            n_modalities)]) for i in range(n_modalities)]
        self.trans = nn.ModuleList(self.trans)

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        self.trans_mems = [self.get_network(
            i, i, mem=True, layers=3) for i in range(n_modalities)]
        self.trans_mems = nn.ModuleList(self.trans_mems)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, mod1, mod2, mem, layers=-1):
        """Create TransformerEncoder network from layer information."""
        if not mem:
            embed_dim = self.embed_dim
            attn_dropout = self.attn_dropout_modalities[mod2]
        else:
            embed_dim = self.n_modalities * self.embed_dim
            attn_dropout = self.attn_dropout

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x, return_features = False):
        """
        Apply MultModel Module to Layer Input.

        Args:
            x: layer input. Has size n_modalities * [batch_size, seq_len, n_features]
        """
        x = [x[v].permute(0, 2, 1)
             for v in x]  # n_modalities * [batch_size, n_features, seq_len]

        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        # Project the textual/visual/audio features
        proj_x = [self.proj[i](x[i]) for i in range(self.n_modalities)]
        proj_x = torch.stack(proj_x)
        # [n_modalities, seq_len, batch_size, proj]
        proj_x = proj_x.permute(0, 3, 1, 2)

        hs = []
        last_hs = []
        for i in range(self.n_modalities):
            h = []
            for j in range(self.n_modalities):
                h.append(self.trans[i][j](proj_x[i], proj_x[j], proj_x[j]))
            h = torch.cat(h, dim=2)
            h = self.trans_mems[i](h)
            # if type(h) == tuple:
            #     h = h[0]
            if self.all_steps:
                hs.append(h)
            else:
                last_hs.append(h[-1])

        if self.all_steps:
            out = torch.cat(hs, dim=2)  # [seq_len, batch_size, out_features]
            out = out.permute(1, 0, 2)  # [batch_size, seq_len, out_features]
        else:
            out = torch.cat(last_hs, dim=1)

        # A residual block
        out_proj = self.proj2(
            F.dropout(F.relu(self.proj1(out)), p=self.out_dropout, training=self.training))
        out_proj += out

        out = self.out_layer(out_proj)

        return {"preds": {"combined": out}, "features": out_proj}

        # return out

class MULTModel_Bench_Uni(nn.Module):
    """
    Implements the MultimodalTransformer Model.

    See https://github.com/yaohungt/Multimodal-Transformer for more.
    """

    class DefaultHyperParams():
        """Set default hyperparameters for the model."""

        # num_heads = 3
        # layers = 3
        # attn_dropout = 0.1
        # attn_dropout_modalities = [0.0] * 1000
        # relu_dropout = 0.1
        # res_dropout = 0.1
        # out_dropout = 0.0
        # embed_dropout = 0.25
        # embed_dim = 9
        # attn_mask = True
        # output_dim = 1
        # all_steps = False
        num_heads = 8
        layers = 4
        attn_dropout = 0.1
        attn_dropout_modalities = [0, 0, 0.1]
        relu_dropout = 0.1
        res_dropout = 0.1
        out_dropout = 0.1
        embed_dropout = 0.2
        embed_dim = 40
        attn_mask = True
        output_dim = 1
        all_steps = False

    def __init__(self, args, encs, hyp_params=DefaultHyperParams):
        """Construct a MulT model."""
        super().__init__()
        n_modalities = 3
        n_features = [74, 35, 300]
        # n_features = [5, 20, 300]
        self.n_modalities = n_modalities
        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_modalities = hyp_params.attn_dropout_modalities
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        self.all_steps = hyp_params.all_steps

        combined_dim = self.embed_dim * n_modalities * n_modalities

        # This is actually not a hyperparameter :-)
        output_dim = hyp_params.output_dim

        # 1. Temporal convolutional layers
        self.proj = [nn.Conv1d(n_features[i], self.embed_dim, kernel_size=1,
                               padding=0, bias=False) for i in range(n_modalities)]
        self.proj = nn.ModuleList(self.proj)

        # 2. Crossmodal Attentions
        self.trans = [nn.ModuleList([self.get_network(i, j, mem=False) for j in range(
            n_modalities)]) for i in range(n_modalities)]
        self.trans = nn.ModuleList(self.trans)

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        self.trans_mems = [self.get_network(
            i, i, mem=True, layers=3) for i in range(n_modalities)]
        self.trans_mems = nn.ModuleList(self.trans_mems)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, mod1, mod2, mem, layers=-1):
        """Create TransformerEncoder network from layer information."""
        if not mem:
            embed_dim = self.embed_dim
            attn_dropout = self.attn_dropout_modalities[mod2]
        else:
            embed_dim = self.n_modalities * self.embed_dim
            attn_dropout = self.attn_dropout

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x, return_features = False):
        """
        Apply MultModel Module to Layer Input.

        Args:
            x: layer input. Has size n_modalities * [batch_size, seq_len, n_features]
        """



        x = [x[v].permute(0, 2, 1)
             for v in x]  # n_modalities * [batch_size, n_features, seq_len]

        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        # Project the textual/visual/audio features
        proj_x = [self.proj[i](x[i]) for i in range(self.n_modalities)]
        proj_x = torch.stack(proj_x)
        # [n_modalities, seq_len, batch_size, proj]
        proj_x = proj_x.permute(0, 3, 1, 2)

        hs = []
        last_hs = []
        for i in range(self.n_modalities):
            h = []
            for j in range(self.n_modalities):
                h.append(self.trans[i][j](proj_x[i], proj_x[j], proj_x[j]))
            h = torch.cat(h, dim=2)
            h = self.trans_mems[i](h)
            # if type(h) == tuple:
            #     h = h[0]
            if self.all_steps:
                hs.append(h)
            else:
                last_hs.append(h[-1])

        if self.all_steps:
            out = torch.cat(hs, dim=2)  # [seq_len, batch_size, out_features]
            out = out.permute(1, 0, 2)  # [batch_size, seq_len, out_features]
        else:
            out = torch.cat(last_hs, dim=1)

        # A residual block
        out_proj = self.proj2(
            F.dropout(F.relu(self.proj1(out)), p=self.out_dropout, training=self.training))
        out_proj += out

        out = self.out_layer(out_proj)

        return {"preds": {"combined": out}, "features": out_proj}

        # return out


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers.

    Each layer is a :class:`TransformerEncoderLayer`.

    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        """Initialize Transformer Encoder.

        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of heads
            layers (int): Number of layers
            attn_dropout (float, optional): Probability of dropout in attention mechanism. Defaults to 0.0.
            relu_dropout (float, optional): Probability of dropout after ReLU. Defaults to 0.0.
            res_dropout (float, optional): Probability of dropout in residual layer. Defaults to 0.0.
            embed_dropout (float, optional): Probability of dropout in embedding layer. Defaults to 0.0.
            attn_mask (bool, optional): Whether to apply a mask to the attention or not. Defaults to False.
        """
        super().__init__()
        self.dropout = embed_dropout  # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for _ in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        """
        Apply Transformer Encoder to layer input.

        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            # Add positional embedding
            x += self.embed_positions(x_in.transpose(0, 1)
                                      [:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                # Add positional embedding
                x_k += self.embed_positions(x_in_k.transpose(0, 1)
                                            [:, :, 0]).transpose(0, 1)
                # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)
                                            [:, :, 0]).transpose(0, 1)
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """Implements encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        """Instantiate TransformerEncoderLayer Module.

        Args:
            embed_dim (int): Embedding dimension
            num_heads (int, optional): Number of heads. Defaults to 4.
            attn_dropout (float, optional): Dropout for attention mechanism. Defaults to 0.1.
            relu_dropout (float, optional): Dropout after ReLU. Defaults to 0.1.
            res_dropout (float, optional): Dropout after residual layer. Defaults to 0.1.
            attn_mask (bool, optional): Whether to apply an attention mask or not. Defaults to False.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        # The "Add & Norm" part in the paper
        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList(
            [LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Apply TransformerEncoderLayer to Layer Input.

        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self._maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self._maybe_layer_norm(0, x_k, before=True)
            x_v = self._maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self._maybe_layer_norm(0, x, after=True)

        residual = x
        x = self._maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self._maybe_layer_norm(1, x, after=True)
        return x

    def _maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    """Generate buffered future mask.

    Args:
        tensor (torch.Tensor): Tensor to initialize mask from.
        tensor2 (torch.Tensor, optional): Tensor to initialize target mask from. Defaults to None.

    Returns:
        torch.Tensor: Buffered future mask.
    """
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(
        torch.ones(dim1, dim2)), 1 + abs(dim2 - dim1))
    if tensor.is_cuda:
        future_mask = future_mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    """Generate Linear Layer with given parameters and Xavier initialization.

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool, optional): Whether to include a bias term or not. Defaults to True.

    Returns:
        nn.Module: Initialized Linear Module.
    """
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    """Generate LayerNorm Layer with given parameters.

    Args:
        embedding_dim (int): Embedding dimension

    Returns:
        nn.Module: Initialized LayerNorm Module
    """
    m = nn.LayerNorm(embedding_dim)
    return m


def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).

    Args:
        tensor (torch.Tensor): Tensor to generate padding on.
        padding_idx (int): Position numbers start at padding_idx + 1
        left_pad (bool): Whether to pad from the left or from the right.

    Returns:
        torch.Tensor: Padded output
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device()
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name, getattr(
        make_positions, buf_name).type_as(tensor))
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos,
                     out=getattr(make_positions, buf_name))
    mask = tensor.ne(padding_idx)
    positions = getattr(make_positions, buf_name)[
                :tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - \
                    mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """
    This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0):
        """Instantiate SinusoidalPositionalEmbedding Module.

        Args:
            embedding_dim (int): Embedding dimension
            padding_idx (int, optional): Padding index. Defaults to 0.
            left_pad (int, optional): Whether to pad from the left or not. Defaults to 0.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        # device --> actual weight; due to nn.DataParallel :-(
        self.weights = dict()
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Apply PositionalEncodings to Input.

        Input is expected to be of size [bsz x seqlen].

        Args:
            input (torch.Tensor): Layer input

        Returns:
            torch.Tensor: Layer output
        """
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # recompute/expand embeddings if needed
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights[device].index_select(0, positions.reshape(-1)).reshape((bsz, seq_len, -1)).detach()


