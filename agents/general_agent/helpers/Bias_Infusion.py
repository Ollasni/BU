import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import wandb
from collections import defaultdict
from models.OGM_Model import *


def pick_bias_infuser(agent):
    method = agent.config.model.args.get("bias_infusion",{}).get("method", False)
    if method == "OGM":
        bi = Bias_Infusion_OGM(agent)
    elif method == "OGM_GE":
        bi = Bias_Infusion_OGM(agent)
    elif method == "OGM-Mine":
        bi = Bias_Infusion_MLB(agent)
    elif method == "OGM-Mine_3d":
        bi = Bias_Infusion_MLB_3d(agent)
    elif method == "OGM-Mine_3d_Reg":
        bi = Bias_Infusion_MLB_3d_Reg(agent)
    elif method == "AGM":
        bi = Bias_Infusion_AGM(agent)
    elif method == "AGM_3mod":
        bi = Bias_Infusion_AGM_3mod(agent)
    elif method == "AGM_3mod_reg":
        bi = Bias_Infusion_AGM_3mod_reg(agent)
    elif method == "Prototype":
        bi = Bias_Infusion_Prototype(agent)
    elif method == "MSLR":
        bi = Bias_Infusion_MSLR(agent)
    else:
        bi = General_Bias_Infusion(agent)
    return bi

class General_Bias_Infusion():
    def __init__(self, agent):
        self.agent = agent

        super(General_Bias_Infusion, self).__init__()

    def before_backward(self, total, output_losses, **kwargs):
        return total, output_losses, False

    def on_backward_end(self, **kwargs):
        return

    def on_epoch_begin(self, **kwargs):
        pass

    def plot_bias(self, **kwargs):
        pass

class Bias_Infusion_Prototype(General_Bias_Infusion):
    """
    NGMine = Norm-Gradient my version
    """
    def __init__(self, agent):
        super(Bias_Infusion_Prototype, self).__init__(agent)

        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.agent.logs["ratio_logs"] = defaultdict(list)
        self.agent.logs["prototypes"] = {}
        self.losses = []
        self.momentum_coef = self.agent.config.model.args.bias_infusion.get("momentum_coef",0.0)
        self.alpha = self.agent.config.model.args.bias_infusion.alpha
        self.starting_epoch = self.agent.config.model.args.bias_infusion.get("starting_epoch", 0)
        self.ending_epoch = self.agent.config.model.args.bias_infusion.get("ending_epoch", 1500)

        self.bn_c = nn.BatchNorm1d(512, affine=False).cuda()
        self.bn_g = nn.BatchNorm1d(512, affine=False).cuda()

    def on_epoch_begin(self, current_epoch):
        self._calculate_prototype(current_epoch)

    def _calculate_prototype(self, current_epoch):

        audio_prototypes = torch.zeros(self.agent.config.model.args.num_classes, self.agent.config.model.args.d_model).to(self.agent.device)
        visual_prototypes = torch.zeros(self.agent.config.model.args.num_classes, self.agent.config.model.args.d_model).to(self.agent.device)
        count_class = [0 for _ in range(self.agent.config.model.args.num_classes)]

        # calculate prototype
        self.agent.model.eval()
        with torch.no_grad():
            sample_count = 0
            all_num = len(self.agent.data_loader.train_loader)
            for step, served_dict in enumerate(self.agent.data_loader.train_loader):
                for i in served_dict["data"]:
                    served_dict["data"][i] = served_dict["data"][i].to(self.agent.accelerator.device).float()  # B x 257 x 1004
                label = served_dict["label"].to(self.agent.accelerator.device)  # B

                out = self.agent.model(served_dict["data"], return_features=True)  # gray colored
                out["features"]["c"] = self.bn_c(out["features"]["c"])
                out["features"]["g"] = self.bn_g(out["features"]["g"])
                for c, l in enumerate(label):
                    l = l.long()
                    count_class[l] += 1
                    audio_prototypes[l, :] += out["features"]["c"].flatten(start_dim=1)[c, :]
                    visual_prototypes[l, :] += out["features"]["g"].flatten(start_dim=1)[c, :]

                if sample_count >= all_num // 10 and current_epoch!=0:
                    break
                sample_count +=1

        for c in range(audio_prototypes.shape[0]):
            audio_prototypes[c, :] /= count_class[c]
            visual_prototypes[c, :] /= count_class[c]

        if current_epoch <= 0:
            self.agent.logs["prototypes"]["a_proto"] = audio_prototypes
            self.agent.logs["prototypes"]["v_proto"] = visual_prototypes
        else:
            if self.agent.logs["prototypes"]["a_proto"].device != audio_prototypes.device:
                self.agent.logs["prototypes"]["a_proto"] = self.agent.logs["prototypes"]["a_proto"].to(audio_prototypes.device)
            if self.agent.logs["prototypes"]["v_proto"].device != visual_prototypes.device:
                self.agent.logs["prototypes"]["v_proto"] = self.agent.logs["prototypes"]["v_proto"].to(visual_prototypes.device)
            self.agent.logs["prototypes"]["a_proto"] = (1 - self.momentum_coef) * audio_prototypes + self.momentum_coef * self.agent.logs["prototypes"]["a_proto"]
            self.agent.logs["prototypes"]["v_proto"] = (1 - self.momentum_coef) * visual_prototypes + self.momentum_coef * self.agent.logs["prototypes"]["v_proto"]

        if torch.isnan(self.agent.logs["prototypes"]["a_proto"]).any():
            raise ValueError("nan in a_proto")
        if torch.isnan(self.agent.logs["prototypes"]["v_proto"]).any():
            raise ValueError("nan in v_proto")

    def before_backward(self, total, output_losses, w_loss, loss_fun, data, label, output, **kwargs):
        def EU_dist(x1, x2):
            d_matrix = torch.zeros(x1.shape[0], x2.shape[0]).to(x1.device)
            for i in range(x1.shape[0]):
                for j in range(x2.shape[0]):
                    d = torch.sqrt(torch.dot((x1[i] - x2[j]), (x1[i] - x2[j])))
                    d_matrix[i, j] = d
            return d_matrix

        # print(output["features"]["c"].sum(dim=1))
        # print(output["features"]["g"].sum(dim=1))
        # print(self.agent.logs["prototypes"]["a_proto"].sum(dim=1))
        # print(self.agent.logs["prototypes"]["v_proto"].sum(dim=1))
        audio_sim = -EU_dist(self.bn_c(output["features"]["c"]), self.agent.logs["prototypes"]["a_proto"])  # B x n_class
        visual_sim = -EU_dist(self.bn_g(output["features"]["g"]), self.agent.logs["prototypes"]["v_proto"])

        output["preds"]["proto_a"] = audio_sim
        output["preds"]["proto_v"] = visual_sim
        # print(audio_sim)
        # print(visual_sim)

        score_a_p = sum([self.softmax(audio_sim)[i][label[i]] for i in range(audio_sim.size(0))])
        score_v_p = sum([self.softmax(visual_sim)[i][label[i]] for i in range(visual_sim.size(0))])
        # print(score_a_p, score_v_p)
        ratio_a_p = score_a_p / (score_v_p + 0.00001)

        if torch.isnan(score_a_p) or torch.isnan(score_v_p) or torch.isnan(ratio_a_p):
            raise ValueError("Nan value detected")
        # score_v = sum([self.softmax(output["preds"]["g"])[i][label[i]] for i in range(output["preds"]["g"].size(0))])
        # score_a = sum([self.softmax(output["preds"]["c"])[i][label[i]] for i in range(output["preds"]["c"].size(0))])
        # ratio_a = score_a / score_v

        loss_proto_a = self.criterion(audio_sim, label)
        loss_proto_v = self.criterion(visual_sim, label)


        if ratio_a_p > 1:
            beta = 0  # audio coef
            lam = 1 * self.alpha  # visual coef
        elif ratio_a_p < 1:
            beta = 1 * self.alpha
            lam = 0
        else:
            beta = 0
            lam = 0

        if self.starting_epoch <= self.agent.logs["current_epoch"] <= self.ending_epoch:
            total = self.criterion(output["preds"]["combined"], label) + beta * loss_proto_a + lam * loss_proto_v
        else:
            total = self.criterion(output["preds"]["combined"], label)

        output_losses['proto_a'] = beta * loss_proto_a
        output_losses['proto_v'] = lam * loss_proto_v

        self.agent.logs["ratio_logs"]["ratio_PMR"].append(ratio_a_p.detach().cpu().numpy())
        self.agent.logs["ratio_logs"]["coeff_color"].append(beta)
        self.agent.logs["ratio_logs"]["coeff_gray"].append(lam)

        wandb.log({"ratio": {"ratio_PMR": ratio_a_p.detach().cpu().numpy(), "coeff_a": beta, "coeff_v": lam}})

        return total, output_losses, False
class Bias_Infusion_OGM(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_OGM, self).__init__(agent)
        logging.info("Bias Infusion OGM is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.agent.logs["ratio_logs"] = {
            "ratio_color":[],
            "ratio_gray":[],
            "coeff_color":[],
            "coeff_gray":[],
        }

    def on_backward_end(self, label, out_color, out_gray):

        score_0 = sum([self.softmax(out_color)[i][label[i]] for i in range(out_color.size(0))]).detach()
        score_1 = sum([self.softmax(out_gray)[i][label[i]] for i in range(out_gray.size(0))]).detach()

        ratio_0 = score_0 / score_1
        ratio_1 = 1 / ratio_0


        """
        Below is the Eq.(10) in our CVPR paper:
                1 - tanh(alpha * rho_t_u), if rho_t_u > 1
        k_t_u = 1,                         else
        
        coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
        """

        if ratio_0 > 1:
            coeff_0 = 1 - self.tanh(
                self.agent.config.model.args.bias_infusion.alpha * self.relu(ratio_0)).cpu().numpy()
            coeff_1 = 1
        else:
            coeff_1 = 1 - self.tanh(
                    self.agent.config.model.args.bias_infusion.alpha * self.relu(ratio_1)).cpu().numpy()
            coeff_0 = 1

        self.agent.logs["ratio_logs"]["ratio_gray"].append(ratio_1.cpu().numpy())
        self.agent.logs["ratio_logs"]["ratio_color"].append(ratio_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["coeff_color"].append(coeff_0)
        self.agent.logs["ratio_logs"]["coeff_gray"].append(coeff_1)

        wandb_output = {
            "ratio": {"ratio_gray": ratio_1.cpu().numpy(),"ratio_color":ratio_0.cpu().numpy(),
                      "coeff_color":coeff_0,"coeff_gray":coeff_1}
        }

        wandb.log(wandb_output)

        self._equalize_gradients(coeff_0=coeff_0, coeff_1=coeff_1)

    def _equalize_gradients(self, coeff_0=1, coeff_1=1):
        if not self.agent.config.model.args.bias_infusion.use: return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs["current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            for name, parms in self.agent.model.named_parameters():
                if parms.grad is None: continue
                if "mod0" in name or "enc_0" in name:
                    if self.agent.config.model.args.bias_infusion.method == 'OGM_GE':  # bug fixed
                        parms.grad = parms.grad * coeff_0 + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    elif self.agent.config.model.args.bias_infusion.method == 'OGM':
                        parms.grad *= coeff_0
                    elif self.agent.config.model.args.bias_infusion.method == 'Acc':
                        parms.grad *= coeff_0
                if "mod1" in name or "enc_1" in name:
                    if self.agent.config.model.args.bias_infusion.method == 'OGM_GE':  # bug fixed
                        parms.grad = parms.grad * coeff_1 + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    elif self.agent.config.model.args.bias_infusion.method == 'OGM':
                        parms.grad *= coeff_1
                    elif self.agent.config.model.args.bias_infusion.method == 'Acc':
                        parms.grad *= coeff_1

    def plot_bias(self):

        if not self.agent.config.model.args.bias_infusion.get("plot",False): return

        this_ratio = {ratio: np.array([self.agent.logs["ratio_logs"][i][ratio] for i in self.agent.logs["ratio_logs"]]).flatten()
                      for ratio in self.agent.logs["ratio_logs"][self.agent.logs["current_step"]]}

        plt.figure()
        plt.plot(np.array(this_ratio["ratio_color"]), label="Ratio Color")
        plt.plot(np.array(this_ratio["ratio_gray"]), label="Ratio Gray")
        plt.legend()
        plt.xlabel("Opt Steps")
        plt.ylabel("Ratio Value")
        plt.title("Ratio Color/Gray {:.2f}".format(np.array(this_ratio["ratio_color"]).mean()))
        plt.show()

        plt.figure()
        plt.plot(np.array(this_ratio["coeff_gray"]), label="Gray")
        plt.plot(np.array(this_ratio["coeff_color"]), label="Color")
        plt.legend()
        plt.xlabel("Opt Steps")
        plt.ylabel("Coeff Value")
        plt.title("Coeff Color {:.2f} - Gray {:.2f}".format(np.array(this_ratio["coeff_color"]).mean(), np.array(this_ratio["coeff_gray"]).mean()))
        plt.show()
class Bias_Infusion_MLB(General_Bias_Infusion):
    def __init__(self, agent):
        super().__init__(agent)
        logging.info("Bias Infusion OGM-Mine is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.agent.logs["ratio_logs"] = {
            "ratio_audiodivvideo":[],
            "coeff_audio":[],
            "coeff_video":[],
        }
        self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", False)
        # self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", "whatev")
        self.alpha = self.agent.config.model.args.bias_infusion.alpha
        self.tanh_mode = self.agent.config.model.args.bias_infusion.get("tanh_mode", False)
        self.tanh_mode_beta = self.agent.config.model.args.bias_infusion.get("tanh_mode_beta", False)

        if self.tanh_mode == 2:
            print("Tanh mode 2 is being employed with beta {}".format(self.tanh_mode_beta))

    # def before_backward(self, total, output_losses, **kwargs):
    #
    #     opt_done = False
    #     if self.balance_mode == "balance_only_multi":
    #         (output_losses["ce_loss_combined"]).backward(retain_graph=True)
    #         # (output_losses["ce_loss_combined"] + self.output_losses["ce_loss_c"] + self.output_losses["ce_loss_g"]).backward(retain_graph=True)
    #         self.output_losses = output_losses
    #         opt_done = True
    #
    #     return total, output_losses, opt_done

    def on_backward_end(self, label, out_color, out_gray):

        score_0 = torch.mean(torch.stack([self.softmax(out_color)[i][label[i]] for i in range(out_color.size(0))])).detach()
        score_1 = torch.mean(torch.stack([self.softmax(out_gray)[i][label[i]] for i in range(out_gray.size(0))])).detach()
        # score_1 = sum([self.softmax(out_gray)[i][label[i]] for i in range(out_gray.size(0))]).detach()

        ratio_0 = score_1 / score_0
        ratio_1 = 1 / ratio_0

        coeff_0 = 1 + self.tanh(self.alpha * (ratio_0-1))
        coeff_1 = 1 + self.tanh(self.alpha * (ratio_1-1))
        if self.tanh_mode == 2:
            if ratio_0 > 1:
                coeff_0 = 1 + (self.tanh_mode_beta-1)*self.tanh(self.alpha * (ratio_0 - 1))
            else:
                coeff_0 = 1 + self.tanh(self.alpha * (ratio_0 - 1))
            if ratio_1 > 1:
                coeff_1 = 1 + (self.tanh_mode_beta-1)*self.tanh(self.alpha * (ratio_1 - 1))
            else:
                coeff_1 = 1 + self.tanh(self.alpha * (ratio_1 - 1))

        # print("Score 0: {} Score 1: {} Coeff 0: {} Coeff 1: {}".format(score_0, score_1, coeff_0, coeff_1))
        # coeff_0 = coeff_0.cpu().numpy()
        # coeff_1 = coeff_1.cpu().numpy()
        # print("Score 0: {} Score 1: {} Coeff 0: {} Coeff 1: {}".format(score_0, score_1, coeff_0, coeff_1))
        # if ratio_0 > 1:
        #     if self.agent.config.model.args.bias_infusion.method == 'OGM-Mine':
        #         coeff_0 = 1 - self.tanh(
        #             self.agent.config.model.args.bias_infusion.alpha * self.relu(ratio_0)).cpu().numpy()
        #         coeff_1 = 1
        #     elif self.agent.config.model.args.bias_infusion.method == 'Acc':
        #         coeff_0 = 1
        #         coeff_1 = 1 + self.tanh(self.agent.config.model.args.bias_infusion.alpha * self.relu(ratio_0)).cpu().numpy()
        #
        # else:
        #     if self.agent.config.model.args.bias_infusion.method == 'OGM-Mine':
        #         coeff_1 = 1 - self.tanh(
        #             self.agent.config.model.args.bias_infusion.alpha * self.relu(ratio_1)).cpu().numpy()
        #         coeff_0 = 1
        #     elif self.agent.config.model.args.bias_infusion.method == 'Acc':
        #         coeff_1 = 1
        #         coeff_0 = 1 + self.tanh(self.agent.config.model.args.bias_infusion.alpha * self.relu(ratio_1)).cpu().numpy()

        self.agent.logs["ratio_logs"]["ratio_audiodivvideo"].append(ratio_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["coeff_audio"].append(coeff_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["coeff_video"].append(coeff_1.cpu().numpy())

        wandb_output = {
            "ratio": {"ratio_0": ratio_0.cpu().numpy(),
                      "ratio_1": ratio_1.cpu().numpy(),
                      "coeff_0":coeff_0.cpu().numpy(),
                      "coeff_1":coeff_1.cpu().numpy()}
        }

        wandb.log(wandb_output)

        self._equalize_gradients(coeff_0=coeff_0, coeff_1=coeff_1)

        # if self.balance_mode == "balance_only_multi":
        #     (self.output_losses["ce_loss_c"] + self.output_losses["ce_loss_g"]).backward()
        #     del self.output_losses
        #     self.agent.optimizer.step()


    def _equalize_gradients(self, coeff_0=1, coeff_1=1):
        if not self.agent.config.model.args.bias_infusion.use: return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs["current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            for name, parms in self.agent.model.named_parameters():
                if parms.grad is None: continue
                if "mod0" in name or "enc_0" in name or "fc_0_lin.weight" in name:
                    parms.grad *= coeff_0
                if "mod1" in name or "enc_1" in name or "fc_1_lin.weight" in name:
                    parms.grad *= coeff_1
class Bias_Infusion_MLB_3d(General_Bias_Infusion):
    def __init__(self, agent):
        super().__init__(agent)
        logging.info("Bias Infusion OGM-Mine is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.agent.logs["ratio_logs"] = {
            "ratio_0":[],
            "ratio_1":[],
            "ratio_2":[],
            "coeff_v":[],
            "coeff_l":[],
            "coeff_f":[]
        }
        self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", False)
        # self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", "whatev")
        self.alpha = self.agent.config.model.args.bias_infusion.alpha
        self.tanh_mode = self.agent.config.model.args.bias_infusion.get("tanh_mode", False)
        self.tanh_mode_beta = self.agent.config.model.args.bias_infusion.get("tanh_mode_beta", False)

        if self.tanh_mode == 2:
            print("Tanh mode 2 is being employed with beta {}".format(self.tanh_mode_beta))



    # def before_backward(self, total, output_losses, **kwargs):
    #
    #     opt_done = False
    #     if self.balance_mode == "balance_only_multi":
    #         (output_losses["ce_loss_combined"]).backward(retain_graph=True)
    #         # (output_losses["ce_loss_combined"] + self.output_losses["ce_loss_c"] + self.output_losses["ce_loss_g"]).backward(retain_graph=True)
    #         self.output_losses = output_losses
    #         opt_done = True
    #
    #     return total, output_losses, opt_done

    def on_backward_end(self, label, out_color, out_gray, out_f):

        score_0 = torch.mean(torch.stack([self.softmax(out_color)[i][label[i]] for i in range(out_color.size(0))])).detach()
        score_1 = torch.mean(torch.stack([self.softmax(out_gray)[i][label[i]] for i in range(out_gray.size(0))])).detach()
        score_2 = torch.mean(torch.stack([self.softmax(out_f)[i][label[i]] for i in range(out_f.size(0))])).detach()

        # score_1 = sum([self.softmax(out_gray)[i][label[i]] for i in range(out_gray.size(0))]).detach()

        ratio_0 =  ((score_1 + score_2)/2)/score_0
        ratio_1 =  ((score_0 + score_2)/2)/score_1
        ratio_2 =  ((score_0 + score_1)/2)/score_2

        coeff_0 = 1 + self.tanh(self.alpha * (ratio_0-1)).cpu().numpy()
        coeff_1 = 1 + self.tanh(self.alpha * (ratio_1-1)).cpu().numpy()
        coeff_2 = 1 + self.tanh(self.alpha * (ratio_2-1)).cpu().numpy()

        if self.tanh_mode == 2:
            if ratio_0 > 1:
                coeff_0 = 1 + (self.tanh_mode_beta-1)*self.tanh(self.alpha * (ratio_0 - 1))
            else:
                coeff_0 = 1 + self.tanh(self.alpha * (ratio_0 - 1))
            if ratio_1 > 1:
                coeff_1 = 1 + (self.tanh_mode_beta-1)*self.tanh(self.alpha * (ratio_1 - 1))
            else:
                coeff_1 = 1 + self.tanh(self.alpha * (ratio_1 - 1))
            if ratio_2 > 1:
                coeff_2 = 1 + (self.tanh_mode_beta-1)*self.tanh(self.alpha * (ratio_2 - 1))
            else:
                coeff_2 = 1 + self.tanh(self.alpha * (ratio_2 - 1))

        self.agent.logs["ratio_logs"]["ratio_0"].append(ratio_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["ratio_1"].append(ratio_1.cpu().numpy())
        self.agent.logs["ratio_logs"]["ratio_2"].append(ratio_2.cpu().numpy())
        self.agent.logs["ratio_logs"]["coeff_v"].append(coeff_0)
        self.agent.logs["ratio_logs"]["coeff_l"].append(coeff_1)
        self.agent.logs["ratio_logs"]["coeff_f"].append(coeff_2)

        wandb_output = {
            "ratio": {"ratio_0": ratio_0.cpu().numpy(),
                      "ratio_1": ratio_1.cpu().numpy(),
                        "ratio_2": ratio_2.cpu().numpy(),
                          "coeff_v":coeff_0,
                          "coeff_l":coeff_1,
                          "coeff_f":coeff_2
        }}

        wandb.log(wandb_output, step=self.agent.logs["current_step"])

        self._equalize_gradients(coeff_0=coeff_0, coeff_1=coeff_1, coeff_2=coeff_2)



    def _equalize_gradients(self, coeff_0=1, coeff_1=1, coeff_2=1):
        if not self.agent.config.model.args.bias_infusion.use: return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs["current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            for name, parms in self.agent.model.named_parameters():
                if parms.grad is None: continue
                if "mod0" in name or "enc_0" in name or "fc_0_lin.weight" in name:
                    parms.grad *= coeff_0
                if "mod1" in name or "enc_1" in name or "fc_1_lin.weight" in name:
                    parms.grad *= coeff_1
                if "mod2" in name or "enc_2" in name or "fc_2_lin.weight" in name:
                    parms.grad *= coeff_2
class Bias_Infusion_MLB_3d_Reg(General_Bias_Infusion):
    def __init__(self, agent):
        super().__init__(agent)
        logging.info("Bias Infusion OGM-Mine is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.agent.logs["ratio_logs"] = {
            "ratio_0":[],
            "ratio_1":[],
            "ratio_2":[],
            "coeff_v":[],
            "coeff_l":[],
            "coeff_f":[]
        }
        self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", False)
        # self.balance_mode = self.agent.config.model.args.bias_infusion.get("balance_mode", "whatev")
        self.alpha = self.agent.config.model.args.bias_infusion.alpha
        self.tanh_mode = self.agent.config.model.args.bias_infusion.get("tanh_mode", False)
        self.tanh_mode_beta = self.agent.config.model.args.bias_infusion.get("tanh_mode_beta", False)

        if self.tanh_mode == 2:
            print("Tanh mode 2 is being employed with beta {}".format(self.tanh_mode_beta))



    # def before_backward(self, total, output_losses, **kwargs):
    #
    #     opt_done = False
    #     if self.balance_mode == "balance_only_multi":
    #         (output_losses["ce_loss_combined"]).backward(retain_graph=True)
    #         # (output_losses["ce_loss_combined"] + self.output_losses["ce_loss_c"] + self.output_losses["ce_loss_g"]).backward(retain_graph=True)
    #         self.output_losses = output_losses
    #         opt_done = True
    #
    #     return total, output_losses, opt_done

    def on_backward_end(self, label, out_color, out_gray, out_f):

        score_0 = F.mse_loss(out_color.squeeze(), label)
        score_1 = F.mse_loss(out_gray.squeeze(), label)
        score_2 = F.mse_loss(out_f.squeeze(), label)

        # score_1 = sum([self.softmax(out_gray)[i][label[i]] for i in range(out_gray.size(0))]).detach()

        ratio_0 =  score_0/((score_1 + score_2)/2)
        ratio_1 =  score_1/((score_0 + score_2)/2)
        ratio_2 =  score_2/((score_0 + score_1)/2)

        coeff_0 = 1 + self.tanh(self.alpha * (ratio_0-1)).cpu().numpy()
        coeff_1 = 1 + self.tanh(self.alpha * (ratio_1-1)).cpu().numpy()
        coeff_2 = 1 + self.tanh(self.alpha * (ratio_2-1)).cpu().numpy()

        if self.tanh_mode == 2:
            if ratio_0 > 1:
                coeff_0 = 1 + (self.tanh_mode_beta-1)*self.tanh(self.alpha * (ratio_0 - 1))
            else:
                coeff_0 = 1 + self.tanh(self.alpha * (ratio_0 - 1))
            if ratio_1 > 1:
                coeff_1 = 1 + (self.tanh_mode_beta-1)*self.tanh(self.alpha * (ratio_1 - 1))
            else:
                coeff_1 = 1 + self.tanh(self.alpha * (ratio_1 - 1))
            if ratio_2 > 1:
                coeff_2 = 1 + (self.tanh_mode_beta-1)*self.tanh(self.alpha * (ratio_2 - 1))
            else:
                coeff_2 = 1 + self.tanh(self.alpha * (ratio_2 - 1))

        self.agent.logs["ratio_logs"]["ratio_0"].append(ratio_0.cpu().numpy())
        self.agent.logs["ratio_logs"]["ratio_1"].append(ratio_1.cpu().numpy())
        self.agent.logs["ratio_logs"]["ratio_2"].append(ratio_2.cpu().numpy())
        self.agent.logs["ratio_logs"]["coeff_v"].append(coeff_0)
        self.agent.logs["ratio_logs"]["coeff_l"].append(coeff_1)
        self.agent.logs["ratio_logs"]["coeff_f"].append(coeff_2)

        wandb_output = {
            "ratio": {"ratio_0": ratio_0.cpu().numpy(),
                      "ratio_1": ratio_1.cpu().numpy(),
                        "ratio_2": ratio_2.cpu().numpy(),
                          "coeff_v":coeff_0,
                          "coeff_l":coeff_1,
                          "coeff_f":coeff_2
        }}

        wandb.log(wandb_output, step=self.agent.logs["current_step"])

        self._equalize_gradients(coeff_0=coeff_0, coeff_1=coeff_1, coeff_2=coeff_2)



    def _equalize_gradients(self, coeff_0=1, coeff_1=1, coeff_2=1):
        if not self.agent.config.model.args.bias_infusion.use: return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs["current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            for name, parms in self.agent.model.named_parameters():
                if parms.grad is None: continue
                if "mod0" in name or "enc_0" in name or "fc_0_lin.weight" in name:
                    parms.grad *= coeff_0
                if "mod1" in name or "enc_1" in name or "fc_1_lin.weight" in name:
                    parms.grad *= coeff_1
                if "mod2" in name or "enc_2" in name or "fc_2_lin.weight" in name:
                    parms.grad *= coeff_2
class Bias_Infusion_MSLR(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_MSLR, self).__init__(agent)
        self.agent.logger.info("Bias Infusion MSLR is used")

        self._initialize_logs_n_utils()

    def before_backward(self, total, output_losses, **kwargs):
        return output_losses["ce_loss_combined"], output_losses, False
    def _initialize_logs_n_utils(self):

        self.init_learning_rate = self.agent.config.model.args.bias_infusion.get("init_learning_rate", {"c":1, "g":1})
        self.coeff_memory = defaultdict(list)
        self.keep_memory_epoch = self.agent.config.model.args.bias_infusion.get("keep_memory_epoch", 5)
        self.starting_epoch = self.agent.config.model.args.bias_infusion.get("starting_epoch", 0)
        self.ending_epoch = self.agent.config.model.args.bias_infusion.get("ending_epoch", 1500)
        self.ratio = defaultdict(lambda: 1)
        self.softmax = nn.Softmax(dim=1)
        self.agent.logs["ratio_logs"] = {
            "ratio_color":[],
            "ratio_gray":[],
            "coeff_color":[],
            "coeff_gray":[],
        }

    def on_epoch_begin(self, **kwargs):

        if self.agent.logs["current_epoch"]<1: return


        acc_0 = self.agent.logs["val_logs"][list(self.agent.logs["val_logs"].keys())[-1]]["acc"]["c"]
        acc_1 = self.agent.logs["val_logs"][list(self.agent.logs["val_logs"].keys())[-1]]["acc"]["g"]

        if self.agent.logs["current_epoch"] > self.keep_memory_epoch:
            ratio_0 = acc_0 / np.array(self.coeff_memory[0][-self.keep_memory_epoch:]).mean()
            ratio_1 = acc_1 / np.array(self.coeff_memory[1][-self.keep_memory_epoch:]).mean()
        else:
            ratio_0, ratio_1 = 1, 1

        self.ratio[0] = ratio_0
        self.ratio[1] = ratio_1

        self.coeff_memory[0].append(acc_0)
        self.coeff_memory[1].append(acc_1)

    def on_backward_end(self, label, out_color, out_gray):

        coeff_0 = self.ratio[0] * self.init_learning_rate["c"]
        coeff_1 = self.ratio[1] * self.init_learning_rate["g"]

        self._equalize_gradients(coeff_0=coeff_0, coeff_1=coeff_1)

        self.agent.logs["ratio_logs"]["ratio_gray"].append(self.ratio[0])
        self.agent.logs["ratio_logs"]["ratio_color"].append(self.ratio[1])
        self.agent.logs["ratio_logs"]["coeff_color"].append(coeff_0)
        self.agent.logs["ratio_logs"]["coeff_gray"].append(coeff_1)
        wandb.log({"ratio":self.agent.logs["ratio_logs"]})


    def _equalize_gradients(self, coeff_0=1, coeff_1=1):
        if not self.agent.config.model.args.bias_infusion.use: return
        if self.starting_epoch <= self.agent.logs["current_epoch"] <= self.ending_epoch:
            for name, params in self.agent.model.named_parameters():
                if params.grad is None: continue
                if "mod0" in name or "enc_0" in name:
                    params.grad *= coeff_0
                if "mod1" in name or "enc_1" in name:
                    params.grad *= coeff_1
class Bias_Infusion_AGM(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_AGM, self).__init__(agent)

        self.agent.logger.info("Bias_Infusion_AGM is used")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.bias_infuser = self.agent.config.model.args.get("bias_infusion", {})
        self.train_score_a, self.train_score_v = 0, 0
        self.ra_score_a, self.ra_score_v = 0, 0
        self.agent.logs["ratio_logs"] = {
            "ratio_color":[],
            "ratio_color_ra":[],
            "ratio_gray":[],
            "ratio_gray_ra":[],
            "coeff_color":[],
            "coeff_gray":[],
        }

    def before_backward(self, total, output_losses, label, output, data, **kwargs):

        # if self.bias_infuser.get("starting_epoch",0) <= self.agent.logs["current_epoch"] <= self.bias_infuser.get("ending_epoch",1500):
        #     return total, output_losses, False

        # self.agent.model.eval()
        # if 0 in data and 1 in data:
        #     this_data = copy.deepcopy(data)
        #     this_data[0] = torch.zeros_like(this_data[0])
        #     pad_audio = self.agent.model(this_data, return_features=True)["preds"]["combined"]
        #     this_data = copy.deepcopy(data)
        #     this_data[1] = torch.zeros_like(this_data[1])
        #     pad_video = self.agent.model(this_data, return_features=True)["preds"]["combined"]
        # elif 2 in data and 3 in data:
        #     this_data = copy.deepcopy(data)
        #     this_data[2] = torch.zeros_like(this_data[2])
        #     pad_audio = self.agent.model(this_data, return_features=True)["preds"]["combined"]
        #     this_data = copy.deepcopy(data)
        #     this_data[3] = torch.zeros_like(this_data[3])
        #
        #     pad_video = self.agent.model(this_data, return_features=True)["preds"]["combined"]
        #
        # self.agent.model.train()

        # output["preds"]["c"] = 0.5 * (output["preds"]["combined"] - pad_audio + pad_video).detach()
        # output["preds"]["g"] = 0.5 * (output["preds"]["combined"] - pad_video + pad_audio).detach()
        # del pad_audio, pad_video, this_data

        out_color = output["preds"]["c"]
        out_gray = output["preds"]["g"]

        score_audio = 0.
        score_visual = 0.
        for k in range(out_color.size(0)):
            if torch.isinf(self.softmax(out_color)[k][label[k]]) or self.softmax(out_color)[k][label[k]] < 1e-8:
                score_audio += - torch.log(torch.tensor(1e-8, dtype=out_color.dtype, device=out_color.device))
            else:
                score_audio += - torch.log(self.softmax(out_color)[k][label[k]])
            if torch.isinf(self.softmax(out_gray)[k][label[k]]) or self.softmax(out_gray)[k][label[k]] < 1e-8:
                score_visual += - torch.log(torch.tensor(1e-8, dtype=out_gray.dtype, device=out_gray.device))
            else:
                score_visual += - torch.log(self.softmax(out_gray)[k][label[k]])

        score_audio = score_audio / out_color.size(0)
        score_visual = score_visual / out_gray.size(0)

        r_a = math.exp((score_visual.item() - score_audio.item()) )
        r_v = math.exp((score_audio.item() - score_visual.item()) )

        optimal_ratio_a = math.exp((self.train_score_v - self.train_score_a))
        optimal_ratio_v = math.exp((self.train_score_a - self.train_score_v))

        coeff_a = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_a - r_a, 10)))
        coeff_v = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_v - r_v, 10)))

        #Shouldnt this go above optimal_ratio?
        iteration = self.agent.logs["current_step"]
        self.train_score_a = self.train_score_a * iteration / (iteration + 1) + score_audio.item() / (iteration + 1)
        self.train_score_v = self.train_score_v * iteration / (iteration + 1) + score_visual.item() / (iteration + 1)

        self.ra_score_a = self.ra_score_a * iteration / (iteration + 1) + score_audio.item() / (iteration + 1)
        self.ra_score_v = self.ra_score_v * iteration / (iteration + 1) + score_visual.item() / (iteration + 1)

        self.agent.logs["ratio_logs"]["ratio_color"].append(self.train_score_a)
        self.agent.logs["ratio_logs"]["ratio_color_ra"].append(self.ra_score_a)
        self.agent.logs["ratio_logs"]["ratio_gray"].append(self.train_score_v)
        self.agent.logs["ratio_logs"]["ratio_gray_ra"].append(self.ra_score_v)
        self.agent.logs["ratio_logs"]["coeff_color"].append(coeff_a)
        self.agent.logs["ratio_logs"]["coeff_gray"].append(coeff_v)

        wandb_output = {
            "ratio": {"ratio_gray": self.train_score_a,
                      "ratio_color": self.train_score_v,
                      "ratio_gray_ra": self.ra_score_a,
                      "ratio_color_ra": self.ra_score_v,
                      "coeff_color":coeff_a,"coeff_gray":coeff_v}
        }

        wandb.log(wandb_output, step=self.agent.logs["current_step"])

        self.agent.model.update_scale(coeff_a, coeff_v)

        return total, output_losses, False
class Bias_Infusion_AGM_3mod(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_AGM_3mod, self).__init__(agent)

        self.agent.logger.info("Bias_Infusion_AGM is used")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.bias_infuser = self.agent.config.model.args.get("bias_infusion", {})
        self.train_score_a, self.train_score_v, self.train_score_f = 0, 0, 0
        self.ra_score_a, self.ra_score_v = 0, 0
        self.agent.logs["ratio_logs"] = {
            "ratio_color":[],
            "ratio_gray":[],
            "ratio_flow":[],
            "coeff_color":[],
            "coeff_gray":[],
            "coeff_flow":[],
        }



    def before_backward(self, total, output_losses, label, output, data, **kwargs):


        out_color = output["preds"]["c"]
        out_gray = output["preds"]["g"]
        out_flow = output["preds"]["f"]

        score_visual = 0.
        score_audio = 0.
        score_flow = 0.

        for k in range(out_color.size(0)):
            if torch.isinf(self.softmax(out_color)[k][label[k]]) or self.softmax(out_color)[k][label[k]] < 1e-8:
                score_audio += - torch.log(torch.tensor(1e-8, dtype=out_color.dtype, device=out_color.device))
            else:
                score_audio += - torch.log(self.softmax(out_color)[k][label[k]])
            if torch.isinf(self.softmax(out_gray)[k][label[k]]) or self.softmax(out_gray)[k][label[k]] < 1e-8:
                score_visual += - torch.log(torch.tensor(1e-8, dtype=out_gray.dtype, device=out_gray.device))
            else:
                score_visual += - torch.log(self.softmax(out_gray)[k][label[k]])

            if torch.isinf(self.softmax(out_flow)[k][label[k]]) or self.softmax(out_flow)[k][label[k]] < 1e-8:
                score_flow += - torch.log(torch.tensor(1e-8, dtype=out_flow.dtype, device=out_gray.device))
            else:
                score_flow += - torch.log(self.softmax(out_flow)[k][label[k]])

        score_audio = score_audio / out_color.size(0)
        score_visual = score_visual / out_gray.size(0)
        score_flow = score_flow / out_flow.size(0)

        mean_score = (score_visual.item() + score_audio.item() + score_flow.item()) / 3

        r_v = math.exp((mean_score - score_visual.item()) * 3 / 2)
        r_a = math.exp((mean_score - score_audio.item()) * 3 / 2)
        r_f = math.exp((mean_score - score_flow.item()) * 3 / 2)

        iteration = self.agent.logs["current_step"]

        self.train_score_a = self.train_score_a * iteration / (iteration + 1) + score_audio.item() / (iteration + 1)
        self.train_score_v = self.train_score_v * iteration / (iteration + 1) + score_visual.item() / (iteration + 1)
        self.train_score_f = self.train_score_f * iteration / (iteration + 1) + score_flow.item() / (iteration + 1)

        optimal_mean_score = (self.train_score_v + self.train_score_a+ self.train_score_f)/3
        optimal_ratio_a = math.exp((optimal_mean_score - self.train_score_a) * 3 / 2)
        optimal_ratio_v = math.exp((optimal_mean_score - self.train_score_v) * 3 / 2)
        optimal_ratio_f = math.exp((optimal_mean_score - self.train_score_f) * 3 / 2)

        coeff_a = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_a - r_a, 7)))
        coeff_v = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_v - r_v, 7)))
        coeff_f = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_f - r_f, 7)))

        self.agent.logs["ratio_logs"]["ratio_color"].append(self.train_score_a)
        self.agent.logs["ratio_logs"]["ratio_gray"].append(self.train_score_v)
        self.agent.logs["ratio_logs"]["ratio_gray"].append(self.train_score_f)

        self.agent.logs["ratio_logs"]["coeff_color"].append(coeff_a)
        self.agent.logs["ratio_logs"]["coeff_gray"].append(coeff_v)
        self.agent.logs["ratio_logs"]["coeff_flow"].append(coeff_f)

        wandb.log({"coeff_v":coeff_a,
                   "coeff_l":coeff_v,
                   "coeff_f":coeff_f,
                   "running_ratio_v": self.train_score_a,
                   "running_ratio_l": self.train_score_v,
                   "running_ratio_f": self.train_score_f
                   }, step=self.agent.logs["current_step"])

        self.agent.model.update_scale(coeff_c=coeff_a, coeff_g=coeff_v, coeff_f=coeff_f)

        return total, output_losses, False
class Bias_Infusion_AGM_3mod_reg(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_AGM_3mod_reg, self).__init__(agent)

        self.agent.logger.info("Bias_Infusion_AGM is used")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.criterion = nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.bias_infuser = self.agent.config.model.args.get("bias_infusion", {})
        self.train_score_a, self.train_score_v, self.train_score_f = 0, 0, 0
        self.ra_score_a, self.ra_score_v = 0, 0
        self.agent.logs["ratio_logs"] = {
            "ratio_color":[],
            "ratio_gray":[],
            "ratio_flow":[],
            "coeff_color":[],
            "coeff_gray":[],
            "coeff_flow":[],
        }



    def before_backward(self, total, output_losses, label, output, data, **kwargs):


        out_color = output["preds"]["c"]
        out_gray = output["preds"]["g"]
        out_flow = output["preds"]["f"]

        score_visual = self.criterion(out_gray.squeeze(), label)
        score_audio = self.criterion(out_color.squeeze(), label)
        score_flow = self.criterion(out_flow.squeeze(), label)

        score_audio = score_audio / out_color.size(0)
        score_visual = score_visual / out_gray.size(0)
        score_flow = score_flow / out_flow.size(0)

        score_visual = 1/score_visual
        score_audio = 1/score_audio
        score_flow = 1/score_flow

        mean_score = (score_visual.item() + score_audio.item() + score_flow.item()) / 3

        r_v = math.exp((mean_score - score_visual.item()) * 3 / 2)
        r_a = math.exp((mean_score - score_audio.item()) * 3 / 2)
        r_f = math.exp((mean_score - score_flow.item()) * 3 / 2)

        iteration = self.agent.logs["current_step"]

        self.train_score_a = self.train_score_a * iteration / (iteration + 1) + score_audio.item() / (iteration + 1)
        self.train_score_v = self.train_score_v * iteration / (iteration + 1) + score_visual.item() / (iteration + 1)
        self.train_score_f = self.train_score_f * iteration / (iteration + 1) + score_flow.item() / (iteration + 1)

        optimal_mean_score = (self.train_score_v + self.train_score_a+ self.train_score_f)/3
        optimal_ratio_a = math.exp((optimal_mean_score - self.train_score_a) * 3 / 2)
        optimal_ratio_v = math.exp((optimal_mean_score - self.train_score_v) * 3 / 2)
        optimal_ratio_f = math.exp((optimal_mean_score - self.train_score_f) * 3 / 2)

        coeff_a = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_a - r_a, 7)))
        coeff_v = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_v - r_v, 7)))
        coeff_f = math.exp(self.bias_infuser.alpha * (min(optimal_ratio_f - r_f, 7)))

        self.agent.logs["ratio_logs"]["ratio_color"].append(self.train_score_a)
        self.agent.logs["ratio_logs"]["ratio_gray"].append(self.train_score_v)
        self.agent.logs["ratio_logs"]["ratio_gray"].append(self.train_score_f)

        self.agent.logs["ratio_logs"]["coeff_color"].append(coeff_a)
        self.agent.logs["ratio_logs"]["coeff_gray"].append(coeff_v)
        self.agent.logs["ratio_logs"]["coeff_flow"].append(coeff_f)

        wandb.log({"coeff_v":coeff_a,
                   "coeff_l":coeff_v,
                   "coeff_f":coeff_f,
                   "running_ratio_v": self.train_score_a,
                   "running_ratio_l": self.train_score_v,
                   "running_ratio_f": self.train_score_f
                   }, step=self.agent.logs["current_step"])

        self.agent.model.update_scale(coeff_c=coeff_a, coeff_g=coeff_v, coeff_f=coeff_f)

        return total, output_losses, False




