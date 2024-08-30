import copy
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score
from collections import defaultdict

class Validator_Tester():
    def __init__(self, agent):
        self.agent = agent
        self.multi_supervised = False

        if self.agent.config.get("task", "classification") == "classification":
            self.valtest_step_func = "valtest_one_step" #self._find_train_step_func()
        elif self.agent.config.get("task", "classification") == "regression":
            self.valtest_step_func = "valtest_one_step_regression"

        # self.test_step_func = "test_one_step" #self._find_test_step_func()
        self.this_valtest_step_func = getattr(self, self.valtest_step_func)
        # self.this_test_step_func = getattr(self, self.test_step_func)
        self._get_loss_weights()

    def validate(self, best_model = False, test_set= False):
        """
        One cycle of model validation
        :return:
        """
        # if best_model:
        #     self.agent.best_model.eval()
        #     self.agent.best_model.train(False)

        self.agent.model.eval()
        self.agent.model.train(False)

        this_evaluator = self.agent.evaluators.test_evaluator if test_set else self.agent.evaluators.val_evaluator
        this_dataloader = self.agent.data_loader.test_loader if test_set else self.agent.data_loader.valid_loader
        this_evaluator.reset()
        with torch.no_grad():
            pbar = tqdm(enumerate(this_dataloader),
                        total=len(this_dataloader),
                        desc="Validation",
                        leave=False,
                        disable=True,
                        position=1)
            for batch_idx, served_dict in pbar:

                step_outcome = self.this_valtest_step_func(served_dict, best_model=best_model)

                all_outputs = self.agent.accelerator.gather(step_outcome)
                if all_outputs["process_flag"]:
                    this_evaluator.process(all_outputs)

                del step_outcome, served_dict

                mean_batch_loss, mean_batch_loss_message = this_evaluator.mean_batch_loss()

                pbar_message = "Validation batch {0:d}/{1:d} with {2:}".format(batch_idx,
                                                                             len(this_dataloader) - 1,
                                                                             mean_batch_loss_message)
                pbar.set_description(pbar_message)
                pbar.refresh()

    def valtest_one_step(self, served_dict, best_model=False):

            data = {view: served_dict["data"][view].cuda() for view in
                                   served_dict["data"] if type(served_dict["data"][view]) is torch.Tensor }
            data.update({view: data[view].float() for view in data if type(view) == int})

            label = served_dict["label"].squeeze().type(torch.LongTensor).cuda()


            if len(label.shape) == 0: label = label.unsqueeze(dim=0)

            # if best_model:
            #     output = self.agent.best_model(data)
            # else:
            output = self.agent.model(data, label=label)

            # data_shuffled_color = copy.deepcopy(data)
            # data_shuffled_color[0] = data_shuffled_color["0_random_indistr"]
            # output_shuffled_color = self.agent.best_model(data_shuffled_color) if best_model else  self.agent.model(data_shuffled_color)
            # output["preds"].update({i+"_shc":output_shuffled_color["preds"][i] for i in output_shuffled_color["preds"]})
            #
            # data_shuffled_gray = copy.deepcopy(data)
            # data_shuffled_gray[1] = data_shuffled_color["1_random_indistr"]
            # output_shuffled_gray = self.agent.best_model(data_shuffled_gray) if best_model else  self.agent.model(data_shuffled_gray)
            #
            # output["preds"].update({i+"_shg":output_shuffled_gray["preds"][i] for i in output_shuffled_gray["preds"]})

            # if "0_random" in data_shuffled_gray:
            #     data_random_color = copy.deepcopy(data)
            #     data_random_color[0] = data_shuffled_gray["0_random"]
            #     output_random_color = self.agent.best_model(data_random_color) if best_model else  self.agent.model(data_random_color)
            #     output["preds"].update({i+"_rc":output_random_color["preds"][i] for i in output_random_color["preds"]})
            #
            # if "1_random" in data_shuffled_gray:
            #     data_random_gray = copy.deepcopy(data)
            #     data_random_gray[1] = data_shuffled_gray["1_random"]
            #     output_random_gray = self.agent.best_model(data_random_gray) if best_model else  self.agent.model(data_random_gray)
            #     output["preds"].update({i+"_rg":output_random_gray["preds"][i] for i in output_random_gray["preds"]})

            total_loss =  torch.zeros(1).squeeze().cuda()
            output_losses, ce_loss = {}, {}

            if self.agent.config.model.args.get("validation_loss_w", {"combined":1}):
                for k, v in output["preds"].items():
                    if k in self.agent.config.model.args.multi_loss.multi_supervised_w and self.agent.config.model.args.multi_loss.multi_supervised_w[k] != 0:
                        if len(label) > 1:  # TODO: Check if this one needs to be one or zero
                            ce_loss[k] = self.agent.loss(v, label)
                            total_loss += self.w_loss[k] * ce_loss[k]
                            ce_loss[k] = ce_loss[k].detach()
                            output_losses.update({"ce_loss_{}".format(k): ce_loss[k]})

            total_loss = total_loss.detach()
            output_losses.update({"total": total_loss})
            for i in output["preds"]:  output["preds"][i] =  output["preds"][i].detach()

            process_flag = False
            if len(label) > 1:
                process_flag = True

            return {"loss": output_losses,
                    "pred" : output["preds"],
                    "features": output["features"],
                    "label": label,
                    "process_flag": process_flag}

    def valtest_one_step_regression(self, served_dict, best_model=False):

            data = {view: served_dict["data"][view].cuda() for view in
                                   served_dict["data"] if type(served_dict["data"][view]) is torch.Tensor }
            data.update({view: data[view].float() for view in data if type(view) == int})

            label = served_dict["label"].cuda()

            # if best_model:
            #     output = self.agent.best_model(data)
            # else:
            output = self.agent.model(data)


            total_loss = torch.nn.L1Loss()(output["preds"]["combined"].flatten(), label)
            output_losses = {"l1_loss_combined": total_loss}

            total_loss = total_loss.detach()
            output_losses.update({"total": total_loss})
            for i in output["preds"]:  output["preds"][i] =  output["preds"][i].detach()

            process_flag = False
            if len(label) > 1:
                process_flag = True

            return {"loss": output_losses,
                    "pred" : output["preds"],
                    # "features": output["features"],
                    "label": label,
                    "process_flag": process_flag}

    def _get_loss_weights(self):

        w_loss = defaultdict(int)
        w_loss["total"] = 1

        # w_loss["total"] = torch.Tensor([1]).cuda()

        ws = self.agent.config.model.args.get("validation_loss_w", {"combined":1})
        for k, v in ws.items():
            w_loss[k] = v

        self.w_loss = w_loss

