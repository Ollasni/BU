from utils.config import process_config, process_config_default
from colorama import Fore, Style
from models.OGM_Model import *
from models.ViT_OGM import *
from models.Mosei_Models import *
from datasets.CREMAD.CREMAD_Dataset import *
from datasets.UCF101.UCF101_Dataset import *
from datasets.MOSEI.multibenchmark_mosei_dataloader import *
from datasets.AVE.AVE_Dataset import *

class Importer():
    def __init__(self, config_name:str,  device:str="cuda:0", default_files: list=None, fold:int=0):
        # print("Loading config from {}".format(config_name))
        if default_files is not None:
            self.config = process_config_default(json_file=config_name, default_files=default_files, printing=False)
        else:
            self.config = process_config( json_file=config_name, printing=False)
        self.device = device
        self.fold = fold

    def load_checkpoint(self):
        # self.config.model.save_dir = self.config.model.save_dir.format(self.fold)
        if "save_base_dir" in self.config.model:
            self.config.model.save_dir = os.path.join(self.config.model.save_base_dir, self.config.model.save_dir)

        if "model" not in self.config:
            # print("Loading from {}".format(self.config.save_dir))
            self.checkpoint = torch.load(self.config.save_dir, map_location="cpu")
        else:
            # print("Loading from {}".format(self.config.model.save_dir))
            self.checkpoint = torch.load(self.config.model.save_dir, map_location="cpu")

    def change_config(self, attr, value, c = None):
        if c == None: c = self.config
        attr_splits = attr.split(".")
        for attr_split in attr_splits[:-1]:
            c = getattr(c, attr_split)
        setattr(c, attr_splits[-1], value)

    def get_dataloaders(self):

        dataloader = globals()[self.config.dataset.dataloader_class]
        data_loader = dataloader(config=self.config)
        if hasattr(self, "checkpoint") and hasattr(self.checkpoint, "metrics"):
            data_loader.load_metrics_ongoing(self.checkpoint["metrics"])
        if hasattr(self, "checkpoint") and hasattr(self.checkpoint, "logs") and hasattr(self.checkpoint['logs'], "weights"):
            data_loader.weights = self.checkpoint['logs']["weights"]

        return data_loader

    def get_model(self, model = None, return_model:str = "best_model"):

        if not model:
            model_class = globals()[self.config.model.model_class]

            if "save_base_dir" in self.config.model and "swin_backbone" in self.config.model.args:
                self.config.model.args.swin_backbone = os.path.join(self.config.model.save_base_dir,
                                                                          self.config.model.args.swin_backbone)
            if "save_base_dir" in self.config.model and "pretraining_paths" in self.config.model.args:
                self.config.model.args.pretraining_paths = {i: os.path.join(self.config.model.save_base_dir,
                                                                                  self.config.model.args.pretraining_paths[
                                                                                      i]) for i in
                                                                  self.config.model.args.pretraining_paths}

            enc = self._load_encoder(encoders=self.config.model.get("encoders", []))
            model = model_class(encs=enc, args=self.config.model.args)

            model = model.to(self.device)
            # model = nn.DataParallel(model, device_ids=[torch.device(i) for i in self.config.training_params.gpu_device])

        if return_model == "untrained_model":
            return model
        elif return_model == "best_model":
            # if len(enc)>0 and "VaVL" not in enc[0].__class__.__name__:
            #     print("Replacing module")
            self.checkpoint["best_model_state_dict"] = {key.replace("module.", ""): value for key, value in
                                              self.checkpoint["best_model_state_dict"].items()}
            self.checkpoint["best_model_state_dict"] = {key.replace("parametrizations.weight.original0", "weight_g"): value for key, value in
                                              self.checkpoint["best_model_state_dict"].items()}
            self.checkpoint["best_model_state_dict"] = {key.replace("parametrizations.weight.original1", "weight_v"): value for key, value in
                                              self.checkpoint["best_model_state_dict"].items()}

            # self.checkpoint["best_model_state_dict"] = {key.replace("bias_lin", "fc_0_lin.bias"): value for key, value in
            #                                   self.checkpoint["best_model_state_dict"].items()}

            # model.load_state_dict(self.checkpoint["best_model_state_dict"], strict=False)
            model.load_state_dict(self.checkpoint["best_model_state_dict"])
            return model
        elif return_model == "running_model":
            model.load_state_dict(self.checkpoint["model_state_dict"])
            return model
        else:
            raise ValueError(
                'Return such model does not exits as option, choose from "best_model","running_model", "untrained_model" ')

    def print_progress(self, multi_fold_results, verbose=True, print_entropy=False, latex_version=False, print_post_test=True):

        val_metrics = self.checkpoint["logs"]["best_logs"]
        # print(val_metrics.keys())
        # return

        multi_fold_results[self.fold] = val_metrics
        if verbose:print("-- Best Validation --")
        latex_message = {}
        if "acc" not in val_metrics:
            current_epoch = self.checkpoint["logs"]["current_epoch"] if "current_epoch" not in val_metrics else val_metrics["current_epoch"]
            message = Style.BRIGHT + Fore.WHITE + "Epoch: {}, No_improve: {} ".format(current_epoch, self.checkpoint["logs"][
                "steps_no_improve"])
            if "loss" in val_metrics:
                for i, v in val_metrics["loss"].items():
                    message += Fore.RED + "{} : {:.6f} ".format(i, val_metrics["loss"][i])
            if verbose:print(message + Style.RESET_ALL)

        else:
            current_epoch = self.checkpoint["logs"]["current_epoch"] if "current_epoch" not in val_metrics else val_metrics["current_epoch"]

            for pred in val_metrics["acc"]:
                message = Style.BRIGHT + Fore.WHITE + "Step: {}, No_improve: {} ".format( current_epoch, self.checkpoint["logs"]["steps_no_improve"])
                # print(message+ Style.RESET_ALL)
                # break
                latex_message[pred] = "{} & ".format(pred)
                if "loss" in val_metrics:
                    for i, v in val_metrics["loss"].items():
                        if pred in i or i =="total":
                            message += Fore.RED + "{} : {:.6f} ".format(i, val_metrics["loss"][i])
                # if "val_loss" in val_metrics:
                #     for i, v in val_metrics["val_loss"].items(): message += Fore.RED + "{} : {:.6f} ".format(i, v)
                if "acc" in val_metrics:
                    if pred in val_metrics["acc"]:
                        message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(pred, val_metrics["acc"][pred] * 100)
                        latex_message[pred] += " {:.1f} &".format(val_metrics["acc"][pred] * 100)
                if "k" in val_metrics:
                    if pred in val_metrics["k"]:
                        message += Fore.LIGHTGREEN_EX + "K_{}: {:.3f} ".format(pred, val_metrics["k"][pred])
                        latex_message[pred] += " {:.3f} &".format(val_metrics["k"][pred])
                if "f1" in val_metrics:
                    if pred in val_metrics["f1"]:
                        message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(pred, val_metrics["f1"][pred] * 100)
                        latex_message[pred] += " {:.1f} &".format(val_metrics["f1"][pred] * 100)
                if "perclassf1" in val_metrics:
                    if pred in val_metrics["perclassf1"]:
                        message += Fore.BLUE + "F1_perclass_{}: {} ".format(pred,"{}".format(str(list((val_metrics["perclassf1"][pred] * 100).round(2)))))
                        for i in list((val_metrics["perclassf1"][pred] * 100).round(2)):
                            latex_message[pred] += " {:.1f} &".format(i)

                if verbose:print(message+ Style.RESET_ALL)
                # print(latex_message[pred]+ Style.RESET_ALL)

        if self.config.training_params.rec_test and "test_logs" in self.checkpoint["logs"] and len(self.checkpoint["logs"]["test_logs"])>0 and "step" in val_metrics:
            if verbose:print("-- Best Test --")

            test_best_logs = self.checkpoint["logs"]["test_logs"][val_metrics["step"]]

            if "acc" not in test_best_logs:
                if "test_acc" in test_best_logs:
                    #replace all elements without the test prefix
                    test_best_logs = {k.replace("test_", ""): v for k, v in test_best_logs.items()}
            for pred in test_best_logs["acc"]:

                message = Style.BRIGHT + Fore.WHITE + "Best Test "
                latex_message[pred] = "{} & ".format(pred)
                if "loss" in test_best_logs:
                    for i, v in test_best_logs["loss"].items():
                        if pred in i or i == "total":
                            message += Fore.RED + "{} : {:.6f} ".format(i, test_best_logs["loss"][i])
                if "acc" in test_best_logs:
                    if pred in test_best_logs["acc"]:
                        message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(pred, test_best_logs["acc"][pred] * 100)
                        latex_message[pred] += " {:.1f} &".format(test_best_logs["acc"][pred] * 100)
                if "k" in test_best_logs:
                    if pred in test_best_logs["k"]:
                        message += Fore.LIGHTGREEN_EX + "K_{}: {:.3f} ".format(pred, test_best_logs["k"][pred])
                        latex_message[pred] += " {:.3f} &".format(test_best_logs["k"][pred])
                if "f1" in test_best_logs:
                    if pred in test_best_logs["f1"]:
                        message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(pred, test_best_logs["f1"][pred] * 100)
                        latex_message[pred] += " {:.1f} &".format(test_best_logs["f1"][pred] * 100)
                if "perclassf1" in test_best_logs:
                    if pred in test_best_logs["perclassf1"]:
                        message += Fore.BLUE + "F1_perclass_{}: {} ".format(pred, "{}".format(
                            str(list((test_best_logs["perclassf1"][pred] * 100).round(2)))))
                        for i in list((test_best_logs["perclassf1"][pred] * 100).round(2)):
                            latex_message[pred] += " {:.1f} &".format(i)

                if verbose:print(message)


        def _print_test_results(metrics, verbose, description, multi_fold_results, print_entropy=False):
            # description = "--- Post Test ---"
            latex_message = {}
            # message = Style.BRIGHT + Fore.WHITE + "{} ".format(description)
            if verbose: print( Style.BRIGHT + Fore.WHITE +  "{} ".format(description))
            for pred in metrics["acc"]:
                message = "{} ".format(pred)
                latex_message[pred] = "{} & ".format(pred)

                if "acc" in metrics:
                    if pred in metrics["acc"]:
                        message += Fore.LIGHTBLUE_EX + "Acc: {:.1f} ".format(metrics["acc"][pred] * 100)
                        latex_message[pred] += " {:.1f} &".format(metrics["acc"][pred] * 100)


                if "k" in metrics:
                    if pred in metrics["k"]:
                        message += Fore.LIGHTGREEN_EX + "K: {:.3f} ".format(metrics["k"][pred])
                        latex_message[pred] += " {:.3f} &".format(metrics["k"][pred])

                if "f1" in metrics:
                    if pred in metrics["f1"]:
                        message += Fore.LIGHTGREEN_EX + "F1: {:.1f} ".format(metrics["f1"][pred] * 100)
                        latex_message[pred] += " {:.1f} &".format(metrics["f1"][pred] * 100)

                if "ece" in metrics:
                    if pred in metrics["ece"]:
                        message += Fore.LIGHTRED_EX + "ECE: {:.3f} ".format(metrics["ece"][pred])
                        latex_message[pred] += " {:.3f} &".format(metrics["ece"][pred])

                # if "f1_perclass" in metrics:
                #     if pred in metrics["f1_perclass"]:
                #         message += Fore.BLUE + "F1_perclass: {} ".format("{}".format(
                #             str(list((metrics["f1_perclass"][pred] * 100).round(1)))))
                #         for i in list((metrics["f1_perclass"][pred] * 100).round(2)):
                #             latex_message[pred] += " {:.1f} &".format(i)

                if verbose:print(message + Style.RESET_ALL)
                if verbose:print(latex_message[pred] + Style.RESET_ALL)

                #TODO: Make sure that this works to accumulate both the skipped and the normal cases, combined tags could get confused together
                multi_fold_results.update({self.fold: metrics})
                if "step" not in val_metrics: val_metrics["step"] = -1
                multi_fold_results[self.fold]["best_step"] = int(val_metrics["step"] / self.config.early_stopping.validate_every)
                multi_fold_results[self.fold]["steps_no_improve"] = self.checkpoint["logs"]["steps_no_improve"]

                if print_entropy:
                    for pred in metrics["entropy"]:
                        message = ""
                        if "entropy" in metrics:
                            if pred in metrics["entropy"]:
                                message += Fore.LIGHTRED_EX + "E_{}: {:.4f} ".format(pred, metrics["entropy"][pred])
                        if "entropy_var" in metrics:
                            if pred in metrics["entropy_var"]:
                                message += Fore.LIGHTRED_EX + "E_var_{}: {:.4f} ".format(pred, metrics["entropy_var"][pred])
                        if "entropy_correct" in metrics:
                            if pred in metrics["entropy_correct"]:
                                message += Fore.LIGHTMAGENTA_EX + "EC_{}: {:.4f} ".format(pred, metrics["entropy_correct"][pred])
                        if "entropy_correct_var" in metrics:
                            if pred in metrics["entropy_correct_var"]:
                                message += Fore.LIGHTMAGENTA_EX + "EC_{}: {:.4f} ".format(pred, metrics["entropy_correct_var"][pred])
                        if "entropy_wrong" in metrics:
                            if pred in metrics["entropy_wrong"]:
                                message += Fore.LIGHTYELLOW_EX + "EW_{}: {:.4f} ".format(pred, metrics["entropy_wrong"][pred])
                        if "entropy_wrong_var" in metrics:
                            if pred in metrics["entropy_wrong_var"]:
                                message += Fore.LIGHTYELLOW_EX + "EW_var_{}: {:.4f} ".format(pred, metrics["entropy_wrong_var"][pred])

                        if verbose:print(message + Style.RESET_ALL)
            return metrics

        test_results = False
        if "post_test_results" in self.checkpoint and print_post_test:
            # test_flag = True
            metrics = self.checkpoint["post_test_results"]
            test_results = _print_test_results(metrics=metrics, verbose=verbose, description="--- Post Test ---", print_entropy=print_entropy, multi_fold_results = multi_fold_results)
        else:
            if "test_best_logs" in locals():
                test_results = test_best_logs
                # multi_fold_results.update({self.fold: test_best_logs})

        step = self.checkpoint["logs"]["current_step"] if "step" not in val_metrics else val_metrics["step"]
        val_metrics["best_step"] = int(step / self.config.early_stopping.validate_every)
        val_metrics["steps_no_improve"] = self.checkpoint["logs"]["steps_no_improve"]
        val_metrics["current_epoch"] = self.checkpoint["logs"]["current_epoch"]

        return val_metrics, test_results

    def print_progress_adv(self, multi_fold_results, print_entropy=False, latex_version=False, print_post_test=True):

        def _print_test_results(metrics, description, epsilon, multi_fold_results, print_entropy=False):
            # description = "--- Post Test ---"
            latex_message = {}
            # message = Style.BRIGHT + Fore.WHITE + "{} ".format(description)
            print( Style.BRIGHT + Fore.WHITE +  "{} ".format(description))
            for pred in metrics["acc"]:
                message = "{} ".format(pred)
                latex_message[pred] = "{} & ".format(pred)

                if "acc" in metrics:
                    if pred in metrics["acc"]:
                        message += Fore.LIGHTBLUE_EX + "Acc: {:.1f} ".format(metrics["acc"][pred] * 100)
                        latex_message[pred] += " {:.1f} &".format(metrics["acc"][pred] * 100)

                if "k" in metrics:
                    if pred in metrics["k"]:
                        message += Fore.LIGHTGREEN_EX + "K: {:.3f} ".format(metrics["k"][pred])
                        latex_message[pred] += " {:.3f} &".format(metrics["k"][pred])

                if "f1" in metrics:
                    if pred in metrics["f1"]:
                        message += Fore.LIGHTGREEN_EX + "F1: {:.1f} ".format(metrics["f1"][pred] * 100)
                        latex_message[pred] += " {:.1f} &".format(metrics["f1"][pred] * 100)

                if "f1_perclass" in metrics:
                    if pred in metrics["f1_perclass"]:
                        message += Fore.BLUE + "F1_perclass: {} ".format("{}".format(
                            str(list((metrics["f1_perclass"][pred] * 100).round(1)))))
                        for i in list((metrics["f1_perclass"][pred] * 100).round(2)):
                            latex_message[pred] += " {:.1f} &".format(i)
                print(message + Style.RESET_ALL)
                # print(latex_message[pred] + Style.RESET_ALL)

                #TODO: Make sure that this works to accumulate both the skipped and the normal cases, combined tags could get confused together
                multi_fold_results.update({epsilon: metrics})

                if print_entropy:
                    for pred in metrics["entropy"]:
                        message = ""
                        if "entropy" in metrics:
                            if pred in metrics["entropy"]:
                                message += Fore.LIGHTRED_EX + "E_{}: {:.4f} ".format(pred, metrics["entropy"][pred])
                        if "entropy_var" in metrics:
                            if pred in metrics["entropy_var"]:
                                message += Fore.LIGHTRED_EX + "E_var_{}: {:.4f} ".format(pred, metrics["entropy_var"][pred])
                        if "entropy_correct" in metrics:
                            if pred in metrics["entropy_correct"]:
                                message += Fore.LIGHTMAGENTA_EX + "EC_{}: {:.4f} ".format(pred, metrics["entropy_correct"][pred])
                        if "entropy_correct_var" in metrics:
                            if pred in metrics["entropy_correct_var"]:
                                message += Fore.LIGHTMAGENTA_EX + "EC_{}: {:.4f} ".format(pred, metrics["entropy_correct_var"][pred])
                        if "entropy_wrong" in metrics:
                            if pred in metrics["entropy_wrong"]:
                                message += Fore.LIGHTYELLOW_EX + "EW_{}: {:.4f} ".format(pred, metrics["entropy_wrong"][pred])
                        if "entropy_wrong_var" in metrics:
                            if pred in metrics["entropy_wrong_var"]:
                                message += Fore.LIGHTYELLOW_EX + "EW_var_{}: {:.4f} ".format(pred, metrics["entropy_wrong_var"][pred])

                        print(message + Style.RESET_ALL)
            return multi_fold_results

        if "post_test_results_adv" in self.checkpoint and print_post_test:
            multi_fold_results[self.fold] = {}
            for eps in sorted(self.checkpoint["post_test_results_adv"].keys()):
                metrics = self.checkpoint["post_test_results_adv"][eps]
                multi_fold_results[self.fold] = _print_test_results(metrics=metrics, description="--- Post Test {} ---".format(eps), print_entropy=print_entropy, epsilon= eps, multi_fold_results = multi_fold_results[self.fold])
            # multi_fold_results = _print_test_results(metrics=metrics, description="--- Post Test ---", print_entropy=print_entropy, multi_fold_results = multi_fold_results)
        # if "post_test_results_skipped" in self.checkpoint:
        #     metrics = self.checkpoint["post_test_results_skipped"]
        #     multi_fold_results = _print_test_results(metrics=metrics, description="--- Post Test Skipped ---", print_entropy=print_entropy, multi_fold_results= multi_fold_results)


        return multi_fold_results

    def print_progress_aggregated(self, multi_fold_results, latex_version=False):
        init_results = {"acc":[], "f1":[], "k":[], "f1_perclass":[], "ece":[] }
        aggregated_results = {}
        if multi_fold_results is None:
            return
        for fold_i in multi_fold_results:
            metrics = multi_fold_results[fold_i]
            if "test_acc" in metrics: metrics["acc"] = metrics["test_acc"]
            if "test_f1" in metrics: metrics["f1"] = metrics["test_f1"]
            if "test_k" in metrics: metrics["k"] = metrics["test_k"]
            if "test_perclassf1" in metrics: metrics["f1_perclass"] = metrics["test_perclassf1"]
            if "acc" not in metrics: continue
            for pred in metrics["acc"]:
                if pred not in aggregated_results:
                    aggregated_results[pred] = copy.deepcopy(init_results)
                # if "combined_sh" in pred:
                #     this_pred = pred
                #     if this_pred not in aggregated_results:
                #         aggregated_results[this_pred] = copy.deepcopy(init_results)
                #     aggregated_results[pred]["acc"].append(metrics["acc"][pred]/metrics["acc"]["combined"])
                #     aggregated_results[pred]["f1"].append(0)
                #     aggregated_results[pred]["k"].append(0)
                #     aggregated_results[pred]["f1_perclass"].append([0,0,0,0,0,0])
                #     continue
                aggregated_results[pred]["acc"].append(metrics["acc"][pred])
                aggregated_results[pred]["f1"].append(metrics["f1"][pred])
                aggregated_results[pred]["k"].append(metrics["k"][pred])
                if "ece" in metrics:
                    if pred in metrics["ece"]:
                        aggregated_results[pred]["ece"].append(metrics["ece"][pred])
                # aggregated_results[pred]["f1_perclass"].append(metrics["f1_perclass"][pred])

        pred = "diff"
        if pred not in aggregated_results:
            aggregated_results[pred] = copy.deepcopy(init_results)
        if "combined_shc" in aggregated_results:
            multi_fold_results[fold_i]["acc"][pred] = aggregated_results["combined_shc"]["acc"][-1] - aggregated_results["combined_shg"]["acc"][-1]
            aggregated_results[pred]["acc"].append(aggregated_results["combined_shc"]["acc"][-1] - aggregated_results["combined_shg"]["acc"][-1])
            multi_fold_results[fold_i]["f1"][pred] = 0
            multi_fold_results[fold_i]["k"][pred] = 0
            # multi_fold_results[fold_i]["f1_perclass"][pred] = [0, 0, 0, 0, 0, 0]
            aggregated_results[pred]["f1"].append(0)
            aggregated_results[pred]["k"].append(0)
            # aggregated_results[pred]["f1_perclass"].append([0, 0, 0, 0, 0, 0])

        for pred in aggregated_results:
            for k in aggregated_results[pred]:
                aggregated_results[pred][k] = {"mean": np.array(aggregated_results[pred][k]).mean(axis=0), "std": np.array(aggregated_results[pred][k]).std(axis=0)}
                # aggregated_results[pred][k] /= len(multi_fold_results)
            # # message = Style.BRIGHT + Fore.WHITE + "{} ".format(description)

        print("-- Aggregared Results {} folds --".format(len(multi_fold_results)))
        latex_message = {}
        latex_message_variance = {}
        if len(multi_fold_results)%3!=0: return multi_fold_results
        for pred in aggregated_results:
            latex_message[pred] = "{} & ".format(pred)
            latex_message_variance[pred] = "{} & ".format(pred)
            message = Style.BRIGHT + Fore.WHITE + "AGG Results "

            message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.3f} ".format(pred, aggregated_results[pred]["acc"]["mean"] * 100)
            latex_message[pred] += " {:.2f}{{\\tiny$\pm${:.2f}}} &".format(aggregated_results[pred]["acc"]["mean"] * 100, aggregated_results[pred]["acc"]["std"] * 100)

            # message += Fore.LIGHTGREEN_EX + "K_{}: {:.3f} ".format(pred, aggregated_results[pred]["k"]["mean"])
            # latex_message[pred] += " {:.4f} {{\\tiny$\pm${:.4f}}} &".format(aggregated_results[pred]["k"]["mean"], aggregated_results[pred]["k"]["std"])
            #
            # message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(pred, aggregated_results[pred]["f1"]["mean"] * 100)
            # latex_message[pred] += " {:.2f} {{\\tiny$\pm${:.4f}}} &".format(aggregated_results[pred]["f1"]["mean"] * 100, aggregated_results[pred]["f1"]["std"] * 100)

            # message += Fore.BLUE + "F1_perclass_{}: {} ".format(pred, "{}".format(str(list((aggregated_results[pred]["f1_perclass"]["mean"] * 100).round(2)))))
            # for i in range(len(list((aggregated_results[pred]["f1_perclass"]["mean"])))):
            #     latex_message[pred] += " {:.1f} {{\\tiny$\pm${:.1f}}} &".format((aggregated_results[pred]["f1_perclass"]["mean"][i]*100).round(2),(aggregated_results[pred]["f1_perclass"]["std"][i]*100).round(2))
            # for i in list((aggregated_results[pred]["f1_perclass"]["std"] * 100).round(2)):
            #     latex_message_variance[pred] += " {:.1f} &".format(i)

            message += Fore.LIGHTGREEN_EX + "ECE_{}: {:.3f} ".format(pred, aggregated_results[pred]["ece"]["mean"])
            latex_message[pred] += " {:.4f} {{\\tiny$\pm${:.4f}}} &".format(aggregated_results[pred]["ece"]["mean"], aggregated_results[pred]["ece"]["std"])
            #

            print(message + Style.RESET_ALL)
            if latex_version:
                print(latex_message[pred] + Style.RESET_ALL)
            # print(latex_message_variance[pred] + Style.RESET_ALL)

        latex_message = ""
        pred = "combined"
        latex_message += " {:.2f}{{\\tiny$\pm${:.2f}}} &".format(aggregated_results[pred]["acc"]["mean"] * 100,
                                                                       aggregated_results[pred]["acc"]["std"] * 100)

        if "c" in list(aggregated_results.keys()):
            pred = "c"
            latex_message += " {:.2f}{{\\tiny$\pm${:.2f}}} &".format(aggregated_results[pred]["acc"]["mean"] * 100,
                                                                           aggregated_results[pred]["acc"]["std"] * 100)
        if "g" in list(aggregated_results.keys()):
            pred = "g"
            latex_message += " {:.2f}{{\\tiny$\pm${:.2f}}} &".format(aggregated_results[pred]["acc"]["mean"] * 100,
                                                                           aggregated_results[pred]["acc"]["std"] * 100)
        if "flow" in list(aggregated_results.keys()):
            pred = "flow"
            latex_message += " {:.2f}{{\\tiny$\pm${:.2f}}} &".format(aggregated_results[pred]["acc"]["mean"] * 100,
                                                                           aggregated_results[pred]["acc"]["std"] * 100)
        pred = "combined_shg"
        latex_message += " {:.2f}{{\\tiny$\pm${:.2f}}} &".format(aggregated_results[pred]["acc"]["mean"] * 100,
                                                                       aggregated_results[pred]["acc"]["std"] * 100)
        pred = "combined_shc"
        latex_message += " {:.2f}{{\\tiny$\pm${:.2f}}}".format(aggregated_results[pred]["acc"]["mean"] * 100,
                                                                       aggregated_results[pred]["acc"]["std"] * 100)

        print(latex_message + Style.RESET_ALL)

        return multi_fold_results

    def _my_numel(self, m: torch.nn.Module, only_trainable: bool = False, verbose = True):
        """
        returns the total number of parameters used by `m` (only counting
        shared parameters once); if `only_trainable` is True, then only
        includes parameters with `requires_grad = True`
        """
        parameters = list(m.parameters())
        if only_trainable:
            parameters = [p for p in parameters if p.requires_grad]
        unique = {p.data_ptr(): p for p in parameters}.values()
        model_total_params =  sum(p.numel() for p in unique)
        if verbose:
            print("Total number of trainable parameters are: {}".format(model_total_params))
        # for n, p in m.named_parameters()::
        #     if p.requires_grad:
        #         print(n, end=" - ")
        #         unique = {i.data_ptr(): i for i in p}.values()
        #         model_total_params = sum(i.numel() for i in unique)
        #         print(model_total_params)

        return model_total_params


    def print_progress_old(self, multi_fold_results):

        val_metrics = self.checkpoint["logs"]["best_logs"]

        print("-- Best Validation --")

        step = int(val_metrics["step"] / self.config.early_stopping.validate_every)
        message = Style.BRIGHT + Fore.WHITE + "Step: {}, No_improve: {} ".format( step, self.checkpoint["logs"]["steps_no_improve"])
        message += Fore.RED + "Loss : {:.6f} ".format(val_metrics["val_loss"])
        message += Fore.LIGHTBLUE_EX + "Acc: {:.2f} ".format(val_metrics["val_acc"] * 100)
        message += Fore.LIGHTGREEN_EX + "F1: {:.2f} ".format(val_metrics["val_f1"] * 100)
        message += Fore.LIGHTGREEN_EX + "K: {:.4f} ".format(val_metrics["val_k"])
        message += Fore.BLUE + "F1_perclass: {} ".format("{}".format(str(list((val_metrics["val_perclassf1"] * 100).round(2)))))
        print(message+ Style.RESET_ALL)


        if self.config.training_params.rec_test:
            print("-- Best Test --")
            test_best_logs = self.checkpoint["logs"]["test_logs"][self.checkpoint["logs"]["best_logs"]["step"]]
            print("Acc: {0:.1f}, Kappa: {1:.3f}, F1: {2:.1f}, f1_per_class: {3:.1f} {4:.1f} {5:.1f} {6:.1f} {7:.1f}".format(
                test_best_logs["accuracy"]*100,
                test_best_logs["k"],
                test_best_logs["f1"]*100,
                test_best_logs["preclass_f1"][0]*100,
                test_best_logs["preclass_f1"][1]*100,
                test_best_logs["preclass_f1"][2]*100,
                test_best_logs["preclass_f1"][3]*100,
                test_best_logs["preclass_f1"][4]*100
            ))

        if "post_test_results" in self.checkpoint:
            multi_fold_results.update({self.fold:self.checkpoint["post_test_results"]})
            print("-- Best Test --")
            print("Acc: {0:.1f}, Kappa: {1:.3f}, F1: {2:.1f}, f1_per_class: {3:.1f} {4:.1f} {5:.1f} {6:.1f} {7:.1f}".format(
                self.checkpoint["post_test_results"]["accuracy"]*100,
                self.checkpoint["post_test_results"]["k"],
                self.checkpoint["post_test_results"]["f1"]*100,
                self.checkpoint["post_test_results"]["preclass_f1"][0]*100,
                self.checkpoint["post_test_results"]["preclass_f1"][1]*100,
                self.checkpoint["post_test_results"]["preclass_f1"][2]*100,
                self.checkpoint["post_test_results"]["preclass_f1"][3]*100,
                self.checkpoint["post_test_results"]["preclass_f1"][4]*100
            ))

        return multi_fold_results

    def _load_encoder(self, encoders):
        # encs = []
        # for num_enc in range(len(encoders)):
        #
        #     enc_class = globals()[encoders[num_enc]["model"]]
        #     args = encoders[num_enc]["args"]
        #     enc = enc_class(args = args)
        #     enc = nn.DataParallel(enc, device_ids=[torch.device(0)])
        #
        #     if encoders[num_enc]["pretrainedEncoder"]["use"]:
        #         print("Loading encoder from {}".format(encoders[num_enc]["pretrainedEncoder"]["dir"]))
        #         checkpoint = torch.load(encoders[num_enc]["pretrainedEncoder"]["dir"])
        #         enc.load_state_dict(checkpoint["encoder_state_dict"])
        #     encs.append(enc)
        # return encs

        encs = []
        for num_enc in range(len(encoders)):
            enc_class = globals()[encoders[num_enc]["model"]]
            args = encoders[num_enc]["args"]
            # print(enc_class)
            if "encoders" in encoders[num_enc]:
                enc_enc = self._load_encoder(encoders = encoders[num_enc]["encoders"])
                enc = enc_class(encs=enc_enc, args=args)
            else:
                enc = enc_class(args=args, encs=[])
            # enc = nn.DataParallel(enc, device_ids=[torch.device(i) for i in self.config.training_params.gpu_device])
            pretrained_enc_args = encoders[num_enc].get("pretrainedEncoder", {})
            if pretrained_enc_args.get("use", False):
                pre_dir = pretrained_enc_args.get("dir", "")
                pre_dir = pre_dir.format(self.fold)
                if "save_base_dir" in self.config.model:
                    pre_dir = os.path.join(self.config.model.save_base_dir, pre_dir)
                print("Loading encoder from {}".format(pre_dir))
                checkpoint = torch.load(pre_dir)
                if "encoder_state_dict" in checkpoint:
                    checkpoint["encoder_state_dict"] = {key.replace("module.", ""): value for key, value in checkpoint["encoder_state_dict"].items()}
                    enc.load_state_dict(checkpoint["encoder_state_dict"])
                elif "model_state_dict" in checkpoint:
                    if "VaVL" not in encoders[num_enc]["model"]:
                        print("Replacing module")
                        checkpoint["best_model_state_dict"] = {key.replace("module.", ""): value for key, value in
                                                               checkpoint["best_model_state_dict"].items()}
                    # checkpoint["model_state_dict"] = {key.replace("module.", ""): value for key, value in checkpoint["model_state_dict"].items()}
                    enc.load_state_dict(checkpoint["best_model_state_dict"])

            encs.append(enc)
        return encs

    def print_aggregated_test(self, config_list, multi_fold_results):
        message = f"{'Model Name':75} - {'Steps':^6}- {'Epoch':^6}- {'No Impr':^7} - {'Norm':^5} - {'C':^5} - {'G':^5} - {'ShC':^5} - {'ShG':^5} - {'RC':^5} - {'RG':^5} - {'DSH':^5} - {'DR':^5}\n"
        # message = f"{'Model Name':55} - {'Norm':^5} - {'C':^5} - {'G':^5} - {'ShC':^5} - {'ShG':^5} - {'DSH':^5} \n"
        prev_name = ""
        for k in multi_fold_results:
            if "best_step" not in multi_fold_results[k]: continue
            dataset_name = config_list[int(k)].split("/")[2]
            name = dataset_name + "-" + config_list[int(k)].split("/")[-1]
            if k == 0:
                prev_name = dataset_name
            elif prev_name != dataset_name:
                prev_name = dataset_name
                # message += "\n"

            message += f"{name:75} -"
            message += f"{multi_fold_results[k]['best_step']:6} -" if "best_step" in multi_fold_results[
                k] else f"{' ':6} -"
            message += f"{multi_fold_results[k]['current_epoch']:6} -" if "current_epoch" in multi_fold_results[
                k] else f"{' ':6} -"
            message += f"{multi_fold_results[k]['steps_no_improve']:7}  -" if "steps_no_improve" in \
                                                                              multi_fold_results[
                                                                                  k] else f"{' ':7}  -"

            if "test_f1" in multi_fold_results[k]:
                for pred in multi_fold_results[k]['test_f1']:
                    if "combined" in pred or pred == "c" or pred == "g":
                        message += " {:.2f} &".format(multi_fold_results[k]['test_acc'][pred] * 100)
                if "combined_shg" in multi_fold_results[k]['test_f1'] and "combined_shc" in multi_fold_results[k][
                    'test_f1']:
                    message += " {:.2f} &".format(np.abs(
                        multi_fold_results[k]['test_acc']["combined_shg"] * 100 - multi_fold_results[k]['test_acc'][
                            "combined_shc"] * 100))
                if "combined_rg" in multi_fold_results[k]['test_f1'] and "combined_rc" in multi_fold_results[k][
                    'test_f1']:
                    message += " {:.2f} &".format(np.abs(
                        multi_fold_results[k]['test_acc']["combined_rg"] * 100 - multi_fold_results[k]['test_acc'][
                            "combined_rc"] * 100))
            elif "acc" in multi_fold_results[k]:
                for pred in multi_fold_results[k]['acc']:
                    if "combined" in pred or pred == "c" or pred == "g" or pred=="flow":
                        message += " {:.2f} &".format(multi_fold_results[k]['acc'][pred] * 100)
                if "combined_shg" in multi_fold_results[k]['acc'] and "combined_shc" in multi_fold_results[k][
                    'acc']:
                    message += " {:.2f} &".format(np.abs(
                        multi_fold_results[k]['acc']["combined_shg"] * 100 - multi_fold_results[k]['acc'][
                            "combined_shc"] * 100))
            elif "top5_acc" in multi_fold_results[k]:
                for pred in multi_fold_results[k]['top5_acc']:
                    if "combined" in pred or pred == "c" or pred == "g":
                        message += " {:.2f} &".format(multi_fold_results[k]['top5_acc'][pred] * 100)
            elif "val_acc" in multi_fold_results[k]:
                for pred in multi_fold_results[k]['val_acc']:
                    if "combined" in pred or pred == "c" or pred == "g":
                        message += " {:.2f} &".format(multi_fold_results[k]['val_acc'][pred] * 100)

            message += "\n"
        print(message)

