from colorama import Fore
from utils.config import process_config, setup_logger, process_config_default
from collections import defaultdict
import argparse
from posthoc.Helpers.Helper_Importer import Importer
import numpy as np

def print_search(config_path, default_config_path, args):
    setup_logger()

    importer = Importer(config_name=config_path, default_files=default_config_path, device="cuda:0")
    m = ""
    if "fold" in args and args.fold is not None and args.fold != "None":
        m += "fold{}".format(args.fold)
    if "alpha" in args and args.alpha is not None and args.alpha != "None":
        m += "_alpha{}".format(args.alpha)
    if "tanh_mode" in args and args.tanh_mode is not None and args.tanh_mode != "None":
        m += "_tanhmode{}".format(args.tanh_mode)
    if "tanh_mode_beta" in args and args.tanh_mode_beta is not None and args.tanh_mode_beta != "None":
        m += "_beta{}".format(args.tanh_mode_beta)
    if "reg_by" in args and args.reg_by is not None and args.reg_by != "None":
        m += "_regby{}".format(args.reg_by)
    if "clip" in args and args.clip is not None and args.clip != "None":
        m += "_clip{}".format(args.clip)
    if "l" in args and args.l is not None and args.l != "None":
        m += "_l{}".format(args.l)
    if "l_diffsq" in args and args.l_diffsq is not None and args.l_diffsq != "None":
        m += "_ldiffsq{}".format(args.l_diffsq)
    if "lib" in args and args.lib is not None and args.lib != "None":
        m += "_lib{}".format(args.lib)
    if "kmepoch" in args and args.kmepoch is not None and args.kmepoch != "None":
        m += "_kmepoch{}".format(args.kmepoch)
    if "mmcosine_scaling" in args and args.mmcosine_scaling is not None and args.mmcosine_scaling != "None":
        m += "_mmcosinescaling{}".format(args.mmcosine_scaling)
    if "ilr_c" in args and "ilr_g" in args and args.ilr_c is not None and args.ilr_g is not None and args.ilr_c != "None" and args.ilr_g != "None":
        m += "_ilrcg{}_{}".format(args.ilr_c, args.ilr_g)
    if "ending_epoch" in args and args.ending_epoch is not None and args.ending_epoch != "None":
        m += "_endingepoch{}".format(args.ending_epoch)
    if "num_samples" in args and args.num_samples is not None and args.num_samples != "None":
        m += "_numsamples{}".format(args.num_samples)
    if "pow" in args and args.pow is not None and args.pow != "None":
        m += "_pow{}".format(args.pow)
    if "nstep" in args and args.nstep is not None and args.nstep != "None":
        m += "_nstep{}".format(args.nstep)
    if "contr_coeff" in args and args.contr_coeff is not None and args.contr_coeff != "None":
        m += "_contrcoeff{}".format(args.contr_coeff)
    if "kde_coeff" in args and args.kde_coeff is not None and args.kde_coeff != "None":
        m += "_kde_coeff{}".format(args.kde_coeff)
    if "etube" in args and args.etube is not None and args.etube != "None":
        m += "_etube{}".format(args.etube)
    if "temperature" in args and args.temperature is not None and args.temperature != "None":
        m += "_temp{}".format(args.temperature)
    if "shuffle_type" in args and args.shuffle_type is not None and args.shuffle_type != "None":
        m += "_st{}".format(args.shuffle_type)
    if "contr_type" in args and args.contr_type is not None and args.contr_type != "None":
        m += "_contrtype{}".format(args.contr_type)
    if "validate_with" in args and args.validate_with is not None and args.validate_with != "None":
        m += "_vld{}".format(args.validate_with)
    if "base_alpha" in args and args.base_alpha is not None and args.base_alpha != "None":
        m += "_basealpha{}".format(args.base_alpha)
    if "alpha_var" in args and args.alpha_var is not None and args.alpha_var != "None":
        m += "_alphavar{}".format(args.alpha_var)
    if "base_beta" in args and args.base_beta is not None and args.base_beta != "None":
        m += "_basebeta{}".format(args.base_beta)
    if "beta_var" in args and args.beta_var is not None and args.beta_var != "None":
        m += "_betavar{}".format(args.beta_var)

    if "lr" in args and args.lr is not None:
        m += "_lr{}".format(args.lr)
    if "wd" in args and args.wd is not None:
        m += "_wd{}".format(args.wd)
    if "cls" in args and args.cls is not None:
        m += "_{}".format(args.cls)
    if "batch_size" in args and args.batch_size is not None:
        m += "_bs{}".format(args.batch_size)

    importer.config.model.save_dir = importer.config.model.save_dir.format(m)

    try:
        importer.load_checkpoint()
    except:
        print("We could not load {}".format(importer.config.model.save_dir))
        return 0, 0

    val_metrics, test_metric = importer.print_progress(multi_fold_results={},
                                                 verbose=False,
                                                 latex_version=False)


    message = Fore.WHITE + "{}  ".format(importer.config.model.save_dir.split("/")[-1])
    if "acc" in val_metrics:
        for i, v in val_metrics["acc"].items():
            if i == "combined":
                message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i, v * 100)
    if test_metric and "acc" in test_metric:
        for i, v in test_metric["acc"].items():
            if i == "combined":
                message += Fore.MAGENTA + "Test_Acc_{}: {:.2f} ".format(i, v * 100)
    if args.printing is True:
        print(message + Fore.RESET)
    return val_metrics, test_metric

def print_mean(m: dict, val=True):
    agg = {}
    counts = defaultdict(int)  # Keep track of counts for non-dict metrics

    # Step 1: Collect values
    for fold in m:
        for metric in m[fold]:
            if isinstance(m[fold][metric], dict):
                if metric not in agg:
                    agg[metric] = defaultdict(list)
                if metric == "f1_perclass":
                    continue
                for pred in m[fold][metric]:
                    if pred == "combined":
                        agg[metric][pred].append(m[fold][metric][pred])
            else:
                if metric not in agg:
                    agg[metric] = []
                agg[metric].append(m[fold][metric])
                counts[metric] += 1

    # Step 2: Compute mean and std, and prepare the message
    message = ""
    if val:
        message += Fore.RED + "Val  "
    else:
        message += Fore.GREEN + "Test  "

    for metric in agg:
        if "acc" == metric:
            if isinstance(agg[metric], defaultdict):
                for pred in agg[metric]:
                    mean_value = np.mean(agg[metric][pred])
                    std_value = np.std(agg[metric][pred])
                    message += Fore.WHITE + "{}_{}: ".format(metric, pred)
                    message += Fore.LIGHTGREEN_EX + "{:.2f} + {:.2f} ".format(100 * mean_value, 100 * std_value)
            else:
                mean_value = np.mean(agg[metric])
                std_value = np.std(agg[metric])
                message += Fore.WHITE + "{}: ".format(metric)
                message += Fore.LIGHTGREEN_EX + "{:.2f} + {:.2f} ".format(100 * mean_value, 100 * std_value)
        if "acc_7" == metric:
            if isinstance(agg[metric], defaultdict):
                for pred in agg[metric]:
                    mean_value = np.mean(agg[metric][pred])
                    std_value = np.std(agg[metric][pred])
                    message += Fore.GREEN + "{}_{}: ".format(metric, pred)
                    message += Fore.LIGHTGREEN_EX + "{:.4f} + {:.4f} ".format(mean_value, std_value)
        elif "f1" == metric:
            if isinstance(agg[metric], defaultdict):
                for pred in agg[metric]:
                    mean_value = np.mean(agg[metric][pred])
                    std_value = np.std(agg[metric][pred])
                    message += Fore.GREEN + "{}_{}: ".format(metric, pred)
                    message += Fore.LIGHTGREEN_EX + "{:.4f} + {:.4f} ".format(mean_value, std_value)
        elif "ece" == metric:
            if isinstance(agg[metric], defaultdict):
                for pred in agg[metric]:
                    mean_value = np.mean(agg[metric][pred])
                    std_value = np.std(agg[metric][pred])
                    message += Fore.GREEN + "{}_{}: ".format(metric, pred)
                    message += Fore.LIGHTGREEN_EX + "{:.4f} + {:.4f} ".format(mean_value, std_value)
        elif "ceu" == metric:
            pred = "combined"
            for each_ceu in agg[metric][pred][0]:
                mean_value = np.concatenate([np.array([i[each_ceu]]) for i in agg[metric][pred]]).mean()
                message += Fore.LIGHTBLUE_EX + "{}_{}: {:.2f} ".format(metric, each_ceu, mean_value)

    if args.printing is True:
        end_char = "\n" if val else "\n\n"
        print(message, end=end_char)

    mean_acc_combined = np.mean(agg["acc"]["combined"])
    return mean_acc_combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My Command Line Program")
    parser.add_argument('--config', help="Number of config file")
    parser.add_argument('--default_config', help="Number of config file")
    parser.add_argument('--fold', help="Fold")
    parser.add_argument('--alpha', help="Alpha")
    parser.add_argument('--tanh_mode', help="tanh_mode")
    parser.add_argument('--tanh_mode_beta', help="tanh_mode_beta")
    parser.add_argument('--reg_by', help="reg_by")
    parser.add_argument('--clip', help="Gradient Clip Value")
    parser.add_argument('--batch_size', help="batch_size")
    parser.add_argument('--l', help="L for Gat")
    parser.add_argument('--l_diffsq', help="L for Gat")
    parser.add_argument('--lib', help="L for Gat")
    parser.add_argument('--kmepoch', help="keep memory epoch")
    parser.add_argument('--num_samples', help="Number of samples for Gat")
    parser.add_argument('--pow', help="ShuffleGrad power")
    parser.add_argument('--nstep', help="ShuffleGrad nstep Reg-Dist-Sep")
    parser.add_argument('--contr_coeff', help="ShuffleGrad Contrastive Coefficient")
    parser.add_argument('--kde_coeff', help="ShuffleGrad kde_coeff Coefficient")
    parser.add_argument('--etube', help="ShuffleGrad Etube")
    parser.add_argument('--temperature', help="ShuffleGrad Contrastive Temperature")
    parser.add_argument('--contr_type', help="ShuffleGrad Contrastive type")
    parser.add_argument('--shuffle_type', help="shuffle_type")
    parser.add_argument('--validate_with', help="validate_with")
    parser.add_argument('--base_alpha', help="Synthetic Alpha")
    parser.add_argument('--alpha_var', help="Synthetic Alpha Variance")
    parser.add_argument('--base_beta', help="Synthetic Beta")
    parser.add_argument('--beta_var', help="Synthetic Beta Variance")
    parser.add_argument('--optim_method', help="Optim for Gat")
    parser.add_argument('--ilr_c', help="Initial Learning Rate Audio")
    parser.add_argument('--ilr_g', help="Initial Learning Rate Video")
    parser.add_argument('--mmcosine_scaling', help="mmcosine_scaling")
    parser.add_argument('--ending_epoch', help="Ending epoch")
    parser.add_argument('--load_ongoing', help="Ending epoch")
    parser.add_argument('--commonlayers', help="Fusion with Conformer Layers")
    parser.add_argument('--lr', required=False, help="Learning Rate", default=None)
    parser.add_argument('--wd', required=False, help="Weight Decay", default=None)
    parser.add_argument('--cls', required=False, help="CLS linear, nonlinear, highlynonlinear", default=None)
    parser.add_argument('--printing', required=False, help="print_results", default=True)
    args = parser.parse_args()

    config_li = list(args.config.split(","))
    val = {}
    test = {}
    if len(config_li) == 1:
        if "UCF101" in args.config:
            for i in range(1,4):
                args.fold = i
                val_metric, test_metric = print_search(config_path=args.config, default_config_path=args.default_config, args=args)
                val[i] = val_metric
                test[i] = test_metric
        else:
            if args.fold is None:
                val_metric, test_metric = print_search(config_path=args.config, default_config_path=args.default_config,
                                                       args=args)
                val[0] = val_metric
                test[0] = test_metric
            else:
                for i in range(3):
                    args.fold = i
                    val_metric, test_metric = print_search(config_path=args.config, default_config_path=args.default_config, args=args)
                    val[i] = val_metric
                    test[i] = test_metric
    else:
        for i in config_li:
            val_metric, test_metric = print_search(config_path=i, default_config_path=args.default_config, args=args)
            val[i] = val_metric
            test[i] = test_metric

    # try:
    print_mean(val, val=True)
    mean_test = print_mean(test, val=False)

        # import sys
        # sys.exit(mean_test)
    # except:
    #     print("There was an error in the print_mean function")
    #     pass