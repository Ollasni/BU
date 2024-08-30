
from utils.config import process_config, setup_logger, process_config_default
from agents.general_agent import *

# xrandr --output DP-4 --scale 0.8x0.8

import argparse
import logging

def main(config_path, default_config_path, args):
    setup_logger()

    config = process_config_default(config_path, default_config_path)

    m = ""

    if "fold" in args and args.fold is not None:
        config.dataset.data_split.fold = int(args.fold)
        config.dataset.fold = int(args.fold)
        m += "fold{}".format(args.fold)
        seeds = [0, 109, 19, 337] if "UCF" in config_path else [109, 19, 337]
        config.training_params.seed = int(seeds[int(args.fold)])
        if "norm_wav_path" in config.dataset:
            config.dataset.norm_wav_path = config.dataset.norm_wav_path.format(args.fold)
        if "norm_face_path" in config.dataset:
            config.dataset.norm_face_path = config.dataset.norm_face_path.format(args.fold)
        if hasattr(config.model, "encoders"):
            for i in range(len(config.model.encoders)):
                config.model.encoders[i].pretrainedEncoder.dir = config.model.encoders[i].pretrainedEncoder.dir.format(args.fold)
    if "alpha" in args and args.alpha is not None:
        config.model.args.bias_infusion.alpha = float(args.alpha)
        m += "_alpha{}".format(args.alpha)
    if "tanh_mode" in args and args.tanh_mode is not None:
        config.model.args.bias_infusion.tanh_mode = int(args.tanh_mode)
        m += "_tanhmode{}".format(args.tanh_mode)
    if "tanh_mode_beta" in args and args.tanh_mode_beta is not None:
        config.model.args.bias_infusion.tanh_mode_beta = float(args.tanh_mode_beta)
        m += "_beta{}".format(args.tanh_mode_beta)
    if "reg_by" in args and args.reg_by is not None:
        config.model.args.bias_infusion.reg_by = args.reg_by
        m += "_regby{}".format(args.reg_by)
    if "clip" in args and args.clip is not None:
        config.model.args.clip_grad = True
        config.model.args.clip_value = float(args.clip)
        m += "_clip{}".format(args.clip)
    if "l" in args and args.l is not None:
        config.model.args.bias_infusion.l = float(args.l)
        m += "_l{}".format(args.l)
    if "l_diffsq" in args and args.l_diffsq is not None:
        config.model.args.bias_infusion.l_diffsq = float(args.l_diffsq)
        m += "_ldiffsq{}".format(args.l_diffsq)
    if "lib" in args and args.lib is not None:
        config.model.args.bias_infusion.lib = float(args.lib)
        m += "_lib{}".format(args.lib)
    if "kmepoch" in args and args.kmepoch is not None:
        config.model.args.bias_infusion.keep_memory_epoch = int(args.kmepoch)
        m += "_kmepoch{}".format(args.kmepoch)

    if "mmcosine_scaling" in args and args.mmcosine_scaling is not None:
        config.model.args.bias_infusion.mmcosine_scaling = float(args.mmcosine_scaling)
        m += "_mmcosinescaling{}".format(args.mmcosine_scaling)

    if "load_ongoing" in args and args.load_ongoing is not None:
        config.model.load_ongoing = args.load_ongoing

    if "ilr_c" in args and "ilr_g" in args and args.ilr_c is not None and args.ilr_g is not None:
        config.model.args.bias_infusion.init_learning_rate = {
          "c" : float(args.ilr_c),
          "g" : float(args.ilr_g)
        }
        m += "_ilrcg{}_{}".format(args.ilr_c, args.ilr_g)

    if "ending_epoch" in args and args.ending_epoch is not None:
        config.model.args.bias_infusion.ending_epoch = int(args.ending_epoch)
        m += "_endingepoch{}".format(args.ending_epoch)
    if "num_samples" in args and args.num_samples is not None:
        config.model.args.bias_infusion.num_samples = int(args.num_samples)
        m += "_numsamples{}".format(args.num_samples)
    if "pow" in args and args.pow is not None:
        config.model.args.bias_infusion.pow = int(args.pow)
        m += "_pow{}".format(args.pow)
    if "nstep" in args and args.nstep is not None:
        config.model.args.bias_infusion.nstep = int(args.nstep)
        m += "_nstep{}".format(args.nstep)
    if "contr_coeff" in args and args.contr_coeff is not None:
        config.model.args.bias_infusion.contr_coeff = float(args.contr_coeff)
        m += "_contrcoeff{}".format(args.contr_coeff)
    if "kde_coeff" in args and args.kde_coeff is not None:
        config.model.args.bias_infusion.kde_coeff = float(args.kde_coeff)
        m += "_kde_coeff{}".format(args.kde_coeff)
    if "etube" in args and args.etube is not None:
        config.model.args.bias_infusion.etube = float(args.etube)
        m += "_etube{}".format(args.etube)
    if "temperature" in args and args.temperature is not None:
        config.model.args.bias_infusion.temperature = float(args.temperature)
        m += "_temp{}".format(args.temperature)
    if "shuffle_type" in args and args.shuffle_type is not None:
        config.model.args.bias_infusion.shuffle_type = args.shuffle_type
        m += "_st{}".format(args.shuffle_type)
    if "contr_type" in args and args.contr_type is not None:
        config.model.args.bias_infusion.contr_type = args.contr_type
        m += "_contrtype{}".format(args.contr_type)
    if "validate_with" in args and args.validate_with is not None:
        config.early_stopping.validate_with = args.validate_with
        m += "_vld{}".format(args.validate_with)
    if "base_alpha" in args and args.base_alpha is not None:
        config.dataset.base_alpha = float(args.base_alpha)
        m += "_basealpha{}".format(args.base_alpha)
    if "alpha_var" in args and args.alpha_var is not None:
        config.dataset.alpha_var = float(args.alpha_var)
        m += "_alphavar{}".format(args.alpha_var)
    if "base_beta" in args and args.base_beta is not None:
        config.dataset.base_beta = float(args.base_beta)
        m += "_basebeta{}".format(args.base_beta)
    if "beta_var" in args and args.beta_var is not None:
        config.dataset.beta_var = float(args.beta_var)
        m += "_betavar{}".format(args.beta_var)
        if hasattr(config.model, "encoders"):
            for i in range(len(config.model.encoders)):
                m_enc = ""
                m_enc += "_basealpha{}".format(args.base_alpha)
                m_enc += "_alphavar{}".format(args.alpha_var)
                m_enc += "_basebeta{}".format(args.base_beta)
                m_enc += "_betavar{}".format(args.beta_var)
                config.model.encoders[i].pretrainedEncoder.dir = config.model.encoders[i].pretrainedEncoder.dir.format(m_enc)
    if "optim_method" in args and args.optim_method is not None:
        config.model.args.bias_infusion.optim_method = args.optim_method
        m += "_optim{}".format(args.optim_method)
    if "lr" in args and args.lr is not None:
        config.optimizer.learning_rate = float(args.lr)
        m += "_lr{}".format(args.lr)
    if "wd" in args and args.wd is not None:
        config.optimizer.weight_decay = float(args.wd)
        m += "_wd{}".format(args.wd)
    if "cls" in args and args.cls is not None:
        config.model.args.cls_type = args.cls
        m += "_{}".format(args.cls)
    if "batch_size" in args and args.batch_size is not None:
        config.training_params.batch_size = int(args.batch_size)
        m += "_bs{}".format(args.batch_size)
    if "commonlayers" in args and args.commonlayers is not None:
        config.model.args.common_layer = int(args.commonlayers)
        m += "_commonlayers{}".format(args.commonlayers)

    config.model.save_dir = config.model.save_dir.format(m)

    logging.info("save_dir: {}".format(config.model.save_dir))
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    agent.finalize()


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
parser.add_argument('--lib', help="lib for Gat")
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
args = parser.parse_args()

for var_name in vars(args):
    var_value = getattr(args, var_name)
    if var_value == "None":
        setattr(args, var_name, None)

print(args)


main(config_path=args.config, default_config_path=args.default_config, args=args)