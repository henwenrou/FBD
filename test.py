import os
import sys
sys.path.append(os.getcwd())
import argparse
from torch.utils.data import DataLoader
import numpy as np
import glob
from omegaconf import OmegaConf
from main import instantiate_from_config
import torch
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    parser.add_argument(
        "--ignore_base_data",
        action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.",
    )
    return parser

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    ckpt = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt_dir = os.path.join(logdir, "checkpoints")
            for f in os.listdir(ckpt_dir):
                if 'latest' in f:
                    ckpt = os.path.join(ckpt_dir, f)
            if ckpt is None:
                # fallback: pick any checkpoint (sorted) if latest not present
                ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")))
                if len(ckpts) > 0:
                    ckpt = ckpts[-1]
        print(f"logdir:{logdir}")
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.base = base_configs+opt.base

    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    model_config = config.pop("model", OmegaConf.create())
    print(model_config)

    gpu = True
    eval_mode = True
    show_config = False
    model = instantiate_from_config(model_config)

    if ckpt is None:
        raise ValueError("No checkpoint found. Please provide a checkpoint file or a logdir with checkpoints.")

    pl_sd=torch.load(ckpt, map_location="cpu")

    # Filter out band_mask_cache from checkpoint (can be recomputed on first forward)
    state_dict = pl_sd['model']
    state_dict = {k: v for k, v in state_dict.items() if 'band_mask_cache' not in k}

    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    val_loader = DataLoader(data.datasets["validation"], batch_size=1, num_workers=1)
    test_loader = DataLoader(data.datasets["test"], batch_size=1, num_workers=1)
    from engine import prediction_wrapper
    label_name=data.datasets["train"].all_label_names
    out_prediction_list, dsc_table, error_dict, domain_names = prediction_wrapper(model, test_loader, 0, label_name, save_prediction=True, device=device)

