import hydra
import os
import numpy as np
import torch
from glob import glob
# import wandb

import utils
import model
import dataset

from riemdiffpp import HyperSphere, get_sm_loss, get_likelihood_fn, get_prob_ode

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf

from omegaconf import OmegaConf, open_dict


def load_hydra_config_from_run(load_dir):
    cfg_path = os.path.join(load_dir, ".hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg



def run_eval(eval_cfg, cfg, work_dir):

    manifold = HyperSphere
    logger = utils.get_logger(os.path.join(work_dir, "logs"))
    def mprint(msg):
        logger.info(msg)

    device = torch.device('cuda')
    mprint(device)

    # load in model
    score_net = model.LDM(cfg.size, cfg.model.dim, cfg.model.num_blocks, cfg.model.dropout, cfg.sigma_min, cfg.sigma_max, cfg.model.precond).to(device)
    ema = utils.ExponentialMovingAverage(score_net.parameters(), cfg.model.ema)
    path = sorted(glob(os.path.join(eval_cfg.load_dir, "ckpts/*.pth")))[-1]
    mprint(path)
    utils.load_checkpoint(path, score_net, ema, device)
    score_net.eval()
    ema.copy_to(score_net.parameters())
    l_fun = lambda x: get_likelihood_fn(manifold, exact_div=eval_cfg.exact_div)(get_prob_ode(manifold, score_net, sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max), x)
    
    
    # evaluate valid
    val_data = dataset.get_data(cfg.root, cfg.data, "valid", dim=cfg.size).to(device)
    val_prob = l_fun(val_data)
    cutoff = torch.quantile(val_prob, eval_cfg.p)
    mprint(f"{cutoff}, {val_prob.min()}, {val_prob.max()}")
    
    datasets = ["SVHN", "places365", "LSUN", "iSUN", "dtd"]

    for d in datasets:
        data = dataset.get_data(cfg.root, cfg.data, d, dim=cfg.size).to(device)
        prob = l_fun(data)
        mprint(f"{d}, {data.shape[0]}, {(prob > cutoff).sum()}")

from eval import run_eval

@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.run.dir if hydra_cfg.mode == RunMode.RUN else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    utils.makedirs(work_dir)

    logger = utils.get_logger(os.path.join(work_dir, "logs"))

    if hydra_cfg.mode == RunMode.RUN:
        wandb_name = work_dir.split("/")[-1]
    else:
        group = work_dir.split("/")[-2]
        run_id = work_dir.split("/")[-1]
        wandb_name = f"{group}_{run_id}"

    if hydra_cfg.mode != RunMode.RUN:
        logger.info(f"Run id: {hydra_cfg.job.id}, {wandb_name}")

    load_cfg = load_hydra_config_from_run(cfg.load_dir)

    try:
        run_eval(cfg, load_cfg, work_dir)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()