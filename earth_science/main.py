import hydra
import os
import numpy as np
import torch
import wandb

import utils
import model
import dataset

from riemdiffpp import Sphere, get_sm_loss, get_prob_ode, get_likelihood_fn, get_fm_loss

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf



def run_train(cfg, work_dir):

    manifold = Sphere
    logger = utils.get_logger(os.path.join(work_dir, "logs"))
    def mprint(msg):
        logger.info(msg)

    device = torch.device('cuda')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mprint(device)

    train_set, val_set, test_set = dataset.get_data(cfg)
    mprint(f"Sizes: {train_set.shape}, {val_set.shape}, {test_set.shape}.")
    train_loader = utils.cycle_loader(torch.utils.data.DataLoader(
        train_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=1, pin_memory=True
    ))
    train_iter = iter(train_loader)
    val_set, test_set = val_set.to(device), test_set.to(device)
    score_net = model.Model(cfg).to(device)
    ema = utils.ExponentialMovingAverage(score_net.parameters(), cfg.model.ema)
    opt = torch.optim.Adam(score_net.parameters(), lr=cfg.train.lr)

    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    mprint(score_net)
    mprint(cfg)
    mprint(f"EMA: {ema}")
    mprint(f"Optimizer: {opt}")

    loss_fn = get_sm_loss(manifold, sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max, sliced=cfg.ssm)
    best_val = float('inf')
    test_ll = None
    
    nll_func = lambda x: -get_likelihood_fn(manifold, sm_ode=True)(get_prob_ode(manifold, score_net, sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max), x)

    for itr in range(cfg.train.itrs):
        data = next(train_iter).to(device)

        opt.zero_grad()
        data = data.to(device)

        loss = loss_fn(score_net, data).mean()
        loss.backward()
        if itr % 100 == 0:
            mprint(f"Iter: {itr}. Loss: {loss.detach().item():.2f}.")
        opt.step()
        ema.update(score_net.parameters())

        if itr % 1000 == 0 and itr >= 25000:
            ema.store(score_net.parameters())
            ema.copy_to(score_net.parameters())
            val_ll = nll_func(val_set).mean()

            if val_ll < best_val:
                best_val = val_ll
                test_ll = nll_func(test_set).mean()

            ema.restore(score_net.parameters())

            mprint(f"Iter: {itr}, Valid NLL: {val_ll.detach().item():.2f}, Test NLL: {test_ll.detach().item():.2f}")

    mprint(f"Test LL: {test_ll}")

from main import run_train

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.run.dir if hydra_cfg.mode == RunMode.RUN else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    utils.makedirs(work_dir)

    logger = utils.get_logger(os.path.join(work_dir, "logs"))

    try:
        run_train(cfg, work_dir)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()
