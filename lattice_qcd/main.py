import hydra
import os
import numpy as np
import torch
# import wandb

import utils
import model
import dataset

from riemdiffpp import SU3, get_sm_loss

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf

torch.backends.cudnn.benchmark = True


def run_train(cfg, work_dir, wandb_name):

    manifold = SU3
    logger = utils.get_logger(os.path.join(work_dir, "logs"))
    def mprint(msg):
        logger.info(msg)

    # mprint('Initializing WANDB')
    # wandb.init(project="riemdiffpp_ood", dir=work_dir, name=wandb_name, group=cfg.data,
    #            config={"model": OmegaConf.to_container(cfg.model), "train": OmegaConf.to_container(cfg.train)})
    # mprint("WANDB Initialized")

    device = torch.device('cuda')
    mprint(device)

    train_set, val_set = dataset.get_data(cfg.root, cfg.val)

    mprint(f"Sizes: {train_set.shape}, {val_set.shape}.")
    train_loader = utils.cycle_loader(torch.utils.data.DataLoader(
        train_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=1, pin_memory=True
    ))
    val_loader = utils.cycle_loader(torch.utils.data.DataLoader(
        train_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=1, pin_memory=True
    ))

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    score_net = model.Model(cfg.model.num_blocks, cfg.model.dim, cfg.model.dropout, cfg.model.attn, cfg.sigma_min, cfg.sigma_max).to(device)
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

    loss_fn = get_sm_loss(manifold, sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max)

    # data = next(train_iter).to(device)
    for itr in range(cfg.train.itrs):
        data = next(train_iter).to(device)

        opt.zero_grad()

        loss = loss_fn(score_net, data).mean()
        loss.backward()

        opt.step()
        ema.update(score_net.parameters())

        if itr % 50 == 0:
            # wandb.log({"training_loss": loss}, step=itr)
            mprint(f"Iter: {itr}. Loss: {loss.detach().item():.2f}.")

        if itr % 100 == 0:
            with torch.no_grad():
                ema.store(score_net.parameters())
                ema.copy_to(score_net.parameters())

                val_data = next(val_iter).to(device)
                val_loss = loss_fn(score_net, val_data).mean()
                # wandb.log({"valid_loss": val_loss}, step=itr)
                mprint(f"Iter: {itr}. Validation Loss: {val_loss.detach().item():.2f}.")

                ema.restore(score_net.parameters())

        if itr % 50000 == 0 and itr > 0:
            # save model
            checkpoint_dir = os.path.join(work_dir, "ckpts")
            if not os.path.exists(checkpoint_dir):
                utils.makedirs(checkpoint_dir)
            utils.save_checkpoint(os.path.join(checkpoint_dir, f"ckpt_{itr}.pth"), score_net, ema)

from main import run_train

@hydra.main(version_base=None, config_path="configs", config_name="config")
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

    try:
        run_train(cfg, work_dir, wandb_name)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()