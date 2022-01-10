import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer
import random
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))

    # Initialize the random seeds for random,numpy, and pytorch
    if cfg.random_seed:
        cfg.random_seed = cfg.random_seed
    else:
        cfg.random_seed = random.randrange(2 ** 32 - 1)

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    trainer = Trainer(cfg)
    trainer.train()
    #
    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
