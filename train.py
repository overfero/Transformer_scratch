import os

import torch
from torch.distributed import destroy_process_group, init_process_group

from config import get_default_config
from training.training import train_model

config = get_default_config()
config.local_rank = int(os.environ["LOCAL_RANK"])
config.global_rank = int(os.environ["RANK"])

if config.local_rank == 0:
    print("Configuration:")
    for key, value in config.__dict__.items():
        print(f"{key:>20}: {value}")

init_process_group(backend="nccl")
torch.cuda.set_device(config.local_rank)

train_model(config)

destroy_process_group()
