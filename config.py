from dataclasses import dataclass


@dataclass
class ModelConfig:
    batch_size: int
    num_epochs: int
    lr: float
    seq_len: int
    d_model: int
    lang_src: str
    lang_tgt: str
    model_folder: str
    model_basename: str
    preload: str
    tokenizer_file: str
    local_rank: int = -1
    global_rank: int = -1


# def get_config():
#     return {
#         "lang_src": "en",
#         "lang_tgt": "id",
#         "batch_size": 8,
#         "seq_len": 600,
#         "tokenizer_file": "tokenizer_{0}.json",
#         "num_epochs": 20,
#         "lr": 0.0001,
#         "model_folder": "weights",
#         "model_basename": "tmodel_",
#         "preload": None,
#         "experiment_name": "runs/tmodel",
#         "d_model": 512,
#     }


def get_default_config() -> ModelConfig:

    return ModelConfig(
        batch_size=8,
        num_epochs=30,
        lr=1e-4,
        seq_len=600,
        d_model=512,
        lang_src="en",
        lang_tgt="id",
        model_folder="weights",
        model_basename="tmodel_{0:02d}.pt",
        preload="latest",
        tokenizer_file="tokenizer_{0}.json",
    )
