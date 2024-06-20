def get_config():
    return {
        "lang_src": "en",
        "lang_tgt": "id",
        "batch_size": 8,
        "seq_len": 600,
        "tokenizer_file": "tokenizer_{0}.json",
        "num_epochs": 20,
        "lr": 0.0001,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "experiment_name": "runs/tmodel",
        "d_model": 512,
    }
