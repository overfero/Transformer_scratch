from datasets import load_dataset  # type: ignore
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from data.dataset import BilingualDataset
from data.tokenizer import get_or_build_tokenizer


def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item["translation"][lang]


def get_dataset(config):
    dataset_raw = load_dataset(
        "Helsinki-NLP/opus-100",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train",
    )

    print(f"GPU {config.local_rank} - Loading tokenizers...")
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config["lang_tgt"])

    train_ds_size = int(0.9 * len(dataset_raw))
    val_ds_size = len(dataset_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(dataset_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_len_src = 0
    max_len_tgt = 0
    for item in dataset_raw:
        max_len_src = max(
            max_len_src,
            len(tokenizer_src.encode(item["translation"][config["lang_src"]]).ids),
        )
        max_len_tgt = max(
            max_len_tgt,
            len(tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids),
        )

    print(f"GPU {config.local_rank} - Max length of source sentence: {max_len_src}")
    print(f"GPU {config.local_rank} - Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        shuffle=False,
        sampler=DistributedSampler(train_ds, shuffle=True),
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def load_nex_batch(config, device):
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    batch = next(iter(val_dataloader))
    encoder_input = batch["encoder_input"].to(device)
    # encoder_mask = batch["encoder_mask"].to(device)
    decoder_input = batch["decoder_input"].to(device)
    # decoder_mask = batch["decoder_mask"].to(device)

    encoder_input_token = [
        tokenizer_src.id_to_token(x) for x in encoder_input[0].cpu().numpy()
    ]
    decoder_input_token = [
        tokenizer_tgt.id_to_token(x) for x in decoder_input[0].cpu().numpy()
    ]

    # model_output = greedy_decode(
    #     model,
    #     encoder_input,
    #     encoder_mask,
    #     tokenizer_src,
    #     tokenizer_tgt,
    #     config["seq_len"],
    #     device,
    # )

    return batch, encoder_input_token, decoder_input_token
