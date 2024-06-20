from pathlib import Path

import torch
import tqdm  # type: ignore
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from data.data_utils import get_dataset
from training.train_utils import (
    get_latest_weights_file_path,
    get_model,
    get_weights_file_path,
    run_validation,
)


def train_model(config):
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    device = torch.device("cuda")
    print(f"GPU {config.local_rank} - Using device: {device}")

    Path(config.model_folder).mkdir(parents=True, exist_ok=True)

    print(f"GPU {config.local_rank} - Loading dataset...")
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    writer = SummaryWriter(config.experiment_name)

    initial_epoch = 0
    global_step = 0
    if config.preload != "":
        if config.preload == "latest":
            # Get the filename of the latest checkpoint
            model_filename = get_latest_weights_file_path(config)
        else:
            # In case we want to preload a specific checkpoint
            model_filename = get_weights_file_path(config, int(config.preload))

        if model_filename is not None:
            print(f"GPU {config.local_rank} - Preloading model {model_filename}")
            state = torch.load(model_filename)
            model.load_state_dict(state["model_state_dict"])
            initial_epoch = state["epoch"] + 1
            optimizer.load_state_dict(state["optimizer_state_dict"])
            global_step = state["global_step"]
            del state
        else:
            # If we couldn't find a model to preload, just start from scratch
            print(
                f"GPU {config.local_rank} - Could not find model to preload: {config.preload}. Starting from scratch"
            )

    model = DistributedDataParallel(model, device_ids=[config.local_rank])

    for epoch in range(initial_epoch, config.num_epochs):
        batch_iterator = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            encoder_output = model.module.encode(encoder_input, encoder_mask)
            decoder_output = model.module.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            proj_output = model.module.project(decoder_output)

            label = batch["label"].to(device)
            loss = criterion(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            batch_iterator.set_postfix({"loss": f"{loss.item(): 6.3f}"})

            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        if config.local_rank == 0:
            run_validation(
                model,
                val_dataloader,
                tokenizer_src,
                tokenizer_tgt,
                config.seq_len,
                device,
                lambda msg: batch_iterator.write(msg),
                global_step,
                writer,
            )

            model_filename = get_weights_file_path(config, epoch)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                },
                model_filename,
            )
