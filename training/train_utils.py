from pathlib import Path

import torch

from config import ModelConfig
from training.build_transformer import build_transformer


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"]
    )
    return model


def get_weights_file_path(config, epochs):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epochs}.pt"
    return str(Path(".") / model_folder / model_filename)


def get_latest_weights_file_path(config: ModelConfig) -> str:
    model_folder = config.model_folder
    # model_basename = config.model_basename
    # Check all files in the model folder
    model_files = Path(model_folder).glob("*.pt")
    # Sort by epoch number (ascending order)
    model_files = sorted(model_files, key=lambda x: int(x.stem.split("_")[-1]))
    if len(model_files) == 0:
        return None
    # Get the last one
    model_filename = model_files[-1]
    return str(model_filename)


def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(src, src_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(src).to(device)
    while True:
        if decoder_input.shape[1] >= max_len:
            break
        decoder_mask = (
            (
                torch.triu(
                    torch.ones(1, decoder_input.shape[1], decoder_input.shape[1]),
                    diagonal=1,
                )
                == 0
            )
            .type_as(src_mask)
            .to(device)
        )
        decoder_output = model.decode(
            encoder_output, src_mask, decoder_input, decoder_mask
        )
        prob = model.project(decoder_output[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(src).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model,
    val_dataset,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_state,
    writer,
    num_examples=2,
):
    model.eval()
    count = 0

    console_width = 80
    with torch.no_grad():
        for item in val_dataset:
            count += 1
            encoder_input = item["encoder_input"].to(device)
            encoder_mask = item["encoder_mask"].to(device)

            assert encoder_input.shape[0] == 1, "Batch size must be 1 for validation"
            model_output = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            src_text = item["src_text"][0]
            expected = item["tgt_text"][0]
            predicted = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            print_msg("-" * console_width)
            print_msg(f"Source: {src_text}")
            print_msg(f"Expected: {expected}")
            print_msg(f"Predicted: {predicted}")

            if count == num_examples:
                break
