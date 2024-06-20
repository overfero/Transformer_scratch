import torch
import torch.nn as nn


class BilingualDataset(nn.Module):
    def __init__(
        self, dataset, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt

        self.sos_token = torch.tensor(
            [tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src = torch.tensor(
            self.tokenizer_src.encode(item["translation"][self.lang_src]).ids,
            dtype=torch.int64,
        )
        tgt = torch.tensor(
            self.tokenizer_tgt.encode(item["translation"][self.lang_tgt]).ids,
            dtype=torch.int64,
        )

        enc_num_pad_tokens = self.seq_len - src.shape[0] - 2
        dec_num_pad_tokens = self.seq_len - tgt.shape[0] - 1

        if enc_num_pad_tokens < 0 or dec_num_pad_tokens < 0:
            raise ValueError("Sentence too long")

        encoder_input = torch.cat(
            [
                self.sos_token,
                src,
                self.eos_token,
                self.pad_token.repeat(enc_num_pad_tokens),
            ]
        )
        decoder_input = torch.cat(
            [self.sos_token, tgt, self.pad_token.repeat(dec_num_pad_tokens)]
        )
        label = torch.cat(
            [tgt, self.eos_token, self.pad_token.repeat(dec_num_pad_tokens)]
        )

        assert encoder_input.shape[0] == self.seq_len
        assert decoder_input.shape[0] == self.seq_len
        assert label.shape[0] == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & (
                torch.triu(torch.ones(1, self.seq_len, self.seq_len), diagonal=1) == 0
            ).int(),
            "label": label,
            "src_text": item["translation"][self.lang_src],
            "tgt_text": item["translation"][self.lang_tgt],
        }
