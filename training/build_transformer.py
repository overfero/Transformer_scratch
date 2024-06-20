import torch.nn as nn

from transformer_model.attentions.multi_head_attention import MultiHeadAttentionBlock
from transformer_model.decoders.decoder import Decoder
from transformer_model.decoders.decoder_block import DecoderBlock
from transformer_model.encoders.encoder import Encoder
from transformer_model.encoders.encoder_block import EncoderBlock
from transformer_model.feed_forwards.feed_forward_block import FeedForwardBlock
from transformer_model.feed_forwards.projection_layer import ProjectionLayer
from transformer_model.inputs.input_embedding import InputEmbeddings
from transformer_model.inputs.positional_encoding import PositionalEncodings
from transformer_model.transformer import Transformer


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PositionalEncodings(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncodings(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_blocks.append(
            EncoderBlock(
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout,
            )
        )
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    decoder_blocks = []
    for _ in range(N):
        decoder_blocks.append(
            DecoderBlock(
                MultiHeadAttentionBlock(d_model, h, dropout),
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout,
            )
        )
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    proj_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj_layer
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
