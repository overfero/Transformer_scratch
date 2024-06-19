import torch.nn as nn

from transformers.decoders.decoder import Decoder
from transformers.encoders.encoder import Encoder
from transformers.feed_forwards.projection_layer import ProjectionLayer
from transformers.inputs.input_embedding import InputEmbeddings
from transformers.inputs.positional_encoding import PositionalEncodings


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncodings,
        tgt_pos: PositionalEncodings,
        proj_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, enc_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)

    def project(self, x):
        return self.proj_layer(x)
