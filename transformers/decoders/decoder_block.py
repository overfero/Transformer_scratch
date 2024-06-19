import torch.nn as nn

from transformers.attentions.multi_head_attention import MultiHeadAttentionBlock
from transformers.feed_forwards.feed_forward_block import FeedForwardBlock
from transformers.feed_forwards.residual_connection import ResidualConnection


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_att_block: MultiHeadAttentionBlock,
        cross_att_block: MultiHeadAttentionBlock,
        feed_frwd_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_att_block = self_att_block
        self.cross_att_block = cross_att_block
        self.feed_frwd_block = feed_frwd_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_att_block(x, x, x, tgt_mask)
        )
        x = self.residual_connection[1](
            x, lambda x: self.cross_att_block(x, enc_output, enc_output, src_mask)
        )
        x = self.residual_connection[2](x, self.feed_frwd_block)
        return x
