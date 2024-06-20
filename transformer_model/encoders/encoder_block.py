import torch.nn as nn

from transformer_model.attentions.multi_head_attention import MultiHeadAttentionBlock
from transformer_model.feed_forwards.feed_forward_block import FeedForwardBlock
from transformer_model.feed_forwards.residual_connection import ResidualConnection


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_att_block: MultiHeadAttentionBlock,
        feed_frwd_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_att_block = self_att_block
        self.feed_frwd_block = feed_frwd_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_att_block(x, x, x, src_mask)
        )
        x = self.residual_connection[1](x, self.feed_frwd_block)
        return x
