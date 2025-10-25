import torch

from logger import Logging, LogLevel
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from layer_norm import LayerNorm
from gpt2_config import config as gpt2_124m_config


class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = MultiHeadAttention(
            num_input_features=cfg["emb_dim"],
            num_output_features=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = torch.nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


if __name__ == "__main__":
    torch.manual_seed(123)
    Logging.set_log_level(LogLevel.INFO)

    x = torch.rand(2, 4, gpt2_124m_config["emb_dim"])
    tb = TransformerBlock(gpt2_124m_config)

    y = tb(x)

    Logging.log(LogLevel.INFO, f"x.shape: {x.shape}")
    Logging.log(LogLevel.INFO, f"y.shape: {y.shape}")
