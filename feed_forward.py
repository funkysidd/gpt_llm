import torch

from logger import Logging, LogLevel
from gpt2_config import config as gpt2_124m_config
from gelu import GELU


class FeedForward(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            torch.nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
