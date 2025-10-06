import torch
import tiktoken

from logger import Logging, LogLevel
from layer_norm import LayerNorm
from transformer_block import TransformerBlock
from gpt2_config import config as gpt2_124m_config

class GPTModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = torch.nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = torch.nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = torch.nn.Sequential(
            *[TransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )

        # NOTE: Really, all computations internally are on tensors of dimenion `emb_dim`.

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = torch.nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # Ideally, the batch should have a seq_len that matches the context_length. But the inputs in this case are 4
        # tokens each.
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            # The `device=in_idx.device` param causes the returned tensor to be on the same device (GPU?) where `in_idx`
            # is.
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

if __name__ == '__main__':
    torch.manual_seed(123)
    Logging.set_log_level(LogLevel.INFO)

    tokenizer = tiktoken.get_encoding('gpt2')

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch = []

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))

    # Batch dimension is (2, 4). This is passed to the model.
    batch = torch.stack(batch, dim=0)

    Logging.log(LogLevel.INFO, f'Input: {batch.shape}')

    model = GPTModel(gpt2_124m_config)

    # The logits contains the same number of tokens as in the input text. The last token is the predicted token. The
    # first `n-1` overlap with the input.
    logits = model(batch)

    Logging.log(LogLevel.INFO, f'Output: {logits.shape}')
