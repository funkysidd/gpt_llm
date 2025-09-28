import torch
import tiktoken

from logger import Logging, LogLevel

class DummyGPTModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = torch.nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = torch.nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = torch.nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )

        # NOTE: Really, all computations internally are on tensors of dimenion `emb_dim`.

        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = torch.nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        Logging.log(LogLevel.INFO, f'{in_idx.shape}')

        # Ideally, the batch should have a seq_len that matches the context_length. But the inputs in this case are 4
        # words/tokens each.
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x

if __name__ == '__main__':
    torch.manual_seed(123)
    Logging.set_log_level(LogLevel.INFO)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    tokenizer = tiktoken.get_encoding('gpt2')

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch = []

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))

    # Batch dimension is (2, 4). This is passed to the model.
    batch = torch.stack(batch, dim=0)

    model = DummyGPTModel(GPT_CONFIG_124M)

    # The logits contains the same number of tokens as in the input text. The last token is the predicted token. The
    # first `n-1`` overlap with the input.
    logits = model(batch)

    Logging.log(LogLevel.INFO, f'Output: {logits.shape}')
