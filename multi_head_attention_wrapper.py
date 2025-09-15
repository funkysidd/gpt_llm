import torch

from logger import Logging, LogLevel

from causal_attention import CausalAttention

class MultiHeadAttentionWrapper(torch.nn.Module):
    def __init__(self, input_feature_count:int, output_feature_count:int, context_length:int, dropout:float,
                 num_heads:int, qkv_bias=False):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [CausalAttention(input_feature_count, output_feature_count, context_length, dropout, qkv_bias)
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

if __name__ == '__main__':
    torch.manual_seed(123)
    Logging.set_log_level(LogLevel.INFO)

    # Token embeddings; made up.
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89], # Your     (x^1)
            [0.55, 0.87, 0.66], # journey  (x^2)
            [0.57, 0.85, 0.64], # starts   (x^3)
            [0.22, 0.58, 0.33], # with     (x^4)
            [0.77, 0.25, 0.10], # one      (x^5)
            [0.05, 0.80, 0.55]  # step     (x^6)
        ]
    )

    batch = torch.stack((inputs, inputs), dim=0)
    mhaw = MultiHeadAttentionWrapper(inputs.shape[1], 1, batch.shape[1], 0.0, 2, False)
    context_vectors = mhaw(batch)

    Logging.log(LogLevel.INFO, f'context_vectors: {context_vectors}')