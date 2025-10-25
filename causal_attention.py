import torch

from logger import Logging, LogLevel


class CausalAttention(torch.nn.Module):
    def __init__(
        self,
        input_feature_count: int,
        output_feature_count: int,
        context_length,
        dropout: float,
        qkv_bias=False,
    ):
        super().__init__()

        self.input_feature_count = input_feature_count
        self.output_feature_count = output_feature_count
        self.W_queries = torch.nn.Linear(input_feature_count, output_feature_count, bias=qkv_bias)
        self.W_keys = torch.nn.Linear(input_feature_count, output_feature_count, bias=qkv_bias)
        self.W_values = torch.nn.Linear(input_feature_count, output_feature_count, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)

        # Registering a buffer with a module causes it to move it the GPU too, when the model iteself is moved.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length, dtype=bool), diagonal=1),
        )

    def forward(self, x):
        """
        Incoming input, x, is specified in batches. Its dimensions are (2, 6, 3), where 2 is the number of batches.
        """

        # `keys`, `queries`, and `values` are of dimensions (2, 6, 2).
        keys = self.W_keys(x)
        queries = self.W_queries(x)
        values = self.W_values(x)

        """
        Essentially we are transposing every batch separately. Here, dimensions 1 and 2 are transposed in every batch.

        `attention_scores` and `attention_weights` are of dimensions (2, 6, 6).
        """
        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill_(self.mask, -torch.inf)
        attention_weights = torch.softmax(attention_scores / self.output_feature_count**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # `context_vectors` is of dimensions (2, 6, 2).
        context_vectors = attention_weights @ values
        return context_vectors


if __name__ == "__main__":
    torch.manual_seed(123)
    Logging.set_log_level(LogLevel.INFO)

    # Token embeddings; made up.
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],  # step     (x^6)
        ]
    )

    batch = torch.stack((inputs, inputs), dim=0)
    ca = CausalAttention(inputs.shape[1], 2, batch.shape[1], 0.0)
    context_vectors = ca(batch)

    Logging.log(LogLevel.INFO, f"context_vectors: {context_vectors}")
