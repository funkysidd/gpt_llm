import torch

from logger import Logging, LogLevel


class SelfAttention_Simple(torch.nn.Module):
    def __init__(
        self,
        input_feature_count: int,
        output_feature_count: int,
        W_queries: torch.tensor,
        W_keys: torch.tensor,
        W_values: torch.tensor,
    ):
        super().__init__()

        self.input_feature_count = input_feature_count
        self.output_feature_count = output_feature_count
        self.W_queries = torch.nn.Parameter(W_queries)
        self.W_keys = torch.nn.Parameter(W_keys)
        self.W_values = torch.nn.Parameter(W_values)

    # Returns an instance initilized with random weights
    @classmethod
    def from_random_weights(self, input_feature_count: int, output_feature_count: int):
        return self(
            input_feature_count,
            output_feature_count,
            torch.rand(input_feature_count, output_feature_count),
            torch.rand(input_feature_count, output_feature_count),
            torch.rand(input_feature_count, output_feature_count),
        )

    def forward(self, inputs):
        queries = inputs @ self.W_queries
        keys = inputs @ self.W_keys
        values = inputs @ self.W_values

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / self.output_feature_count**0.5, dim=1)

        context_vectors = attention_weights @ values
        return context_vectors


class SelfAttention_Linear(torch.nn.Module):
    def __init__(self, input_feature_count: int, output_feature_count: int):
        super().__init__()

        self.input_feature_count = input_feature_count
        self.output_feature_count = output_feature_count
        self.W_queries = torch.nn.Linear(input_feature_count, output_feature_count, bias=False)
        self.W_keys = torch.nn.Linear(input_feature_count, output_feature_count, bias=False)
        self.W_values = torch.nn.Linear(input_feature_count, output_feature_count, bias=False)

    def forward(self, inputs):
        queries = self.W_queries(inputs)
        keys = self.W_keys(inputs)
        values = self.W_values(inputs)

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / self.output_feature_count**0.5, dim=1)

        context_vectors = attention_weights @ values
        return context_vectors


if __name__ == "__main__":
    torch.manual_seed(789)
    Logging.set_log_level(LogLevel.INFO)

    # Token embeddings; made up.
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],
        ]  # step     (x^6)
    )
    sa_linear = SelfAttention_Linear(3, 2)
    context_vectors_linear = sa_linear(inputs)

    Logging.log(LogLevel.INFO, f"context_vectors (linear): {context_vectors_linear}")

    sa_simple = SelfAttention_Simple.from_random_weights(3, 2)
    context_vectors_simple = sa_simple(inputs)
    Logging.log(LogLevel.INFO, f"context_vectors (simple): {context_vectors_simple}")

    # The weights in the linear variant are stored are stored as transposes of the initially specified dimensions. In
    # this case, the weights are stores as a 2x3 matrix, so they are transposed to retrieve the original 3x2 matrix.
    sa_simple = SelfAttention_Simple(
        3,
        2,
        sa_linear.W_queries.weight.T,
        sa_linear.W_keys.weight.T,
        sa_linear.W_values.weight.T,
    )
    context_vectors_simple = sa_simple(inputs)
    Logging.log(LogLevel.INFO, f"context_vectors (simple): {context_vectors_simple}")
