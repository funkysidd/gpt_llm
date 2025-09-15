import torch

from logger import Logging, LogLevel

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_input_features:int, num_output_features:int, context_length:int, dropout:float,
                 num_heads:int, qkv_bias=False):

        super().__init__()
        assert (num_output_features % num_heads == 0), \
            "num_output_features must be divisible by num_heads"

        self.num_output_features = num_output_features
        self.num_heads = num_heads
        self.head_dim = num_output_features // num_heads # Rounds the value to a int, as opposed to a float

        self.W_queries = torch.nn.Linear(num_input_features, num_output_features, bias=qkv_bias)
        self.W_keys = torch.nn.Linear(num_input_features, num_output_features, bias=qkv_bias)
        self.W_values = torch.nn.Linear(num_input_features, num_output_features, bias=qkv_bias)

        # Unlike others, `out_proj` uses the same value for input and output features.
        self.out_proj = torch.nn.Linear(num_output_features, num_output_features)

        self.dropout = torch.nn.Dropout(dropout)

        self.register_buffer(
           'mask',
           torch.triu(torch.ones(context_length, context_length, dtype=bool), diagonal=1)
        )

    def forward(self, x):
        batch, num_tokens, d_in = x.shape
        keys = self.W_keys(x)
        queries = self.W_queries(x)
        values = self.W_values(x)

        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)

        # Flips `num_tokens`` with `num_heads``; this splits the matrices on a per head basis, each of which has the
        # dimension `head_dim`.
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # In causal_atention, this was (1, 2); since there is an additional dimension here, that changed to (2, 3).
        attention_scores = queries @ keys.transpose(2, 3)
        attention_scores.masked_fill_(self.mask, -torch.inf)

        # In causal_attention, we used `self.num_output_features` as opposed to `keys[-1]` which is 1 and not 2. That
        # is representative of `head_dim`, which is `self.output_feature_count // num_heads. Really, using `keys[-1]`
        # is a better idea since it expands to 1 or 2 depending on `num_heads.`
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Post multipication, flips `num_heads` with `num_tokens`. This in preparation to resotre to a state, prior to
        # splitting. The actual restoration happens just after.
        # Also, using contiguous() doesn't seem necessary.
        context_vectors = (attention_weights @ values).transpose(1, 2)
        context_vectors = context_vectors.contiguous().view(batch, num_tokens, self.num_output_features)

        # Projects computed `context_vectors`. This was not present in causal_attention. Disabling this operation
        # causes the output to match from causal_attention.
        context_vectors = self.out_proj(context_vectors)
        return context_vectors

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

    # Initialize input params
    _ , context_length, num_input_features = batch.shape
    num_output_features = 2
    num_heads = 2
    dropout = 0.0

    mha = MultiHeadAttention(num_input_features, 2, context_length, dropout, num_heads, False)
    context_vectors = mha(batch)

    Logging.log(LogLevel.INFO, f'context_vectors: {context_vectors}')