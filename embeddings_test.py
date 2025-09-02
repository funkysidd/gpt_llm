import torch
import tiktoken

from logger import Logging, LogLevel

if __name__ == '__main__':
    Logging.set_log_level(LogLevel.INFO)
    torch.manual_seed(123)

    # embedding_layer = torch.nn.Embedding(num_embeddings=6, embedding_dim=3)
    # layers = torch.nn.Sequential(
    #     torch.nn.Embedding(6, 2),
    #     torch.nn.Linear(2, 30), # by default, `bias is true
    #     torch.nn.ReLU(), # activation unit
    #     torch.nn.Linear(30, 20),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(20, 3)
    # )

    # Logging.log(LogLevel.INFO, layers)
    # token_embedding_layer = layers[0]

    tokenizer = tiktoken.encoding_for_model(model_name='gpt2')
    max_token_value = tokenizer.max_token_value
    max_tokens = max_token_value + 1

    Logging.log(LogLevel.INFO, f'Maximum number of tokens: {max_tokens}')

    token_embeddings_layer = torch.nn.Embedding(num_embeddings=max_tokens, embedding_dim=256)

    # The weights associated with the embedding layer requires gradient
    Logging.log(LogLevel.INFO, token_embeddings_layer.weight.requires_grad) # Evaluates to true

    # Look-ups into the embedding layer requires passing a tensor. The incoming tensor itself can be multi-dimensional.
    # For every element in the tensor, the corresponding value (itself a tensor) is looked up.
    #
    # In other words, if the incoming tensor is of dimension (a, b), then the outgoing tensor is of dimension (a, b, d),
    # where `d` is the dimension of the tensor in the look-up table.
    #
    # `arange` returns a 1D tensor that has values in the range [`start`, `end``).
    # Logging.log(LogLevel.INFO, token_embedding_layer(torch.arange(end=6)))

    Logging.log(LogLevel.INFO, token_embeddings_layer.weight.shape)

    with open(file='./data/the-verdict.txt', encoding="utf-8") as f:
        raw_text = f.read()

    input_arr = torch.zeros(8, 4, dtype=torch.int32)
    if raw_text:
        tokens = tokenizer.encode(raw_text)
        start_idx = 0
        for row in range(8):
            input_arr[row] = torch.tensor(tokens[start_idx : start_idx+4])
            start_idx += 4

    Logging.log(LogLevel.INFO, f'input_arr: {input_arr}')

    token_embeddings = token_embeddings_layer(input_arr)
    Logging.log(LogLevel.INFO, f'token_embeddings shape: {token_embeddings.shape}')
