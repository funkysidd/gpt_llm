from enum import Enum

# fmt: off
gpt2_base_config = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False,      # Query-Key-Value bias
}
# fmt: off

class GPT2Config(Enum):
    SMALL_124M = 0
    MEDIUM_355M = 1
    LARGE_774M = 2
    XLARGE_1558M = 3

openai_config_overrides = {
    GPT2Config.SMALL_124M: {"emb_dim": 768, "n_layers": 12, "n_heads": 12, "qkv_bias": True},
    GPT2Config.MEDIUM_355M: {"emb_dim": 1024, "n_layers": 24, "n_heads": 16, "qkv_bias": True},
    GPT2Config.LARGE_774M: {"emb_dim": 1280, "n_layers": 36, "n_heads": 20, "qkv_bias": True},
    GPT2Config.XLARGE_1558M: {"emb_dim": 1600, "n_layers": 48, "n_heads": 25, "qkv_bias": True},
}

def get_gpt2_config(config:GPT2Config):
    ret = gpt2_base_config.copy()
    match config:
        case GPT2Config.SMALL_124M:
            ret.update(openai_config_overrides[GPT2Config.SMALL_124M])
        case GPT2Config.MEDIUM_355M:
            ret.update(openai_config_overrides[GPT2Config.MEDIUM_355M])
        case GPT2Config.LARGE_774M:
            ret.update(openai_config_overrides[GPT2Config.LARGE_774M])
        case GPT2Config.XLARGE_1558M:
            ret.update(openai_config_overrides[GPT2Config.XLARGE_1558M])

    return ret
