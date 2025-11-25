import torch
import tiktoken

from gpt_model import GPTModel
from logger import Logging, LogLevel
from gpt2_configs import GPT2Config, get_gpt2_config
from gpt_download import load_gpt2
from gpt_utils import generate_text, text_to_token_ids, token_ids_to_text, load_weights_into_gpt

if __name__ == "__main__":
    torch.manual_seed(123)
    Logging.set_log_level(LogLevel.INFO)

    gpt2_config = get_gpt2_config(GPT2Config.SMALL_124M)
    Logging.log(LogLevel.INFO, f"gpt2_config: {gpt2_config}")

    Logging.log(LogLevel.INFO, "Creating model...")
    model = GPTModel(gpt2_config)

    Logging.log(LogLevel.INFO, "Reading pre-trained weights from file(s)...")
    settings, params = load_gpt2("124M", "datasets//gpt2")

    Logging.log(LogLevel.INFO, "Loading pre-trained weights into model...")
    load_weights_into_gpt(model, params)

    Logging.log(LogLevel.INFO, "Choosing device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Logging.log(LogLevel.INFO, f"Moving model to {device}...")
    model.to(device)

    Logging.log(LogLevel.INFO, "Switching model to evaluation mode...")
    model.eval()

    Logging.log(LogLevel.INFO, "Generating text...")
    generated_tokens = generate_text(
        model=model,
        tokens=text_to_token_ids("The capital of India is", tokenizer=tiktoken.get_encoding("gpt2")).to(device),
        max_new_tokens=10,
        context_length=model.pos_emb.weight.shape[0],
        top_k=10,
        temperature=1.0,
    )
    Logging.log(
        LogLevel.INFO, f"{token_ids_to_text(token_ids=generated_tokens, tokenizer=tiktoken.get_encoding("gpt2"))}"
    )
