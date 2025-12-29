import os
import torch
import tiktoken

from gpt_model import GPTModel
from logger import Logging, LogLevel
from gpt2_configs import GPT2Config, get_gpt2_config
from gpt_download import load_gpt2
from gpt_utils import generate_tokens, text_to_token_ids, token_ids_to_text, load_weights_into_gpt


def clear_console():
    """Clears the console screen based on the operating system."""
    if os.name == "nt":
        # Command for Windows
        _ = os.system("cls")
    else:
        # Commands for Linux/macOS/Posix
        _ = os.system("clear")


def decode_tokens_and_print_text(token_next: torch.tensor, callback_data: list):
    clear_console()

    # The incoming token could be formed of multiple words
    output = token_ids_to_text(token_ids=token_next, tokenizer=tiktoken.get_encoding("gpt2"))
    callback_data += output
    Logging.log(LogLevel.INFO, "".join(callback_data))


if __name__ == "__main__":
    torch.manual_seed(123)
    Logging.set_log_level(LogLevel.INFO)

    gpt2_config = get_gpt2_config(GPT2Config.MEDIUM_355M)
    Logging.log(LogLevel.INFO, f"gpt2_config: {gpt2_config}")

    Logging.log(LogLevel.INFO, "Creating model...")
    model = GPTModel(gpt2_config)

    Logging.log(LogLevel.INFO, "Reading pre-trained weights from file(s)...")
    settings, params = load_gpt2("355M", "datasets//gpt2")

    Logging.log(LogLevel.INFO, "Loading pre-trained weights into model...")
    load_weights_into_gpt(model, params)

    Logging.log(LogLevel.INFO, "Choosing device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Logging.log(LogLevel.INFO, f"Moving model to {device}...")
    model.to(device)

    Logging.log(LogLevel.INFO, "Switching model to evaluation mode...")
    model.eval()

    Logging.log(LogLevel.INFO, "Generating text...")

    # test_str = "A young computer programmer, determined to build something amazing, decided to invent something even better - a powerful, groundbreaking, and groundbreaking computer."
    test_str = "The tiny house was cozy, yet it was also incredibly large."

    streaming_output = []
    generate_tokens(
        model=model,
        tokens=text_to_token_ids(test_str, tokenizer=tiktoken.get_encoding("gpt2")).to(device),
        max_new_tokens=256,
        context_length=model.pos_emb.weight.shape[0],
        eos_id=tiktoken.get_encoding("gpt2").encode("<|endoftext|>", allowed_special="all")[0],
        top_k=10,
        temperature=1.0,
        callback=decode_tokens_and_print_text,
        callback_data=streaming_output,
    )
