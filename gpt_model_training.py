import torch
import tiktoken

from logger import Logging, LogLevel
from gpt_model import GPTModel
from gpt_utils import generate_text_simple, create_dataloader_v1, calc_loss_loader
from gpt2_config import config_small_context as gpt2_124m_config

if __name__ == "__main__":
    torch.manual_seed(123)
    Logging.set_log_level(LogLevel.INFO)

    file_path = "./data/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    # Wouldn't it be a better idea to split using tokens? This could split mid word, as `text_data` is one long string.
    training_text_len = int(len(text_data) * 0.9)
    training_text = text_data[:training_text_len]
    validation_text = text_data[training_text_len:]

    training_loader = create_dataloader_v1(
        training_text,
        batch_size=2,
        max_length=gpt2_124m_config["context_length"],
        stride=gpt2_124m_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = create_dataloader_v1(
        validation_text,
        batch_size=2,
        max_length=gpt2_124m_config["context_length"],
        stride=gpt2_124m_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    for i, (input, output) in enumerate(training_loader):
        Logging.log(LogLevel.INFO, f"Training batch {i}: {input.shape, output.shape}")

    for i, (input, output) in enumerate(validation_loader):
        Logging.log(LogLevel.INFO, f"Validation batch {i}: {input.shape, output.shape}")

    model = GPTModel(gpt2_124m_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logging.log(LogLevel.INFO, f"Chosen device: {device}")

    model.to(device)
    with torch.no_grad():
        training_loss = calc_loss_loader(model, training_loader, device)
        validation_loss = calc_loss_loader(model, validation_loader, device)

    Logging.log(LogLevel.INFO, f"Training loss: {training_loss}")
    Logging.log(LogLevel.INFO, f"Validation loss: {validation_loss}")
