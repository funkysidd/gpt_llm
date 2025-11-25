import torch
import tiktoken

from logger import Logging, LogLevel
from gpt_model import GPTModel
from gpt_utils import (
    generate_text_simple,
    generate_text,
    create_dataloader_v1,
    calc_loss_loader,
    train_model_simple,
    text_to_token_ids,
    token_ids_to_text,
)
from gpt2_configs import gpt2_base_config

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

    Logging.log(LogLevel.INFO, "Adjusting context_length to 256...")
    gpt2_config = gpt2_base_config.copy()
    gpt2_config["context_length"] = 256

    Logging.log(LogLevel.INFO, "Creating training loader...")
    training_loader = create_dataloader_v1(
        training_text,
        batch_size=2,
        max_length=gpt2_config["context_length"],
        stride=gpt2_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    Logging.log(LogLevel.INFO, "Creating validation loader...")
    validation_loader = create_dataloader_v1(
        validation_text,
        batch_size=2,
        max_length=gpt2_config["context_length"],
        stride=gpt2_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    if Logging.log_level == LogLevel.VERBOSE:
        for i, (input, output) in enumerate(training_loader):
            Logging.log(LogLevel.VERBOSE, f"Training batch {i}: {input.shape, output.shape}")

        for i, (input, output) in enumerate(validation_loader):
            Logging.log(LogLevel.VERBOSE, f"Validation batch {i}: {input.shape, output.shape}")

    Logging.log(LogLevel.INFO, "Creating model...")
    model = GPTModel(gpt2_config)

    Logging.log(LogLevel.INFO, "Choosing device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Logging.log(LogLevel.INFO, f"Moving model to {device}...")
    model.to(device)

    Logging.log(LogLevel.INFO, "Creating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    Logging.log(LogLevel.INFO, "Starting training...")
    train_model_simple(
        model=model,
        training_loader=training_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
        eval_freq=5,
        eval_iter=5,
        start_context="Every step moves you",
    )

    model.eval()

    torch.manual_seed(123)
    generated_tokens = generate_text_simple(
        model=model,
        tokens=text_to_token_ids("Every step moves you", tokenizer=tiktoken.get_encoding("gpt2")).to(device),
        max_new_tokens=25,
        context_length=model.pos_emb.weight.shape[0],
    )
    Logging.log(
        LogLevel.INFO, f"{token_ids_to_text(token_ids=generated_tokens, tokenizer=tiktoken.get_encoding("gpt2"))}"
    )

    torch.manual_seed(123)
    generated_tokens = generate_text(
        model=model,
        tokens=text_to_token_ids("Every effort moves you", tokenizer=tiktoken.get_encoding("gpt2")).to(device),
        max_new_tokens=15,
        context_length=model.pos_emb.weight.shape[0],
        top_k=25,
        temperature=1.4,
    )
    Logging.log(
        LogLevel.INFO, f"{token_ids_to_text(token_ids=generated_tokens, tokenizer=tiktoken.get_encoding("gpt2"))}"
    )

    # model.eval()
    # with torch.no_grad():
    #     training_loss = calc_loss_loader(model=model, data_loader=training_loader, device=device)
    #     validation_loss = calc_loss_loader(model=model, data_loader=validation_loader, device=device)

    # Logging.log(LogLevel.INFO, f"Training loss: {training_loss}")
    # Logging.log(LogLevel.INFO, f"Validation loss: {validation_loss}")
