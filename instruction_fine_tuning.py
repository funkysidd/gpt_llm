import torch
import json
import time
import tiktoken
import argparse

from functools import partial
from torch.utils.data import DataLoader

from logger import Logging, LogLevel
from gpt_model import GPTModel
from instruction_dataset import InstructionDataset, format_input
from gpt2_configs import GPT2Config, get_gpt2_config
from gpt_download import load_gpt2
from gpt_utils import (
    custom_collate_function,
    load_weights_into_gpt,
    calc_loss_loader,
    train_model_simple,
)


def load_database(file_path: str, max_io_length: int = -1, truncate_size: int = -1):
    dataset = []

    with open(file_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    truncated_dataset = []
    if max_io_length != -1:
        for item in dataset:
            if len(item["input"]) <= max_io_length and len(item["output"]) <= max_io_length:
                truncated_dataset.append(item)
    else:
        truncated_dataset = dataset

    if truncate_size != -1:
        truncated_dataset = truncated_dataset[:truncate_size]

    dataset_size = len(truncated_dataset)

    Logging.log(LogLevel.INFO, f"Dataset size: {dataset_size}")

    training_list_size = int(0.85 * dataset_size)
    testing_list_size = int(0.1 * dataset_size)

    training_list = truncated_dataset[:training_list_size]
    testing_list = truncated_dataset[training_list_size : training_list_size + testing_list_size]
    validation_list = truncated_dataset[training_list_size + testing_list_size :]

    return training_list, testing_list, validation_list


def save_model(model: GPTModel, optimizer: torch.optim.AdamW, file_path: str):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        file_path,
    )


if __name__ == "__main__":
    torch.manual_seed(123)
    Logging.set_log_level(LogLevel.INFO)

    parser = argparse.ArgumentParser(
        description="A utility to fine tune a GPT2 class neural network using a instruction dataset."
    )
    parser.add_argument("input", help="Path to a JSON file containing the instruction dataset.")
    parser.add_argument("output", help="Path to a file where the model weights are written.")
    parser.add_argument("--batch_size", "-b", help="The batch size in a dataloader.", type=int, default=4)
    parser.add_argument(
        "--max_io_length", "-m", help="The max size of inputs and outputs for instructions.", type=int, default=-1
    )

    args = parser.parse_args()

    training_list, testing_list, validation_list = load_database(file_path=args.input, max_io_length=args.max_io_length)

    Logging.log(LogLevel.INFO, f"Training dataset size: {len(training_list)}")
    Logging.log(LogLevel.INFO, f"Testing dataset size: {len(testing_list)}")
    Logging.log(LogLevel.INFO, f"Validation dataset size: {len(validation_list)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logging.log(LogLevel.INFO, f"Device: {device}")

    specialized_collate_function = partial(custom_collate_function, device=device, allowed_max_length=1024)
    tokenizer = tiktoken.get_encoding("gpt2")

    num_workers = 0
    batch_size = args.batch_size

    training_dataset = InstructionDataset(training_list, tokenizer)
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        collate_fn=specialized_collate_function,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    validation_dataset = InstructionDataset(validation_list, tokenizer)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        collate_fn=specialized_collate_function,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    gpt2_config = get_gpt2_config(GPT2Config.MEDIUM_355M)
    Logging.log(LogLevel.INFO, f"gpt2_config: {gpt2_config}")

    Logging.log(LogLevel.INFO, "Creating model...")
    model = GPTModel(gpt2_config)

    Logging.log(LogLevel.INFO, "Reading pre-trained weights from file(s)...")
    settings, params = load_gpt2("355M", "datasets//gpt2")

    Logging.log(LogLevel.INFO, "Loading pre-trained weights into model...")
    load_weights_into_gpt(model, params)

    Logging.log(LogLevel.INFO, f"Moving model to device: {device}...")
    model.to(device)

    Logging.log(LogLevel.INFO, "Switching model to evaluation mode...")
    model.eval()

    Logging.log(LogLevel.INFO, "Computing initial losses...")
    with torch.no_grad():
        training_loss = calc_loss_loader(model=model, data_loader=training_loader, device=device, num_batches=5)
        validation_loss = calc_loss_loader(model=model, data_loader=validation_loader, device=device, num_batches=5)

    Logging.log(LogLevel.INFO, f"Initial training loss: {training_loss}")
    Logging.log(LogLevel.INFO, f"Initial validation loss: {validation_loss}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    num_epochs = 2
    start_context = format_input(validation_list[0])

    Logging.log(LogLevel.INFO, f"Start context: {start_context}")

    Logging.log(LogLevel.INFO, "Starting training...")
    start_time = time.time()
    train_model_simple(
        model=model,
        tokenizer=tiktoken.get_encoding("gpt2"),
        training_loader=training_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
        eval_freq=5,
        eval_iter=5,
        start_context=start_context,
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    Logging.log(LogLevel.INFO, f"Training completed in {execution_time_minutes:.2f} minutes.")

    Logging.log(LogLevel.INFO, f"Saving model and optimizer...")
    save_model(model, optimizer, file_path=args.output)
    Logging.log(LogLevel.INFO, f"... done")
