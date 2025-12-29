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
from gpt_utils import (
    generate_text,
    replace_linear_with_lora
)


def load_database(file_path: str, max_io_length: int = None, truncate_size: int = None):
    dataset = []

    with open(file_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    truncated_dataset = []
    if max_io_length is not None:
        for item in dataset:
            if len(item["input"]) <= max_io_length and len(item["output"]) <= max_io_length:
                truncated_dataset.append(item)
    else:
        truncated_dataset = dataset

    if truncate_size is not None:
        truncated_dataset = truncated_dataset[:truncate_size]

    dataset_size = len(truncated_dataset)

    Logging.log(LogLevel.INFO, f"Dataset size: {dataset_size}")

    training_list_size = int(0.85 * dataset_size)
    testing_list_size = int(0.1 * dataset_size)

    training_list = truncated_dataset[:training_list_size]
    testing_list = truncated_dataset[training_list_size : training_list_size + testing_list_size]
    validation_list = truncated_dataset[training_list_size + testing_list_size :]

    return training_list, testing_list, validation_list


if __name__ == "__main__":
    torch.manual_seed(123)
    Logging.set_log_level(LogLevel.INFO)

    parser = argparse.ArgumentParser(
        description="A utility to fine tune a GPT2 class neural network using a instruction dataset."
    )
    parser.add_argument(
        "--enable_lora", "-l", help="Use LORA tuned dataset.", type=bool, default=True
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logging.log(LogLevel.INFO, f"Device: {device}")

    gpt2_config = get_gpt2_config(GPT2Config.MEDIUM_355M)
    Logging.log(LogLevel.INFO, f"gpt2_config: {gpt2_config}")

    Logging.log(LogLevel.INFO, "Creating model...")
    model = GPTModel(gpt2_config)

    instruction_fine_tuning_path = "./datasets/alpaca_cleaned_fine_tuned.pth"
    if args.enable_lora:
        Logging.log(LogLevel.INFO, "Replacing linear modules with LORA...")
        instruction_fine_tuning_path = "./datasets/alpaca_cleaned_fine_tuned_lora.pth"
        replace_linear_with_lora(model, 16, 16)

    Logging.log(LogLevel.INFO, f"Moving model to device: {device}...")
    model.to(device)

    Logging.log(LogLevel.INFO, f"Reading pre-trained weights from file: {instruction_fine_tuning_path}")
    checkpoint = torch.load(instruction_fine_tuning_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    Logging.log(LogLevel.INFO, "Switching model to evaluation mode...")
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    # Constants
    eos_id = tokenizer.encode("<|endoftext|>", allowed_special="all")[0]
    max_new_tokens = 256
    temperature = 1.0
    top_k = 10

    while True:
        user_input = input("Enter a prompt: ")
        if user_input.lower() in ["quit", "q", "exit"]:
            break

        entry = {"instruction": user_input, "input": ""}
        start_context = format_input(entry)

        decoded_text = generate_text(
            model, tokenizer, device, start_context, max_new_tokens, eos_id, temperature, top_k
        )

        response_splitted = decoded_text.split("Response:")
        if len(response_splitted) > 1:
            formmatted_response = response_splitted[1].strip()
            Logging.log(LogLevel.INFO, f"Response: \033[31m{formmatted_response}\033[0m")
        else:
            Logging.log(LogLevel.WARNING, f"Invalid response: \033[97m{decoded_text}\033[0m")

    Logging.log(LogLevel.INFO, "Exiting...")
