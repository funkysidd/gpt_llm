import torch
import json
import time
import tiktoken
import argparse

from functools import partial
from rich.console import Console
from rich.status import Status
from torch.utils.data import DataLoader

from logger import Logging, LogLevel
from gpt_model import GPTModel
from instruction_dataset import InstructionDataset, format_input
from gpt2_configs import GPT2Config, get_gpt2_config
from gpt_utils import generate_text, generate_tokens, token_ids_to_text, text_to_token_ids, replace_linear_with_lora


class CallbackData:
    def __init__(self, status: Status = None):
        self.streaming_output = []
        self.status = status


def decode_tokens_and_print_text(token_next: torch.tensor, callback_data: CallbackData):
    # The incoming token could be formed of multiple words
    output = token_ids_to_text(token_ids=token_next, tokenizer=tiktoken.get_encoding("gpt2"))
    callback_data.streaming_output += output
    decoded_text = "".join(callback_data.streaming_output)

    response_splitted = decoded_text.split("Response:")
    if len(response_splitted) > 1:
        formatted_response = response_splitted[1].strip()
        callback_data.status.update(f"{formatted_response}")


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
    console = Console()

    parser = argparse.ArgumentParser(
        description="A utility to fine tune a GPT2 class neural network using a instruction dataset."
    )
    parser.add_argument("--enable-lora", "-l", help="Use LORA tuned dataset.", action="store_true")

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
    max_new_tokens = 1024
    temperature = 1.0
    top_k = 10

    callback_data = CallbackData()

    console.print("Ready to go :tada:", style="bold green", emoji=True)
    while True:
        user_input = input("Enter a prompt: ")
        if user_input.lower() in ["quit", "q", "exit"]:
            break

        entry = {"instruction": user_input, "input": ""}
        start_context = format_input(entry)

        callback_data.streaming_output.clear()

        with console.status("Generating...") as status:
            callback_data.status = status
            tokens = text_to_token_ids(start_context, tokenizer).to(device)
            generate_tokens(
                model,
                tokens,
                max_new_tokens,
                model.pos_emb.weight.shape[0],
                eos_id,
                temperature,
                top_k,
                decode_tokens_and_print_text,
                callback_data,
            )

        decoded_text = "".join(callback_data.streaming_output)
        response_splitted = decoded_text.split("Response:")
        if len(response_splitted) > 1:
            formatted_response = response_splitted[1].strip()
            console.print(f"{formatted_response}", style="grey70")
        else:
            console.print(f"{decoded_text}", style="orange1")

    console.print("Bye :wave:", style="bold green", emoji=True)
    Logging.log(LogLevel.INFO, "Exiting...")
