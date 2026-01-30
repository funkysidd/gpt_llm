import torch
import tiktoken
import argparse

from rich.console import Console
from rich.status import Status

from logger import Logging, LogLevel
from gpt_model import GPTModel
from instruction_dataset import format_input
from gpt2_configs import GPT2Config, get_gpt2_config
from gpt_utils import generate_tokens, token_ids_to_text, text_to_token_ids, replace_linear_with_lora


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


class Runner:
    def __init__(self, args):
        self.TOP_K = 10
        self.TEMPERATURE = 1.0
        self.MAX_NEW_TOKENS = 1024

        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.EOS_ID = self.tokenizer.encode("<|endoftext|>", allowed_special="all")[0]

        self.model = None
        self.device = None
        self.console = Console()
        self.callback_data = CallbackData()

    def initialize(self):
        with self.console.status("Initializing...") as status:
            status.update("Initializing: Choosing device...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            Logging.log(LogLevel.INFO, f"Device: {self.device}")

            gpt2_config = get_gpt2_config(GPT2Config.MEDIUM_355M)
            Logging.log(LogLevel.INFO, f"gpt2_config: {gpt2_config}")

            status.update("Initializing: Creating model...")
            self.model = GPTModel(gpt2_config)

            instruction_fine_tuning_path = "./datasets/alpaca_cleaned_fine_tuned.pth"
            if args.enable_lora:
                status.update("Initializing: Replacing linear modules with LORA...")
                instruction_fine_tuning_path = "./datasets/alpaca_cleaned_fine_tuned_lora.pth"
                replace_linear_with_lora(self.model, 16, 16)

            status.update(f"Initializing: Moving model to device: {self.device}...")
            self.model.to(self.device)

            status.update(f"Initializing: Reading pre-trained weights from file: {instruction_fine_tuning_path}")
            checkpoint = torch.load(instruction_fine_tuning_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            status.update("Initializing: Switching model to evaluation mode...")
            self.model.eval()

    def run(self):
        self.console.print("Ready to go :tada:", style="bold green", emoji=True)
        while True:
            user_input = input("Enter a prompt: ")
            if user_input.lower() in ["quit", "q", "exit"]:
                break

            entry = {"instruction": user_input, "input": ""}
            start_context = format_input(entry)

            self.callback_data.streaming_output.clear()

            with self.console.status("Generating...") as status:
                self.callback_data.status = status
                tokens = text_to_token_ids(start_context, self.tokenizer).to(self.device)
                generate_tokens(
                    self.model,
                    tokens,
                    self.MAX_NEW_TOKENS,
                    self.model.pos_emb.weight.shape[0],
                    self.EOS_ID,
                    self.TEMPERATURE,
                    self.TOP_K,
                    decode_tokens_and_print_text,
                    self.callback_data,
                )

            decoded_text = "".join(self.callback_data.streaming_output)
            response_splitted = decoded_text.split("Response:")
            if len(response_splitted) > 1:
                formatted_response = response_splitted[1].strip()
                self.console.print(f"{formatted_response}", style="grey70")
            else:
                self.console.print(f"{decoded_text}", style="orange1")

        self.console.print("Bye :wave:", style="bold green", emoji=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A utility to test a GPT2 class neural network trained on an instruction dataset."
    )
    parser.add_argument("--enable-lora", "-l", help="Use LORA tuned dataset.", action="store_true")
    parser.add_argument("--enable-logging", "-L", help="Enabled logging.", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(123)
    Logging.set_log_level(LogLevel.INFO if args.enable_logging else LogLevel.WARNING)

    runner = Runner(args)
    runner.initialize()
    runner.run()

    Logging.log(LogLevel.INFO, "Exiting...")
