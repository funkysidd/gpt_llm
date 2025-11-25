import torch
import tiktoken

from torch.utils.data import Dataset

### The format_input function used in Chapter 7 is slightly different from this.
def format_input(entry: dict):
    formatted_text = ""
    has_input = entry["input"] != ""
    formatted_text = (
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a "
        "response that appropriately completes the request.\n\n"
        if has_input
        else "Below is an instruction that describes a task. Write a response that appropriately completes the "
        "request.\n\n"
    )
    formatted_text += f"### Instruction:\n{entry["instruction"]}\n\n"
    formatted_text += f"### Input:\n{entry["input"]}\n\n" if has_input is True else ""

    return formatted_text


def format_entry(entry: dict):
    formatted_text = format_input(entry) + f"### Response:\n{entry["output"]}\n"
    return formatted_text

class InstructionDataset(Dataset):
    def __init__(self, dataset, tokenizer: tiktoken.Encoding):
        self.entries = []
        for entry in dataset:
            tokens = tokenizer.encode(format_entry(entry))
            self.entries.append(tokens)

    def __getitem__(self, index):
        return self.entries[index]

    def __len__(self):
        return len(self.entries)


if __name__ == "__main__":
    entry = {
        "instruction": "Give three tips for staying healthy.",
        "input": "hoha",
        "output": "1. Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule.",
    }

    print(format_entry(entry))
