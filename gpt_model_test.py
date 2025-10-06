import torch
import tiktoken

from logger import Logging, LogLevel
from gpt_model import GPTModel
from neural_net_utils import generate_text_simple
from gpt2_config import config as gpt2_124m_config

if __name__ == '__main__':
    torch.manual_seed(123)
    Logging.set_log_level(LogLevel.INFO)

    model = GPTModel(gpt2_124m_config)

    tokenizer = tiktoken.get_encoding('gpt2')
    txt = "Hello, I am"
    batch = torch.tensor(tokenizer.encode(txt)).unsqueeze(0)

    model.eval()
    out = generate_text_simple(model=model, idx=batch, max_new_tokens=6, context_length=gpt2_124m_config['context_length'])

    Logging.log(LogLevel.INFO, f'Input: {txt}')
    Logging.log(LogLevel.INFO, f'Output: {tokenizer.decode(out.squeeze(0).tolist())}')
