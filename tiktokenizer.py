import tiktoken
from logger import Logging, LogLevel

if __name__ == "__main__":
    Logging.set_log_level(LogLevel.INFO)

    tokenizer = tiktoken.get_encoding("gpt2")

    text = "Hello, my name is Siddharth."

    tokens = tokenizer.encode(text)
    Logging.log(LogLevel.INFO, "Encoded text : {}".format(tokens))
    Logging.log(LogLevel.INFO, "Decoded text : {}".format(tokenizer.decode(tokens)))
