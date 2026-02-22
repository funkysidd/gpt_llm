# GPT LLM

A GPT2 class LLM implementation based on Sebastian Raschaka's Build a Large Language Model (From Scratch) [1].

The sole intent is to understand the LLM architecture and build a functional implementation from scratch.

# Requirements

See `requirements.txt` for using `instruction_testing.py`.

The requirements can be installed by using,

```python
python3 -m pip install -r requirements.txt
```

Additionally, download the following weights file and place it within `datasets` directory relative to the `README.md`.

@todo:

# Usage

Starting the LLM,

```python
python3 ./instruction_testing.py -l
```

To exit the LLM, type `q`, `quit`, or `exit` on the prompt.

# References

1. [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
