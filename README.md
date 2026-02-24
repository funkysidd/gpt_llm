# GPT LLM

A GPT-2 class LLM implementation based on `Sebastian Raschaka's Build a Large Language Model (From Scratch)` [1]. The sole
intent was to understand the LLM architecture and build a functional implementation from scratch.

The model was was initialized with the 355M weights from OpenAI [2], and instruction fine tuned using the
AlpacaDataCleaned dataset [3].

A chat prompt is implemented to interact with the model.

# Requirements

See `requirements.txt` for using `instruction_testing.py`.

The requirements can be installed by using,

```python
python3 -m pip install -r requirements.txt
```

Additionally,

1. Download the weights file from the Google Drive [link](https://drive.google.com/file/d/1O3dYk_9_XyTnys-2AMIIc8RDvk_6XE2n/view?usp=drive_link).
2. Place it within a `datasets` directory relative to the `README.md`.

# Usage

To bring up a chat prompt, type the following on the terminal,

```python
python3 ./instruction_testing.py -l
```

As part of initializing the model, PyTorch looks for a CUDA capable device. If not found, the CPU is used.

To exit the LLM, type `q`, `quit`, or `exit` on the prompt.

# References

1. [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
2. [GPT-2](https://github.com/openai/gpt-2/tree/master)
3. [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned/tree/main)

