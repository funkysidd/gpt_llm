import re

split_separators = r'([,.:;?_!"()\']|--|\s)'

class Vocabulary:
    def __init__(self, filepath:str):
        self.filepath = filepath
        self.str_to_int = {}
        self.generate()

    def generate(self):
        raw_text =  ""
        with open(file=self.filepath, encoding="utf-8") as f:
            raw_text = f.read()

        if raw_text:
            all_words = re.split(split_separators, raw_text)
            all_words = [item.strip() for item in all_words if item.strip()]
            processed = sorted(set(all_words))
            self.str_to_int = {k:i for i,k in enumerate(processed)}

    def get(self):
        return self.str_to_int

class Tokenizer:
    def __init__(self, str_to_int):
        self.str_to_int = str_to_int
        self.int_to_str = {k:i for i,k in str_to_int.items()}

        self.unknown_token = (65536, "<Unknown>")

    def encode(self, text):
        processed = re.split(split_separators, text)
        processed = [item.strip() for item in processed if item.strip()]
        return [self.str_to_int[item] if item in self.str_to_int else self.unknown_token[0] for item in processed]

    def decode(self, ids):
        text = ' '.join([self.int_to_str[item] if item in self.int_to_str else self.unknown_token[1] for item in ids])
        print(f'Joined  string (round 0): {text}')
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text) # Subset of punctuations
        print(f'Decoded string (round 1): {text}')
        text = re.sub(r'^(["])\s+', r'\1', text) # Replaces ` "` at the beginning of text with `"`.
        return text

def test_tokenizer():
    vocab = Vocabulary("./data/the-verdict.txt")

    tokenizer = Tokenizer(vocab.get())

    test_str = r'"Why _has_ he chucked painting?" I asked abruptly. Hullabaloo?'
    encoded = tokenizer.encode(test_str)
    print(f'Encoded string is: {encoded}')
    print(f'Decoded string (round n): {tokenizer.decode(encoded)}')

if __name__ == "__main__":
    test_tokenizer()

