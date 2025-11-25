import torch
import tiktoken
import numpy as np

from torch.utils.data import DataLoader

from logger import Logging, LogLevel
from gpt_dataset import GPTDatasetV1


def compute_accuracy(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
    model.eval()

    correct = 0.0
    total_examples = 0
    for features, labels in dataloader:
        with torch.no_grad():
            logits = model(features)

        # logits contains an entry for every dimension in the input. argmax returns the index associated with the max
        # element in logits. In the case of two dimensions (or classes), this returns 0 or 1. The labels array contains
        # one class each for every input, which is why we compare labels with predictions.
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples


def compute_accuracy_gpu(rank, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
    model.eval()

    correct = 0.0
    total_examples = 0
    for features, labels in dataloader:
        features, labels = features.to(rank), labels.to(rank)
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Transforms a `n` sized tensor to (1, n)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # Transforms a (1, n) sized tensor `n`
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model, tokens, max_new_tokens, context_length, eos_id=None):
    for _ in range(max_new_tokens):
        # Incoming idx may be artbitrarily large array of tokens; `-context_length` ensures that we only process the
        # last `context_length` tokens. Not to mention `idx` keeps on growing till we have processed `max_new_tokens`.
        # Also, if the incoming aray of tokens is smaller than `content_length`, it doesn't oveflow.
        tokens_current = tokens[:, -context_length:]
        with torch.no_grad():
            logits = model(tokens_current)

        # `-1` indicates the last column. That last column in an array of `vocab_size` elements. The softmax is
        # computed on that array. This also effectively collapses the second dimension, i.e., if logits had the shape
        # (1, 3, 50257), the resulting shape is (1, 50257).
        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        token_next = torch.argmax(probabilities, dim=-1, keepdim=True)
        
        if token_next == eos_id:
            break

        tokens = torch.cat((tokens, token_next), dim=1)

    return tokens


# An advanced implemention of the generate_text_simple function above that uses top-k sampling and temperature scaling.
def generate_text(model, tokens, max_new_tokens, context_length, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        tokens_current = tokens[:, -context_length:]
        with torch.no_grad():
            logits = model(tokens_current)

        logits = logits[:, -1, :]

        # Top-k sampling: The top `k` values are only used for considertion, remaining are set to -inf. That causes
        # their corrresponding softmax values to be 0.
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # Temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)
            token_next = torch.multinomial(probabilities, num_samples=1)
        else:
            token_next = torch.argmax(logits, dim=-1, keepdim=True)

        if token_next == eos_id:
            break

        tokens = torch.cat((tokens, token_next), dim=1)

    return tokens


def generate_and_print_sample(model, tokenizer, device, start_context, eos_id=None) -> str:
    model.eval()
    context_length = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    token_ids = generate_text_simple(model=model, tokens=encoded, max_new_tokens=50, context_length=context_length, eos_id=eos_id)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    decoded_text = decoded_text.replace("\n", " ")
    model.train()

    return decoded_text


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


def calc_loss_batch(model, input_batch, target_batch, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    # fmt: off
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),    # Flatens along the batch dimension, i.e., the number of elements in a batch. For eg.
                                 # (2, 3, 50257) becomes (6, 50257).
        target_batch.flatten(),  # Flatens the entire array i.e., (2, 3) becomes 6. Also, target_batch are tokens as-is.
                                 # They are not a probability disrtibution over 50257 tokens in the vocabulary, but the 
                                 # indicies in the vocablary of size 50257 tokens.
    )
    # fmt: on

    return loss


def calc_loss_loader(model, data_loader, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)  # Returns the number of batches
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def evaluate_model(model, training_loader, validation_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        training_loss = calc_loss_loader(model=model, data_loader=training_loader, device=device, num_batches=eval_iter)
        validation_loss = calc_loss_loader(
            model=model, data_loader=validation_loader, device=device, num_batches=eval_iter
        )
    model.train()

    return training_loss, validation_loss


def train_model_simple(
    model, training_loader, validation_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context
):
    training_losses, validation_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    tokenizer = tiktoken.get_encoding("gpt2")

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in training_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(model=model, input_batch=input_batch, target_batch=target_batch, device=device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                training_loss, validation_loss = evaluate_model(
                    model=model,
                    training_loader=training_loader,
                    validation_loader=validation_loader,
                    device=device,
                    eval_iter=eval_iter,
                )
                training_losses.append(training_loss)
                validation_losses.append(validation_loss)
                track_tokens_seen.append(tokens_seen)
                Logging.log(
                    LogLevel.INFO,
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Training loss {training_loss:.3f}, "
                    f"Validation loss {validation_loss:.3f}",
                )

        decoded_text = generate_and_print_sample(model, tokenizer, device, start_context)
        Logging.log(LogLevel.INFO, decoded_text)

    return training_losses, validation_losses, track_tokens_seen


def custom_collate_function(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    max_item_length = max(len(item) for item in batch)

    inputs = []
    targets = []
    for item in batch:
        input_list = item.copy()

        # `+` simply adds an item to an existing list
        padded_items = max_item_length - len(item)
        input_list += padded_items * [pad_token_id]
        target_list = input_list[1:] + [pad_token_id]

        # Find the offset of the first item added
        offset = max_item_length - padded_items
        target_list[offset:] = (max_item_length - offset) * [ignore_index]

        if allowed_max_length is not None:
            input_list = input_list[:allowed_max_length]
            target_list = target_list[:allowed_max_length]

        # Appends a list to collection of lists
        inputs.append(input_list)
        targets.append(target_list)

    inputs_tensor = torch.tensor(inputs).to(device)
    targets_tensor = torch.tensor(targets).to(device)

    return inputs_tensor, targets_tensor


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, " "Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(model, params):
    model.pos_emb.weight = assign(model.pos_emb.weight, params["wpe"])
    model.tok_emb.weight = assign(model.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        model.trf_blocks[b].att.W_queries.weight = assign(model.trf_blocks[b].att.W_queries.weight, q_w.T)
        model.trf_blocks[b].att.W_keys.weight = assign(model.trf_blocks[b].att.W_keys.weight, k_w.T)
        model.trf_blocks[b].att.W_values.weight = assign(model.trf_blocks[b].att.W_values.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        model.trf_blocks[b].att.W_queries.bias = assign(model.trf_blocks[b].att.W_queries.bias, q_b)
        model.trf_blocks[b].att.W_keys.bias = assign(model.trf_blocks[b].att.W_keys.bias, k_b)
        model.trf_blocks[b].att.W_values.bias = assign(model.trf_blocks[b].att.W_values.bias, v_b)

        model.trf_blocks[b].att.out_proj.weight = assign(
            model.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        model.trf_blocks[b].att.out_proj.bias = assign(
            model.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        model.trf_blocks[b].ff.layers[0].weight = assign(
            model.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        model.trf_blocks[b].ff.layers[0].bias = assign(
            model.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        model.trf_blocks[b].ff.layers[2].weight = assign(
            model.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        model.trf_blocks[b].ff.layers[2].bias = assign(
            model.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        model.trf_blocks[b].norm1.scale = assign(model.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        model.trf_blocks[b].norm1.shift = assign(model.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        model.trf_blocks[b].norm2.scale = assign(model.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        model.trf_blocks[b].norm2.shift = assign(model.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

    model.final_norm.scale = assign(model.final_norm.scale, params["g"])
    model.final_norm.shift = assign(model.final_norm.shift, params["b"])
    model.out_head.weight = assign(model.out_head.weight, params["wte"])
