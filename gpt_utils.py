import torch
import tiktoken

from torch.utils.data import DataLoader

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


def generate_text_simple(model, tokens, max_new_tokens, context_length):
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
        probas = torch.softmax(logits, dim=-1)
        token_next = torch.argmax(probas, dim=-1, keepdim=True)
        tokens = torch.cat((tokens, token_next), dim=1)

    return tokens


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, tokens=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


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
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),  # Flatens along the batch dimension, i.e., the number of elements in a batch. For eg.
        # (2, 3, 50257) becomes (6, 50257).
        target_batch.flatten(),  # Flatens the entire array. For eg., (2, 3) becomes 6.
    )

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


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()

    return train_loss, val_loss


def train_model_simple(
    model,
    training_loader,
    validation_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    training_losses, validation_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in training_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, training_loader, validation_loader, device, eval_iter)
                training_losses.append(train_loss)
                validation_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

    return training_losses, validation_losses, track_tokens_seen
