import torch

def compute_accuracy(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader):
    model = model.eval()

    correct = 0.0
    total_examples = 0
    for (features, labels) in dataloader:
        with torch.no_grad():
            logits = model(features)

        # logits contains an entry for every dimension in the input. argmax returns the index associated with the max
        # element in logits. In the case of two dimensions (or classes), this returns 0 or 1. The labels array contains
        # one class each for every input, which is why we compare labels with predictions.
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples)

def compute_accuracy_gpu(rank, model:torch.nn.Module, dataloader:torch.utils.data.DataLoader):
    model = model.eval()

    correct = 0.0
    total_examples = 0
    for (features, labels) in dataloader:
        features, labels = features.to(rank), labels.to(rank)
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples)

def generate_text_simple(model, idx, max_new_tokens, context_length): 
    for _ in range(max_new_tokens):
        # Incoming idx may be artbitrarily large array of tokens; `-context_length` ensures that we only process the
        # last `context_length` tokens. Not to mention `idx` keeps on growing till we have processed `max_new_tokens`.
        # Also, if the incoming aray of tokens is smaller than `content_length`, it doesn't oveflow.
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond)

        # `-1` indicates the last column. That last colums in an array of `vocab_size` elements. The softmax is computed
        # on that array.
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx