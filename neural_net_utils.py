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