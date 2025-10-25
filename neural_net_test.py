import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from neural_net import NeuralNetwork
from gpt_utils import compute_accuracy
from dataset import SampleDataset
from logger import Logging, LogLevel

if __name__ == '__main__':
    Logging.set_log_level(LogLevel.INFO)

    # Global torch options
    torch.manual_seed(123)
    torch.set_printoptions(sci_mode=False)

    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    train_ds = SampleDataset(X_train, y_train)
    test_ds = SampleDataset(X_test, y_test)

    # for i, (x, y) in enumerate(test_ds):
    #     print(f'Item {i}: X: {x}, y: {y}')

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, drop_last=True, num_workers=1)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=1)

    # for i, (features, labels) in enumerate(train_loader):
    #     print(f'Batch {i}: X: {features}, \n\ty: {labels}')

    model = NeuralNetwork(input_features=2, output_features=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    Logging.log(LogLevel.VERBOSE, f"Number of trainable params: {model.get_trainable_param_count()}")

    num_epochs = 3
    for epoch in range(3):
        # Invoked every epoch
        model.train()

        for batch_idx, (features, labels) in enumerate(train_loader):
            logits = model(features)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Logging.log(LogLevel.INFO, "Epoch: {0:03d}/{1:03d}, batch: {2:03d}/{3:03d}, loss: {4:.08f}".format(
                epoch+1,
                num_epochs,
                batch_idx,
                len(train_loader),
                loss))

    # Logging.log(LogLevel.INFO, "Evaluating model with training dataset")
    # with torch.no_grad():
    #     y_train_val = model(X_train)
    #     probablities = torch.softmax(y_train_val, dim=1)
    #     predictions = torch.argmax(probablities, dim=1)
    #     predictions_alt = torch.argmax(y_train_val, dim=1)

    # print(f'Eval result: {y_train_val}, \nprobablities: {probablities}, \npredictions: {predictions},'
    #       f'\npredictions_alt: {predictions_alt}')

    Logging.log(LogLevel.INFO, f'Accuracy (training): {compute_accuracy(model, train_loader)}')
    Logging.log(LogLevel.INFO, f'Accuracy (testing): {compute_accuracy(model, test_loader)}')
