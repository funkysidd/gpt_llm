import torch
import torch.nn.functional as F

from neural_net import NeuralNetwork
from torch.utils.data import Dataset, DataLoader
from logger import Logging, LogLevel

class SampleDataset(Dataset):
    def __init__(self, features:torch.tensor, labels:torch.tensor):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        return (self.features[index], self.labels[index])
    
    def __len__(self):
        return self.labels.shape[0]
    
if __name__ == '__main__':
    Logging.set_log_level(LogLevel.INFO)
    
    torch.manual_seed(123)

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

    model.train()

    num_epochs = 3
    for epoch in range(3):    
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

    Logging.log(LogLevel.INFO, "Evaluating model with training dataset")
    model.eval()
    with torch.no_grad():
        y_train_val = model(X_train)

    print(f"Eval result: {y_train_val}")
