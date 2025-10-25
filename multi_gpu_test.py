import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from neural_net import NeuralNetwork
from gpt_utils import compute_accuracy, compute_accuracy_gpu
from dataset import SampleDataset
from logger import Logging, LogLevel


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def prepare_dataset():
    X_train = torch.tensor([[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor(
        [
            [-0.8, 2.8],
            [2.6, -1.6],
        ]
    )
    y_test = torch.tensor([0, 1])

    train_ds = SampleDataset(X_train, y_train)
    test_ds = SampleDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(train_ds),
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(test_ds),
    )

    return train_loader, test_loader


# Code executed per process, each of which executes the model on the device (GPU)
def main(rank, world_size, num_epochs):
    # Given this is a new process, logging needs to be initialized again.
    Logging.set_log_level(LogLevel.INFO)

    ddp_setup(rank, world_size)
    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(input_features=2, output_features=2)
    model.to(rank)  # Model is transfered to the device
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model = DDP(model, device_ids=[rank])
    Logging.log(LogLevel.INFO, f"[GPU {rank}] About to begin training...")
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(rank), labels.to(rank)  # Operating data is transfered to the device

            logits = model(features)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Logging.log(
                LogLevel.INFO,
                "[GPU {:03d}] Epoch: {:03d}/{:03d}, batch: {:03d}/{:03d}, loss: {:.08f}".format(
                    rank, epoch + 1, num_epochs, batch_idx, len(train_loader), loss
                ),
            )

    Logging.log(
        LogLevel.INFO,
        f"[GPU {rank}] Accuracy (training): {compute_accuracy_gpu(rank, model, train_loader)}",
    )
    Logging.log(
        LogLevel.INFO,
        f"[GPU {rank}] Accuracy (testing): {compute_accuracy_gpu(rank, model, test_loader)}",
    )

    destroy_process_group()


if __name__ == "__main__":
    Logging.set_log_level(LogLevel.INFO)

    if torch.cuda.is_available():
        Logging.log(LogLevel.INFO, f"Number of GPUs available: {torch.cuda.device_count()}")
        torch.manual_seed(123)
        num_epochs = 3
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
        Logging.log(LogLevel.INFO, f"Done...")
    else:
        Logging.log(LogLevel.ERROR, "Cuda is not available, exiting!")
