from torchvision.datasets import MNIST
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from time import time
from torch.backends import mps
from torch import cuda, device

DOWNLOAD_DATA = True
BATCH_SIZE = 256
NUM_WORKERS = 1
EPOCHS = 5
RANDOM_SEED = 0
VALID_FRACTION = 0.1


def get_device() -> device:
    device_using = None
    if cuda.is_available():
        device_using = device("cuda:0")
    elif mps.is_available():
        if mps.is_built():
            device_using = device("mps")
        else:
            device_using = device("cpu")
    else:
        device_using = device("cpu")
    print(f"Using Device: {device_using}")
    return device_using


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolve = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classify = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.convolve(x)  # NCHW
        x = torch.flatten(x, 1)  # C*H*W
        x = self.classify(x)
        return x


def validate(dataloader: DataLoader):
    correct_predictions = 0
    total_samples = 0

    for _, (features, labels) in enumerate(dataloader):
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model.forward(features)
        _, predicted_labels = torch.max(logits, 1)

        correct_predictions += (labels == predicted_labels).sum()
        total_samples += features.size(0)

    loss = F.cross_entropy(logits, labels)
    accuracy = correct_predictions / total_samples * 100
    return loss, accuracy


if __name__ == "__main__":
    # TODO Set random seeds

    DEVICE = get_device()

    transforms = Compose([Resize((32, 32)), ToTensor(), Normalize((0.5,), (0.5,))])

    train_dataset = MNIST(
        root="./data", download=DOWNLOAD_DATA, train=True, transform=transforms
    )
    valid_dataset = MNIST(
        root="./data", download=DOWNLOAD_DATA, train=True, transform=transforms
    )
    test_dataset = MNIST(
        root="./data", download=DOWNLOAD_DATA, train=False, transform=transforms
    )

    train_size = len(train_dataset)
    valid_size = VALID_FRACTION * train_size
    train_indices = torch.arange(0, train_size - valid_size)
    valid_indices = torch.arange(train_size - valid_size, train_size)

    train_sampler = SubsetRandomSampler(
        [int(i.item()) for i in train_indices]
    )  # Messy to get around type warning
    valid_sampler = SubsetRandomSampler([int(i.item()) for i in valid_indices])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        sampler=train_sampler,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        sampler=valid_sampler,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )

    model = LeNet5().to(DEVICE)

    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, mode="max")

    tracker_dict = {}

    start_time = time()

    for epoch in range(EPOCHS):
        tracker_dict[epoch] = {}
        model.train()
        for batch, (features, labels) in enumerate(train_dataloader):
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model.forward(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tracker_dict[epoch][batch] = loss.item()

        model.eval()
        with torch.no_grad():
            train_loss, train_accuracy = validate(train_dataloader)
            valid_loss, valid_accuracy = validate(valid_dataloader)
            tracker_dict[epoch]["train_accuracy"] = train_accuracy
            tracker_dict[epoch]["valid_accuracy"] = valid_accuracy

        scheduler.step(valid_loss)

        print(
            f"Epoch:{epoch} | Train Accuracy: {train_accuracy:.2f}% | Valid Accuracy: {valid_accuracy:.2f}%"
        )

    test_loss, test_accuracy = validate(test_dataloader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    elapsed = (time() - start_time) / 60
    print(f"Elapsed time: {elapsed:.2f} seconds")
