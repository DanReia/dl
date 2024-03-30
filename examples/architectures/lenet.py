from dl.utils.datasets import MNIST
from dl.utils import get_device
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torch.nn.functional as F

DOWNLOAD_DATA = False
BATCH_SIZE = 256
EPOCHS = 5
RANDOM_SEED = 0
DEVICE = get_device()


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


if __name__ == "__main__":
    # TODO Set random seeds

    transforms = Compose([Resize((32, 32)), ToTensor(), Normalize((0.5,), (0.5,))])

    train_dataset = MNIST(
        root="./data", download=DOWNLOAD_DATA, train=True, transform=transforms
    )
    test_dataset = MNIST(
        root="./data", download=DOWNLOAD_DATA, train=False, transform=transforms
    )

    train_dataloader = DataLoader(train_dataset, batch_size=256, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=1)

    model = LeNet5().to(DEVICE)

    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, mode="max", verbose=True)

    for epoch in range(EPOCHS):
        model.train()
        for batch, (features, labels) in enumerate(train_dataloader):
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model.forward(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
