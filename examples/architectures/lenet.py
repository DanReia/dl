from dl.utils.datasets import MNIST
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

DOWNLOAD_DATA = False


class LeNet(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


if __name__ == "__main__":
    transforms = Compose([Resize((32, 32)), ToTensor(), Normalize((0.5,), (0.5,))])

    train_dataset = MNIST(
        root="./data", download=DOWNLOAD_DATA, train=True, transform=transforms
    )
    test_dataset = MNIST(
        root="./data", download=DOWNLOAD_DATA, train=False, transform=transforms
    )

    train_dataloader = DataLoader(train_dataset, batch_size=256, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=1)
