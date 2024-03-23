from dl.utils.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

if __name__ == "__main__":
    train_data = MNIST(root="./data", download=False, train=True, transform=ToTensor())

    img, target = train_data[0]
    dataloader = DataLoader(train_data, batch_size=256, num_workers=1)
    print(next(iter(dataloader)))
