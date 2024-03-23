from dl.utils.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

train_data = MNIST(root="./data", download=False, train=True, transform=ToTensor())

img, target = train_data[0]
dataloader = DataLoader(train_data, batch_size=256)
print(next(iter(dataloader)))
