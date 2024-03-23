from torch.utils.data import Dataset
from typing import Union, Callable, Optional, Any, Tuple
from pathlib import Path
from urllib.request import urlretrieve
import gzip
import numpy as np
from PIL import Image


class MNIST(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        download: bool,
        train: bool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        if isinstance(root, str):
            self.root = Path(root).resolve()
        else:
            self.root = root
        self.download = download
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.download_url = "http://yann.lecun.com/exdb/mnist/"

        self.train_resources = {
            "images": "train-images-idx3-ubyte.gz",
            "labels": "train-labels-idx1-ubyte.gz",
        }
        self.test_resources = {
            "images": "t10k-images-idx3-ubyte.gz",
            "labels": "t10k-labels-idx1-ubyte.gz",
        }
        if self.train:
            self.resources = self.train_resources
        else:
            self.resources = self.test_resources

        self.workdir = self.root.joinpath(self.__class__.__name__)
        self.raw_path = self.workdir.joinpath("raw")

        if self.download:
            self._download()

        self.data, self.targets = self._read_data()

        assert len(self.data) == len(self.targets)

    def _download(self) -> None:
        self.raw_path.mkdir(exist_ok=True, parents=True)

        for resource in self.resources.values():
            urlretrieve(
                self.download_url + resource,
                self.raw_path.joinpath(resource),
            )

    def _read_data(self) -> Tuple[Any, Any]:
        with gzip.open(self.raw_path.joinpath(self.resources["images"]), "r") as f:
            f.read(16)
            buf = f.read()
            images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            images = images.reshape(-1, 28, 28, 1)

        with gzip.open(self.raw_path.joinpath(self.resources["labels"]), "r") as f:
            f.read(8)
            buf = f.read()
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return images, labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.squeeze(), mode=None)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
