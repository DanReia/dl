from torch.backends import mps
from torch import cuda, device


def get_device() -> device:
    if cuda.is_available():
        return device("cuda:0")

    elif mps.is_available():
        if mps.is_built():
            return device("mps")
        else:
            return device("cpu")
    else:
        return device("cpu")
