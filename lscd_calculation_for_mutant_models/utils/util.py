import numpy as np
import torch


class obj:
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def z_score_normalization(input, rmax=1, rmin=0):
    """
    scale the input to be between 0 and 1 (normalization)
    :param input: the layer's output tensor
    :param rmax: the upper bound of scale
    :param rmin: the lower bound of scale
    :return: scaled input
    """
    divider = input.max() - input.min()
    if divider == 0:
        return torch.zeros(input.shape)
    X_std = (input - input.min()) / divider
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def tensor2numpy(input, detach=False):
    if isinstance(input, np.ndarray):
        return input
    elif isinstance(input, torch.Tensor):
        return input.data.cpu().numpy()
    else:
        raise NotImplementedError("invalid input type for tensor2numpy")
