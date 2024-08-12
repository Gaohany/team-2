import cv2
import kornia
import torch
import torch.nn.functional as F
from skimage.util import random_noise

# NOTE: rotation and shear should not be used unless the network is trained for it


def translation(img, params):
    _, height, width = img.shape
    M = torch.tensor([[1.0, 0.0, params], [0.0, 1.0, params]])
    dst = kornia.geometry.warp_affine(img.unsqueeze(dim=0), M.unsqueeze(dim=0), (height, width)).squeeze(dim=0)
    return dst


def scale(img, params):
    _, height, width = img.shape
    res = F.interpolate(img.unsqueeze(dim=0), scale_factor=(params, params), mode="bicubic").squeeze(dim=0)
    c, h, w = res.shape
    if params > 1:
        # need to crop
        startx = w // 2 - width // 2
        starty = h // 2 - height // 2
        return res[:, starty : starty + height, startx : startx + width]
    elif params < 1:
        # need to pad
        sty = int((height - h) / 2)
        stx = int((width - w) / 2)
        return F.pad(input=res, pad=[stx, width - w - stx, sty, height - h - sty], mode="constant", value=0)
    return res


def shear(img, params):
    _, height, width = img.shape
    factor = params * (-1.0)
    M = torch.tensor([[1, factor, 0], [0, 1, 0]])
    dst = kornia.geometry.warp_affine(img.unsqueeze(dim=0), M.unsqueeze(dim=0), (height, width)).squeeze(dim=0)
    return dst


def rotation(img, params):
    _, height, width = img.shape
    M = kornia.geometry.get_rotation_matrix2d(
        center=torch.tensor([height / 2, width / 2]).unsqueeze(dim=0),
        angle=torch.ones(1) * params,
        scale=torch.ones(1),
    )
    dst = kornia.geometry.warp_affine(img.unsqueeze(dim=0), M, dsize=(width, height), flags="nearest").squeeze(dim=0)
    return dst


def contrast(img, params):
    alpha = params
    new_img = alpha * img
    # mul_img = img*alpha
    # new_img = cv2.add(mul_img, beta)  # new_img = img*alpha + beta
    return new_img


def brightness(img, params):
    beta = params
    new_img = beta + img
    # new_img = cv2.add(img, beta)  # new_img = img*alpha + beta
    return new_img


def blur(img, params):
    channel, height, width = img.shape
    blur = None
    if params == 1:
        blur = kornia.filters.box_blur(img.unsqueeze(0), (3, 3)).squeeze(0)
    if params == 2:
        blur = kornia.filters.box_blur(img.unsqueeze(0), (4, 4)).squeeze(0)
    if params == 3:
        blur = kornia.filters.box_blur(img.unsqueeze(0), (5, 5)).squeeze(0)
    if params == 4:
        sigma = 0.3 * ((3 - 1) * 0.5 - 1) + 0.8
        blur = kornia.filters.gaussian_blur2d(img.unsqueeze(0), (3, 3), (sigma, sigma)).squeeze(0)
    if params == 5:
        sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
        blur = kornia.filters.gaussian_blur2d(img.unsqueeze(0), (5, 5), (sigma, sigma)).squeeze(0)
    if params == 6:
        sigma = 0.3 * ((7 - 1) * 0.5 - 1) + 0.8
        blur = kornia.filters.gaussian_blur2d(img.unsqueeze(0), (7, 7), (sigma, sigma)).squeeze(0)
    if params == 7:
        blur = kornia.filters.median_blur(img.unsqueeze(0), (3, 3)).squeeze(0)
    if params == 8:
        blur = kornia.filters.median_blur(img.unsqueeze(0), (3, 3)).squeeze(0)
    if params == 9:
        img = img.reshape(width, height, channel).numpy()
        blur = torch.from_numpy(cv2.bilateralFilter(img, 6, 50, 50).reshape(channel, width, height))

    return blur


def pixel_change(img, params):
    # random change 1 - 5 pixels from 0 -255
    img1d = img.view(-1)
    arr = torch.randint(0, len(img1d), (params,))
    for i in arr:
        img1d[i] = torch.randint(0, 256, (1,))
    new_img = img1d.reshape(img.shape)
    return new_img


def image_noise(img, params):
    if params == 1:
        # Gaussian-distributed additive noise.
        mean = 0
        var = 0.2
        sigma = var**0.1
        gauss = torch.empty(img.shape).normal_(mean, sigma)
        noisy = img + gauss
        return noisy
    elif params == 2:
        # Replaces random pixels with 0 or 1.
        out = torch.tensor(random_noise(img, mode="s&p", salt_vs_pepper=0.3, clip=True, amount=0.5))
        return out
    elif params == 3:
        # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
        gauss = torch.randn(img.shape)
        noisy = torch.add(img, img * gauss)
        return noisy
