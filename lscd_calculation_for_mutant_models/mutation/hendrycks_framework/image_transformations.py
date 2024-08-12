import numpy as np
import skimage as sk
from PIL import Image, ImageEnhance, ImageOps


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
    Returns:
      An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
    Returns:
      A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.0


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def np_to_pil(func):
    def wrapper(*args):
        org_image = np.clip(args[0] * 255.0, 0, 255).astype(np.uint8)
        # Check the number of channels
        if org_image.shape[-1] == 1:
            # Grayscale image
            org_image = org_image.squeeze(axis=-1)
            np_to_image = Image.fromarray(org_image, mode="L")
        elif org_image.shape[-1] == 3:
            # RGB image
            np_to_image = Image.fromarray(org_image, mode="RGB")
        else:
            # Handle other cases or raise an error
            raise ValueError("Unsupported number of channels")

        np_to_image = Image.fromarray(org_image)
        tras_img = func(np_to_image, args[1])
        return (np.array(tras_img) / 255.0).astype(np.float32)

    return wrapper


@np_to_pil
def autocontrast(pil_img, _):
    cut_off = np.random.randint(1, 10)
    return ImageOps.autocontrast(pil_img, cutoff=cut_off)


@np_to_pil
def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


@np_to_pil
def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


@np_to_pil
def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


@np_to_pil
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


@np_to_pil
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


@np_to_pil
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    if level < 0.6:
        level = 0.6
    return ImageEnhance.Brightness(pil_img).enhance(level)


@np_to_pil
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


def saturate(org_img, saturation=1.5, scale=0.1):
    if org_img.shape[-1] == 1:
        trans_img = np.clip(org_img * saturation + scale, 0, 1)
    else:
        trans_img = sk.color.rgb2hsv(org_img)
        trans_img[:, :, 1] = np.clip(trans_img[:, :, 1] * saturation + scale, 0, 1)
        trans_img = sk.color.hsv2rgb(trans_img)

    return trans_img
