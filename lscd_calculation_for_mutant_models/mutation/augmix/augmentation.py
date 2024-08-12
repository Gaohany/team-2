# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""

import cv2
import numpy as np
import skimage as sk
from PIL import Image, ImageOps, ImageEnhance
from skimage.filters import gaussian

# KIA & A2D2 image size.
IMAGE_SIZE = 300


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


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE), Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


def spatter(pil_img, loc=0.65, scale=0.5, sigma=0.3, sigma2=0.7, threshold=0.65):
    """
    Implementation of spatter from https://github.com/hendrycks/robustness/tree/master.
    """
    c_img = np.array(pil_img, dtype=np.float32) / 255.0
    liquid_layer = np.random.normal(size=c_img.shape[:2], loc=loc, scale=scale)

    liquid_layer = gaussian(liquid_layer, sigma=sigma)
    liquid_layer[liquid_layer < threshold] = 0
    if np.random.rand(1) < 0.5:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        # ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        # ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        factor = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        factor /= np.max(factor, axis=(0, 1))
        factor *= sigma2

        # water is pale turqouise
        bgr = np.concatenate(
            (
                175 / 255.0 * np.ones_like(factor[..., :1]),
                238 / 255.0 * np.ones_like(factor[..., :1]),
                238 / 255.0 * np.ones_like(factor[..., :1]),
            ),
            axis=2,
        )

        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2BGRA)

        c_img = cv2.cvtColor(np.clip(c_img + factor * bgra, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        factor = np.where(liquid_layer > threshold, 1, 0)
        factor = gaussian(factor.astype(np.float32), sigma=sigma2)
        factor[factor < 0.8] = 0

        # mud brown
        bgr = np.concatenate(
            (
                63 / 255.0 * np.ones_like(c_img[..., :1]),
                42 / 255.0 * np.ones_like(c_img[..., :1]),
                20 / 255.0 * np.ones_like(c_img[..., :1]),
            ),
            axis=2,
        )

        bgr *= factor[..., np.newaxis]
        c_img *= 1 - factor[..., np.newaxis]

        c_img = np.clip(c_img + bgr, 0, 1) * 255

    c_img = c_img.astype(np.uint8)
    c_img = Image.fromarray(c_img)

    return c_img


def specklenoise(pil_img, scale=0.2):
    """
    Implementation of specklenoise from https://github.com/hendrycks/robustness/tree/master.
    """
    c_img = np.array(pil_img) / 255.0
    c_img = np.clip(c_img + c_img * np.random.normal(size=c_img.shape, scale=scale), 0, 1) * 255
    c_img = c_img.astype(np.uint8)
    c_img = Image.fromarray(c_img)

    return c_img


def gaussianblur(pil_img, sigma=8):
    """
    Implementation of gaussianblur from https://github.com/hendrycks/robustness/tree/master.
    """
    c_img = gaussian(np.array(pil_img) / 255.0, sigma=sigma, channel_axis=2)
    c_img = gaussian(c_img, sigma=sigma, channel_axis=2)
    c_img = np.clip(c_img, 0, 1) * 255

    c_img = c_img.astype(np.uint8)
    c_img = Image.fromarray(c_img)

    return c_img


def saturate(pil_img, saturation=1.5, scale=0.1):
    """
    Implementation of saturate from PIL.
    """
    c_img = np.array(pil_img) / 255.0
    c_img = sk.color.rgb2hsv(c_img)
    c_img[:, :, 1] = np.clip(c_img[:, :, 1] * saturation + scale, 0, 1)
    c_img = sk.color.hsv2rgb(c_img)
    c_img = np.clip(c_img, 0, 1) * 255
    c_img = c_img.astype(np.uint8)
    c_img = Image.fromarray(c_img)

    return c_img


augmentations_fuzzer = [
    autocontrast,
    equalize,
    posterize,
    color,
    sharpness,
    specklenoise,
    spatter,
    gaussianblur,
    saturate,
    solarize,
    contrast,
    brightness,
]

augmentations_fuzzer_name = [
    "autocontrast",
    " equalize",
    " posterize",
    " color",
    " sharpness",
    " specklenoise",
    " spatter",
    " gaussianblur",
    " saturate",
    " solarize",
    " contrast",
    " brightness",
]

augmentations_all = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    color,
    contrast,
    brightness,
    sharpness,
]
