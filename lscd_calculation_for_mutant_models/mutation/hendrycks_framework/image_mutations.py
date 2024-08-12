from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from skimage.filters import gaussian


def gaussian_blur(x, severity=1):
    c = [0.5, 0.75, 1, 1.25, 1.5][severity - 1]

    x = gaussian(np.array(x) / 255.0, sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.1, 1, 1), (0.5, 1, 1), (0.6, 1, 2), (0.7, 2, 1), (0.9, 2, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(64 - c[1], c[1], -1):
            for w in range(64 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255.0, sigma=c[0], multichannel=True), 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(0.5, 0.6), (1, 0.1), (1.5, 0.1), (2.5, 0.01), (3, 0.1)][severity - 1]

    x = np.array(x) / 255.0
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x64x64 -> 64x64x3

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1):
    c = [(10, 1), (10, 1.5), (10, 2), (10, 2.5), (12, 3)][severity - 1]

    output = BytesIO()
    x.save(output, format="PNG")
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

    if x.shape != (64, 64):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, severity=1):
    c = [
        np.arange(1, 1.06, 0.01),
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.01),
        np.arange(1, 1.26, 0.01),
    ][severity - 1]

    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def gaussianblur(pil_img, sigma=8):
    """
    Implementation of gaussianblur from https://github.com/hendrycks/robustness/tree/master.
    """
    c_img = gaussian(np.array(pil_img) / 255.0, sigma=sigma, multichannel=True)
    c_img = gaussian(c_img, sigma=sigma, multichannel=True)
    c_img = np.clip(c_img, 0, 1) * 255

    c_img = c_img.astype(np.uint8)
    c_img = Image.fromarray(c_img)

    return c_img
