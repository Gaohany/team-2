from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian
from wand.api import library as wandlibrary
from wand.image import Image as WandImage


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


def glass_blur(org_img, severity=1):
    # sigma, max_delta, iterations
    c = [(0.1, 1, 1), (0.5, 1, 1), (0.6, 1, 2), (0.7, 2, 1), (0.9, 2, 2)][severity - 1]

    x = np.uint8(gaussian(org_img, sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(64 - c[1], c[1], -1):
            for w in range(64 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255.0, sigma=c[0], multichannel=True), 0, 1)


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def defocus_blur(org_img, severity=1):
    c = [(0.5, 0.6), (1, 0.1), (1.5, 0.1), (2.5, 0.01), (3, 0.1)][severity - 1]

    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(org_img.shape[-1]):
        channels.append(cv2.filter2D(org_img[:, :, d], -1, kernel))
    trans_image = np.array(channels).transpose((1, 2, 0))  # 3x64x64 -> 64x64x3

    return trans_image


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def motion_blur(org_img, severity=1):
    c = [(10, 1), (10, 1.5), (10, 2), (10, 2.5), (12, 3)][severity - 1]
    img = Image.fromarray((org_img * 255.0).astype(np.uint8))
    output = BytesIO()
    img.save(output, format="PNG")
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
    trans_image = np.clip(x[..., [2, 1, 0]], 0, 255)

    return (trans_image / 255.0).astype(np.float32)


def motion_blur_gray(org_image, severity=1):
    c = [(10, 1), (10, 1.5), (10, 2), (10, 2.5), (12, 3)][severity - 1]

    image = np.asarray(org_image * 255, dtype="uint8")

    kernel_size, sigma, angle = c[0], c[1], np.random.uniform(-45, 45)

    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    angle_rad = np.deg2rad(angle)
    center = (kernel_size - 1) / 2

    # Calculate motion blur direction
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    # Set motion blur pixels
    for i in range(kernel_size):
        x = int(center + i * dx)
        y = int(center + i * dy)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0 / kernel_size

    # Apply blur using filter2D
    trans_image = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    trans_image = np.expand_dims(trans_image, axis=2)

    return (trans_image / 255.0).astype(np.float32)


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    w = img.shape[1]

    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))
    top = (h - ch) // 2

    cw = int(np.ceil(w / zoom_factor))
    right = (w - cw) // 2

    img = scizoom(img[top : top + ch, right : right + cw], (zoom_factor, zoom_factor, 1), order=1)

    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    trim_right = (img.shape[1] - w) // 2

    return img[trim_top : trim_top + h, trim_right : trim_right + w]


def zoom_blur(org_img, severity=1):
    # c = [np.arange(1, 1.06, 0.01), np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
    # np.arange(1, 1.21, 0.01), np.arange(1, 1.26, 0.01)][severity - 1]
    c = np.arange(1, 1.06, 0.01)
    out = np.zeros_like(org_img)
    for zoom_factor in c:
        out += clipped_zoom(org_img, zoom_factor)

    trans_image = (org_img + out) / (len(c) + 1)
    return trans_image


def gaussian_blur(org_img, sigma=8):
    # Sigms higher, more blur.
    trans_image = gaussian(org_img, sigma=sigma, channel_axis=2)
    # c_img = np.clip(c_img, 0, 1) * 255

    return trans_image
