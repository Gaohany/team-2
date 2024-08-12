from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import zoom as scizoom
from wand.api import library as wandlibrary
from wand.image import Image as WandImage


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def plasma_fractal(mapsize=32, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


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


def fog(org_img, factor=1.5, wibbledecay=1.5):
    height, width = org_img.shape[0], org_img.shape[1]
    max_val = org_img.max()

    org_img += factor * plasma_fractal(mapsize=2048, wibbledecay=wibbledecay)[:height, :width][..., np.newaxis]
    trans_img = np.clip(org_img * max_val / (max_val + factor), 0, 1)
    return (trans_img).astype(np.float32)


def frost(org_img, severity=1):
    height, width = org_img.shape[0], org_img.shape[1]
    c = [(1, 0.2), (1, 0.3), (0.75, 0.45), (0.9, 0.4), (0.85, 0.4)][severity - 1]
    idx = np.random.randint(3)
    frost_img_path = "/home/vekariya/Documents/testdnn/mutation/hendrycks_framework/frost/"
    filename = [frost_img_path + "1.jpg", frost_img_path + "2.jpg", frost_img_path + "3.jpg"][idx]
    frost = cv2.imread(filename)
    assert frost is not None, f"Could not load image {filename} needed for Frost perturbation"

    frost_height, frost_width = frost.shape[0], frost.shape[1]
    factor = 1.5

    new_frost_width = width * factor
    new_frost_height = new_frost_width * frost_height / frost_width

    if new_frost_height < height * factor or new_frost_width < width * factor:
        new_frost_height = height * factor
        new_frost_width = new_frost_height * frost_width / frost_height

    # Frost image should be at least factor x bigger than width and height
    frost = cv2.resize(frost, (int(new_frost_width), int(new_frost_height)))
    frost_height, frost_width = frost.shape[0], frost.shape[1]

    # randomly crop and convert to rgb
    y_start, x_start = np.random.randint(0, frost_height - height), np.random.randint(0, frost_width - width)
    frost = frost[y_start : y_start + height, x_start : x_start + width][..., [2, 1, 0]]

    trans_img = np.clip(c[0] * org_img + c[1] * frost, 0, 255)

    return (trans_img / 255.0).astype(np.float32)


def snow(org_img, severity=1):
    c = [
        (0.1, 0.2, 1, 0.6, 8, 3, 0.95),
        (0.1, 0.2, 1, 0.5, 10, 4, 0.9),
        (0.3, 0.3, 1.25, 0.65, 14, 12, 0.8),
        (0.15, 0.3, 1.75, 0.55, 10, 4, 0.9),
        (0.25, 0.3, 2.25, 0.6, 12, 6, 0.85),
    ][severity - 1]

    snow_layer = np.random.normal(size=org_img.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome
    height, width = org_img.shape[0], org_img.shape[1]

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode="L")
    output = BytesIO()
    snow_layer.save(output, format="PNG")
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED) / 255.0
    snow_layer = snow_layer[..., np.newaxis]

    trans_img = c[6] * org_img + (1 - c[6]) * np.maximum(
        org_img, cv2.cvtColor(org_img, cv2.COLOR_RGB2GRAY).reshape(height, width, 1) * 1.5 + 0.5
    )
    trans_img = np.clip(trans_img + snow_layer + np.rot90(snow_layer, k=2), 0, 1).astype(np.float32)
    return trans_img
