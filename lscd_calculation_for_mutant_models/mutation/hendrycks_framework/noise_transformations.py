import cv2
import numpy as np
import skimage as sk
from skimage.filters import gaussian


def spatter(org_img, loc=0.65, scale=0.5, sigma=0.3, sigma2=0.7, threshold=0.65):
    """
    This function aims to create a more complex, water-like splatter effect with a combination of noise, edge detection, and color manipulation. (TO DO: Only works for RGB. Extend to gray scale.)
    """
    loc = 0.65
    liquid_layer = np.random.normal(size=org_img.shape[:2], loc=loc, scale=scale)

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

        if org_img.shape[-1] == 3:
            org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2BGRA)
            trans_img = cv2.cvtColor(np.clip(org_img + factor * bgra, 0, 1), cv2.COLOR_BGRA2BGR)
        else:
            org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2BGRA)
            trans_img = cv2.cvtColor(np.clip(org_img + factor * bgra, 0, 1), cv2.COLOR_BGRA2GRAY)
    else:
        factor = np.where(liquid_layer > threshold, 1, 0)
        factor = gaussian(factor.astype(np.float32), sigma=sigma2)
        factor[factor < 0.8] = 0

        # mud brown
        bgr = np.concatenate(
            (
                63 / 255.0 * np.ones_like(org_img[..., :1]),
                42 / 255.0 * np.ones_like(org_img[..., :1]),
                20 / 255.0 * np.ones_like(org_img[..., :1]),
            ),
            axis=2,
        )

        bgr *= factor[..., np.newaxis]
        org_img *= 1 - factor[..., np.newaxis]
        trans_img = np.clip(org_img + bgr, 0, 1)

        if (org_img.shape[-1] != trans_img.shape[-1]) and org_img.shape[-1] == 1:
            trans_img = cv2.cvtColor(trans_img, cv2.COLOR_BGRA2GRAY)
        elif (org_img.shape[-1] != trans_img.shape[-1]) and org_img.shape[-1] != 1:
            raise NotImplementedError
        else:
            pass

    return trans_img


def spatter_gray(org_img, amount=0.02, size=0.07):
    """
    This function adds simple, random noise particles to a grayscale or RGB image, offering straightforward splatter noise.
    """
    image = (org_img * 255.0).astype(np.uint8)

    height, width, channels = image.shape
    noise_num = int(amount * height * width)

    # Generate random noise coordinates and intensities
    noise_x = np.random.randint(0, width, noise_num)
    noise_y = np.random.randint(0, height, noise_num)
    noise_intensity = np.random.random(noise_num) * 255

    # Apply noise to each channel independently
    for channel in range(channels):
        image[:, :, channel][noise_y, noise_x] = np.minimum(
            image[:, :, channel][noise_y, noise_x] + noise_intensity.astype(int), 255
        )

    return (image / 255.0).astype(np.float32)


def speckle_noise(org_img, scale=0.2):
    trans_image = np.clip(org_img + org_img * np.random.normal(size=org_img.shape, scale=scale), 0, 1)

    return trans_image


def gaussian_noise(org_img, severity=1):
    c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]

    return np.clip(org_img + np.random.normal(size=org_img.shape, scale=c), 0, 1)


def shot_noise(org_img, severity=1):
    org_img = np.clip(org_img, 0, None)
    c = [500, 250, 100, 75, 50][severity - 1]

    return np.clip(np.random.poisson(org_img * c) / c, 0, 1)


def impulse_noise(org_img, severity=1):
    c = [0.01, 0.02, 0.03, 0.05, 0.07][severity - 1]

    trans_img = sk.util.random_noise(org_img, mode="s&p", amount=c)
    return np.clip(trans_img, 0, 1)
