from __future__ import print_function

import random

import torch
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

from fuzzer.fuzz_queue import Seed
from mutation.augmix import augmentation
from mutation.hendrycks_framework.blur_transformations import *
from mutation.hendrycks_framework.image_transformations import *
from mutation.hendrycks_framework.noise_transformations import *
from mutation.hendrycks_framework.weather_transformations import *

# blur_noise_transformations = [defocus_blur, motion_blur, zoom_blur, gaussian_blur, gaussian_noise, speckle_noise, spatter, impulse_noise, shot_noise]
# image_weather_transformations = [autocontrast, equalize, posterize,color, sharpness, brightness, contrast, saturate, solarize, fog]

transformations = [
    defocus_blur,
    motion_blur,
    zoom_blur,
    gaussian_blur,
    gaussian_noise,
    speckle_noise,
    spatter,
    impulse_noise,
    shot_noise,
    autocontrast,
    equalize,
    posterize,
    color,
    sharpness,
    brightness,
    contrast,
    saturate,
    solarize,
    fog,
]

transformations_name = [
    "defocus_blur",
    "motion_blur",
    "zoom_blur",
    "gaussian_blur",
    "gaussian_noise",
    "speckle_noise",
    "spatter",
    "impulse_noise",
    "shot_noise",
    "autocontrast",
    "equalize",
    "posterize",
    "color",
    "sharpness",
    "brightness",
    "contrast",
    "saturate",
    "solarize",
    "fog",
]


def denormalize(image, config):
    """
    Args:
      image: torch tensor with channel-first.
    Returns:
      image: np array with channel-last [0, 1] ready to apply transformations / metric calculations.
    """

    mean = torch.tensor(getattr(config, "norm_mean_" + config.detection_model.image_set)).view(-1, 1, 1)
    std = torch.tensor(getattr(config, "norm_std_" + config.detection_model.image_set)).view(-1, 1, 1)
    image = (image * std) + mean
    image = np.array(image, dtype=np.float32)
    image = np.transpose(image, (1, 2, 0))

    return image


def normalize(image, config):
    """
    Args:
      image: np array with channel-last.
    Returns:
      image: np array with channel-last.
    """

    mean = torch.tensor(getattr(config, "norm_mean_" + config.detection_model.image_set)).view(-1, 1, 1)
    std = torch.tensor(getattr(config, "norm_std_" + config.detection_model.image_set)).view(-1, 1, 1)
    image = image.transpose(2, 0, 1)  # Switch to channel-first
    mean, std = np.array(mean), np.array(std)
    image = (image - mean) / std
    return image.transpose(1, 2, 0)


def apply_op(image, op, severity):
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img) / 255.0


def augment_and_mix(org_img, seed_img, config, severity=2, width=3, depth=-1, alpha=1):
    """Perform AugMix augmentations and compute mixture.Input & Output to transformations is torch tensor with channel-first (normalized).
    Args:
      image: Raw input image as float32 np.ndarray of shape (h, w, c)
      severity: Severity of underlying augmentation operators (between 1 to 10).
      width: Width of augmentation chain
      depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
        from [1, 3]
      alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
      mixed: Augmented and mixed image.
    """

    if config.data == "gtsrb":
        seed_img = denormalize(seed_img, config)
        org_img = denormalize(org_img, config)
    else:
        seed_img = np.transpose(np.array(seed_img, dtype=np.float32), (1, 2, 0))
        org_img = np.transpose(np.array(org_img, dtype=np.float32), (1, 2, 0))

    opacities = np.float32(np.random.dirichlet([alpha] * width))
    factor = np.float32(np.random.beta(alpha, alpha))
    op_names = ""
    mix = np.zeros_like(seed_img)
    for i in range(width):
        image_aug = seed_img.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op_select = random.randint(0, len(augmentation.augmentations_fuzzer) - 1)
            # op = np.random.choice(augmentation.augmentations_fuzzer)
            op = augmentation.augmentations_fuzzer[op_select]
            op_name = augmentation.augmentations_fuzzer_name[op_select]
            op_names = op_names + "_" + op_name
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += opacities[i] * normalize(image_aug, config)

    mixed = (1 - factor) * normalize(seed_img, config) + factor * mix
    mixed = torch.tensor(mixed, dtype=torch.float32)
    mixed = mixed.permute(2, 0, 1)

    trans_img = denormalize(mixed, config)
    ssim_value = ssim(org_img, trans_img, data_range=org_img.max() - org_img.min(), channel_axis=2)
    mse_value = mean_squared_error(org_img, trans_img)

    sub = np.subtract(org_img, trans_img)
    abs_diff = np.abs(sub)
    l0 = (abs_diff != 0).sum()
    l2 = np.linalg.norm(abs_diff)
    linf = np.max(np.absolute(abs_diff))
    # linf = np.linalg.norm(abs_diff, ord=np.inf)

    return mixed, op_names, l0, l2, linf, ssim_value, mse_value


def apply_transformation(org_img, seed_img, config, severity=2):
    """Input & Output to transformations is torch tensor with channel-first (normalized)."""

    if config.data == "mnist" or config.data == "gtsrb" or config.data == "gtsrb-gray" or config.data == "svhn":
        seed_img = denormalize(seed_img, config)
        org_img = denormalize(org_img, config)
    else:
        raise NotImplementedError("Please extend Mutator with proper inputs.")
        # seed_img = np.transpose(np.array(seed_img, dtype=np.float32), (1, 2, 0))
        # org_img = np.transpose(np.array(org_img, dtype=np.float32), (1, 2, 0))

    if config.random_severity:
        severity_level = [1,2,3,4,5]
        severity = random.choice(severity_level)
        # print("Severity:", severity)

    sample_number = random.randint(0, len(transformations) - 1)
    selected_transformation = transformations[sample_number]
    selected_transformation_name = transformations_name[sample_number]

    trans_img = selected_transformation(seed_img, severity)  # After transformation np.ndarray [0,1]

    if len(trans_img.shape) <= 2:
        trans_img = np.expand_dims(trans_img, axis=-1)
    else:
        pass

    if org_img.shape[-1] == trans_img.shape[-1]:
        ssim_value = ssim(org_img, trans_img, data_range=org_img.max() - org_img.min(), channel_axis=2)
        mse_value = mean_squared_error(org_img, trans_img)
        sub = np.subtract(org_img, trans_img)
        abs_diff = np.abs(sub)
        l0 = (abs_diff != 0).sum()
        l2 = np.linalg.norm(abs_diff)
        linf = np.max(np.absolute(abs_diff))
        # linf = np.linalg.norm(abs_diff, ord=np.inf)

        if config.data == "mnist" or config.data == "gtsrb" or config.data == "gtsrb-gray" or config.data == "svhn":
            trans_img = normalize(trans_img, config)
        else:
            raise NotImplementedError("Please extend Mutator with proper inputs.")

        trans_img = torch.tensor(trans_img, dtype=torch.float32)
        trans_img = trans_img.permute(2, 0, 1)

        return trans_img, selected_transformation_name, l0, l2, linf, ssim_value, mse_value
    else:
        print("Unsucessful mutations:", selected_transformation_name, ssim_value)
        return trans_img, selected_transformation_name, l0, l2, linf, ssim_value, mse_value


def mutate_one(reference_img, seed, transformation_class, l0_ref, l2_ref, linf_ref, ssim_ref, mse_ref, config):
    seed = seed.to("cpu")
    reference_img = reference_img.to("cpu")

    if config.mutation_criteria == "augmix":
        img_mutated, transformation_app, l0, l2, linf, ssim_value, mse_value = augment_and_mix(
            reference_img, seed, config
        )
    elif config.mutation_criteria == "transformations":
        img_mutated, transformation_app, l0, l2, linf, ssim_value, mse_value = apply_transformation(
            reference_img, seed, config
        )
    else:
        raise NotImplementedError("This mutation criteria has not been implemented.")

    transformation_ret = str(transformation_class) + "-" + transformation_app

    if ssim_value > 0.5:
        return reference_img, img_mutated, transformation_ret, 1, l0, l2, linf, ssim_value, mse_value
    else:
        # print("Unsucessful mutations:", transformation_class)
        return reference_img, seed, transformation_class, 0, l0_ref, l2_ref, linf_ref, ssim_ref, mse_ref


def mutator(seed: Seed, num_mutants: int, config: dict, print_logs: bool = False):
    """
    Mutates the input seed based on AugMix transformations.
    :param seed: information about the image being read in, object of class seed
    :param num_mutants: number of mutated images to try and generate
    :param config: config dict
    :param print_logs: whether to print any log messages
    :return: reference images, mutated images, transformation type applied, l0 reference and l infinity reference
    """
    data = torch.load(seed.file_name)
    imgs = data["images"]
    reference_img, seed_img = imgs[0], imgs[1]
    (
        reference_images,
        mutated_seeds,
        transformation_classes,
        l0_values,
        l2_values,
        linf_values,
        ssim_values,
        mse_values,
    ) = ([], [], [], [], [], [], [], [])

    max_tries = 10
    try_count = 0

    while len(mutated_seeds) < num_mutants and try_count <= max_tries:
        (
            reference_img_out,
            mutated_seed,
            transformation,
            successful_mutation,
            l0_val,
            l2_val,
            linf_val,
            ssim_val,
            mse_val,
        ) = mutate_one(
            reference_img,
            seed_img,
            seed.transformation_class,
            seed.l0_ref,
            seed.l2_ref,
            seed.linf_ref,
            seed.ssim_ref,
            seed.mse_ref,
            config,
        )
        if successful_mutation:
            reference_images.append(reference_img_out)
            mutated_seeds.append(mutated_seed)
            transformation_classes.append(transformation)
            l0_values.append(l0_val)
            l2_values.append(l2_val)
            linf_values.append(linf_val)
            ssim_values.append(ssim_val)
            mse_values.append(mse_val)
        else:
            if print_logs:
                print("SSIM < 0.5 thus discarding mutant image.")
            try_count += 1

    if mutated_seeds and not (try_count > max_tries):
        mutated_images = torch.stack(mutated_seeds)
        reference_images = torch.stack(reference_images)
        return (
            reference_images,
            mutated_images,
            transformation_classes,
            l0_values,
            l2_values,
            linf_values,
            ssim_values,
            mse_values,
        )
    else:
        # If we can't generate successful batch of mutants.. we return original images. This seed will be deleted as it won't have new bit.
        # TO DO: if we have atleast 10/20 mutants.. how to handle them?
        if print_logs:
            print("Returning original seeds.")
        if len(reference_images) != 0:
            reference_images = torch.stack(reference_images)
        else:
            reference_images = torch.stack([reference_img, reference_img])  # This is to handle empty list.
            transformation_classes, l0_values, l2_values, linf_values, ssim_values, mse_values = (
                ["none"],
                [0],
                [0],
                [0],
                [1],
                [0],
            )
        reference_images = reference_images[0].unsqueeze(0).repeat(num_mutants, 1, 1, 1)
        transformation_classes, l0_values, l2_values = (
            [transformation_classes[0]] * num_mutants,
            [l0_values[0]] * num_mutants,
            [l2_values[0]] * num_mutants,
        )
        linf_values, ssim_values, mse_values = (
            [linf_values[0]] * num_mutants,
            [ssim_values[0]] * num_mutants,
            [mse_values[0]] * num_mutants,
        )
        return (
            reference_images,
            reference_images,
            transformation_classes,
            l0_values,
            l2_values,
            linf_values,
            ssim_values,
            mse_values,
        )


if __name__ == "__main__":
    print("main Test.")
