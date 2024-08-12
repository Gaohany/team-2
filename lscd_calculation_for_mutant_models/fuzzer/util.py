import os

import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from torchvision.transforms import transforms

from utils.util import z_score_normalization


def save_seeds(seeds, k, output_path, prefix):
    """
    save selected seeds
    :param seeds: images to be saved
    :param k: number of samples to be picked from each class/cluster
    :param output_path: dir to save seeds in
    :param prefix: filename prefix to indicate class/cluster id
    :return:
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    batches = np.split(seeds, k, axis=0)
    trans = transforms.ToPILImage()
    for i, batch in enumerate(batches):
        batch.squeeze_()
        file_name = prefix + str(i) + ".png"
        trans(batch).save(os.path.join(output_path, file_name), "PNG")


def save_seeds_rgb(image_name, image_path, output_path):
    """
    Save selected images from SVHN/GTSRB RGB dataset.
    :param image_name:
    :param image_path: root path for selected seed
    :param output_path: Output path to store the seed
    :return:
    """
    image = Image.open(image_path, mode="r").convert("RGB")
    file_name = image_name + ".png"
    image.save(os.path.join(output_path, file_name), "PNG")


def save_seeds_gray(image_name, image_path, output_path):
    """
    Save selected images from GTSRB dataset.
    :param image_name:
    :param image_path: root path for selected seed
    :param output_path: Output path to store the seed
    :return:
    """

    image = (image_path[0] * 255).astype(np.uint8)

    if len(image.shape) == 3:
        image = image[0]

    image = Image.fromarray(image, mode="L")
    file_name = image_name + ".png"
    image.save(os.path.join(output_path, file_name), "PNG")


@torch.no_grad()
def predict(model, data, network_type, config, normalize=False):
    """
    :param model:
    :param data: needs to be pre-scaled if needed
    :return:
    """
    if network_type == "classification":
        images, labels = data
        if normalize:
            data = z_score_normalization(images)
        if len(images.shape) == 3:
            images = images.unsqueeze(dim=0).to(config.device)
        else:
            images = images.to(config.device)
        detections, op_class_prob, op_probabilities, feature_vector = model.inference(images, req_feature_vec=True)
        layers_outputs = model.intermediate_outputs(images)
        output_dict = {"op_class": detections, "op_class_prob": op_class_prob, "op_probs": op_probabilities}

        return layers_outputs, detections, output_dict, feature_vector

    else:
        raise NotImplementedError("Please extend predict method to given network class.")


def profile_dict_to_gpu(profile_dict, device):
    for neuron_name in profile_dict.keys():
        temp_var = profile_dict[neuron_name]
        [mean_new, squared_mean, std_deviation, lower_bound, upper_bound] = temp_var
        temp_var = [
            mean_new.to(device),
            squared_mean.to(device),
            std_deviation.to(device),
            lower_bound.to(device),
            upper_bound.to(device),
        ]
        profile_dict[neuron_name] = temp_var

    del temp_var
    return profile_dict
