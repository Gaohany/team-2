import argparse
import json
import os
import random

import toml
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataset.GTSRB.gtsrb_dataset import GTSRBDataset
from dataset.GTSRB.gtsrb_dataset_gray import GTSRBDataset_gray
from dataset.MNIST.mnist_dataset import MNISTDataset
from dataset.SVHN.svhn_dataset import SVHNDataset
from fuzzer.model_structure import model_structure
from models.network_type import ClassificationNetworks
from utils.util import obj

# def create_balanced_dataloader(org_dataset, batch_size, num_samples_per_class):
#     labels = org_dataset.classes
#     labels_new = [int(labels[i]) for i in range(len(labels))]
#     class_indices = {label: [] for label in set(labels_new)}

#     for i, label in enumerate(labels_new):
#         class_indices[label].append(i)

#     sampled_indices = []

#     for label, indices in class_indices.items():
#         random.shuffle(indices)
#         sampled_indices.extend(indices[:num_samples_per_class])

#     sub_dataloader = torch.utils.data.Subset(org_dataset, sampled_indices)
#     #sub_dataloader_1 = sub_dataloader.dataset

#     return sub_dataloader


def get_test_dataloader(config):
    if config.data == "mnist":
        test_set = MNISTDataset(
            config=config, image_set=config.detection_model.image_set, mode=config.mode, augmentation=False
        )
        return test_set

    elif config.data == "gtsrb":
        test_set = GTSRBDataset(
            config=config, image_set=config.detection_model.image_set, mode=config.mode, augmentation=False
        )
        return test_set

    elif config.data == "gtsrb-gray":
        test_set = GTSRBDataset_gray(
            config=config, image_set=config.detection_model.image_set, mode=config.mode, augmentation=False
        )
        return test_set

    elif config.data == "svhn":
        test_set = SVHNDataset(
            config=config, image_set=config.detection_model.image_set, mode=config.mode, augmentation=False
        )
        return test_set
    else:
        print("please extend seed constructor for this dataset")
        raise NotImplementedError


def construct_seeds(model, network, config, output_path, partial_dataset=False, num_samples_per_class=50):
    """

    :param model: trained model for inference
    :param network: network type (for seed selection strategies implementation)
    :param config: configuration settings
    :param output_path: output directory where seed is stored.
    :return:
    """
    print("constructing initial seeds")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    test_loader = get_test_dataloader(
        config
    )  # TO DO: KIA & A2D2 to return dataset or dataloader (Current: Returns dataset).

    # Seed selection
    if config.network_type == "classification":
        network.seed_selection(model, test_loader, output_path, config, partial_dataset, num_samples_per_class)
    else:
        print(f"Please extend the initial seeds constructor for {config.network_type}")
        exit(0)

    print("initial seeds constructed")
    print("initial seeds stored at:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="control experiment")
    parser.add_argument("-output_path", help="Out path", default="/data/disk1/init_seeds/gtsrb_test/")
    parser.add_argument("-config_file", help="which config file to load", default="config/gtsrb.toml")
    args = parser.parse_args()
    config = toml.load(args.config_file)
    config = json.loads(json.dumps(config), object_hook=obj)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config.device = device
    torch.set_grad_enabled(False)

    if config.network_type == "classification" and (config.data == "mnist" or config.data == "gtsrb"):
        network = ClassificationNetworks(config.init_seed_selection, num_classes=config.num_classes)
    else:
        raise NotImplementedError("Network type specified has not been implemented")

    if config.model == "lenet5":
        model = model_structure[config.model]()
    elif config.model == "gtsrb-new":
        model = model_structure[config.model]()
    elif config.model == "gtsrb-lenet":
        model = model_structure[config.model](output_dim=config.num_classes)
    elif config.model == "gtsrb-lenet5_selftrained":
        model = model_structure[config.model]()

    else:
        raise NotImplementedError("Model specified has not been implemented.")

    model.load_state_dict(torch.load(model.model_path, map_location=device))
    model.to(device)
    model.eval()
    construct_seeds(model, network, config, args.output_path)
