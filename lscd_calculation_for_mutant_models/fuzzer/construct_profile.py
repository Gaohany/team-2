from __future__ import print_function

import argparse
import collections
import json
import pickle

import toml
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm

from dataset.GTSRB.gtsrb_dataset import GTSRBDataset
from dataset.GTSRB.gtsrb_dataset_gray import GTSRBDataset_gray
from dataset.MNIST.mnist_dataset import MNISTDataset
from dataset.SVHN.svhn_dataset import SVHNDataset
from fuzzer.model_structure import model_structure
from fuzzer.util import predict
from models.network_type import ClassificationNetworks
from utils.util import obj


class DNNProfile:
    def __init__(self, network_type, model, data, only_layer=""):
        """
        Initialize the models to be tested
        coverage : [mean_value_new, squared_mean_value, standard_deviation, lower_bound, upper_bound]
        :param network_type: task to be performed
        :param model: model to be tested
        :param only_layer Only these layers are considered for neuron coverage
        """
        self.network_type = network_type
        self.model = model
        self.layer_to_compute = model.intermediate_layers
        self.layer_neuron_num = model.num_neurons
        self.coverage = collections.OrderedDict()
        self.data = data

        if only_layer != "":
            self.layer_to_compute = [only_layer]
            self.layer_neuron_num = self.layer_neuron_num[self.layer_to_compute.index(only_layer)]

        print("* target layer list:", self.layer_to_compute)

        for idx, layer_name in enumerate(self.layer_to_compute):
            for index in range(self.layer_neuron_num[idx]):
                self.coverage[(layer_name, index)] = [0.0, 0.0, 0.0, None, None]

    def generate_coverage(self, input_data, config):
        """
        coverage : [mean_value_new, squared_mean_value, standard_deviation, lower_bound, upper_bound]
        update coverage for each neuron in each layer for each data sample
        :param input_data: data to generate coverage using
        :param normalize_data: bool
        :return:
        """
        if (
            self.data == "kia"
            or self.data == "a2d2"
            or self.data == "pascal"
            or self.data == "gtsrb"
            or self.data == "gtsrb-gray"
            or self.data == "mnist"
            or self.data == "kitti"
            or self.data == "svhn"
        ):
            with torch.no_grad():
                layers_outputs = self.model.intermediate_outputs(input_data)
        else:
            layers_outputs, outputs = predict(self.model, input_data, self.network_type, config, config.normalize_data)

        for layer_idx, layer_name in enumerate(self.layer_to_compute):
            layer_outputs = layers_outputs[layer_name]
            # handle the layer output by each data, i is the number of data
            for i, layer_output in enumerate(layer_outputs):
                for neuron_idx in range(layer_output.shape[0]):
                    neuron_output = torch.mean(layer_output[neuron_idx, ...])

                    mean, squared_mean, _, lower_bound, upper_bound = self.coverage[(layer_name, neuron_idx)]

                    mean_new = (neuron_output + mean * i) / (i + 1)
                    squared_mean = (neuron_output * neuron_output + squared_mean * i) / (i + 1)

                    std_deviation = torch.sqrt(abs(squared_mean - mean_new * mean_new))

                    if (lower_bound is None) and (upper_bound is None):
                        lower_bound, upper_bound = neuron_output, neuron_output
                    else:
                        if neuron_output < lower_bound:
                            lower_bound = neuron_output
                        if neuron_output > upper_bound:
                            upper_bound = neuron_output

                    self.coverage[(layer_name, neuron_idx)] = [
                        mean_new.to("cpu"),
                        squared_mean.to("cpu"),
                        std_deviation.to("cpu"),
                        lower_bound.to("cpu"),
                        upper_bound.to("cpu"),
                    ]

    def dump(self, output_file):
        """
        save generated profile in a pickle file
        :param output_file: name of the pickle file to be dumped in
        :return:
        """
        print("*profiling neuron size:", len(self.coverage.items()))
        with open(output_file, "wb") as f:
            pickle.dump(self.coverage, f)


def construct_profile(model, config):
    print(f"Creating profile for {config.model}")

    if config.data == "mnist":
        profiler = DNNProfile(config.network_type, model, config.data)
        train_set = MNISTDataset(
            config=config, image_set=config.detection_model.image_set, mode="train", augmentation=False
        )
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        for data in tqdm(train_loader):
            images, classes = data
            images = images.to(config.device)
            profiler.generate_coverage(images, config)

    elif config.data == "gtsrb":
        profiler = DNNProfile(config.network_type, model, config.data)
        train_set = GTSRBDataset(
            config=config, image_set=config.detection_model.image_set, mode="train", augmentation=False
        )  # config.detection_model.image_set
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        for data in tqdm(train_loader):
            images, classes = data
            images = images.to(config.device)
            profiler.generate_coverage(images, config)

    elif config.data == "svhn":
        profiler = DNNProfile(config.network_type, model, config.data)
        train_set = SVHNDataset(
            config=config, image_set=config.detection_model.image_set, mode="train", augmentation=False
        )  # config.detection_model.image_set
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        for data in tqdm(train_loader):
            images, classes = data
            images = images.to(config.device)
            profiler.generate_coverage(images, config)

    elif config.data == "gtsrb-gray":
        profiler = DNNProfile(config.network_type, model, config.data)
        train_set = GTSRBDataset_gray(
            config=config, image_set=config.detection_model.image_set, mode="train", augmentation=False
        )
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        for data in tqdm(train_loader):
            images, classes = data
            images = images.to(config.device)
            profiler.generate_coverage(images, config)

    else:
        print("Please extend the profile constructor for this dataset")
        raise NotImplementedError

    profiler.dump(config.model_profile_path)

    print("profiling coverage results written to {0}".format(config.model_profile_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="neuron output profiling")
    parser.add_argument("-config_file", help="choose configuration with which to run", default="config/kitti.toml")
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
    elif config.model == "gtsrb-alex":
        model = model_structure[config.model](output_dim=config.num_classes)
    elif config.model == "gtsrb-lenet5_selftrained":
        model = model_structure[config.model]()

    else:
        raise NotImplementedError("Model specified has not been implemented.")

    model.load_state_dict(torch.load(model.model_path, map_location=device))
    model.to(device)
    model.eval()

    construct_profile(model, config)
