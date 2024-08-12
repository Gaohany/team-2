from __future__ import absolute_import, division, print_function

import argparse
import datetime
import json
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

import toml
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, datasets

sys.path.append("..")

from dataset.corner_case_dataset import CornerCaseDataset
from mutagen.all_classes import *


def arguments():
    parser = argparse.ArgumentParser(description="Pre-Process Data for Calculations")
    parser.add_argument(
        "-config_file",
        help="choose configuration with which to run",
        default="../config/mnist.toml",
    )
    return parser.parse_args()

def export(root_folder: Path, filename: str, dataset_list: list, num_classes: int):

    file = root_folder / filename

    obj = {
        "img_cc_nc": dataset_list[0],
        "lbl_cc_nc": dataset_list[1],
        "path_cc_nc": dataset_list[2],
        "ori_img_cc_nc": dataset_list[3],
        "img_cc_kmnc": dataset_list[4],
        "lbl_cc_kmnc": dataset_list[5],
        "path_cc_kmnc": dataset_list[6],
        "ori_img_cc_kmnc": dataset_list[7],
        "img_cc_nbc": dataset_list[8],
        "lbl_cc_nbc": dataset_list[9],
        "path_cc_nbc": dataset_list[10],
        "ori_img_cc_nbc": dataset_list[11],
        "num_classes": num_classes,
    }

    torch.save(obj, file)
    print("Saved", file)

def load_ori_image(img_paths, config, preprocess):
    ori_imgs_loaded = []
    if config.data == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((config.norm_mean_mnist), (config.norm_std_mnist))
])
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform= transform)
    for path in tqdm(img_paths):
        if config.data == "gtsrb-gray":
            image = Image.open(Path(f"{config.data_path}/Test/{path:0{5}}.png"), mode="r").convert("L")
        elif config.data == "mnist":
            image = test_dataset.test_data[path].unsqueeze(0)
        else:
            image = Image.open(Path(f"{config.data_path}/Test/{path:0{5}}.png"), mode="r").convert("RGB")
        
        if config.data == "mnist":
            ori_imgs_loaded.append(image)
        else:
            ori_imgs_loaded.append(preprocess(image))

    normalized_ori_imgs = torch.stack(
        [ori_imgs_loaded[i] for i in tqdm(range(len(ori_imgs_loaded)))]
    )

    return normalized_ori_imgs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch using device:", device)

    args = arguments()
    config = json.loads(json.dumps(toml.load(args.config_file)), object_hook=obj)
    config.device = device
    torch.set_grad_enabled(False)

    root_folder = Path("data") / config.detection_model.dataset
    root_folder.mkdir(exist_ok=True, parents=True)
    experiment_paths = [
        "E:/datasets/mnist_1_nc/mnist_1_1",
        "E:/datasets/mnist_1_kmnc/mnist_2_1",
        "E:/datasets/mnist_1_nbc/mnist_3_1",
    ]
    splits = ["cc_nc", "cc_kmnc", "cc_nbc"]
    total_dataset_list = []

    for i, selected_path in enumerate(experiment_paths):
        imgs_loaded = []
        if config.data == "mnist":
            config.experiment_path = [selected_path]
            dataset = CornerCaseDataset(
                config=config,
                image_set=config.detection_model.image_set,
                mode=splits[i],
                augmentation=False,
            )

            preprocess = torchvision.transforms.Compose(
                [
                    transforms.Resize([config.input_height, config.input_width]),
                    transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (config.norm_mean_mnist), (config.norm_std_mnist)
                    )
                ]
            )
            gt_img_paths = dataset._image_paths
            for path in tqdm(gt_img_paths):
                if config.data == "gtsrb-gray" or config.data == "mnist":
                    image = Image.open(Path(path), mode="r").convert("L")
                else:
                    image = Image.open(Path(path), mode="r").convert("RGB")
                imgs_loaded.append(preprocess(image))

            normalized_imgs = torch.stack(
                [imgs_loaded[i] for i in tqdm(range(len(imgs_loaded)))]
            )
            gt_labels = torch.tensor(dataset.classes).squeeze().to(torch.uint8)
            img_paths = dataset.image_ids
            img_paths = torch.tensor(
                [int(img_paths[i].split("_")[-1]) for i in tqdm(range(len(img_paths)))], dtype=torch.int64
            ) # id_000000_27 (e.g.)
            ori_imgs = load_ori_image(img_paths, config, preprocess)
        elif config.data == "gtsrb":
            config.experiment_path = [selected_path]
            dataset = CornerCaseDataset(
                config=config,
                image_set=config.detection_model.image_set,
                mode=splits[i],
                augmentation=False,
            )
            preprocess = torchvision.transforms.Compose(
                [

                    transforms.Resize([config.input_height, config.input_width]),
                    transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (config.norm_mean_gtsrb), (config.norm_std_gtsrb)
                    )
                ]
            )
            gt_img_paths = dataset._image_paths
            for path in tqdm(gt_img_paths):
                if config.data == "gtsrb-gray" or config.data == "mnist":
                    image = Image.open(Path(path), mode="r").convert("L")
                else:
                    image = Image.open(Path(path), mode="r").convert("RGB")
                imgs_loaded.append(preprocess(image))

            normalized_imgs = torch.stack(
                [imgs_loaded[i] for i in tqdm(range(len(imgs_loaded)))]
            )
            gt_labels = torch.tensor(dataset.classes).squeeze().to(torch.uint8)
            img_paths = dataset.image_ids
            img_paths = torch.tensor(
                [int(img_paths[i].split("_")[-1]) for i in tqdm(range(len(img_paths)))], dtype=torch.int64
            ) 
            ori_imgs = load_ori_image(img_paths, config, preprocess)
        else:
            raise NotImplementedError(
                "Please extend it to input dataset or use supported dataset."
            )
        print(
            "Split:",
            splits[i],
            normalized_imgs.shape,
            gt_labels.shape,
            img_paths.shape,
        )

        total_dataset_list.append(normalized_imgs)
        total_dataset_list.append(gt_labels)
        total_dataset_list.append(
            img_paths
        )
        total_dataset_list.append(
            ori_imgs
        )


    export(
        root_folder=root_folder,
        filename=f"{config.data}_fuzzing_data_normalized.pth",
        dataset_list=total_dataset_list,
        num_classes=config.num_classes,
    )
