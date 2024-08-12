import os
import pdb
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch_intermediate_layer_getter import IntermediateLayerGetter
from torchvision.datasets import MNIST
from tqdm import tqdm

from models.default_model import ClassificationModel
from utils.util import z_score_normalization


class MNIST_Lenet5(ClassificationModel):
    def __init__(self, num_classes=10, drop_rate=0.0):
        super().__init__()

        self.model_name = "pytorch_classification_mnist_lenet5"
        self.model_path = os.path.join("models/mnist/lenet5/weights", "{}.pth".format(self.model_name))
        print(self.model_path)

        self.dropout = drop_rate

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.fc_1 = nn.Linear(4 * 4 * 16, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.relu = nn.ReLU()
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """
        :param x: [b, 1, 28, 28]
        :return:
        """
        batch_size = x.shape[0]
        x = self.conv_1(x)

        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)

        x = self.conv_2(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)
        x = x.view(batch_size, 4 * 4 * 16)

        x = self.fc_1(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)

        x = self.fc_2(self.relu(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)

        feature_vector = self.fc_3(self.relu(x))

        x = F.log_softmax(feature_vector, dim=1)

        return x, feature_vector

    def inference(self, x, req_feature_vec=False):
        x_pred_softmax, feature_vector = self.forward(x)
        feature_vector = feature_vector[0].cpu().numpy()
        _, x_pred_tags = torch.max(x_pred_softmax, dim=1)
        pred_probs = torch.exp(x_pred_softmax)
        pred_probs = pred_probs.cpu().numpy()
        x_pred_tags = x_pred_tags.cpu().numpy()
        x_pred_prob = pred_probs[0][x_pred_tags]

        if req_feature_vec:
            return x_pred_tags, x_pred_prob, pred_probs, feature_vector
        else:
            return x_pred_tags, x_pred_prob, pred_probs

    def good_seed_detection(self, test_loader, predictor, config):
        """
        from the test set detect seeds which can be considered as initial test seeds for fault detection
        (ones with correct prediction)
        :param test_loader:
        :param predictor: method for prediction
        :return: indexes of seeds to be considered
        """
        gt_labels, pred_labels, pred_op_prob, pred_probs = [], [], [], []

        for i in tqdm(range(len(test_loader))):
            data = test_loader[i]
            images, labels = data
            # labels = labels.numpy()
            layers_outputs, detections, output_dict, feature_vector = predictor(
                self, data, config.network_type, config
            )
            gt_labels.extend(labels)
            pred_labels.extend(detections)
            pred_op_prob.extend(output_dict["op_class_prob"])
            pred_probs.extend(output_dict["op_probs"])

        return gt_labels, pred_labels, pred_op_prob, pred_probs
