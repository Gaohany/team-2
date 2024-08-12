import torch
import torch.nn as nn
import torch.nn.functional as F
from models.default_model import ClassificationModel
import os
import tqdm
from pathlib import Path

class MNIST_Lenet5(ClassificationModel):
    def __init__(self, num_classes: int = 10, drop_rate: float = 0.0):
        super(MNIST_Lenet5, self).__init__()

        self.dropout = nn.Dropout(p=drop_rate)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        # CNN layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=True)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.ac1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 16, kernel_size=5, stride=1, padding=0, bias=True)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.ac2 = nn.ReLU()

        self.fc1 = nn.Linear(4 * 4 * self.conv2.out_channels, 120)
        self.fc2 = nn.Linear(self.fc1.out_features, 84)
        self.fc3 = nn.Linear(self.fc2.out_features, num_classes)

    def forward(self, x):

        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.ac1(x)
        x = self.avgpool1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.ac2(x)
        x = self.avgpool2(x)
        x = self.dropout(x)
        x = x.view(batch_size, 4 * 4 * 16) 
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        feature_vector = self.fc3(x)

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    m = MNIST_Lenet5()
    print(count_parameters(m))
