import torch
import torch.nn as nn
import torch.nn.functional as F
from models.default_model import ClassificationModel
from tqdm import tqdm


class GTSRB_new(ClassificationModel):
    def __init__(self, num_classes: int = 43, drop_out: float = 0.5):
        super(GTSRB_new, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv2d(3,  out_channels=100, kernel_size=5, bias=True)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.ac1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(self.conv1.out_channels, out_channels=150, kernel_size=3, bias=True)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.ac2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(self.conv2.out_channels, out_channels=250, kernel_size=3, bias=True)
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)
        self.ac3 = nn.LeakyReLU()

        self.conv_drop = nn.Dropout2d(p=drop_out)
        self.fc_drop = nn.Dropout(p=drop_out)

        self.fc1 = nn.Linear(self.conv3.out_channels * 2 * 2, out_features=350, bias=True)
        self.fc2 = nn.Linear(self.fc1.out_features, out_features=num_classes, bias=True)
        self.ac4 = nn.LeakyReLU()

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform forward pass
        x = self.bn1(F.max_pool2d(self.ac1(self.conv1(x)), 2))
        x = self.conv_drop(x)
        x = self.bn2(F.max_pool2d(self.ac2(self.conv2(x)), 2))
        x = self.conv_drop(x)
        x = self.bn3(F.max_pool2d(self.ac3(self.conv3(x)), 2))
        x = self.conv_drop(x)
        x = x.view(-1, self.conv3.out_channels * 2 * 2)
        x = self.ac4(self.fc1(x))
        x = self.fc_drop(x)
        feature_vector = self.fc2(x)
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
    m = GTSRB_new()
    print(count_parameters(m))
