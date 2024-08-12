import json
import os
from datetime import datetime

import numpy as np
import torch

COMPANY = "tum"
ALGORITHM = "gtsrb_new "
VERSION = "v1-GR"
TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class_name_dict = {
    "gtsrb": [
        "Speed limit (20km/h)",
        "Speed limit (30km/h)",
        "Speed limit (50km/h)",
        "Speed limit (60km/h)",
        "Speed limit (70km/h)",
        "Speed limit (80km/h)",
        "End of speed limit (80km/h)",
        "Speed limit (100km/h)",
        "Speed limit (120km/h)",
        "No passing",
        "No passing for vehicles over 3.5 metric tons",
        "Right-of-way at the next intersection",
        "Priority road",
        "Yield",
        "Stop",
        "No vehicles",
        "Vehicles over 3.5 metric tons prohibited",
        "No entry",
        "General caution",
        "Dangerous curve to the left",
        "Dangerous curve to the right",
        "Double curve",
        "Bumpy road",
        "Slippery road",
        "Road narrows on the right",
        "Road work",
        "Traffic signals",
        "Pedestrians",
        "Children crossing",
        "Bicycles crossing",
        "Beware of ice/snow",
        "Wild animals crossing",
        "End of all speed and passing limits",
        "Turn right ahead",
        "Turn left ahead",
        "Ahead only",
        "Go straight or right",
        "Go straight or left",
        "Keep right",
        "Keep left",
        "Roundabout mandatory",
        "End of no passing",
        "End of no passing by vehicles over 3.5 metric tons",
    ],
    "svhn": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "mnist": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
}


def save_predictions(
    average_precision: dict,
    height: int,
    width: int,
    file_name_json: str,
    dataset: str,
    criteria: str,
    print_logs: bool = True,
) -> None:
    if print_logs:
        print("Saving predictions in .json...")

    if dataset == "gtsrb" or dataset == "gtsrb-gray":
        class_name = class_name_dict["gtsrb"]
    else:
        class_name = class_name_dict[dataset]

    header = dict(
        __version_entry__=[
            dict(__Tool__="{}{}".format(COMPANY, ALGORITHM), __Version__="{}".format(VERSION), __Time__=TIME)
        ]
    )

    eval_metrics = dict(
        __evaluations_metrics__=[
            dict(
                op_class=int(average_precision["op_class"]),
                op_class_name=class_name[int(average_precision["op_class"])],
                op_class_prob=round(float(average_precision["op_class_prob"]), 2),
            )
        ]
    )

    org_metrics = dict(
        __ground_truth_metrics__=[
            dict(
                op_class=int(average_precision["org_pred_class"]),
                op_class_name=class_name[int(average_precision["org_pred_class"])],
                op_class_prob=round(float(average_precision["org_class_pred_prob"]), 2),
            )
        ]
    )

    add_info_1 = dict(
        __image_quality_metrics__=[
            dict(
                org_image=average_precision["org_image_name"],
                SSIM=round(float(average_precision["ssim_ref"]), 2),
                MSE=round(float(average_precision["mse_ref"]), 2),
                l0_norm=int(average_precision["l0_ref"]),
                l2_norm=round(float(average_precision["l2_ref"]), 2),
                linf_norm=round(float(average_precision["linf_ref"]), 2),
                transformations=average_precision["transformation_class"],
            )
        ]
    )

    _gt_labels = dict(
        __gt_labels__=[
            dict(
                __gt_class__=int(average_precision["gt_classes"]),
                __gt_classname__=class_name[int(average_precision["gt_classes"])],
            )
        ]
    )

    if criteria == "lscd":
        add_info_2 = dict(
            __latent_space_distances__=[
                dict(org_dist=average_precision["org_dist"], curr_dist=average_precision["curr_dist"])
            ]
        )

        file = {**header, **eval_metrics, **org_metrics, **add_info_1, **add_info_2, **_gt_labels}
    else:
        file = {**header, **eval_metrics, **org_metrics, **add_info_1, **_gt_labels}

    with open(os.path.join(file_name_json), "w") as json_file:
        json.dump(file, json_file, indent=4)

    if print_logs:
        print("Done.")
