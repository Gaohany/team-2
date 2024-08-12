import json
import os
import pickle
import sys
from math import sqrt

import numpy as np
import toml
import torch
from numpy import mean, var
from scipy.stats import ttest_rel, wilcoxon
from tqdm import tqdm

sys.path.append("/home/vekariya/Documents/practicum_dl_testing/testing_framework_fortiss_classifiers")

experiment_path = ["/data/disk1/experiments/new_1/gtsrb_3/"]
num_classes = 43
op_log_file = "crashes_classes_new.json"
data_type = "queue"
analyse = ["queue"]  # queue, crashes_rec

ip_file_paths = []
for dir in experiment_path:
    ip_path = os.path.join(dir, analyse[0])
    path = os.listdir(ip_path)
    ip_file_paths.append([os.path.join(ip_path, file_path) for file_path in path])
    # ip_file_paths.append([os.path.join(ip_path,file_path) for file_path in path if file_path.endswith('.json')])

comb_dict = {}

for i in range(len(ip_file_paths)):
    dir_dict = {}
    for j in range(num_classes):
        metric_list = 0
        dir_dict[j] = metric_list

    for j in tqdm(range(len(ip_file_paths[i]))):
        ip_json = ip_file_paths[i][j]
        with open(os.path.join(ip_json), "r") as file:
            if data_type == "crashes":
                data = json.load(file)
                class_id = data["__gt_labels__"][0]["__gt_class__"]
            else:
                data = torch.load(ip_json)
                class_id = int(data["seed_obj"].gt_label_dict)
            dir_dict[class_id] += 1
    comb_dict.update({i: dir_dict})

with open(op_log_file, "w") as json_file:
    json.dump(comb_dict, json_file, indent=4)
    print("Results written to:", op_log_file)
