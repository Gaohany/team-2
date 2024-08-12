import argparse
import json
import os
import pickle
from datetime import datetime

import toml
import torch

from pathlib import Path
import metrics_utils.utils_metrics as utils_metrics
from fuzzer.model_structure import model_structure
from metrics_utils.features_from_input import *
from metrics_utils.gini_index_archive import *
from metrics_utils.visualization_gini_index import *
from utils.util import obj


def compare_pca_variance_with_combined_dataset(metrics_config, config, feature_dict):
        pca_both_dataset = []
        dataset_working = []
        dimensionality_list = metrics_config.measured_dimensionalities
        for l in range(len(metrics_config.modes)):
            if metrics_config.modes[l] == "all":
                vector_path_ = os.path.join(f"{config.data}/combined_vector/combined_{config.data}.pkl")
                if not os.path.exists(vector_path_):
                    collect_all_feature_vectors(f"{config.data}/combined_vector/", config)
                else:

                    with open(vector_path_, "rb") as vector_collected_dataset:
                        feature_dict_all = pickle.load(vector_collected_dataset)

                    data_for_pca = []
                    for i in range(len(feature_dict_all)):
                        array_for_dataset = utils_metrics.load_data(
                            config, feature_dict_all[i], metrics_config.type
                        )
                        data_for_pca.append(array_for_dataset)
                    array_1d = [element for sublist in data_for_pca for element in sublist]
                    np_array_ = np.array(array_1d)
                    cumulative_variance_ratio_list_all = []
                    for j in range(len(dimensionality_list)):
                        pca_all_dataset = PCA(n_components=dimensionality_list[j])
                        pca_all_dataset.fit(np_array_)
                        cumulative_variance_ratio = 0
                        expl_ratio_all = pca_all_dataset.explained_variance_ratio_
                        for k in range(len(expl_ratio_all)):
                            cumulative_variance_ratio += expl_ratio_all[k]
                        cumulative_variance_ratio_list_all.append(cumulative_variance_ratio)
                    pca_both_dataset.append(cumulative_variance_ratio_list_all)
                    dataset_working.append("combined")

            else:
                data_gen_ = utils_metrics.load_data(config, feature_dict, metrics_config.type)
                cumulative_variance_ratio_list_other = []
                for i in range(len(dimensionality_list)):
                    pca_other = PCA(n_components=dimensionality_list[i])
                    pca_other.fit(data_gen_)
                    cumulative_variance_ratio = 0
                    expl_ratio_other = pca_other.explained_variance_ratio_
                    for j in range(len(expl_ratio_other)):
                        cumulative_variance_ratio += expl_ratio_other[j]
                    cumulative_variance_ratio_list_other.append(cumulative_variance_ratio)
                pca_both_dataset.append(cumulative_variance_ratio_list_other)
                dataset_working_now = "train" if metrics_config.mode == "train" else "org" if metrics_config.mode == "test" and metrics_config.type == "org" else metrics_config.experiment_path.split("\\")[-1].split("_")[-1]
                dataset_working.append(dataset_working_now)

            if metrics_config.compare_two_pca_models:
                compare_training_with_all_combined_pca_visualization(
                    metrics_config, pca_both_dataset, dataset_working, "visualize_pca"
                ) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Comparing cumulative explained variance ratio of combined data set with a specified data set")
    parser.add_argument("--mc", help="Choose the metrics config file for getting the parameters", type=str)
    parser.add_argument("--dc", type=str, help="Choose with in which data set this comparison will be made")
    parser.add_argument("--feature_dict_path", type=str, help="Choose which feature vector will be compared with the combined data set")
    
    args = parser.parse_args()

    metrics_config = json.loads(json.dumps(toml.load(args.metrics_config_file)), object_hook=obj)
    config = json.loads(json.dumps(toml.load(args.dataset_config_file)), object_hook=obj)

    compare_pca_variance_with_combined_dataset(
        metrics_config,
        config,
        feature_dict=Path(args.feature_dict_path)
    )