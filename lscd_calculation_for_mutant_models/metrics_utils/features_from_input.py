import json
import os
import pickle
import time
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity, NearestNeighbors
from tqdm import tqdm

from scipy.stats import norm

import metrics_utils.gini_index_archive as gini_index_archive
import metrics_utils.metrics as Metrics
import metrics_utils.utils_metrics as utils_metrics
from dataset.base_dataset import BaseDataset
from dataset.GTSRB.gtsrb_dataset import GTSRBDataset
from dataset.GTSRB.gtsrb_dataset_gray import GTSRBDataset_gray
from dataset.MNIST.mnist_dataset import MNISTDataset
from dataset.SVHN.svhn_dataset import SVHNDataset
from metrics_utils.metrics import cent, equivalence_partitioning, euclidean_dist
from metrics_utils.metrics_dataloader import *
from metrics_utils.metrics_dataloader import metrics_dataloader
from metrics_utils.utils_metrics import (
    create_centroid_positioning_structure_merged,
    create_equivalent_partitioning_structure,
    create_features_dict_merged,
)
from metrics_utils.visualization import *
from models.default_model import ClassificationModel


def get_feature_vectors(model: ClassificationModel, dataset: BaseDataset, index: int, config, type: str = "org"):
    image, gt_class = dataset.__getitem__(index)
    image = image.unsqueeze(dim=0).to(config.device)
    x_pred_tags, x_pred_prob, pred_probs, feature_vector = model.inference(image, req_feature_vec=True)
    # Pass feature vectors only for TPs.
    if (gt_class == x_pred_tags) and (type == "org"):
        return x_pred_tags, gt_class, x_pred_prob, pred_probs, feature_vector
    elif (gt_class != x_pred_tags) and (type == "org"):
        feature_vector = np.array([])
        return x_pred_tags, gt_class, x_pred_prob, pred_probs, feature_vector
    else:
        return x_pred_tags, gt_class, x_pred_prob, pred_probs, feature_vector


def calculate_initial_centroid_radius(model: ClassificationModel, config, centroids_save_path: str, log_filepath: str):
    print("Calculating the initial centroid positioning & radius thresholds from the train dataset.")
    # Mode is hard-coded as "train" as centroids are to be calculated only for training dataset.
    if config.data == "mnist":
        dataset_train = MNISTDataset(
            config=config, image_set=config.detection_model.image_set, mode="train", augmentation=False
        )

    elif config.data == "gtsrb":
        dataset_train = GTSRBDataset(
            config=config, image_set=config.detection_model.image_set, mode="train", augmentation=False
        )

    elif config.data == "gtsrb-gray":
        dataset_train = GTSRBDataset_gray(
            config=config, image_set=config.detection_model.image_set, mode="train", augmentation=False
        )

    elif config.data == "svhn":
        dataset_train = SVHNDataset(
            config=config, image_set=config.detection_model.image_set, mode="train", augmentation=False
        )

    else:
        raise NotImplementedError("Please extend the method to defined dataset.")

    num_classes = config.num_classes

    print("Length of the dataset is: ", len(dataset_train))

    all_centroids = np.zeros((num_classes, num_classes), dtype=np.float32)
    all_features_merged = np.zeros((num_classes, num_classes), dtype=np.float32)
    num_samples_total = np.zeros((num_classes,), dtype=np.uint)

    centroid_distances: List[List[float]] = [[] for _ in range(num_classes)]
    prediction_confidences: List[List[float]] = [[] for _ in range(num_classes)]
    all_radius: List[float] = []

    with torch.no_grad():
        samples_ignored = 0

        print("Computing feature vectors ...")
        for i in tqdm(range(len(dataset_train))):
            x_pred_tags, gt_class, x_pred_prob, _, feature_vector = get_feature_vectors(
                model, dataset=dataset_train, index=i, config=config, type="org"
            )
            x_pred_tags, gt_class = int(x_pred_tags), int(gt_class)

            if (feature_vector.size != 0) and (x_pred_tags == gt_class) and not np.any(np.isnan(feature_vector)):
                all_features_merged[gt_class] = all_features_merged[gt_class] + feature_vector
                num_samples_total[gt_class] += 1

            else:
                samples_ignored += 1

        print(f"Ignored {samples_ignored} out of {len(dataset_train)} total samples.")

        # Compute average feature vector per class, i.e. the centroid
        for class_idx in range(num_classes):
            if num_samples_total[class_idx] > 0:
                centroid = all_features_merged[class_idx] / num_samples_total[class_idx]
            else:
                centroid = np.zeros((num_classes,), dtype=np.float32)

            all_centroids[class_idx] = centroid

        centroid_info_dict = {"all_centroids": all_centroids}
        all_centroids = centroid_info_dict["all_centroids"]

        samples_ignored = 0

        print("Computing centroid radii ...")
        for i in tqdm(range(len(dataset_train))):
            x_pred_tags, gt_class, x_pred_prob, _, feature_vector = get_feature_vectors(
                model, dataset=dataset_train, index=i, config=config, type="org"
            )
            x_pred_tags, gt_class = int(x_pred_tags), int(gt_class)

            if (feature_vector.size != 0) and (x_pred_tags == gt_class) and not np.any(np.isnan(feature_vector)):
                radius = float(euclidean_dist(feature_vector, all_centroids[gt_class]))
                centroid_distances[gt_class].append(radius)  # Radius based on individual object from init_centroid
                prediction_confidences[gt_class].append(x_pred_prob)

            else:
                samples_ignored += 1

        # Compute centroid radius at threshold
        for class_idx in range(num_classes):
            class_centroid_distances = np.array(centroid_distances[class_idx])

            threshold = 0.7
            centroid_radius = np.percentile(class_centroid_distances, threshold)
            all_radius.append(centroid_radius)

            print(
                f"Class idx: {class_idx} - Max Euclidean Distance: {class_centroid_distances.max():.4g} - Centroid Radius: {centroid_radius}"
            )

        centroid_info_dict.update({"all_radius": all_radius})
        centroid_info_dict.update({"all_distances_train_data": centroid_distances})
        centroid_info_dict.update({"all_tp_conf_train_data": prediction_confidences})

        os.makedirs(os.path.dirname(centroids_save_path), exist_ok=True)
        print("Saving centroids to", centroids_save_path)
        torch.save(centroid_info_dict, centroids_save_path)

        if os.path.exists(log_filepath):
            with open(log_filepath) as json_file:
                data = json.load(json_file)
                data.update({"Radius Threshold Values:": centroid_info_dict["all_radius"]})
        else:
            data = {"Radius Threshold Values:": centroid_info_dict["all_radius"]}

        with open(log_filepath, "w") as json_file:
            json.dump(data, json_file, indent=4)
            print("Results written to:", log_filepath)

        print("Initial Centroids & radius threshold values are saved!")

        print(f"{samples_ignored} samples ignored !")


def store_feature_vectors(model, config, type="org", mode="test", output_file_name=None) -> dict:
    # This is to store feature vectors corresponding to test images. To be run once and later can be used for calculations.

    print("Storing feature vectors for the given {} dataset.".format(type))

    dataset_test = metrics_dataloader(config=config, type=type, mode=mode)

    main_dict = {}

    with torch.no_grad():
        for i in tqdm(range(len(dataset_test))):
            img_wise_dict = {}

            image_path = dataset_test._image_paths[i]
            if isinstance(image_path, str):
                image_id = image_path.split("/")[-1]
            else:
                image_id = str(i)
            x_pred_tags, gt_class, x_pred_prob, pred_probs, feature_vector = get_feature_vectors(
                model, dataset=dataset_test, index=i, config=config, type=type
            )
            classes_list = gt_class.tolist()
            img_wise_dict.update(
                {
                    "feature_vector": feature_vector,
                    "classes": classes_list,
                    "pred_probs": pred_probs,
                    "predicted_label": x_pred_tags,
                    "x_pred_prob": x_pred_prob,
                    "pred_prob": x_pred_prob,
                    "image_path": image_path,
                }
            )
            main_dict.update({image_id: img_wise_dict})

    with open(output_file_name, "wb") as f:
        torch.save(main_dict, f)

    return main_dict

def store_feature_vectors_mt(model, config, type, mode, output_file_name, dataset_test) -> dict:
    # This is to store feature vectors corresponding to test images. To be run once and later can be used for calculations.

    main_dict = {}

    with torch.no_grad():
        for i in tqdm(range(len(dataset_test))):
            img_wise_dict = {}

            image_path = dataset_test._image_paths[i]
            if isinstance(image_path, str):
                image_id = image_path.split("/")[-1]
            else:
                image_id = str(i)
            x_pred_tags, gt_class, x_pred_prob, pred_probs, feature_vector = get_feature_vectors(
                model, dataset=dataset_test, index=i, config=config, type=type
            )
            classes_list = gt_class.tolist()
            img_wise_dict.update(
                {
                    "feature_vector": feature_vector,
                    "classes": classes_list,
                    "pred_probs": pred_probs,
                    "predicted_label": x_pred_tags,
                    "x_pred_prob": x_pred_prob,
                    "pred_prob": x_pred_prob,
                    "image_path": image_path,
                }
            )
            main_dict.update({image_id: img_wise_dict})

    with open(output_file_name, "wb") as f:
        torch.save(main_dict, f)

    return main_dict


def calculate_latent_space_distances(img_ids, feature_vectors_dict_main, config):
    classes_distance_all, avg_classes_distance, max_classes_distance = {}, {}, {}
    ignored = 0

    for key in img_ids:
        gt_class = int(feature_vectors_dict_main[key]["classes"][0])
        try:
            distance = feature_vectors_dict_main[key]["distances"][gt_class]

            if not classes_distance_all.get(gt_class):
                classes_distance_all[gt_class] = []

            classes_distance_all[gt_class].extend(distance)
        except:
            ignored += 1

    for cl in range(config.num_classes):
        if not max_classes_distance.get(cl):
            max_classes_distance[cl] = []
        try:
            max_dist = np.max(classes_distance_all[cl])
            max_classes_distance[cl] = round(float(max_dist), 2)
        except:
            print("No samples from class {}.".format(cl))

    for cl in range(config.num_classes):
        if not avg_classes_distance.get(cl):
            avg_classes_distance[cl] = []
        try:
            mean_dist = np.mean(classes_distance_all[cl])
            avg_classes_distance[cl] = round(float(mean_dist), 2)
        except:
            print("No samples from class {}.".format(cl))

    # classes_distance_copy = copy.deepcopy(classes_distance)
    # max_class_distances_copy = copy.deepcopy(max_classes_distance)
    # avg_classes_distance_copy = copy.deepcopy(avg_classes_distance)

    return classes_distance_all, max_classes_distance, avg_classes_distance


def calculate_lscd(
    model,
    config,
    centroids_save_path,
    log_filepath,
    feature_vectors_dict_main=None,
    type="org",
    radius_type="base",
    output_path="new.pickle",
):
    print("Calculating the LSCD from the {} dataset.".format(type))

    # The first part calculates % values of test samples outside radius value. Also, it stores distances of each data point for detailed analysis.
    num_classes = config.num_classes

    if radius_type == "base":
        print("Calculating LSC from base radius values.")
        all_centroids = torch.load(centroids_save_path)
        centroids = all_centroids["all_centroids"]
        radius_values = all_centroids["all_radius"]
        print(radius_values)
    else:
        print("Calculating LSC from automatic radius values from Elbow point.")
        all_centroids = torch.load(centroids_save_path)
        centroids = all_centroids["all_centroids"]
        radius_values = all_centroids["all_radius_automatic"]
        print(radius_values)

    print("Centroids loaded!")

    total_obj_counter = create_centroid_positioning_structure_merged(num_classes=num_classes)
    lscd_counter = create_centroid_positioning_structure_merged(num_classes=num_classes)
    dist_metrics = create_features_dict_merged(num_classes=num_classes)
    lscd_avg = create_features_dict_merged(num_classes=num_classes)
    img_ids = feature_vectors_dict_main.keys()
    samples_ignored = 0

    with torch.no_grad():
        for key in tqdm(img_ids):
            classes_distance = {}
            feature_vector_img = feature_vectors_dict_main[key]["feature_vector"]
            classes_list = int(feature_vectors_dict_main[key]["classes"][0])

            if not feature_vector_img.size == 0:
                if not classes_distance.get(classes_list):
                    classes_distance[classes_list] = []

                    r = float(radius_values[classes_list])

                    if len(centroids[classes_list]) != 0:
                        if isinstance(centroids[classes_list], list):
                            centroids[classes_list] = torch.from_numpy(np.array(centroids[classes_list]))
                        centroid_cl = centroids[classes_list]
                        diff = np.abs(feature_vector_img - centroid_cl)
                        distance = np.linalg.norm(diff)
                        lscd_counter[classes_list] = lscd_counter[classes_list] + cent(
                            feature_vector_img, centroids[classes_list], r
                        )

                        total_obj_counter[classes_list] = total_obj_counter[classes_list] + 1
                        classes_distance[classes_list].append(distance)

                        feature_vectors_dict_main[key].update({"distances": classes_distance})
            else:
                samples_ignored += 1
                pass

    print("Samples ignored {} out of total {} objects.".format(samples_ignored, len(img_ids)))

    for cl in range(num_classes):
        if total_obj_counter[cl] != 0:
            lscd = lscd_counter[cl] / total_obj_counter[cl]
            lscd_avg[cl] = lscd
        else:
            lscd_avg[cl] = None

    with open(output_path, "wb") as f:
        torch.save(feature_vectors_dict_main, f)

    # In second part we use these distances to calculate distance based metrics (Average, Max, Min distance, etc.).
    _, max_classes_distance, avg_classes_distance = calculate_latent_space_distances(
        img_ids, feature_vectors_dict_main, config
    )

    lscd_metric_value = round(np.array(list(avg_classes_distance.values())).mean(),3)
    print("LSCD Value: ", lscd_metric_value)
    
    if os.path.exists(log_filepath):
        with open(log_filepath) as json_file:
            data = json.load(json_file)
            key = "Latent_Space_Class_Dispersion_" + str(config.mode) + "_" + str(type) + "(in %)"
            data.update({key: lscd_avg})
            key_1 = "Average_Latent_Space_Class_Distance_" + str(config.mode) + "_" + str(type)
            data.update({key_1: avg_classes_distance})
            key_2 = "Maximum_Latent_Space_Class_Distance_" + str(config.mode) + "_" + str(type)
            data.update({key_2: max_classes_distance})
            key_3 = "LSCD_" + str(config.mode) + "_" + str(type)
            data.update({key_3: lscd_metric_value})
    else:
        key = "Latent_Space_Class_Dispersion_" + str(config.mode) + "_" + str(type) + "(in %)"
        data.update({key: lscd_avg})
        key_1 = "Average_Latent_Space_Class_Distance_" + str(config.mode) + "_" + str(type)
        data.update({key_1: avg_classes_distance})
        key_2 = "Maximum_Latent_Space_Class_Distance_" + str(config.mode) + "_" + str(type)
        data.update({key_2: max_classes_distance})
        key_3 = "LSCD_" + str(config.mode) + "_" + str(type)
        data.update({key_3: lscd_metric_value})

    with open(log_filepath, "w") as json_file:
        json.dump(data, json_file, indent=4)
        print("Results written to:", log_filepath)

    print(f"{samples_ignored} were ignored !")

    return lscd_avg, avg_classes_distance, max_classes_distance


def calculate_equivalent_partitioning(config, log_filepath, type):
    print("Calculating equivalent partitioning for individual splits in the entire dataset.")

    num_classes = config.num_classes
    dataset = config.data

    if dataset == "gtsrb":
        # dataset_train = A2D2BboxDataset(config=config, image_set=config.ssd_model.image_set, mode='train', augmentation=False)
        if type == "org":
            dataset_train = GTSRBDataset(
                config=config, image_set=config.detection_model.image_set, mode="train", augmentation=False
            )
            dataset_test = GTSRBDataset(
                config=config, image_set=config.detection_model.image_set, mode="test", augmentation=False
            )
        elif type == "crashes":
            print("TO DO")
    else:
        NotImplementedError

    if type == "org":
        datasets_used = [dataset_train, dataset_test]
        datasets_list = ["train", "test"]
    elif type == "crashes":
        datasets_used = [dataset_test]
        datasets_list = ["adv_dataset"]

    ep_metrics = create_equivalent_partitioning_structure(num_classes=num_classes)

    ns_per_class_dict = {}
    epi_per_class_dict = {}

    for i in range(len(datasets_used)):
        ns_per_class = []
        for _ in range(num_classes):
            ns_per_class.append(0)

        classes = datasets_used[i].classes
        difficulties = datasets_used[i].difficulties
        for j in tqdm(range(len(classes))):
            temp = np.array(classes[j])
            temp = temp[~difficulties[j]]
            for cl in temp:
                ns_per_class[cl] += 1

        ns_per_class_dict.update({datasets_list[i]: ns_per_class})
        epi_per_class = equivalence_partitioning(ns_per_class_dict[datasets_list[i]], num_classes)
        epi_per_class_dict.update({datasets_list[i]: epi_per_class})

    ep_metrics["samples_per_class"] = ns_per_class_dict
    ep_metrics["ep_metrics"] = epi_per_class_dict

    if os.path.exists(log_filepath):
        with open(log_filepath) as json_file:
            data = json.load(json_file)
            data.update({"Equivalent Partitioning:": ep_metrics})
    else:
        data = {"Equivalent Partitioning:": ep_metrics}

    with open(log_filepath, "w") as json_file:
        json.dump(data, json_file, indent=4)
        print("Results written to:", log_filepath)

    return ep_metrics


def calculate_boundary_conditioning(model, config, log_filepath, type="org", mode="test", theta1=0.0, theta2=0.5):
    ## To DO: be adapted for multi class.

    print("Calculating the boundary conditioning from ", mode, " dataset.")

    num_classes = config.num_classes
    dataset = config.data

    if dataset == "gtsrb":
        dataset_test = GTSRBDataset(
            config=config, image_set=config.ssd_model.image_set, mode=config.mode, augmentation=False
        )
    else:
        raise NotImplementedError

    print("Length of the dataset is: ", len(dataset_test))

    bc_metrics = {config.mode: 0}

    GTs, TPs, FNs = 0, 0, 0
    all_conf_scores = []


def calculate_annd(config, log_file, feature_vectors_dict_main=None, type="org", output_path="new.pkl"):
    print("Calculating the ANND/KNN-Density from the {} dataset.".format(type))

    num_classes = config.num_classes
    img_keys = feature_vectors_dict_main.keys()
    list_img_keys = list(img_keys)
    samples_ignored = 0
    feature_matrix = np.zeros((len(list_img_keys), num_classes))
    with torch.no_grad():
        for key in tqdm(img_keys):
            feature_vector_img = feature_vectors_dict_main[key]["feature_vector"]
            if feature_vector_img.size != 0:
                feature_matrix[list_img_keys.index(key)] = feature_vector_img
            else:
                samples_ignored += 1
    print("Samples ignored {} out of total {} objects.".format(samples_ignored, len(img_keys)))

    # find nearest neighbour distances
    neigh = NearestNeighbors(n_neighbors=config.annd_num_neighbours)
    neigh.fit(feature_matrix)
    neigh_dist, neigh_ind = neigh.kneighbors()
    annd = np.mean(neigh_dist)

    with open(output_path, "wb") as f:
        pickle.dump(feature_vectors_dict_main, f)

    key = "Average_nearest_neighbour_distance_" + str(config.mode) + "_" + str(type)
    if os.path.exists(log_file):
        with open(log_file) as json_file:
            data = json.load(json_file)
            data.update({key: annd})
    else:
        data = {key: annd}

    with open(log_file, "w") as json_file:
        json.dump(data, json_file, indent=4)
        print("Results written to:", log_file)

    print(f"{samples_ignored} samples were ignored !")

    return annd


def calculate_gini_index_class_based_approach(
    metrics_config,
    config,
    feature_dict,  # feature_vector here
    number_bins_per_dimension: int,
    log_file,
    type_dataset: str,
    num_dimensionality: int,
    dim_type="pca",
    use_traditional_gini_index_method: bool = False,
) -> list:
    """
    Calculates Gini Index of each classes taking into consideration that they are the ground truth classes of the data points we are observing

    Arguments:

        metrics_config: General metric configuration
        config: Configuration for specific data sets (GTSRB, SVHN, MNIST)
        feature_dict: Data points from which class based Gini Index will be calculated
        number_bins_per_dimension: Hyperparameter of number of bins per dimension
        log_file: Log filepath
        type_dataset: Type of the data set
        num_dimensionality: Number of dimensions to be reduced
        dim_type: Method for using dimensionality reduction
        use_traditional_gini_index_method: Boolean for using whether the traditional approach or adapted Gini Index approach

    Returns:
        gini_index_gt_label_based: Dictionary of entries having seperated based on classes
    """
    data_as_array = utils_metrics.load_data(config, feature_dict, type_dataset)
    gen_data = utils_metrics.reduce_dimensionality(metrics_config, config, data_as_array, dim_type)
    functional_range_ = calculate_functional_range(
        metrics_config, config, type_dataset, num_dimensionality, dim_type, store_as_pkl=True
    )

    gt_class_points_array = [[] for _ in range(config.num_classes)]
    # iterating through the array to assign each point to its class array
    ground_truth_classes = []
    for element_dict in feature_dict.values():
        feature_vector = element_dict["feature_vector"]
        if len(feature_vector) == 0:
            continue
        ground_truth_classes.append(element_dict["classes"])

    for data_point, point_gt_label in zip(gen_data, ground_truth_classes):
        gt_class_points_array[point_gt_label[0]].append(data_point)

    gt_class_points_numpy_array = [
        np.reshape(gt_class_points_array[i], (len(gt_class_points_array[i]), num_dimensionality))
        for i in range(len(gt_class_points_array))
    ]
    # calculate gini index for each dimensionality reduced data point clouds based on their ground truth labels
    gini_index_gt_label_based = {}
    for index, class_points_list in zip(range(len(gt_class_points_numpy_array)), gt_class_points_numpy_array):
        histogram_values, binedges = np.histogramdd(
            class_points_list, bins=number_bins_per_dimension, range=functional_range_
        )
        gini_sum = 0
        flattened_histogram = histogram_values.flatten()
        if use_traditional_gini_index_method:
            for i in range(len(flattened_histogram)):
                gini_sum += (flattened_histogram[i] / class_points_list.shape[0]) ** 2
            gini_index = 1 - gini_sum
            gini_index_gt_label_based.update({index: gini_index})
        else:
            for i in range(len(flattened_histogram) - 1):
                for j in range(i + 1, len(flattened_histogram)):
                    gini_sum += np.abs(flattened_histogram[i] - flattened_histogram[j])
            gini_index = gini_sum / (len(flattened_histogram) * len(data_as_array))
            gini_index_gt_label_based.update({index: gini_index})

    return gini_index_gt_label_based


def calculate_gini_index_all_data(
    metrics_config,
    config,
    data_path,
    number_bins_per_dimension: int,
    log_file,
    type_dataset: str,
    num_dimensionality: int,
    dim_type="pca",
    use_traditional_gini_index_method: bool = False,
) -> float:
    """Calculates the Gini Index in the reduced dimensionality by using histogramdd()

    Args:
        config: Configuration parameter of a certain dataset (GTSRB, MNIST, SVHN...)
        data_path: Data set from which Gini Index is going to be computed
        number_bins_per_dimension: List of number of bins per dimension
        log_file: File path for logging the gini index results
        type_dataset: Type of the dataset (train, val, test)
        num_dimensionality: Number of dimensions to reduce the dimensionality of data points. If Gini Index will be calculated class-based, this parameter is not necessary to fill in
        dim_type: Dimensionality reduction method to apply to the data points. Supported methods are "pca" and "umap"
        use_traditional_gini_index_method: If the traditional method used in Gini Index, i.e. the method used in decision tree algorithms, this should be set in default

    Returns:
        Result of Gini Index
    """
    data_as_array = utils_metrics.load_data(config, data_path, type_dataset)

    gen_data = utils_metrics.reduce_dimensionality(metrics_config, config, data_as_array, dim_type)
    functional_range = calculate_functional_range(
        metrics_config, config, type_dataset, num_dimensionality, dim_method=dim_type, store_as_pkl=True
    )
    histogram_values, binedges = np.histogramdd(gen_data, bins=number_bins_per_dimension, range=functional_range)

    gini_sum = 0
    flattened_histogram = histogram_values.flatten()
    # new implementation of Gini Index based on the implementation used in decision trees, doesn't consider the empty bins
    # in order to let the local density to see (without the empty bins), we are getting only the gini sum, not 1- gini_sum

    if use_traditional_gini_index_method:
        for i in range(len(flattened_histogram)):
            gini_sum += (flattened_histogram[i] / (len(data_as_array))) ** 2

        gini_index = gini_sum

    else:
        for i in range(len(flattened_histogram) - 1):
            for j in range(i + 1, len(flattened_histogram)):
                gini_sum += np.abs(flattened_histogram[i] - flattened_histogram[j])
        gini_index = gini_sum / (len(flattened_histogram) * len(data_as_array))

    return gini_index


# implemented mainly for visualization tasks
def calculate_gini_index_in_batches(
    metrics_config,
    config,
    number_bins_per_dimension_list: list,
    log_file,
    type_dataset: str,
    num_dimensionality: int,
    dim_type="pca",
    use_traditional_gini_index_method: bool = False,
    gini_index_type: int = 0,
) -> list[list]:
    """Calculates the Gini Index for each given number of bins per dimension in the selected type of gini index
        Returns the list of the results
    Args:
        config: Configuration parameter of a certain dataset (GTSRB, MNIST, SVHN...)
        data_path: Data set from which Gini Index is going to be computed
        number_bins_per_dimension_list: List of number of bins per dimension
        log_file: File path for logging the gini index results
        type_dataset: Type of the dataset (train, val, test)
        num_dimensionality: Number of dimensions to reduce the dimensionality of data points. If Gini Index will be calculated class-based, this parameter is not necessary to fill in
        dim_type: Dimensionality reduction method to apply to the data points. Supported methods are "pca" and "umap"
        use_traditional_gini_index_method: If the traditional method used in Gini Index, i.e. the method used in decision tree algorithms, this should be set in default
        gini_index_type: Type of Gini Index to make the calculation. 0= calculation of Gini Index with the method histogramdd(), 1= calculation of Gini Index in each class, 2= calculation of Gini Index for whole data points but without histogramdd()

    Returns:
        List of list of Gini Index results for different data sets
    """
    gini_index_different_datasets = []
    dataset_working = []
    iteration = 0
    for dirpath, dirname, filenames in os.walk(config.vector_path):
        if len(filenames) == 0:
            continue
        feature_dict = None
        for filename in filenames:
            vector_path = os.path.join(dirpath, filename)
            print(f"Current vector path is {vector_path}")
            if not vector_path.__contains__(config.data):
                print(
                    f"Vector in the path {vector_path} can't be used for the visualization of different datasets for {config.data}! Continuing..."
                )
                continue
            try:
                with open(vector_path, "rb") as vp:
                    print(f"Loading feature vectors from: {vector_path}")
                    feature_dict = pickle.load(vp)
                    if vector_path.__contains__("train"):
                        dataset_working.append("train")
                    else:
                        dataset_working.append(vector_path.split("\\")[-1].split("_")[-2])
            except FileNotFoundError:
                pass

            if feature_dict is None:
                continue

            if gini_index_type == 0:
                calculated_gini_index = [
                    calculate_gini_index_all_data(
                        metrics_config,
                        config,
                        feature_dict,
                        number_bins_per_dimension_list[i],
                        log_file,
                        type_dataset,
                        num_dimensionality,
                        dim_type,
                        use_traditional_gini_index_method,
                    )
                    for i in range(len(number_bins_per_dimension_list))
                ]

            elif gini_index_type == 1:
                calculated_gini_index = [
                    calculate_gini_index_class_based_approach(
                        config,
                        feature_dict,
                        number_bins_per_dimension_list[i],
                        log_file,
                        type_dataset,
                        use_traditional_gini_index_method,
                    )
                    for i in range(len(number_bins_per_dimension_list))
                ]

            elif gini_index_type == 2:
                # gini index calculation by not using numpy.histogramdd() function but custom built functions inside the framework
                calculated_gini_index = [
                    gini_index_archive.calculate_gini_index_with_pca_without_histogramdd(
                        config,
                        feature_dict,
                        number_bins_per_dimension_list[i],
                        log_file,
                        type_dataset,
                        num_dimensionality,
                        dim_type,
                        use_traditional_gini_index_method,
                    )
                    for i in range(len(number_bins_per_dimension_list))
                ]
            else:
                raise ValueError(
                    "This number does not have any corresponding Gini Index type! Please give gini_index_type parameter a value of 0, 1 or 2"
                )
            gini_index_different_datasets.append(calculated_gini_index)

    return gini_index_different_datasets, dataset_working


def calculate_functional_range(
    metrics_config,
    config,
    type_dataset: str,
    num_components: int = -1,
    dim_method: str = "pca",
    store_as_pkl: bool = True,
) -> list:
    """Calculates the functional range by the given .pkl files of train, crashes and original test dataset under vectors file.
       Returns the list of the min-max tuples for each dimension
    Args:
        config: configuration of the dataset (GTSRB, MNIST, SVHN...),
        type_dataset: type of the dataset (train, val, test)
        num_components: For how many dimensions the functional range should be reduced. If default (-1), no dimensionality reduction will be made
        dim_method: Method to perform dimensionality reduction. Supported methods are "pca" and "umap"
        store_as_pkl: Whether the calculated functional range should be stored as a .pkl file
    Returns:
    minimum and maximum of each dimension/feature contained in a tuple in a list
    """
    if num_components == -1:
        num_components = config.num_classes

    functional_range_path = os.path.join(
        "functional_range\\", "_".join([config.data, str(num_components), dim_method])
    )
    functional_range_pkl = os.path.join(functional_range_path, f"vectors.pkl")

    os.makedirs(functional_range_path, exist_ok=True)
    try:
        with open(functional_range_pkl, "rb") as func_range:
            print("Loading functional range from: ", functional_range_pkl)
            functional_range = pickle.load(func_range)
            print(functional_range)
        return functional_range
    except FileNotFoundError:
        pass

    min_each_dim, max_each_dim = [], []
    for dirpath, dirname, filenames in os.walk(config.calculations_path):
        if len(filenames) == 0:
            continue

        for filename in filenames:
            feature_dict = None
            vector_path = os.path.join(dirpath, filename)
            print("Current vector path is {}".format(vector_path))
            if not vector_path.__contains__(config.data):
                print(
                    f"This is not one of the datasets containing points regarding {config.data}! Passing {vector_path}"
                )
                continue
            else:
                try:
                    with open(vector_path, "rb") as vp:
                        print("Loading feature vectors from: ", vector_path)
                        feature_dict = pickle.load(vp)
                except FileNotFoundError:
                    pass
            if feature_dict == None:
                continue

            gen_data_array = utils_metrics.load_data(config, feature_dict, type_dataset)
            if (
                num_components != config.num_classes
            ):  # means a manual dimensionality is wanted, necessary for list comprehension while creating the tuples in line 694
                gen_data_array = utils_metrics.reduce_dimensionality(
                    metrics_config, config, gen_data_array, dim_method
                )
                gen_data_array_min_max_each_dim_pair = [
                    (np.min(gen_data_array[:, dim]), np.max(gen_data_array[:, dim])) for dim in range(num_components)
                ]
            if len(min_each_dim) > 0 and len(max_each_dim) > 0:
                # compare the previous min max pairs for each dimension with the new data new min max
                for j in range(len(gen_data_array_min_max_each_dim_pair)):
                    if gen_data_array_min_max_each_dim_pair[j][0] < min_each_dim[j]:
                        min_each_dim[j] = gen_data_array_min_max_each_dim_pair[j][0]
                for k in range(len(gen_data_array_min_max_each_dim_pair)):
                    if gen_data_array_min_max_each_dim_pair[k][1] > max_each_dim[k]:
                        max_each_dim[k] = gen_data_array_min_max_each_dim_pair[k][1]
            else:
                # set the min and max per dimension, happens only once
                min_each_dim = [
                    gen_data_array_min_max_each_dim_pair[j][0]
                    for j in range(len(gen_data_array_min_max_each_dim_pair))
                ]
                max_each_dim = [
                    gen_data_array_min_max_each_dim_pair[j][1]
                    for j in range(len(gen_data_array_min_max_each_dim_pair))
                ]

    # take the maximum range and make all the other dimensions equal to that range
    max_range = 0
    for min_dim, max_dim in zip(min_each_dim, max_each_dim):
        if max_dim - min_dim > max_range:
            max_range = max_dim - min_dim
    min_max_each_dim_new = []
    for min_dim, max_dim in zip(min_each_dim, max_each_dim):
        difference = max_range - (max_dim - min_dim)
        dim_min_max_pair = ()
        dim_min_max_pair_new = dim_min_max_pair + (min_dim - (difference / 2),) + (max_dim + (difference / 2),)
        min_max_each_dim_new.append(dim_min_max_pair_new)

    if store_as_pkl:
        with open(functional_range_pkl, "wb") as func_range:
            pickle.dump(min_max_each_dim_new, func_range)

    return min_max_each_dim_new


def calculate_ground_truth_label_means(training_data_feature_dict: dict[str, dict], config) -> list:
    """Calculates cluster means in the given training data by looking at the ground truth label of the data points

    Args:
        training_data_feature_dict: Numpy array of the training data to look at
        config: Configuration parameter of the dataset (GTSRB, MNIST, SVHN...)

    Returns:
        cluster means of each class
    """
    # seperate training data by the classes attribute in feature vector
    feature_vectors = []
    classes = []
    for element_dict in training_data_feature_dict.values():
        feature_vector = element_dict["feature_vector"]
        if len(feature_vector) == 0:
            continue
        feature_vectors.append(feature_vector)
        classes.append(element_dict["classes"])

    class_arrays = [[] for _ in range(config.num_classes)]
    mean_each_class = [[] for _ in range(config.num_classes)]
    for feature_vector_, class_ in zip(feature_vectors, classes):
        class_arrays[class_[0]].append(feature_vector_)

    # calculate mean for each class arrays
    for i, class_array in zip(range(len(mean_each_class)), class_arrays):
        mean_each_class[i].append(np.mean(class_array, axis=0))

    return mean_each_class


def collect_all_feature_vectors(output_path, config):
    feature_dict_total = []
    output_path_pkl = os.path.join(output_path, f"combined_{config.data}.pkl")
    if os.path.exists(output_path_pkl):
        with open(output_path_pkl, "rb") as combined_data:
            return pickle.load(combined_data)

    os.makedirs(output_path, exist_ok=True)
    for dirpath, dirname, filenames in os.walk(config.calculations_path):
        if len(filenames) == 0:
            continue
        feature_dict = None
        for filename in filenames:
            vector_path = os.path.join(dirpath, filename)
            print(f"Current vector path is {vector_path}")
            if not vector_path.__contains__(config.data):
                print(
                    f"Vector in the path {vector_path} can't be used for the whole data which will be used for PCA! Continuing..."
                )
                continue
            try:
                with open(vector_path, "rb") as vp:
                    print(f"Loading feature vectors from: {vector_path}")
                    feature_dict = pickle.load(vp)

            except FileNotFoundError:
                pass

            if feature_dict is None:
                continue
            feature_dict_total.append(feature_dict)

    with open(output_path_pkl, "wb") as feature_dict_storage:
        pickle.dump(feature_dict_total, feature_dict_storage)

    return feature_dict_total


def calculate_voxelisation(
    feature_vectors_dict_main: Dict,
    bin_counts: Union[int, List[int]],
    hypercube_min_max: Tuple[int, int],
    mode: str = "test",
    type_: str = "org",
    compute_bin_metrics: bool = False,
    pca_dims: Union[int, List[int]] = 3,
    log_filepath: str = None,
    max_cuboids_count: int = 10000000000,
    reuse_pca_transform: bool = False,
    pca_transform_save_dir: str = None,
) -> None:
    """PCA-reduce latent space and voxelise into equally sized cubes. Then count the
    number of cubes that contain at least one data point as well as bin metrics,
    such as the median number of points per cube, the median of variances of prediction
    probabilities per cube and the variance of variances of prediction probabilities. Dump
    the results to a log file, if provided.

    Args:
        feature_vectors_dict_main (Dict): Feature vectors dictionary
        bin_counts (Union[int, List[int]]): Number of bins per PCA-dimension.
            If a list of integers is provided, the algorithm will be run once per each.
        hypercube_min_max (Tuple[int, int]): The full size of latent space to consider, in PCA dimensions.
        mode (str, optional): Defaults to "test".
        type_ (str, optional): Defaults to "org".
        compute_bin_metrics (bool, optional): Defaults to False.
        pca_dims (Union[int, List[int]], optional): Number of dimensions in PCA-reduced space. If multiple are provided,
            the algorithm will be run once per each. Defaults to 3.
        log_filepath (str, optional): Filepath where to store the output. Defaults to None.
        max_cuboids_count (int, optional): Don't compute if combination of PCA dims and bin_count supercedes this
            value. Defaults to 10000000000.
        reuse_pca_transform (bool, optional): If True, reuse the PCA transform previously computed and stored to
            pca_transform_path. It is generally recommended to create a transform once on training data
            with reuse_pca_transform==False, and then reuse it on the corner case data with reuse_pca_transform==False.
            Defaults to False.
        pca_transform_save_dir (str, optional): If passed and reuse_pca_transform==False, store the PCA transforms
            at this folder path. If reuse_pca_transform==True, read the transform from this path. Defaults to None.
    """

    if isinstance(bin_counts, int):
        bin_counts = [bin_counts]

    if isinstance(pca_dims, int):
        pca_dims = [pca_dims]

    if reuse_pca_transform and pca_transform_save_dir is None:
        raise ValueError(f"If reuse_pca_transform==True, a pca_transform_save_dir must be provided!")

    hypercube_min, hypercube_max = hypercube_min_max

    feature_vectors = []
    prediction_probabilities = []
    for element_dict in feature_vectors_dict_main.values():
        feature_vector = element_dict["feature_vector"]
        if len(feature_vector) == 0:  # some feature vectors have length zero, somehow
            continue
        feature_vectors.append(feature_vector)

        # extract the prediction probability of the ground truth class, not the predicted class
        gt_class = element_dict["classes"][0]
        gt_pred_prob = element_dict["pred_probs"][0, gt_class]
        prediction_probabilities.append(float(gt_pred_prob))

    all_feature_vectors = np.stack(feature_vectors)  # shape (N, D)
    all_prediction_probabilities = np.array(prediction_probabilities)  # shape (N, )

    voxelisation_dicts = []
    for current_pca_dims in pca_dims:
        try:
            print(
                f"Conducting PCA on {all_feature_vectors.shape[0]} feature vectors. Reducing dimensions {all_feature_vectors.shape[1]} -> {current_pca_dims}"
            )

            if pca_transform_save_dir:
                transform_filepath = os.path.join(pca_transform_save_dir, f"pca_{current_pca_dims}_dims.pkl")

                if reuse_pca_transform:  # read pca transform from file
                    with open(transform_filepath, "rb") as fp:
                        pca = pickle.load(fp)
                else:  # fit new transform and save to file
                    pca = PCA(current_pca_dims)
                    pca.fit(all_feature_vectors)

                    with open(transform_filepath, "wb") as fp:
                        pickle.dump(pca, fp)

                # use the saved transform without fitting again
                reduced_feature_vectors = pca.transform(all_feature_vectors)

            else:  # fit new transform and apply immediately
                pca = PCA(current_pca_dims)
                reduced_feature_vectors = pca.fit_transform(all_feature_vectors)

            min_value = reduced_feature_vectors.min()
            max_value = reduced_feature_vectors.max()
            print(f"Min/Max values after reduction: {min_value} .. {max_value}")

            if np.any((reduced_feature_vectors < hypercube_min) | (reduced_feature_vectors > hypercube_max)):
                print(f"WARNING! Some latent space values exceed hypercube dimensions!")

            histogram_range = [(hypercube_min, hypercube_max)] * current_pca_dims

            for current_bin_count in bin_counts:
                if current_bin_count**current_pca_dims > max_cuboids_count:
                    print(f"Can't compute histogram! Too many bins!")
                    continue

                print(f"Computing histogram for latent space with bin count: {current_bin_count} ...")
                histogram_values, bin_edges = np.histogramdd(
                    reduced_feature_vectors, bins=current_bin_count, range=histogram_range
                )
                filled_cuboids_ratio = np.count_nonzero(histogram_values) / histogram_values.size

                sample_dict = {
                    "pca_dims": current_pca_dims,
                    "bin_count": current_bin_count,
                    "sparsity": filled_cuboids_ratio,
                }

                print(f"Voxelisation Filled Cuboid Ratio: {filled_cuboids_ratio * 100:.2f} %")
                if compute_bin_metrics:
                    median_of_variances, variance_of_variances, median_num_points = _compute_pred_prob_variances(
                        reduced_feature_vectors, all_prediction_probabilities, histogram_values, np.stack(bin_edges)
                    )
                    sample_dict["pred_prob_median_of_variances"] = median_of_variances
                    sample_dict["pred_prob_variance_of_variances"] = variance_of_variances
                    sample_dict["median_num_points"] = median_num_points

                voxelisation_dicts.append(sample_dict)

        except KeyboardInterrupt:
            print(f"Skipping to next pca_dim ...")
            continue

    if log_filepath:
        try:
            with open(log_filepath) as log_fp:
                data = json.load(log_fp)

        except FileNotFoundError:
            data = {}

        field_name = f"Voxelisation_Sparsities_{mode}_{type_}"
        print("Writing results to section", field_name, "...")
        data.update({field_name: voxelisation_dicts})

        with open(log_filepath, "w") as log_fp:
            json.dump(data, log_fp, indent=4)
            print("Results written to log:", log_filepath)


def _compute_pred_prob_variances(
    reduced_feature_vectors: np.ndarray,  # shape (N, pca_dims)
    pred_probabilities: np.ndarray,  # shape (N, )
    hist_values: np.ndarray,  # shape (bin_count, ) * pca_dims
    hist_bin_edges: np.ndarray,  # shape (pca_dims, bin_count + 1)
) -> Tuple[
    float, float, float
]:  # (median of variances of pred probs, variance of variances of pred probs, median num points)
    pca_dims = reduced_feature_vectors.shape[1]

    indices_of_cubes_with_points = np.argwhere(hist_values > 0)

    all_variances = []
    all_num_points = []
    for cube_index in indices_of_cubes_with_points:
        samples_in_cube_mask = np.ones_like(pred_probabilities, dtype=bool)

        for pca_dim in range(pca_dims):
            min_, max_ = hist_bin_edges[pca_dim, cube_index[pca_dim] : cube_index[pca_dim] + 2]
            samples_in_cube_mask &= reduced_feature_vectors[:, pca_dim] > min_
            samples_in_cube_mask &= reduced_feature_vectors[:, pca_dim] < max_

        variance = np.var(pred_probabilities[samples_in_cube_mask])
        num_points = len(pred_probabilities[samples_in_cube_mask])
        all_variances.append(variance)
        all_num_points.append(num_points)

    median_of_variances = np.median(all_variances)
    variance_of_variances = np.var(all_variances)
    median_num_points = np.median(num_points)

    print(
        "Computed bin metrics:",
        f"median of variances: {median_of_variances}",
        f"variance of variances: {variance_of_variances}",
        f"median num points: {median_num_points}",
        sep="\n",
    )
    return median_of_variances, variance_of_variances, median_num_points
