import json
import os
import pickle
import time

import numpy as np
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from metrics_utils.metrics_dataloader import *

# from metrics_utils.visualization import *

fw = None


def create_dict_from_array(x1, x2):
    counter = 0
    dictionary = {}
    for i in range(x2.size()[0]):
        if x2[i]:
            dictionary[counter] = i
            counter += 1

    return dictionary


def create_dict_from_indices(x1, x2):
    dictionary = {}
    for i in range(len(x2)):
        dictionary[i] = x2[i]

    return dictionary


def get_positions(idx):
    positions = []
    if fw == "loc":
        num = 4
    else:
        num = 2
    for i in range(num):
        positions.append((idx, i))
    return positions


def get_pos_in_loc(positions):
    pos_in_loc = []
    for pos in positions:
        if fw == "loc":
            pos_loc = np.unravel_index(np.ravel_multi_index(pos, (8732, 4)), (1, 34928))
        else:
            pos_loc = np.unravel_index(np.ravel_multi_index(pos, (8732, 2)), (1, 17464))
        pos_in_loc.append(pos_loc)
    return pos_in_loc


def get_pos_in_bucket(pos_in_loc):
    idx = pos_in_loc[0][1]

    if fw == "loc":
        sizes = [23103, 8664, 2400, 600, 144, 16]
        shapes = [(1, 23104), (1, 8664), (1, 2400), (1, 600), (1, 144), (1, 16)]
        sizes2 = [23104, 8664, 2400, 600, 144, 16]
        loc_shapes = [(1, 38, 38, 16), (1, 19, 19, 24), (1, 10, 10, 24), (1, 5, 5, 24), (1, 3, 3, 16), (1, 1, 1, 16)]
    else:
        sizes = [11551, 4332, 1200, 300, 72, 8]
        shapes = [(1, 11552), (1, 4332), (1, 1200), (1, 300), (1, 72), (1, 8)]
        sizes2 = [11552, 4332, 1200, 300, 72, 8]
        loc_shapes = [(1, 38, 38, 8), (1, 19, 19, 12), (1, 10, 10, 12), (1, 5, 5, 12), (1, 3, 3, 8), (1, 1, 1, 8)]

    for i in range(1, len(sizes) + 1):
        if idx <= sum(sizes[0:i]):
            which_bucket = i - 1
            break
    # print('which_bucket: ', which_bucket)

    positions_in_bucket = []
    # print(sum(sizes2[0:which_bucket-1]))
    for pos in pos_in_loc:
        if which_bucket != 0:
            pos_bucket = (0, pos[1] - sum(sizes2[0:which_bucket]))
        else:
            pos_bucket = (0, pos[1])
        positions_in_bucket.append(pos_bucket)
    return which_bucket, positions_in_bucket


def get_pos_after_head(which_bucket, positions_in_bucket):
    if fw == "loc":
        shapes = [(1, 23104), (1, 8664), (1, 2400), (1, 600), (1, 144), (1, 16)]
        after_head_shapes = [
            (1, 38, 38, 16),
            (1, 19, 19, 24),
            (1, 10, 10, 24),
            (1, 5, 5, 24),
            (1, 3, 3, 16),
            (1, 1, 1, 16),
        ]
    else:
        shapes = [(1, 11552), (1, 4332), (1, 1200), (1, 300), (1, 72), (1, 8)]
        after_head_shapes = [
            (1, 38, 38, 8),
            (1, 19, 19, 12),
            (1, 10, 10, 12),
            (1, 5, 5, 12),
            (1, 3, 3, 8),
            (1, 1, 1, 8),
        ]

    positions_after_head = []
    for pos in positions_in_bucket:
        pos_af_head = np.unravel_index(
            np.ravel_multi_index(pos, shapes[which_bucket]), after_head_shapes[which_bucket]
        )
        positions_after_head.append(pos_af_head)

    return which_bucket, positions_after_head


def calculate_mean_of_list_of_arrays(mids, lengths):
    mids_mask = np.ma.empty((len(mids), max(lengths)))
    mids_mask.mask = True
    for i in range(len(mids)):
        mid = mids[i]
        mids_mask[i][0 : len(mid)] = mid

    centroid = mids_mask.mean(axis=0)
    return mids_mask, centroid.data


def calculate_mean_of_list_of_arrays_in_buckets(all_mids, max_lengths):
    all_features = create_features_dict(num_classes=1)
    all_centroids = create_features_dict(num_classes=1)
    for b in range(6):
        if all_mids[0][b]:
            # mids = all_mids[0][b]
            # lengths = max_lengths[0][b]
            # mids_mask = np.ma.empty((len(mids), max(lengths)))
            # mids_mask.mask = True
            # for i in range(len(mids)):
            #     mid = mids[i]
            #     mids_mask[i][0:len(mid)] = mid

            # all_features[0][b] = mids_mask

            # centroid = mids_mask.mean(axis=0)
            # all_centroids[0][b] = centroid.data
            all_centroids[0][b] = avgNestedLists(all_mids[0][b])

    # return mids_mask, centroid.data
    return all_features, all_centroids


def calculate_mean_in_buckets(all_mids):
    centroids_per_bucket = {}
    for b in range(6):
        if all_mids[0][b]:
            features_bucket = torch.stack(all_mids[0][b])

            all_centroids.append(features_bucket)


def avgNestedLists(nested_vals):
    """
    Averages a 2-D array and returns a 1-D array of all of the columns
    averaged together, regardless of their dimensions.
    """

    output = []
    maximum = 0
    for lst in nested_vals:
        if len(lst) > maximum:
            maximum = len(lst)
    for index in range(maximum):  # Go through each index of longest list
        temp = []
        for lst in nested_vals:  # Go through each list
            if index < len(lst):  # If not an index error
                temp.append(lst[index])
        output.append(np.mean(np.asarray(temp)))
    return np.asarray(output)


def create_smallest_largest_object_dict(num_classes):
    dictionary = {}
    for i in range(num_classes):
        buckets_dict = {}
        for j in range(6):
            buckets_dict[j] = {"min_height": 3000, "min_width": 3000, "max_height": 0, "max_width": 0}

        dictionary[i] = buckets_dict

    return dictionary


def create_features_dict(num_classes):
    all_features_dict = {}

    for i in range(num_classes):
        buckets_dict = {}
        for j in range(6):
            buckets_dict[j] = []

        all_features_dict[i] = buckets_dict

    return all_features_dict


def create_features_dict_merged(num_classes):
    all_features_dict = {}

    for i in range(num_classes):
        buckets_list = []
        all_features_dict[i] = buckets_list

    return all_features_dict


def create_num_samples_dict(num_classes):
    all_features_dict = {}

    for i in range(1, num_classes):
        buckets_dict = {}
        for j in range(6):
            buckets_dict[j] = 0

        all_features_dict[i] = buckets_dict

    return all_features_dict


def create_num_samples_dict_merged(num_classes):
    all_features_dict = {}

    for i in range(num_classes):
        all_features_dict[i] = 0

    return all_features_dict


def create_features_dict_no_list(num_classes):
    all_features_dict = {}

    for i in range(num_classes):
        buckets_dict = {}
        for j in range(6):
            buckets_dict[j] = {}

        all_features_dict[i] = buckets_dict

    return all_features_dict


def create_centroid_positioning_structure(num_classes):
    all_features_dict = {}

    for i in range(num_classes):
        buckets_dict = {}
        for j in range(6):
            buckets_dict[j] = 0

        all_features_dict[i] = buckets_dict

    return all_features_dict


def create_centroid_positioning_structure_merged(num_classes):
    all_features_dict = {}

    for i in range(num_classes):
        all_features_dict[i] = 0

    return all_features_dict


def create_equivalent_partitioning_structure(num_classes):
    all_dict = {}

    for i in range(2):
        list = ["samples_per_class", "ep_metrics"]
        sub_dict = {}
        for j in range(num_classes):
            sub_dict[j] = 0

        all_dict[list[i]] = sub_dict

    return all_dict


def add_diff_length_arrays(a, b):
    if len(a) < len(b):
        c = b.copy()
        c[: len(a)] += a
    else:
        c = a.copy()
        c[: len(b)] += b
    return c


def eucledian_dist(a, b):
    ab = a - b
    dist = np.linalg.norm(ab.data)
    return dist


def add_vectors_by_masking(a, b):
    max_length = max(len(a), len(b))
    a_masked = np.ma.empty(max_length)
    a_masked.mask = True
    a_masked[0 : len(a)] = a

    b_masked = np.ma.empty(max_length)
    b_masked.mask = True
    b_masked[0 : len(b)] = b

    print(len(a_masked))
    print(len(b_masked))
    a_plus_b = a_masked + b_masked
    return a_plus_b


def calculate_mean_from_summed_vector(vector, max_lengths, samples_per_dimension, cl, b):
    # dicti = {'0': 5.0}
    # my_dict = samples_per_dimension[0][b]

    # key_max = max(my_dict.keys(), key=(lambda k: my_dict[k]))
    # key_min = min(my_dict.keys(), key=(lambda k: my_dict[k]))

    # print('Maximum Value: ',my_dict[key_max])
    # print('Minimum Value: ',my_dict[key_min])

    for dim in range(0, len(vector)):
        # num_samples = sum(i >= (dim+1) for i in max_lengths)
        # if num_samples < len(max_lengths):
        #     print('Smaller')
        vector[dim] = vector[dim] / samples_per_dimension[cl][b][dim]
    return vector


def load_data(config, generated_data_path, type_dataset):
    """Loads the data coming in the feature vector dictionary to a 2d Numpy array

    Args:
        config: Configuration parameters of the dataset
        generated_data_path: Feature vector dictionary to load the data from
        type_dataset: type of the dataset

    Returns:
        2d Numpy array
    """
    print("Loading the data from the {} dataset.".format(type_dataset))

    num_classes = config.num_classes

    img_ids = generated_data_path.keys()
    list_image_ids = list(img_ids)
    samples_ignored = 0
    data_as_array = np.zeros((len(list_image_ids), num_classes))
    with torch.no_grad():
        for key in tqdm(img_ids):
            feature_vector_img = generated_data_path[key]["feature_vector"]
            key_index = list_image_ids.index(key)
            if not feature_vector_img.size == 0:
                for i in range(num_classes):
                    data_as_array[key_index, i] = feature_vector_img[i]
            else:
                samples_ignored += 1
                pass

    print("Samples ignored {} out of total {} objects.".format(samples_ignored, len(img_ids)))
    return data_as_array


def log_gini_index(log_file, config, type_dataset, gini_index, gini_index_type):
    """Logging for Gini Index algorithms"""

    try:
        with open(log_file) as json_file:
            data = json.load(json_file)
            key = "Gini_Index_" + gini_index_type + "_" + str(config.mode) + "_" + str(type_dataset)
            data.update({key: gini_index})
    except FileNotFoundError:
        key = "Gini_Index_" + gini_index_type + "_" + str(config.mode) + "_" + str(type_dataset)
        data = {key: gini_index}

    with open(log_file, "w") as json_file:
        json.dump(data, json_file, indent=4)
        print("Results written to:", log_file)

def fit_model_for_reduction(metrics_config, config, method: str = "pca"):
    pca_all_dataset_pickle = f"pca_pickles/pca_all_dataset_{config.data}_{str(metrics_config.gini_index.number_dimensionality)}.pkl"
    umap_all_dataset_pickle = f"umap_pickles/umap_all_dataset_{config.data}_{str(metrics_config.gini_index.number_dimensionality)}.pkl"
    reduce_dim_method = None
    if method == "pca":
        
        try:
            with open(pca_all_dataset_pickle, "rb") as pca_all:
                return pickle.load(
                    pca_all
                )
        except FileNotFoundError:
            pass

    elif method == "umap":
        try:
            with open(umap_all_dataset_pickle, "rb") as umap_all:
                return pickle.load(
                    umap_all
                )
        except FileNotFoundError:
            pass     
    try:
        with open(f"{config.data}/combined_vector/combined_{config.data}.pkl", "rb") as vector_collected_dataset:
            feature_dict_all = pickle.load(vector_collected_dataset)

            data_for_dim_reduce = []
            for i in range(len(feature_dict_all)):
                array_for_dataset = load_data(
                    config, feature_dict_all[i], metrics_config.type
                )
                data_for_dim_reduce.append(array_for_dataset)
            array_1d = [element for sublist in data_for_dim_reduce for element in sublist]
            np_array_ = np.array(array_1d)

    except FileNotFoundError:
        raise FileNotFoundError(f"Please consider collecting all the feature vectors from {config.data} before going on with dimensionality reduction!")

    if method == "pca":
        os.makedirs("pca_pickles", exist_ok=True)
        if reduce_dim_method is None:
            reduce_dim_method = PCA(n_components=metrics_config.gini_index.number_dimensionality)
            reduce_dim_method.fit(np_array_)

            with open(pca_all_dataset_pickle, "wb") as write_path_pca:
                pickle.dump(reduce_dim_method, write_path_pca)
        
    elif method == "umap":
        os.makedirs("umap_pickles", exist_ok=True)
        
        if reduce_dim_method is None:
            reduce_dim_method = umap.UMAP(n_components=metrics_config.gini_index.number_dimensionality)
            reduce_dim_method.fit(np_array_)

            with open(umap_all_dataset_pickle, "wb") as write_path_umap:
                pickle.dump(reduce_dim_method, write_path_umap)

    else:
        raise ValueError(
            f"The dimensionality reduction method {method} that you have provided is not supported! Please put one of these dimensionality reduction methods: pca, umap"
        )
    
    return reduce_dim_method

def reduce_dimensionality(metrics_config, config, data_as_array, method: str = "pca"):
    """Reduces the dimensionality to the given number of dimensions with the supported methods

    Args:
        metrics_config: Metrics configuration in general
        config: Configuration specifically for each dataset (GTSRB, MNIST, SVHN)
        data_as_array: data points to be dimensionality reduced
        method: dimensionality reduction method to use, available methods are pca and umap

    Returns:
        reduced_data: dimensionality reduced data points
    """
    time_start = time.time()
    reduce_dim_method = fit_model_for_reduction(metrics_config, config, method)

    dim_reduced_data = reduce_dim_method.transform(data_as_array)
    print(f"Dimensionality reduction done! Time elapsed: {time.time() - time_start} seconds")
    return dim_reduced_data
