import time

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

import metrics_utils.features_from_input as features_from_input
import metrics_utils.utils_metrics as utils_metrics


def calculate_gini_index_with_pca_without_histogramdd(
    config,
    data_path,
    number_bins_per_dimension: int,
    log_file,
    type_dataset: str,
    num_dimensionality: int = 3,
    use_traditional_gini_index_method: bool = True,
) -> float:
    """Calculates the Gini Index in the reduced dimensionality by not using histogramdd()
    Args:
        config: Configuration parameter of a certain dataset (GTSRB, MNIST, SVHN...)
        data_path:
        number_bins_per_dimension: List of number of bins per dimension
        log_file: File path for logging the gini index results
        type_dataset: Type of the dataset (train, val, test)
        num_dimensionality: Number of dimensions to reduce the dimensionality of data points. If Gini Index will be calculated class-based, this parameter is not necessary to fill in
        dim_type: Dimensionality reduction method to apply to the data points. Supported methods are "pca", "tsne", "lle" and "umap"
        use_traditional_gini_index_method: If the traditional method used in Gini Index, i.e. the method used in decision tree algorithms, this should be set in default

    Returns:
        Result of Gini Index
    """
    if number_bins_per_dimension <= 0:
        raise ValueError("Please give valid number of bins!")

    data_as_array = utils_metrics.load_data(config, data_path, type_dataset)
    functional_range = features_from_input.calculate_functional_range(config, type_dataset, num_dimensionality)
    gen_data_pca = utils_metrics.reduce_dimensionality(data_as_array, num_dimensions=num_dimensionality)
    all_points_bin = count_points_each_bin(
        number_bins_per_dimension, num_dimensionality, gen_data_pca, functional_range
    )

    # assigning each point into the created cubes
    point_indices_flat = []
    tuple_ = tuple(number_bins_per_dimension for _ in range(num_dimensionality))

    for point_indices in all_points_bin:
        point_index = np.ravel_multi_index(point_indices, tuple_)
        point_indices_flat.append(point_index)

    # counting occurrence per cube
    np_array = np.array(point_indices_flat)
    count = []
    for j in range(1, (number_bins_per_dimension**num_dimensionality)):
        count_per_number = np.count_nonzero(np_array == j - 1)
        count.append(count_per_number)

    gini_sum = 0
    if use_traditional_gini_index_method:
        for i in range(len(count)):
            gini_sum += (count[i] / len(data_as_array)) ** 2
        gini_index = 1 - gini_sum

    else:
        for i in range(len(count)):
            for j in range(len(count)):
                if j == i:
                    continue
            gini_sum += np.abs(count[i] - count[j])
        gini_index = gini_sum / (2 * len(count) * len(data_as_array))

    utils_metrics.log_gini_index(
        log_file,
        config,
        type_dataset,
        gini_index,
        "Pca_Without_Histogram_"
        + str(num_dimensionality)
        + "_"
        + ("traditional" if use_traditional_gini_index_method else "adapted"),
    )
    return gini_index


def count_points_each_bin(number_bins, num_classes, data_as_array, functional_range: list):
    # creating equal distributed edges for each dimension
    edges = np.zeros((num_classes, number_bins))
    for dim in range(num_classes):
        min = np.min(functional_range[dim][0])
        max = np.max(functional_range[dim][1])
        edges[dim, :] = np.linspace(min, max, number_bins)

    # assigning the data into equally distributed bins by using np.digitize

    all_points_bin = []
    for i in range(len(data_as_array)):
        bin_indices = []
        for dim in range(num_classes):
            bin_index = np.digitize(data_as_array[i, dim], edges[dim, :])
            bin_indices.append(bin_index - 1)
        all_points_bin.append(bin_indices)

    return all_points_bin


def count_per_class(all_points_bin, number_bins):
    """Bin counts on every class are calculated"""
    all_counts_bin = []
    transposed_counts_for_feature = np.transpose(all_points_bin)
    for i in range(len(transposed_counts_for_feature)):
        counts = []
        for j in range(1, number_bins + 1):
            count_bin = {j: np.count_nonzero(transposed_counts_for_feature[i] == j)}
            counts.append(count_bin)
        counts_per_feature = {i: counts}
        all_counts_bin.append(counts_per_feature)

    return all_counts_bin


def calculate_kde(bandwidth_range: int, data_as_array: np.ndarray) -> list:
    """Calculates the Kernel Density Estimation of a given array
    Args:
        bandwidth_range: Wanted kernel bandwidth range to look at so that an optimal bandwidth can be obtained for the given array
        data_as_array: Numpy array to make the KDE of each data point in that array
    Returns:
        Kernel Density Estimation of each data point in the multi dimensional space
    """
    start_time = time.time()
    # estimate the optimal size, with that size make the kernel density dist
    optimal_size = estimate_density_kde(bandwidth_range, data_as_array)
    kde = KernelDensity(bandwidth=optimal_size, kernel="gaussian")
    kde.fit(data_as_array)

    overall_densities = kde.score_samples(data_as_array)
    end_time = time.time()
    print(f"KDE for the given dataset is done! Time passed is {end_time - start_time} seconds!")

    return overall_densities


def estimate_density_kde(bandwidth_range_to_find_optimal: int, data_as_array: np.ndarray):
    """Calculates the most optimal bandwidth parameter for the given Numpy array

    Parameters:
    bandwidth_range_to_find_optimal: int --> Optimal range to find the bandwidth between 10 and 1/10
    data_as_array: np.ndarray

    Returns:
    optimal bandwidth
    """

    bandwidths = 10 ** np.linspace(-1, 1, bandwidth_range_to_find_optimal)
    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=10
    )  # since as default KFold is used we don't need to explicitly define it

    grid.fit(data_as_array)

    best_params_ = grid.best_params_

    return best_params_.get("bandwidth")