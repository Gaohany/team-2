import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from metrics_utils import utils_metrics


def project_and_visualize(points, radius, plot_mode="2d"):
    center = points.mean(axis=0, keepdims=True)
    points = np.concatenate([points, center], axis=0)
    distances = np.linalg.norm(center - points, axis=1)
    if plot_mode == "2d":
        pca = PCA(n_components=2)
    else:
        pca = PCA(n_components=3)

    transformed_points = pca.fit_transform(points)
    exceed_radius = transformed_points[distances > radius]
    below_radius = transformed_points[distances < radius]

    if plot_mode == "2d":
        plt.rcParams["figure.figsize"] = (10, 10)
        plt.scatter(exceed_radius[:-1, 0], exceed_radius[:-1, 1], c="red", label="exceed radius ({})".format(radius))
        plt.scatter(below_radius[:-1, 0], below_radius[:-1, 1], c="blue", label="below radius ({})".format(radius))
        plt.scatter(transformed_points[-1, 0], transformed_points[-1, 1], c="green", label="center")
        plt.legend()
        plt.show()

    else:
        plt.rcParams["figure.figsize"] = (10, 10)
        ax = plt.axes(projection="3d")
        ax.scatter3D(
            exceed_radius[:-1, 0],
            exceed_radius[:-1, 1],
            exceed_radius[:-1, 2],
            c="red",
            label="exceed radius ({})".format(radius),
        )
        ax.scatter3D(
            below_radius[:-1, 0],
            below_radius[:-1, 1],
            below_radius[:-1, 2],
            c="blue",
            label="below radius ({})".format(radius),
        )
        ax.scatter3D(
            transformed_points[-1, 0], transformed_points[-1, 1], transformed_points[-1, 2], c="green", label="center"
        )
        plt.legend()
        plt.show()


def scree_plot(feature_matrix, n_components):
    pca = PCA(n_components=n_components)
    pca = pca.fit(feature_matrix)
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, "o-", linewidth=2, color="blue")
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.show()


def plot_cumsum_to_n_components(feature_matrix, n_components=32, desired_explained_variance=0.9):
    # Is used to analyse how much components in PCA are needed in order to hold the explained varience of 0.9 and greater
    components = [i + 1 for i in range(n_components)]
    pca = PCA(n_components=n_components)
    pca.fit_transform(feature_matrix)
    cumsum = pca.explained_variance_ratio_.cumsum()
    colors = ["blue" if value > desired_explained_variance else "grey" for value in cumsum]
    plt.figure(figsize=(10, 10))  # You can adjust the figure size as needed
    plt.bar(components, cumsum, color=colors)  # Creates a bar chart
    plt.xlabel("Components")  # Label for the x-axis
    plt.ylabel("Sum")  # Label for the y-axis
    plt.title("Sum of Components")  # Title of the plot
    plt.xticks(rotation=45, ticks=pca.components_)  # Rotates the x-axis labels if they're too long
    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping


def project_and_visualize_confidence_values(
    metrics_config, config, type_, n_components, feature_dict, functional_range, functional_range_customized=False
):
    """
    Discretizes the data points in a given data set to 4 different confidence value levels and projects them in a plot accordingly

    Arguments:

        metrics_config: General configuration for metrics
        config: Configuration for the specified dataset (GTSRB, SVHN, MNIST...)
        type_: Type of the dataset
        n_components: Specifies to how many dimensions the data should be reduced
        feature_dict: Data points from which the plot is made
        functional_range: Functional range which specifies the bounds of the plot
        functional_range_customized: If yes, global functional range will be used, if not, local bounds will be used

    Returns:

        Matplotlib plot of classified data points with respect to their confidence values in the global functional range, if given.
    """
    if n_components > 3:
        raise ValueError(
            "Number of dimensions to be reduced must not be bigger than 3! Please reduce the dimension accordingly (to either 2 or 3 dimensions!)"
        )

    gen_data_array = utils_metrics.load_data(config, feature_dict, type_)
    gen_data_reduced = utils_metrics.reduce_dimensionality(
        gen_data_array, n_components, metrics_config.gini_index.dimensionality_method
    )

    feature_vectors = []
    prediction_probabilities = []
    for element_dict in feature_dict.values():
        feature_vector = element_dict["feature_vector"]
        if len(feature_vector) == 0:
            continue
        feature_vectors.append(feature_vector)
        prediction_probabilities.append(float(element_dict["pred_probs"][0, element_dict["classes"][0]]))

    outliers, not_confident, confident, most_confident = [], [], [], []
    for i in range(len(prediction_probabilities)):
        image_prob = prediction_probabilities[i]
        if image_prob >= 0.2 and image_prob < 0.6:
            outliers.append(gen_data_reduced[i])
        elif image_prob >= 0.6 and image_prob < 0.8:
            confident.append(gen_data_reduced[i])
        elif image_prob >= 0.8 and image_prob <= 1.0:
            most_confident.append(gen_data_reduced[i])
        elif image_prob >= 0.0 and image_prob < 0.2:
            not_confident.append(gen_data_reduced[i])

    np_outlier = np.array(outliers).reshape((len(outliers), n_components))
    np_not_confident = np.array(not_confident).reshape((len(not_confident), n_components))
    np_confident = np.array(confident).reshape((len(confident), n_components))
    np_most_confident = np.array(most_confident).reshape((len(most_confident), n_components))

    if n_components == 2:
        plt.rcParams["figure.figsize"] = (10, 10)
        plt.scatter(np_outlier[:-1, 0], np_outlier[:-1, 1], c="blue", label="confidence value between 0.2-0.6")
        plt.scatter(
            np_not_confident[:-1, 0], np_not_confident[:-1, 1], c="red", label="confidence value between 0.0-0.2"
        )
        plt.scatter(np_confident[-1, 0], np_confident[-1, 1], c="green", label="confidence value between 0.6-0.8")
        plt.scatter(
            np_most_confident[-1, 0], np_most_confident[-1, 1], c="yellow", label="confidence value between 0.8-1.0"
        )
        plt.legend()
        data_name = metrics_config.experiment_path[0].split("\\")[-1].split("_")[-1]
        plt.savefig("visualization_2d_confidence_values_" + data_name + "_" + config.data)
        plt.show()
    else:
        plt.rcParams["figure.figsize"] = (10, 10)
        ax = plt.axes(projection="3d")
        ax.scatter3D(
            np_not_confident[:-1, 0],
            np_not_confident[:-1, 1],
            np_not_confident[:-1, 2],
            c="red",
            label="confidence value between 0.0-0.2",
        )
        ax.scatter3D(
            np_outlier[:-1, 0],
            np_outlier[:-1, 1],
            np_outlier[:-1, 2],
            c="blue",
            label="confidence value between 0.2-0.6",
        )
        ax.scatter3D(
            np_confident[:-1, 0],
            np_confident[:-1, 1],
            np_confident[:-1, 2],
            c="green",
            label="confidence value between 0.6-0.8",
        )
        ax.scatter3D(
            np_most_confident[:-1, 0],
            np_most_confident[:-1, 1],
            np_most_confident[:-1, 2],
            c="yellow",
            label="confidence value between 0.8-1.0",
        )
        if functional_range_customized:
            ax.set_xlim(functional_range[0])
            ax.set_ylim(functional_range[1])
            ax.set_zlim(functional_range[2])

        data_name = metrics_config.experiment_path[0].split("\\")[-1].split("_")[-1]
        data_name_ = "train" if metrics_config.mode == "train" else data_name
        data_ = config.data
        # plt.title(f"{data_.upper()}_{data_name_} Data Set in the Functional Range Colored Based on Confidence Values")
        plt.legend(loc="upper right")
        print(
            f"Count of confidence values between 0.0 and 0.2 = {len(not_confident)}, Count of confidence values between 0.2 and 0.6 = {len(outliers)}, Count of confidence values between 0.6 and 0.8 = {len(confident)}, Count of confidence values between 0.6 and 0.8 = {len(most_confident)}"
        )
        plt.savefig(f"visualization_3d_confidence_values_{data_name_}_{config.data}.pdf")
        plt.show()


def compare_training_with_all_combined_pca_visualization(
    metrics_config,
    pca_all_explained_ratio_results: list,
    dataset_name_list: list,
    save_path,
    threshold_min: float = 0.9,
    threshold_max: float = 0.99,
):
    fig, ax1 = plt.subplots()
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for pca_explained_ratio, color_name, dataset_name in zip(
        pca_all_explained_ratio_results, mcolors.TABLEAU_COLORS.keys(), dataset_name_list
    ):
        ax1.plot(
            metrics_config.measured_dimensionalities, pca_explained_ratio, c=color_name, label=dataset_name, marker="."
        )

    plt.rcParams["figure.figsize"] = (20, 12)
    ax1.set_xlabel("Reduced dimensionality")
    ax1.set_ylabel("PCA cumulative explained variance ratio")
    ax1.tick_params(axis="y")
    ax1.tick_params(axis="x", length=1)
    ax1.axhline(threshold_min, color="red", linestyle="-", label=f"{threshold_min} Cumulative Variance")
    ax1.axhline(threshold_max, color="green", linestyle="--", label=f"{threshold_max} Cumulative Variance")

    ax1.set_xticks(metrics_config.measured_dimensionalities)
    ax1.legend(loc="lower right")

    plt.title(f"PCA Cumulative Explained Variance Ratio Comparison, {dataset_name[0]} and {dataset_name[1]} Dataset")
    fig.tight_layout()
    plt.savefig(f"{save_path}\\pca_explained_variance_ratio_{dataset_name[0]}_vs_{dataset_name[1]}_.png")

    print(f"Saved successfully as 'pca_explained_variance_ratio_{dataset_name[0]}_vs_{dataset_name[1]}_.png'")
    plt.show()


if __name__ == "__main__":
    N = 1000
    radius = 11.5
    points = np.random.randn(N - 1, 6)
    points_shifted = np.random.randn(N - 1, 6) + 10
    points = np.concatenate([points, points_shifted], axis=0)

    project_and_visualize(points, radius, plot_mode="2d")

    project_and_visualize(points, radius, plot_mode="3d")
