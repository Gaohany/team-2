import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metrics_utils.utils_metrics as utils_metrics


def visualize_gini_index(
    metrics_config,
    config,
    gini_index_calculations_list: list[list],
    dataset_name_list: list,
    save_path,
    optimal_bin_count: int = 0,
):
    """Visualizes the gini index calculations in a list with respect to the number of bins per dimension parameter

    Args:
        config: Configuration parameter for the metric, used for retrieving the number of bins array we have been using for calculating the gini index algorithm
        gini_index_calculations_list: the list in which the Gini Index calculations are stored

    Returns:
        Matplotlib plot of Gini Index calculations w.r.t the number of bins per dimension they have been calculated with
    """

    fig, ax1 = plt.subplots()
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for gini_index_calculation, color_name, dataset_name in zip(
        gini_index_calculations_list, mcolors.TABLEAU_COLORS.keys(), dataset_name_list
    ):
        ax1.plot(
            metrics_config.gini_index.number_bins, gini_index_calculation, c=color_name, label=dataset_name, marker="+"
        )

    plt.rcParams["figure.figsize"] = (12, 12)
    ax1.set_xlabel("Number of bins per dimension")
    ax1.set_ylabel("Gini Index Algorithm Results")
    ax1.tick_params(axis="y")
    ax1.tick_params(axis="x", length=1)

    ax1.set_xticks(metrics_config.gini_index.number_bins)
    ax1.legend(loc="lower right")
    if optimal_bin_count != 0: # for plotting optimal bin count with gini_index
        plt.axvline(x=optimal_bin_count, color="purple")

    plt.title(f"Gini Index vs. Bin Count, {metrics_config.gini_index.dimensionality_method} {metrics_config.gini_index.number_dimensionality} dimensions, {config.data} data set")
    fig.tight_layout()
    joined_dataset_name = "_".join(dataset_name_list)
    # pdf is used as default so that the image quality could be kept
    plt.savefig(f"{save_path}\\{config.data}_{joined_dataset_name}_gini_index_results.pdf")

    print(f"Plot saved successfully under {save_path} as '{config.data}_{joined_dataset_name}.pdf'")
    plt.show()

def visualize_results_as_table(
    metrics_config, config, gini_index_calculations_list: list[list], dataset_name: list, save_path
):
    """
    Creates a table based on the Gini Index calculations

    Arguments:

        metrics_config: General configuration for metrics
        config: Configuration for the specified dataset (GTSRB, SVHN, MNIST...) 
        gini_index_calculations_list: Gini Index results as a list
        dataset_name: The names of data sets that have been calculated
        save_path: The directory in which the table is going to be saved as a .png
    
    Returns:
        Matplotlib plot of tabloid version of Gini Index values
    """
    if len(metrics_config.gini_index.number_bins) != len(gini_index_calculations_list[0]):
        raise ValueError(
            "The dimensions of the number of bins and calculations don't correlate with each other! Please create a new vector with the given parameter in the config!"
        )
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    data = {
        "Number of Bins": metrics_config.gini_index.number_bins,
    }
    data.update({dataset_name[i]: gini_index_calculations_list[i] for i in range(len(gini_index_calculations_list))})
    data_multiple_datasets_as_table = pd.DataFrame(data)

    plt.table(
        cellText=data_multiple_datasets_as_table.values,
        colLabels=data_multiple_datasets_as_table.columns,
        cellLoc="center",
        loc="center",
    )

    plt.axis("off")
    joined_datasets = "_".join(dataset_name)
    plt.savefig(
        f"{save_path}\\{config.data}_{joined_datasets}_table.png", bbox_inches="tight", pad_inches=0.05
    )
    print(f"Table saved successfully under {save_path} as '{config.data}_{joined_datasets}_table.png'")