import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

FIELD_NAME_TO_DISPLAY_NAME = {
    "pred_prob_median_of_variances": "Median of Pred. Prob. Variances",
    "pred_prob_variance_of_variances": "Variance of Pred. Prob. Variances",
    "median_num_points": "Median number of Points",
}

LEGEND_LABEL_MAP = {"test_crashes": "corner cases", "train_org": "train", "test_org": "test"}


def visualize_voxelisation(
    data: Dict[str, Any],
    pca_dims: int = None,
    output_dir: Path = None,
    title_suffix: str = None,
    plot_bin_metric: Optional[str] = None,
    fix_y_lim: Optional[float] = None,
    fix_y2_lim: Optional[float] = None,
    vertical_line: Optional[float] = None,
):
    """Create plot of sparsity measure over bin count, grouped by PCA dims count.

    Args:
        data: log.json contents
        output_dir: path where to save the plot
        title_suffix: string to add to end of plot title
        plot_bin_metric: whether to plot the given bin metric on a separate y-s
    """
    experiments = {key: value for key, value in data.items() if key.startswith("Voxelisation")}

    if not pca_dims:
        unique_pca_dims = set(
            [sample["pca_dims"] for experiment_data in experiments.values() for sample in experiment_data]
        )
        if len(unique_pca_dims) == 1:
            pca_dims = unique_pca_dims[0]
        else:
            raise Exception(f"Found more than one pca_dim in data. Please provide one! Found: {unique_pca_dims}")

    _, ratio_ax = plt.subplots(1, 1, figsize=(10, 6))

    if plot_bin_metric:
        bin_metric_ax = ratio_ax.twinx()
        bin_metric_ax.set_yscale("log")

    for experiment_name, experiment_data, color_name in zip(
        experiments.keys(), experiments.values(), mcolors.TABLEAU_COLORS.keys()
    ):
        bin_counts, sparsities = [], []

        if plot_bin_metric:
            bin_metrics = []

        for data_dict in experiment_data:
            if data_dict["pca_dims"] == pca_dims:
                bin_counts.append(data_dict["bin_count"])
                sparsities.append(data_dict["sparsity"] * 100)

                if plot_bin_metric:
                    bin_metrics.append(data_dict[plot_bin_metric])

        legend_label = LEGEND_LABEL_MAP[experiment_name.replace("Voxelisation_Sparsities_", "")]
        ratio_ax.plot(
            bin_counts,
            sparsities,
            c=color_name,
            label=legend_label,
            marker="+",
        )

        if plot_bin_metric:
            bin_metric_ax.scatter(
                x=bin_counts,
                y=bin_metrics,
                c=color_name,
                label=legend_label,
                marker="x",
            )

    ratio_ax.set_xlabel("Number of bins per dimension")
    ratio_ax.set_ylabel("Filled Cuboid Ratio, percent")
    current_ylim = ratio_ax.get_ylim()
    top_ylim = fix_y_lim if fix_y_lim else current_ylim[1]
    ratio_ax.set_ylim(bottom=0, top=top_ylim)
    current_xlim = ratio_ax.get_xlim()
    ratio_ax.set_xlim(left=0, right=current_xlim[1])
    if plot_bin_metric:
        bin_metric_ax.set_ylabel(FIELD_NAME_TO_DISPLAY_NAME[plot_bin_metric])
        bin_metric_ax.legend(loc="right", title=FIELD_NAME_TO_DISPLAY_NAME[plot_bin_metric])
        current_y2lim = bin_metric_ax.get_ylim()
        top_y2lim = fix_y2_lim if fix_y2_lim else current_y2lim[1]
        bin_metric_ax.set_ylim(bottom=0, top=top_y2lim * 10)

    if vertical_line:
        ratio_ax.axvline(vertical_line, color="lightcoral", linestyle="--")

    ratio_ax.set_title(
        f"Filled Cuboid Ratio{f', {FIELD_NAME_TO_DISPLAY_NAME[plot_bin_metric]}' if plot_bin_metric else ''} vs. Bin Count, {pca_dims} PCA dims{f' - {title_suffix}' if title_suffix else ''}"
    )
    ratio_ax.grid()
    ratio_ax.set_axisbelow(True)
    ratio_ax.legend(loc="upper right", title="Filled Cuboid Ratio")

    output_dir.mkdir(exist_ok=True)
    output_path = (
        output_dir
        / f"voxelisation_{pca_dims}_pca_dims_filled_cuboid{f'_{plot_bin_metric}' if plot_bin_metric else ''}_vs_bin_count.png"
    )
    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("visualize voxelisation sparsity based on log.json")
    parser.add_argument("log_filepath")
    parser.add_argument("--title-suffix", help="Extra text to add to plot title")
    parser.add_argument("--pca-dims", help="Choose which PCA dims count to plot, if multiple are present.", type=int)
    parser.add_argument("--fix-y-lim", type=float, help="Set fixed y-axis upper bound")
    parser.add_argument("--fix-y2-lim", type=float, help="Set fixed second y-axis upper bound")
    parser.add_argument(
        "--plot-bin-metric",
        help="If passed, plot the specified bin metric (e.g. pred_prob_median_of_variances)",
    )
    parser.add_argument("--vertical-line", type=float, help="Plot vertical line at given x position")
    parser.add_argument("-o", "--output-dir", default=".")
    args = parser.parse_args()

    with open(args.log_filepath) as log_fp:
        log_data = json.load(log_fp)

    visualize_voxelisation(
        log_data,
        output_dir=Path(args.output_dir),
        plot_bin_metric=args.plot_bin_metric,
        pca_dims=args.pca_dims,
        title_suffix=args.title_suffix,
        fix_y_lim=args.fix_y_lim,
        fix_y2_lim=args.fix_y2_lim,
        vertical_line=args.vertical_line,
    )
