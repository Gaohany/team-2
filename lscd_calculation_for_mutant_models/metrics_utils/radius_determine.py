import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from kneed import KneeLocator

sys.path.append("/home/vekariya/Documents/practicum_dl_testing_ws_22_23/testing_framework_fortiss_classifiers")

import numpy as np

import metrics_utils.utils_metrics as utils_metrics
from metrics_utils.plot_radius import *

_DEGREE_POLYFIT = 4


def _radius_at_percentage_dataset(distances: list, threshold_p: float) -> float:
    """Finds a cluster radius that separates the central points from outliers.

    The method assumes that p% of the closest points in the train dataset
    are central. A radius in accord with the assumption is identified. The
    p% is passed as a constructor argument.

    Returns:
        The threshold radius that separates central points from outliers.
    """
    # Define the polynomials that describe the change and rate of change
    # in distance from the centroid
    # We use a high degree for the polynomial to improve the fit. We are
    # not extrapolating from it, so over-fitting is not an issue.
    x, y = list(range(len(distances))), distances
    poly = np.polyfit(x, y, deg=_DEGREE_POLYFIT)
    x_max = max(x)
    x_threshold = threshold_p * x_max
    picked_radius = float(np.polyval(poly, x_threshold))

    return x_threshold, picked_radius


def _find_radius_elbow(distances: list, plot_flag: bool = False) -> float:
    """Finds a cluster radius to separate central points from outliers.

    The method takes all ordered distances from the centroid and fits
    a polynomial over the distribution. The derivative of this polynomial
    describes the rate at which the distance from the centroid increases.

    The algorithm then identifies the interval for which the average rate
    of change is the most pronounced. This shift in the distance distribution
    is interpreted as a departure from the central points of the cluster
    towards the outliers. As there is no theoretical upper bound on
    the distance between a centroid and an outlier, a discount factor is
    included to bias towards earlier intervals.

    Note: Since we are working on a 1D distribution, the X-axis does not
    carry any inherent meaning.

    Arguments:
        distances: An ordered array of distance from centroid
        plot_flag: Plot the point distribution, fitted polynomial,
            derivative and chosen radius for publishing or debug
            purposes
    Returns:
        The threshold radius that separates central points from outliers.
    """
    # Define the polynomials that describe the change and rate of change
    # in distance from the centroid
    # We use a high degree for the polynomial to improve the fit. We are
    # not extrapolating from it, so over-fitting is not an issue.
    x, y = list(range(len(distances))), distances
    poly = np.poly1d(np.polyfit(x, y, deg=_DEGREE_POLYFIT))
    poly_derivative = np.polyder(poly)
    # Algorithm set-up: constants and division of X-axis in intervals
    NUM_SPLITS = 12
    x_min, x_max = min(x), max(x)
    smooth_x = np.linspace(x_min, x_max)
    x_splits = np.array_split(smooth_x, NUM_SPLITS)
    # Find the interval with the largest average rate of change
    best_idx, best_val = None, float("-inf")
    DISCOUNT_FACTOR = 0.5  # Bias towards earlier intervals
    for idx, split in enumerate(x_splits):  # Slicing from 30% to avoid early perturbations
        if idx < 5:
            continue
        derivative_val = np.polyval(poly_derivative, split)
        # derivative_val = poly_derivative
        # +3 to compensate for the slicing above; computes
        avg_rate_change = (DISCOUNT_FACTOR**idx) * np.average(derivative_val)
        if avg_rate_change > best_val:
            best_val = avg_rate_change
            best_idx = idx
    # Obtain the radius values and its correspondent coordinate on X
    picked_x_interval = x_splits[best_idx - 1]
    candidates_r = np.polyval(poly, picked_x_interval)

    picked_radius_idx = np.argmin(candidates_r)
    picked_radius = candidates_r[picked_radius_idx]
    picked_radius_x_val = picked_x_interval[picked_radius_idx]

    # picked_radius_idx = int(np.min(picked_x_interval))
    # picked_radius = distances[picked_radius_idx]
    # picked_radius_x_val = picked_radius_idx
    print(picked_radius_x_val, picked_radius)
    return picked_radius_x_val, picked_radius, poly


def knee_point(distances: list, cl=None) -> float:
    x, y = list(range(len(distances))), distances
    if cl == 6 or cl == 14 or cl == 17 or cl == 39 or cl == 42:
        kneedle = KneeLocator(
            x, y, S=1, curve="convex", direction="increasing", interp_method="polynomial", polynomial_degree=2
        )
    else:
        kneedle = KneeLocator(
            x,
            y,
            S=1,
            curve="convex",
            direction="increasing",
            interp_method="polynomial",
            polynomial_degree=_DEGREE_POLYFIT,
        )

    x_threshold_elbow = kneedle.elbow
    selected_radius_elbow = kneedle.elbow_y.astype(np.float64)
    poly = [kneedle.x, kneedle.Ds_y]
    # kneedle.plot_knee_normalized()
    print(x_threshold_elbow, selected_radius_elbow)
    return x_threshold_elbow, selected_radius_elbow, poly


centroids_save_path = "saved_parameters/init_centroids/gtsrb_new.pkl"
log_file = "results/gtsrb_new/gtsrb-new.json"
output_dir = "radius_plot/"
num_classes = 43

all_centroids = torch.load(centroids_save_path)
centroids = all_centroids["all_centroids"]
radius_selected = all_centroids["all_radius"]
radius_values = all_centroids["all_distances_train_data"]
# confidence_values = all_centroids['all_tp_conf_train_data']
new_radius = utils_metrics.create_features_dict_merged(num_classes=num_classes)

for cl in range(num_classes):
    try:
        radius_ip = sorted(radius_values[cl])
        # radius_ip, conf_ip = zip(*sorted(zip(radius_ip, conf_ip)))
        x_threshold, selected_radius = _radius_at_percentage_dataset(radius_ip, 0.7)
        # x_threshold_elbow, selected_radius_elbow, poly = _find_radius_elbow(radius_ip)
        x_threshold_elbow, selected_radius_elbow, poly = knee_point(radius_ip, cl)
        output_path = "gtsrb_" + str(cl) + ".png"
        output_path = os.path.join(output_dir, output_path)
        plot_radius_values(
            radius_ip, x_threshold, selected_radius, x_threshold_elbow, selected_radius_elbow, poly, output_path, i=cl
        )
        new_radius[cl] = str(round(selected_radius_elbow, 3))
    except:
        new_radius[cl] = str(0)


all_centroids.update({"all_radius_automatic": new_radius})

torch.save(all_centroids, centroids_save_path)

if os.path.exists(log_file):
    with open(log_file) as json_file:
        data = json.load(json_file)
        data.update({"Automatic Radius Threshold Values:": all_centroids["all_radius_automatic"]})
else:
    data = {"Automatic Radius Threshold Values:": all_centroids["all_radius_automatic"]}

with open(log_file, "w") as json_file:
    json.dump(data, json_file, indent=4)
    print("Results written to:", log_file)

# with open(log_file, 'w') as json_file:
#     json.dump(data, json_file, indent=4)
#     print('Results written to:', log_file)
