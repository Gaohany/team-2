import json
import os
import sys
from math import sqrt

import numpy as np
from numpy import mean, var
from scipy.stats import wilcoxon

sys.path.append("/home/vekariya/Documents/practicum_dl_testing/testing_framework_fortiss_classifiers")


def cohend(d1, d2):
    """
    function to calculate Cohen's d for independent samples
    """
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    d = (u1 - u2) / s
    d = abs(d)

    result = ""
    if d < 0.2:
        result = "negligible"
    if 0.2 <= d < 0.5:
        result = "small"
    if 0.5 <= d < 0.8:
        result = "medium"
    if d >= 0.8:
        result = "large"

    return result, d


def run_wilcoxon_and_cohend(data1, data2):
    w_statistic, pvalue = wilcoxon(data1, data2, mode="exact")
    cohensd = cohend(data1, data2)
    print(f"P-Value is: {pvalue}")
    print(f"Cohen's D is: {cohensd}")

    return pvalue, cohensd[0]


if __name__ == "__main__":
    avg_dist_dic = {}
    comb_data = []
    num_classes = 43

    stored_results_path = ["results/gtsrb_3_1/gtsrb-3-1.json", "results/gtsrb_4_1/gtsrb-4-1.json"]
    for i in range(len(stored_results_path)):
        if not os.path.exists(stored_results_path[i]):
            raise NotImplementedError("Please create feature vector directory first.")
        else:
            ip_log_file_name = os.path.join(stored_results_path[i])
            with open(ip_log_file_name) as json_file:
                data = json.load(json_file)
                print("Loading feature vectors from: ", ip_log_file_name)
            # avg_dist_dic.update({i:data['Average_Latent_Space_Class_Distance_test_crashes']['mean_distances']})
            comb_data.append(data["Average_Latent_Space_Class_Distance_test_crashes"])

    comb_dict = {}
    for i in range(2):
        metric_list = []
        comb_dict[i] = metric_list

    for i in range(len(comb_data)):
        mean_distances = comb_data[i]["mean_distances"]
        op_list = comb_dict[i]
        for cl in range(num_classes):
            try:
                if mean_distances[str(cl)] is not None:
                    op_list.extend([round(float(mean_distances[str(cl)]), 3)])
                else:
                    op_list.extend([round(float(0.0), 3)])
            except:
                op_list.extend([round(float(0.0), 3)])

    data1, data2 = np.array(comb_dict[0]), np.array(comb_dict[1])

    if data1.size != data2.size:
        raise Exception("Please check the data loading first.")

    pvalue, cohensd = run_wilcoxon_and_cohend(data1, data2)
