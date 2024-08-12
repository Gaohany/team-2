from itertools import combinations
from collections import defaultdict
import os
import torch
from pathlib import Path
import numpy as np
import json
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
from ms_calculations.mutagen.ms_utils import *
#from models.mnist.lenet5.model import Net
import pandas as pd
#from tqdm import tqdm
#from plot_utils import *
#from scipy import stats
from sklearn.ensemble import IsolationForest




def calculate_initial_centroid_radius(gt_labels, output_labels, input_data_frame):
    centroid_dictionary = {}
    sut_name = input_data_frame["sut_name"]
    print(
        "Calculating the initial centroid positioning & radius thresholds from the train dataset for .",
        sut_name[0],
    )

    accuracy = round((gt_labels == output_labels).sum() / gt_labels.shape[0] * 100, 2)
    fp_samples = int((gt_labels != output_labels).sum())

    print(
        "Accuracy on {} is {} % using training dataset.".format(sut_name[0], accuracy)
    )
    # feature_vectors = input_data["latent_space"]
    mask = input_data_frame["label"] == input_data_frame["output"]
    input_data_frame_new = input_data_frame[mask]
    input_data_frame_new.reset_index(drop=True, inplace=True)

    print("FPs: ", fp_samples)
    print(
        "New Dataframe resized to:",
        input_data_frame.shape[0] - input_data_frame_new.shape[0],
    )

    grouped = input_data_frame_new.groupby("label")
    label_pairs_dict = defaultdict(list)

    for label, group in grouped:
        features = group["latent_space"].tolist()
        label_pairs_dict[label] = features
        # print(len(features))

    samples_ignored = 0  # TP values with nan entries.

    for key, class_wise_feature_vectors in label_pairs_dict.items():
        try:
            features_cl_train = torch.tensor(class_wise_feature_vectors)
            if (features_cl_train.size != 0) and not torch.any(
                torch.isnan(features_cl_train)
            ):
                # print(key, features_cl_train.shape)
                mean_feature = torch.mean(features_cl_train, dim=0)
                centroid_dictionary[key] = mean_feature
            else:
                samples_ignored += 1
        except:
            samples_ignored += 1
    print("Samples Ignored:", samples_ignored)
    return centroid_dictionary


def calculate_lscd(centroid_loaded, input_data_frame, num_classes):
    sut_name = input_data_frame["sut_name"]

    print(
        "Calculating the LSCD values for ",
        sut_name[0],
    )

    lscd_values_dict, lscd_values_dict_2, avg_lscd_value, avg_lscd_value_2 = (
        {},
        {},
        0.0,
        0.0,
    )

    input_data_frame_new = input_data_frame.copy(deep=True)
    grouped = input_data_frame_new.groupby("label")
    # input_data_frame_new.reset_index(drop=True, inplace=True)
    label_pairs_dict = defaultdict(list)

    for label, group in grouped:
        features = group["latent_space"].tolist()
        label_pairs_dict[label] = features

    for cl in range(num_classes):
        all_dist, all_dist_2 = [], []
        if cl in centroid_loaded["all_centroids"]:
            cl_centroid = centroid_loaded["all_centroids"][cl]
        else:
            cl_centroid = torch.zeros(num_classes)

        cl_feature_vectors = label_pairs_dict[cl]

        try:
            for vector in cl_feature_vectors:
                # diff = np.abs(np.array(vector) - np.array(cl_centroid))
                # distance = np.linalg.norm(diff)
                diff = torch.abs(torch.tensor(vector) - cl_centroid)
                euc_dist = torch.norm(diff, p=2)
                all_dist.append(euc_dist)
        except:
            pass

        # print(len(all_dist), len(cl_feature_vectors))
        lscd_values_dict[cl] = torch.mean(torch.tensor(all_dist))

    for cl in range(num_classes):
        avg_lscd_value += lscd_values_dict[cl]

    avg_lscd_value = round(float(avg_lscd_value / num_classes),4)

    return lscd_values_dict, avg_lscd_value

   


def calculate_mutation_score(input_data, reference_data, num_classes, split):
    cl_ms_score_dict = {}

    if split == "test":
        input_data_op = input_data["output"]
        ref_data_op = reference_data["output"]
        mutation_score_DMplus = (
            (input_data_op != ref_data_op).sum() / input_data_op.shape[0]
        ) 
    else:
        input_data_op = input_data["output"]
        ref_data_op = reference_data["ori_output"]
        mutation_score_DMplus = (
            (input_data_op != ref_data_op).sum() / input_data_op.shape[0]
        )

    return mutation_score_DMplus, cl_ms_score_dict



def classwise_analysis(lscd_values_dict, cl_ms_score_dict, num_classes):
    cl_cor_values, ms_list, lscd_list = {}, [], []

    for cl in range(num_classes):
        cl_lscd = lscd_values_dict[cl]
        cl_ms = cl_ms_score_dict[cl]
        lscd_list.append(cl_lscd)
        ms_list.append(cl_ms)

    cor, p_value = spearmanr(ms_list, lscd_list)
    # print("Classwise Correlation: ", cor, p_value)

    return cl_cor_values



def correlation_analysis (ms_list, lscd_list, acc_list,log_filepath):

    outlier_methods = ["IQR", "IsolationForest"]
    correlation_methods = ["Spearman", "Pearson"]
    filtered_dict, filtered_dict_list = {}, {}
    
    np.random.seed(0)

    data = pd.DataFrame({
        'ms_list': ms_list,
        'lscd_list': lscd_list,
        'acc_list': acc_list
    })

    # Calculate the Z-scores & Filter out the outliers
    # z_scores = np.abs(stats.zscore(data))
    # threshold = 1.5
    # data_clean = data[(z_scores < threshold).all(axis=1)]

    for i, filter_method in enumerate(outlier_methods):
    
        if filter_method == "IQR":
            # Calculate the IQR (interquartile range) & Filter out the outliers
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.85)
            IQR = Q3 - Q1
            data_clean = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
        elif filter_method == "IsolationForest":
            iso_forest = IsolationForest(contamination=0.1, random_state=0) # can be from (0.0, 0.5]
            # data_isl = np.column_stack((data["ms_list"], data["lscd_list"]))
            y_pred = iso_forest.fit_predict(data)
            data_clean = data[y_pred == 1]
        else:
            data_clean = data

        print("Original Data Shape:", data.shape)
        print("Cleaned Data Shape:", data_clean.shape)

        for correlation_method in correlation_methods:
            if correlation_method == "Spearman":
                cor_1, p_value_1 = spearmanr(data_clean['ms_list'], data_clean['lscd_list'])
                cor_2, p_value_2 = spearmanr(data_clean['acc_list'], data_clean['ms_list'])
                cor_3, p_value_3 = spearmanr(data_clean['acc_list'], data_clean['lscd_list'])
            elif correlation_method == "Pearson":
                cor_1, p_value_1 = pearsonr(data_clean['ms_list'], data_clean['lscd_list'])
                cor_2, p_value_2 = pearsonr(data_clean['acc_list'], data_clean['ms_list'])
                cor_3, p_value_3 = pearsonr(data_clean['acc_list'], data_clean['lscd_list'])
            else:
                pass
            
            print("Correlation MS v/s LSCD: ", cor_1, p_value_1)
            print("Correlation Acc v/s MS: ", cor_2, p_value_2)
            print("Correlation Acc v/s LSCD: ", cor_3, p_value_3)


            # keys = ["ms_list", "lscd_list",  "acc_list"]

            # filtered_dict.update({"outlier_filter_method:": filter_method})
            # filtered_dict.update({"Correlation Coeffiecient:": correlation_method})
            # filtered_dict.update({"Correlation Values: ": round(float(cor_1),3)})
            
            analysis_data = dict(
                        Outlier_method = filter_method,
                        Correlation_Coeffiecient=correlation_method,
                        Correlation_Values=round(float(cor_1),3),
                        p_value=round(float(p_value_1),4))
            
            full_data = dict(
                        MS_list = list(data_clean["ms_list"]),
                        LSCD_list = list(data_clean["lscd_list"]),
                        ACC_list = list(data_clean["acc_list"]),
                        )
            
            # for key in keys:
            #     filtered_dict[key] = []
            #     filtered_dict[key].extend(list(data_clean[key]))        

            data_key = str(i) + "_" + filter_method + "_" + correlation_method
            list_key = str(i) + "_" + filter_method + "_"+ correlation_method + "_data" 
            
            filtered_dict.update({data_key: analysis_data})
            filtered_dict_list.update({list_key: full_data})

    data = {}
    log_filepath_data = Path(str(log_filepath) + ".json")
    log_filepath_list = Path(str(log_filepath) + "list.json")

    if os.path.exists(log_filepath_data):
        with open(log_filepath_data) as json_file:
            data = json.load(json_file)
            del data
            data = filtered_dict
    else:
            data = filtered_dict

    with open(log_filepath_data, "w") as json_file:
        json.dump(data, json_file, indent=4)
        print("Results written to:", log_filepath_data)        
    
    if os.path.exists(log_filepath_list):
        with open(log_filepath_list) as json_file:
            data = json.load(json_file)
            del data
            data = filtered_dict_list
    else:
            data = filtered_dict_list

    with open(log_filepath_list, "w") as json_file:
        json.dump(data, json_file, indent=4)
        print("Results written to:", log_filepath_list)  
    
    return filtered_dict 




