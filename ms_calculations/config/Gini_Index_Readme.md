TEST CASE SCENARIO:
1) We have all the GTSRB and SVHN feature vectors stored in one directory (i.e. "metrics" or "vectors", it doesn't matter the name of the directory)
2) We would like to be able to calculate Gini Indexes and store them in a .json file as logging (used in individual systems) or store in a .pkl file (used in batch systems)

Intermediary results:

-> In order to fit our PCA model to all data set we have in our system, we must first collect all feature vectors in one .pkl, collect_all_feature_vectors yields us this intermediary result.
-> After fitting PCA model with that combined data set, we store our fitted PCA model in a .pkl file under pca_pickles (if it is UMAP, in umap_pickles)
-> In the meantime, we need to have our functional range tailored for the specificed reduced dimensionality and data set (GTSRB or SVHN) to calculate confidence values, this is made by calculate_functional_range in the Gini Index algorithms automatically

before starting: 
It is worth noting that the fitted PCA model will be used throughout the whole calculations if the data set or the number of dimensions are not going to be changed (also the functional range).
This means, in the first iteration of calculating functional range as well as a fitted PCA model for capturing the maximum variance, please give the calculations_path in gtsrb.toml and svhn.toml the directory that contains the most vectors in the system (i.e. metrics data set) otherwise all the calculations would be based on less reliant PCA models and/or functional ranges.

Some notes:
In Gini Index calculation, batch calculations are stored in .pkl files whereas individual calculations of whole_points_approach and also class based approach are stored in metrics/log.json

Please use main_dataset_metrics.py file to first generate feature vectors and store them in "metrics" folder as we rely on that naming structure for .pkl files when retrieving the data set names under gini_index_in_batches

visualization parameter is best used with gini_index_in_batches

How To Use The Framework With Respect To The Gini Index Calculation:

	1. Gini Index calculation with only one number of bins per dimension hyperparameter in one data set:
         --> please give the calculation_path the vector path for looking at the .pickle files to get the combined data set for our fitted model in dimensionality reduction
		--> please change the collect_all_feature_vectors parameter to true as it is crucial for getting used at the dimensionality reduction 
		--> please do the normal data set configurations under dataset_metrics_config.toml such as type, mode and if type == "crashes" experiment_path
		--> under dataset_metrics_config.toml/[gini_index] please change number_bins to the integer which you want to compute and please make everything false other than all_class_approach = true
			Note: If you would like to see class based distribution of Gini Index, please make class_based_approach = true as well
		--> change the other metrics number_dimensionality and dimensionality_method accordingly 
			Note: For number_dimensionality maximum of 4 number_dimensionality is optimal. More than this would cause performance problems.
			Note: Right now, pca and umap are supported. Please write as it is 

		
	2. Gini Index calculation with multiple number of bins per dimension hyperparameter in one data set:

        --> please give the calculation_path the vector path for looking at the .pickle files to get the combined data set for our fitted model in dimensionality reduction
		--> please change the vector_path parameter as following:
			please create a new folder and put the pickle that you want to test into that folder
			and give that folder's path into the vector_path (such as "metrics")
		--> please change the collect_all_feature_vectors parameter to true as it is crucial for getting used at the dimensionality reduction 
		--> please do the normal data set configurations under dataset_metrics_config.toml such as type, mode and if type == "crashes" experiment_path
		--> under dataset_metrics_config.toml/[gini_index] please let number_bins stay as list of number of bins per dimension which you want to compute and please make everything false other than in_batches = true
		--> change the other metrics number_dimensionality and dimensionality_method accordingly 
			Note: For number_dimensionality maximum of 4 number_dimensionality is optimal. More than this would cause performance problems.
			Note: Right now, pca and umap are supported. Please write as it is 

	3. Gini Index calculation with multiple number of bins per dimension hyperparameter in multiple data sets:
		
		--> please change the collect_all_feature_vectors parameter to true as it is crucial for getting used at the dimensionality reduction (if it was already used for getting the combined PCA model then you don't need to put that to true)
		--> for vector_path please give feature_vectors_dir's value so that all the data sets associated with GTSRB or SVHN (whatever is selected) can be gotten from there
		--> please give the calculation_path the vector path for looking at the .pickle files to get the combined data set for our fitted model in dimensionality reduction (if this is already calculated (fitted PCA model) then there is no need for recalculating it)
		--> under dataset_metrics_config.toml/[gini_index] please let number_bins stay as list of number of bins per dimension which you want to compute and please make everything false other than in_batches = true
		--> change the other metrics number_dimensionality and dimensionality_method accordingly 
			Note: For number_dimensionality maximum of 4 number_dimensionality is optimal. More than this would cause performance problems.
			Note: Right now, pca and umap are supported. Please write as it is 

	While running, please give --metrics-config-file and --dataset-config-file what you would like to test, and it is ready to go! Depending on the size of the num_bins and number_dimensionality parameters, it can last from 1 minute up to a day
	The results coming from the first case will be stored individually under gini_index_results directory in gini_index_all_points_approach.json file, results from class based approach
will be stored in gini_index_class_based_approach.json. In 2. and 3. cases, Gini Index algorithm results will be stored as a .pkl file with a descriptive name 



Other programs:
visualization of confidence values : Please change visualization.confidence_values and calculate_functional_range to true.
calculating the label means of ground truth classes: Please change calculate_ground_truth_label_means to true.
Note: Under gini_index_archive.py there are some archived programs such as but not limited to : KDE calculation and an alternative approach for Gini Index calculation which doesn't take histogramdd() function 
to discretize bins but custom built in functions 