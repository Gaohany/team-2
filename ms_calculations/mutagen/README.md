# MUTAGEN - Framework for mutation testing of deep learning models

# Installation
- Required python version: 3.10
- OS: Windows
```
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Update packaging related packages
python -m pip install -U pip wheel setuptools packaging

# Install basic requirements
python -m pip install -r requirements/req_basic.txt

# Install correct pytorch (2.1.1) version corresponding to the local cuda installation
# refer to https://pytorch.org/get-started/locally/#start-locally for the instructions
# For example:
python -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1

# If you want to use the SQL IDE harlequin, install our modifed version (to have support for udfs)
python -m pip install -e ../harlequin
```

# Creation of new mutation operators
New mutation operators can be put in any python file in the `mutation_ops` folder. The framework will search all the python files in this folder and will create the operators.

It looks for all classes that inherits from `TrainingMutation` or `ModelMutation` as well as all methods that do not start with an underscore '_', these are assumed to be training data mutation operators.

# Step 1: Dateset preprocessing
To handle datasets more efficiently, we only use them in preprocessed form everywhere. These are stored using pytorch as dictionary.

A training dataset usually contains the following keys:
- img_train: pytorch float tensor of shape (N, C, 32, 32) containing the training images
- img_valid: pytorch float tensor of shape (N, C, 32, 32) containing the validation images
- lbl_train: pytorch uint8 tensor of shape (N,) containing training labels
- lbl_valid: pytorch uint8 tensor of shape (N,) containing validation labels
- num_classes: number of classes in the dataset

A converted test set can contain these keys:
- img_test: pytorch float tensor of shape (N, C, 32, 32) containing the test images
- lbl_train: pytorch uint8 tensor of shape (N,) containing test labels
- path_test: pytorch int64 tensor of shape (N,) containing a unique ids for the test images (e.g. stem of the filename in the original data)

For the fair comparison we also have datasets containing the following additional keys:
- sample_id: pytorch int64 tensor of shape (N,) containing unique ids for the test images
- image_id: pytorch i,nt64 tensor of shape (N,) containing the ids of the original images the samples were derived from
- transformations: list of strings with length N containing the name of the transformation applied to the corresponding image
- ssim, mse, l_0_norm, l_2_norm, l_inf_norm: pytorch float tensor of shape (N,) containing quality metrics for the images

## Commands
Training data for the model can be preprocessed with the following command:
```bash
python prepare_org_dataset.py -config_file "..\config\mnist.toml"
```
This will download the GTSRB dataset (if not already present) and create the two datasets:
- `data/mnist_data_normalized.pth`


# Step 2: Mutant generation
Now the mutants can be generated. For this you need a config file in which the model, data and the desired mutation operators are configured.

Simple example (configs/small.ini):
```ini
[General]
model = models/gtsrb/model.py
train = models/gtsrb/train.py
eval = models/gtsrb/sut.py
data = data/mnist_data_normalized.pth

[ChangeEpochs]
mutation = ChangeEpochs
epochs = 5

[RemoveSamples]
mutation = remove_samples
percentage = 0.5
```

This can then be passed to the script to generate the mutants:
```bash
python mut_run.py -config_file "./configs/small.ini"
```

This will create a new folder in `results/` based on the current time containing the raw_mutants.

# Step 3: Mutant training
These raw_mutants can then be trained using the `mut_train.py` script:
```bash
python mut_train.py --num_trainings <n> --result_dir "./results/<wanted_folder>"
```


# Step 4: Preprocessing of fuzzing data
The data generated from the fuzzing should be put under self defined folder, to create a folder structure like this:
```
📂 corner_case_gtsrb_kmnc
┣ 📂 crashes_rec
┃ ┗ 📄 id_{id}.json
┃ ┗ 📄 id_{id}.png
┃ ┗ 📄 ...
📂 corner_case_gtsrb_nbc
┣ 📂 crashes_rec
┃ ┗ 📄 id_{id}.json
┃ ┗ 📄 id_{id}.png
┃ ┗ 📄 ...
```
The self defined folder path should be declared in the "experiment_paths" in line 97 in the "prepare_fuzzing_dataset.py"

You should also modify the "splits" in the line 103 corresponding to "experiment_paths"

For example:

experiment_paths = [
    "E:/datasets/mnist_1_nc/mnist_1_1",
    "E:/datasets/mnist_1_kmnc/mnist_2_1",
    "E:/datasets/mnist_1_nbc/mnist_3_1",
]
splits = ["cc_nc", "cc_kmnc", "cc_nbc"]

The preprocessing can then be done with:
```
python prepare_fuzzing_dataset.py -config_file "../config/mnist.toml"
```

Which will create:
- `mutagen/data/mnist/mnist_fuzzing_data_normalized.pth`

# Step 5: Run basic evaluation
Now the evaluation can be carried out on the static data sets(test):
```
python evaluation_accuracy.py -result_dir "./results/<timestamp>" -config_file "../config/mnist.toml"
```
This will write the evalution results to `results/<timestamp>/evaln` with a folder structure like the hive partitioning of duckdb.

For fuzzing datasets:
```
python evaluation_accuracy_fuzzing.py -result_dir "./results/<timestamp>" -model mnist -config_file "../config/mnist.toml"
```

For Metamorphic datasets:
```
python evaluation_accuracy_mt.py -result_dir "./results/<timestamp>" -model mnist -config_file "../config/mnist.toml"
```

Note: 
1. Make sure you have declared the correct paths to the datases under variable "data_sets" in each script.
2. Make sure the "number_classes" in the "CREATE TABLE test_results.latent_space FLOAT[num_classes]" in eval_util.py is correct.

# Step 6: Mutation Score calculation
Now the mutation score for each mutation model and each dataset can be calculated:
```
python ms_calculation.py -result_dir "./results/<timestamp>" -config_file "../config/mnist.toml"
```
The results will be stored  in the "results.xlsx". 

Note: Make sure all datasets wanted to be analysed are stored under list "selected_splits" and "selected_splits_analyze". 