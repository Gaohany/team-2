# Coverage guided fuzz testing framework 

This is a framework -
1. To evaluate quality of dataset using latent space based metrics. 
The current framework is created based on the Coverage guided fuzz testing framework created by Vivek.

## How to use:
### Parts:
1. main.py: Use this main script to run LSCD calculation. Various input arguments are set to default. They can be changed based on requirements.

### General Notes
1. Each mutant model to be tested could be directly copied from the mutant generation framework's results folder, and saved under the base folder. The original models under "raw_mutants" and "trained_mutants" folder should be deleted.
2. A configuration file needed for each model-dataset combination and saved in config/. Need to be thorough when creating these. Now only MNIST and GTSRB are supported.

## Installation:
Run requirements.txt file for installing necessary Python packages. (Use of virtual enviornment is highly recommended)

### MacOS
1. Install Python 3.10 (preferably using [pyenv](https://github.com/pyenv/pyenv#installation))
2. Activate a virtual environment `python -m venv .env && source .env/bin/activate` 
3. Install requirements: `pip install -r requirements_mac.txt`
4. Install `brew install freetype imagemagick` (needs [Homebrew](https://brew.sh/) package manager). Note the installation path of `imagemagick`, which will be printed towards the end of the installation. Depending on the exact version installed, add a line similar to this to your `~/.bashrc` or `~/.zshrc`: `export MAGICK_HOME=/opt/homebrew/Cellar/imagemagick/7.1.1-21`
5. Run `git submodule init && git submodule update` to check out the `flann` submodule.
6. Build and install the FLANN library: 
   1. `brew install cmake`
   2. `cd flann && mkdir build && cd build`
   3. `cmake ..`
   4. `make`
   5. `sudo make install` (enter your system password)
7. Test by running: `python main.py --help`

### Git LFS
Large files like `.pickle`, `.pkl`, `.pth`, `.pt`, etc. are store in [Git LFS](https://docs.gitlab.com/ee/topics/git/lfs/) to avoid cluttering the Git repo. If you add any new file types that are not yet tracked in `.gitattributes`, please make sure to add them. You do this by running e.g. 
```bash
git lfs track "*.gz"
``` 
_(Note: the "" are necessary!)_

### Formatting
Make sure to apply the [isort](https://github.com/PyCQA/isort) import sorter [black](https://github.com/psf/black) code formatter to any Python files you changed.
- In VSCode, simply install the "isort" and "Black Formatter" extensions, and you'll be good to go.
- Otherwise, you can install them through `pip install isort black` and run using `isort [MY_PYTHON_FILE]` and `black [MY_PYTHON_FILE]`. 

# Calculate LSCD 

## Step 1: Dataset storage location
To make main.py run correctly, the paths of each augmented datasets to be used for testing should be declared between line 115 and line 135 in the main.py. The path of original test dataset is declared in the configuration file, for example "./config/gtsrb.toml" line 13 "data_path".

## Step 2: Dataset criteria
Change "dataset_criteria" in line 74 in the main.py to adpat to the dataset you want to test with.
All supported datasets are Original Dataset("org"), Fuzzing Dataset("nc","nbc","kmnc) and Metamorphic Datasets("passed", "failed"). If you want to use other datasets please check and modify the function "metrics_dataloader" in metrics_utils.metrics_dataloader.py

## Step 3: Run the main.py
Run the main.py with correct input arguments:
```
python main.py -mt_models_dir "./mnist_mutant_models"
```

Note: If the results are needed to be stored as .xlsx, youcan call the function "write_results_to_excel" at the very end.










