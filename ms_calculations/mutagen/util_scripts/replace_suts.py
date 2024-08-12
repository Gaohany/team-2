import argparse
import configparser
import importlib
import importlib.util
import shutil
import sys
import json

from pathlib import Path

import torch


def run_training(result_path: Path, new_sut_file: Path):
    output_folder = result_path / "trained_mutants"

    for mutant_folder in sorted(output_folder.iterdir()):
        target_file = mutant_folder / 'sut_new.py'
        shutil.copy(new_sut_file, target_file)
        print(target_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir')
    parser.add_argument('new_sut_file')
    args = parser.parse_args()

    run_training(Path(args.result_dir), Path(args.new_sut_file))
