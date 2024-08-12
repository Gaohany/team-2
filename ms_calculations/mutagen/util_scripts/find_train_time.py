import argparse
import configparser
import importlib
import importlib.util
import shutil
import sys
import json

from pathlib import Path

import torch


def run_training(result_path: Path):
    output_folder = result_path / "trained_mutants"

    total = 0
    count = 0

    for mutant_folder in sorted(output_folder.iterdir()):
        t_file = mutant_folder / 'training.json'
        conf = json.loads(t_file.read_text())
        total += conf['duration']
        count += 1

    print(total, count, total / count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir')
    args = parser.parse_args()

    run_training(Path(args.result_dir))
