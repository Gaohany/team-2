import argparse
import shutil
import json
from typing import Dict, List
from pathlib import Path
import configparser


def run(result_path: Path):
    m_folder = result_path / 'raw_mutants'
    folders = sorted(m for m in m_folder.iterdir())

    config = configparser.ConfigParser(default_section="General")
    config.read(next(folders[0].glob("*.ini")))

    d = {}

    for m in folders:
        m = m / 'data.link'
        actual_file = Path(m.read_text())

        d[actual_file.name] = actual_file.stat().st_size

        # if m.exists():
        #     actual_file = Path(m.read_text())
        #     print(actual_file, actual_file.stat().st_size)

    print("Training data:", sum(d.values()) / (1024 * 1024 * 1024))

    trained_folder = result_path / 'trained_mutants'
    all_trained_files = list(trained_folder.rglob('*'))
    all_trained_files = [f.stat().st_size for f in all_trained_files if f.is_file() and not f.suffix == '.pyc']
    print("Trained mutants", sum(all_trained_files)/ (1024 * 1024 * 1024))
    # candidates, all_folders = get_relevant_mutation_folders(
    #     result_path / "raw_mutants", filename='training.json', key='final_train_acc'
    # )
    trained_folder = result_path / 'evaln'
    all_trained_files = list(trained_folder.rglob('*'))
    all_trained_files = [f.stat().st_size for f in all_trained_files if f.is_file() and not f.suffix == '.pyc']
    print("Eval results", sum(all_trained_files)/ (1024 * 1024 * 1024))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir')
    args = parser.parse_args()

    run(Path(args.result_dir))
