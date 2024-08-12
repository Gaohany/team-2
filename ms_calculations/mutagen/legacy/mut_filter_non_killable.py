import argparse
import shutil
import json
from typing import Dict, List
from pathlib import Path

from stats import get_relevant_mutation_folders, is_diff_sts


def run(result_path: Path):
    candidates, all_folders = get_relevant_mutation_folders(
        result_path / "trained_mutants", filename='training.json', key='final_train_acc'
    )

    original_accuracies = candidates.pop('AAA_Original')

    output_path = result_path / 'killable_mutants'
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir()

    for name, accuracies in candidates.items():
        is_sts, p_value, effect_size = is_diff_sts(original_accuracies, accuracies)
        print(f"{name: <30}", '\u2705' if is_sts else '\u274C')

        if not is_sts:
            del all_folders[name]

    print('=' * 80)

    for k, v in all_folders.items():
        print("Copying folder for", k)
        for folder in v:
            # print(folder)
            shutil.copytree(folder, output_path / folder.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir')
    args = parser.parse_args()

    run(Path(args.result_dir))
