import argparse
import shutil
import json
from typing import Dict, List
from pathlib import Path

from stats import get_relevant_mutation_folders, is_diff_sts


def run(result_path: Path, filename: str,  key: str):
    candidates, all_folders = get_relevant_mutation_folders(
        result_path / "killable_mutants", filename=filename, key=key
    )
    original_accuracies = candidates.pop('AAA_Original')

    output_path = result_path / 'analysis'
    output_path.mkdir(exist_ok=True, parents=True)

    print("Analysis for", key)
    counter = 0
    for name, accuracies in candidates.items():
        is_sts, p_value, effect_size = is_diff_sts(original_accuracies, accuracies)
        print(f"- {name: <30}", '\u2705' if is_sts else '\u274C')

        folder = output_path / name
        folder.mkdir(exist_ok=True, parents=True)

        obj = {
            "accuracies": accuracies,
            "is_sts": bool(is_sts),
            "p_value": p_value,
            "effect_size": effect_size,
        }
        json_file = folder / f'{key}.json'
        json_file.write_text(json.dumps(obj, indent=4))

        counter += bool(is_sts)

    print(f"Final score { counter / len(candidates):07.3%}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir')
    args = parser.parse_args()

    configs = [
        ("eval.json", "test_set"),
        # ("eval.json", "fuzzing_corner_case_gtsrb_kmnc"),
        # ("eval.json", "fuzzing_corner_case_gtsrb_nbc"),
        # ("eval.json", "fuzzing_corner_case_gtsrb_nc"),
    ]

    for c in configs:
        run(Path(args.result_dir), *c)

