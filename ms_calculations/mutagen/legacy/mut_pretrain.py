import argparse
import configparser
import json

from pathlib import Path
from typing import Dict

from mut_train import load_module


def run_training(result_path: Path, data: Path):
    cache_dir = Path.cwd() / 'data' / 'pretrain'
    cache_dir.mkdir(exist_ok=True, parents=True)

    items = [folder for folder in (result_path / "raw_mutants").iterdir() if folder.is_dir()]
    model_files: Dict[str, Path] = {}

    train_script_name = None
    original_train_hash = None

    for mutant_folder in sorted(items):
        conf = json.loads((mutant_folder / "meta.json").read_text())
        model_hash = conf['hashes']['model']

        if mutant_folder.name == "AAA_Original":
            original_train_hash = conf['hashes']['train']

        full_hash = f"{model_hash}_{original_train_hash}"
        if full_hash in model_files:
            continue

        config = configparser.ConfigParser(default_section="General")
        config.read(next(mutant_folder.glob("*.ini")))

        model_file_name = Path(config["General"]['model']).name
        train_script_name = Path(config["General"]['train']).name

        model_files[full_hash] = mutant_folder / model_file_name

    train_path = result_path / "raw_mutants" / "AAA_Original" / train_script_name
    train_module = load_module("pre_train", train_path)

    print("Results folder", result_path, 'needs', len(model_files), "pretrained weights")
    for k, v in model_files.items():
        print(k, v)
        out_file = cache_dir / f'{k}.pth'

        if out_file.exists():
            print(k, "Already trained")
            continue

        model_module = load_module(".".join([v.parent.name, "model"]), v)

        net = model_module.Net(num_classes=10)
        train_module.train(net, str(data), out_file, num_epochs=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir')
    parser.add_argument('data')

    args = parser.parse_args()

    run_training(Path(args.result_dir), Path(args.data))
