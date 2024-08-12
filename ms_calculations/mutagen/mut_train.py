import argparse
import configparser
import importlib
import importlib.util
import shutil
import sys
import json
import os
from typing import Dict
from pathlib import Path
import torch
import numpy as np
sys.path.append("./results") # Or add the direct path to the results folder to the system


def load_module(module_name: str, filepath: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    
    spec = importlib.util.spec_from_file_location(module_name, filepath.resolve())
    
    module = importlib.util.module_from_spec(spec)
    
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    print("Loaded module", module_name)

    return module


def run_training(result_path: Path, num_training: int):
    output_folder = result_path / "trained_mutants"
    output_folder.mkdir(exist_ok=True)
    items = [folder for folder in (result_path / "raw_mutants").iterdir() if folder.is_dir()]
    orig_conf = json.loads((result_path / "raw_mutants" / "AAA_Original" / "meta.json").read_text())
    original_train_hash = orig_conf['hashes']['train']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for mutant_folder in sorted(items):
        config = configparser.ConfigParser(default_section="General")
        config.read(next(mutant_folder.glob("*.ini")))

        files = {k: mutant_folder / Path(config["General"][k]).name for k in ("model", "eval", "train")}
        train_module = load_module(".".join([mutant_folder.name, "train"]), files['train'])
        model_module = load_module(".".join([mutant_folder.name, "model"]), files['model'])
        
        
        print(".".join([mutant_folder.name, "train"]))
    
        with (mutant_folder / "meta.json").open("r") as fp:
            conf = json.load(fp)

        pretrain = Path.cwd() / 'data' / 'pretrain' / f'{conf["hashes"]["model"]}_{original_train_hash}.pth'
        if not pretrain.exists():
            pretrain = None
            #print("pretrain is none")

        data_file = (mutant_folder / 'data.link').read_text()
        data = torch.load(data_file, device)

        for i in range(num_training):
            print(f"[{i:04d}] {mutant_folder.name}")
            training_folder = output_folder / f"{mutant_folder.name}_{i:03d}"
            if training_folder.exists():
                if all((training_folder / f).exists() for f in ("training.json", "model.pth")):
                    print("Mutant", training_folder.name, "already trained")
                    continue

                shutil.rmtree(training_folder)
            training_folder.mkdir()

            copied = [shutil.copy(f, training_folder) for f in mutant_folder.iterdir() if f.is_file()]

            net = model_module.Net() # Creates an instance of NN defined in model_module.
            net.to(device)
            #data = data_norm(data)
            train_module.train(net, data, training_folder / 'model.pth', pretrain=pretrain)
            del net
        del data
        torch.cuda.empty_cache()
        print("=" * 120)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir')
    parser.add_argument('--num_trainings', default=1, type=int)

    args = parser.parse_args()

    run_training(Path(args.result_dir), args.num_trainings)
