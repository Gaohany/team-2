import argparse
import configparser
import json
import time
from typing import Tuple, Optional
from pathlib import Path
import torch
import toml
import pandas as pd
import duckdb
from collections import defaultdict

import eval_util
from mut_train import load_module
from all_classes import *


def load_config(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser(default_section="General")
    config.read(next(path.glob("*.ini")))
    return config


def load_mutant(mutant_folder: Path) -> Mutant:
    config = load_config(mutant_folder)
    sut_module = load_module(
        ".".join([mutant_folder.name, "eval"]), Path(config["General"]["eval"])
    )
    sut = sut_module.SUT(mutant_folder, device="cpu")
    model_train_data = Path((mutant_folder / "data.link").read_text())

    sut_name, sut_training = mutant_folder.name.rsplit("_", maxsplit=1)
    sut_training = int(sut_training)

    with (mutant_folder / "training.json").open() as fp:
        obj = json.load(fp)

    return Mutant(
        sut_name,
        sut_training,
        mutant_folder,
        model_train_data,
        config,
        sut,
        obj["duration"],
    )


def create_tensor_dataset(path: Path, split: str, has_path: bool = True):
    data = torch.load(path, "cpu")

    return Data(
        data[f"img_{split}"].to("cpu"),
        data[f"lbl_{split}"].view(-1).to("cpu"),
        data[f"path_{split}"].to("cpu"),
        data[f"ori_img_{split}"].to("cpu"),
        #torch.arange(len(data[f"lbl_{split}"])),
    )


def create_hive_folder(
    base_path: Path, sut_name: str, sut_training: int, dataset_name: str
):
    data_path = (
        base_path
        / f"dataset={dataset_name}"
        / f"sut_name={sut_name}"
        / f"sut_training={sut_training}"
    )
    data_path.mkdir(exist_ok=True, parents=True)

    return data_path, data_path / f"{dataset_name}.parquet"


def print_progresses(*items: Tuple[int, int], width: int = 20):
    for current, total in items:
        percentage = current / total
        print(f"[{'=' * round(percentage * width):<{width}}]", end="")


def batched(data: Data, batch_size: int):
    length = len(data.lbl)
    for i in range(0, length, batch_size):
        end = min(length, i + batch_size)

        yield data.img[i:end], data.lbl[i:end], data.ids[i:end], data.ori_img[i:end], i, end


def run_eval(mutant: Mutant, data: Data, dataset_name: str, config):
    start = time.time()
    print(
        f"[{mutant.training:03d}][{mutant.name:<30}] {dataset_name:15}",
        end="",
        flush=True,
    )

    p_classes = torch.empty_like(data.lbl)
    comparison = torch.empty_like(data.lbl, dtype=torch.bool)
    probabilities = torch.empty_like(data.lbl, dtype=torch.float32)
    latent_space_vectors = torch.empty((len(data.lbl), config.num_classes), dtype=torch.float32)
    p_ori_classes = torch.empty_like(data.lbl)

    with mutant.sut:
        for img, label, sample_ids, ori_img, i, end in batched(data, batch_size=2048):
            p, classes, outputs = mutant.sut.execute_raw(img)
            ori_p, ori_classes, ori_outputs = mutant.sut.execute_raw(ori_img)
            #classes = classes.view(-1)
            
            classes = classes.cpu()
            ori_classes = ori_classes.cpu()
        
            p_classes[i:end] = classes
            comparison[i:end] = label == classes
            probabilities[i:end] = p
            latent_space_vectors[i:end] = outputs
            p_ori_classes[i:end] = ori_classes



    test_df = pd.DataFrame.from_dict(
        {
            "sut_name": mutant.name,
            "sut_training": mutant.training,
            "dataset": dataset_name,
            "sample_id": data.ids,
            "label": data.lbl,
            "output": p_classes,
            "result": comparison,
            "confidence": probabilities,
            "latent_space": [v.tolist() for v in latent_space_vectors],
            "ori_output": p_ori_classes,
        }
    )
    dur = time.time() - start
    test_df["training_time"] = mutant.training_time
    test_df["evaluation_time"] = dur
    test_df["is_duplicate"] = False

    print(f"took {dur: 7.2f}s", end=" ")
    return test_df


def write_dataframe(
    df: pd.DataFrame,
    eval_path: Path,
    mutant: Mutant,
    dataset_name: str,
    duplicate: Optional[str] = None,
):
    start = time.time()

    sub_folder, outfile = create_hive_folder(
        eval_path, mutant.name, mutant.training, dataset_name
    )

    with duckdb.connect() as db_con:
        db_con.sql(eval_util.TEST_RESULT_TABLE_CREATION)
        db_con.execute("INSERT INTO test_results SELECT * FROM df")
        db_con.sql("SELECT * FROM test_results").write_parquet(
            str(outfile), compression=None
        )

        print(f"| export took {time.time() - start: 7.2f}s", end="")
        start = time.time()

        if duplicate:
            sub_folder, outfile = create_hive_folder(
                eval_path, mutant.name, mutant.training, duplicate
            )

            db_con.execute(
                "UPDATE test_results SET dataset = $1, is_duplicate=true;",
                parameters=[duplicate],
            )
            db_con.sql("SELECT * FROM test_results").write_parquet(
                str(outfile), compression=None
            )

            print(f" | duplicate took {time.time() - start: 7.2f}s", end="")

    print()


def eval(result_path: Path, model: str, args: argparse.ArgumentParser):
    folders = sorted(
        [
            folder
            for folder in (result_path / "trained_mutants").iterdir()
            if folder.is_dir()
        ]
    )
    print(folders)
    original_data_set = Path(load_config(folders[0])["General"]["data"])
    mutants = [load_mutant(folder) for folder in folders]
    config = json.loads(json.dumps(toml.load(args.config_file)), object_hook=obj)
    if model == "mnist":
        data_sets = {
            "cc_nc": (
                Path("../data", "mnist", "mnist_cc_5_data_normalized.pth"),
                "cc_nc",
            ),
            "cc_kmnc": (
                Path("../data", "mnist", "mnist_cc_5_data_normalized.pth"),
                "cc_kmnc",
            ),
            "cc_nbc": (
                Path("../data", "mnist", "mnist_cc_5_data_normalized.pth"),
                "cc_nbc",
            ),
        }
    elif model == "gtsrb":
        data_sets = {
            "cc_nc": (
                Path("./data", "gtsrb", "gtsrb_cc_4_data_normalized.pth"),
                "cc_nc",
            ),
            "cc_kmnc": (
                Path("./data", "gtsrb", "gtsrb_cc_4_data_normalized.pth"),
                "cc_kmnc",
            ),
            "cc_nbc": (
                Path("./data", "gtsrb", "gtsrb_cc_4_data_normalized.pth"),
                "cc_nbc",
            ),
        }
    eval_path = result_path / "evaln"
    eval_path.mkdir(exist_ok=True, parents=True)

    mutants_datasets = defaultdict(list)

    print(type(mutants_datasets))

    for di, (dataset_name, data_set_args) in enumerate(data_sets.items()):
        data_set = create_tensor_dataset(*data_set_args) # Path, split_name

        for i, mutant in enumerate(mutants):
            print_progresses((di, len(data_sets)), (i, len(mutants)))

            if create_hive_folder(
                eval_path, mutant.name, mutant.training, dataset_name
            )[1].exists():
                print(
                    f"[{mutant.training:03d}][{mutant.name:<30}] {dataset_name:15}exists"
                )
                continue

            dupl = ""
            if dataset_name == "otrain":
                if mutant.train_set_path == original_data_set:
                    dupl = "mtrain"
                else:
                    mutants_datasets[mutant.train_set_path].append(mutant)

            results = run_eval(mutant, data_set, dataset_name, config)
            write_dataframe(results, eval_path, mutant, dataset_name, dupl)

        del data_set

    for di, (dataset_path, mutant_list) in enumerate(mutants_datasets.items()):
        data_set = create_tensor_dataset(dataset_path, split="train", has_path=False)

        for i, mutant in enumerate(mutant_list):
            print_progresses((di, len(mutants_datasets)), (i, len(mutant_list)))

            if create_hive_folder(eval_path, mutant.name, mutant.training, "mtrain")[
                1
            ].exists():
                print(f"[{mutant.training:03d}][{mutant.name:<30}] {'mtrain':15}exists")
                continue

            results = run_eval(mutant, data_set, "mtrain", config)
            write_dataframe(results, eval_path, mutant, "mtrain")

        del data_set


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-result_dir",
        default=Path("./results/mnist"),
        help="Result Path",
    )
    parser.add_argument("-model", default="mnist", help="Model Type")
    parser.add_argument(
        "-config_file",
        help="choose configuration with which to run",
        default="../config/mnist.toml",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()

    eval(result_path=args.result_dir, model=args.model, args = args)
