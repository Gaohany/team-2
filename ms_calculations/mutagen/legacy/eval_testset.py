import argparse
import configparser
from typing import List, Protocol, Union, Iterable, Tuple
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import torchvision
import numpy as np

from eval_util import TestResult, write_output_db, write_eval_result

from mut_train import load_module


class SUT_Proto(Protocol):
    def execute(self, images: Union[torch.Tensor, Iterable[np.array]]) -> Tuple[List[int], List[float]]: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, value, traceback): ...


def run_eval(sut: SUT_Proto, sut_folder: Path, data_loader: DataLoader) -> List[TestResult]:
    sut_name, sut_training = sut_folder.parent.name.rsplit("_", maxsplit=1)

    predictions = []
    labels = []

    with sut:
        for img, label in data_loader:
            classes, probabilities = sut.execute(img)
            predictions.append(torch.tensor(classes))
            labels.append(label)

    result_tensor = torch.stack([torch.cat(labels), torch.cat(predictions)], dim=1)
    results = [
        TestResult(sut_name, sut_training, t[0].item(), t[1].item(), "equality", t[0].item() == t[1].item())
        for t in result_tensor
    ]

    return results


def run(result_path: Path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ])

    test_set = torchvision.datasets.GTSRB(str('data'), split="test", download=True, transform=data_transforms)
    test_loader = DataLoader(test_set, batch_size=4096, num_workers=8, shuffle=False)

    folders = [folder for folder in (result_path / "killable_mutants").iterdir() if folder.is_dir()]

    db_tuples = []
    for mutant_folder in sorted(folders):
        config = configparser.ConfigParser(default_section="General")
        config.read(next(mutant_folder.glob("*.ini")))

        sut_module = load_module(".".join([mutant_folder.name, "eval"]), Path(config["General"]["eval"]))
        # sut_module = load_module(".".join([mutant_folder.name, "eval"]), mutant_folder / Path(config["General"]["eval"]).name)

        sut = sut_module.SUT(mutant_folder, device=device)

        results = run_eval(sut, mutant_folder, test_loader)
        db_tuples.extend(results)

        acc = sum(r.result for r in results) / len(results)
        write_eval_result(mutant_folder, name='test_set', acc=acc)

    # write_output_db(result_path, db_tuples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir')
    args = parser.parse_args()

    run(Path(args.result_dir))
