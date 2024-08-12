import argparse
import json
import configparser
from typing import Tuple, Optional, Callable, NamedTuple
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
import torch
import torchvision
from torchvision.datasets import VisionDataset
import PIL

from eval_util import TestResult, write_output_db, write_eval_result
from eval_testset import run_eval
from mut_train import load_module


class FuzzingDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        model: str,
        method: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._target_folder = Path(root) / 'fuzzing' / model / method / 'crashes_rec'

        self._images = sorted(self._target_folder.glob('*.png'))
        self._infos = [f.with_suffix('.json') for f in self._images]

    def __len__(self) -> int:
        return len(self._infos)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, int, int]:
        # sample = PIL.Image.open(self._images[index]).convert("RGB")
        sample = PIL.Image.open(self._images[index])

        with self._infos[index].open("r") as fp:
            target = json.load(fp)['__ground_truth_metrics__'][0]['op_class']

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        _, *nums = self._infos[index].stem.split("_")
        nums = [int(n) for n in nums]

        return sample, target, nums[0] << 32 | nums[1]


class FuzzingFairItem(NamedTuple):
    sample: torch.Tensor
    label: int
    original_image_id: int
    ssim: float
    mse: float
    l_0_norm: float
    l_2_norm: float
    l_inf_norm: float
    transformation: str
    combined_id: int


class FuzzingDatasetFair(VisionDataset):
    def __init__(
        self,
        root: str,
        model: str,
        method: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._target_folder = Path(root) / 'fuzzing' / model / method / 'crashes_rec'

        self._images = sorted(self._target_folder.glob('*.png'))
        self._infos = [f.with_suffix('.json') for f in self._images]

    def __len__(self) -> int:
        return len(self._infos)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, int, int]:
        # sample = PIL.Image.open(self._images[index]).convert("RGB")
        sample = PIL.Image.open(self._images[index])

        if self.transform is not None:
            sample = self.transform(sample)

        with self._infos[index].open("r") as fp:
            conf = json.load(fp)
        target = conf['__ground_truth_metrics__'][0]['op_class']
        qm = conf['__image_quality_metrics__'][0]

        if self.target_transform is not None:
            target = self.target_transform(target)

        _, *nums = self._infos[index].stem.split("_")
        nums = [int(n) for n in nums]

        return FuzzingFairItem(sample, target, int(qm['org_image']), qm['SSIM'], qm['MSE'], qm['l0_norm'], qm['l2_norm'], qm['linf_norm'], qm['transformations'], nums[0] << 32 | nums[1])


def run(result_path: Path, model: str, fuzzing_key: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ])

    test_set = FuzzingDataset('data', model, fuzzing_key, transform=data_transforms)
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
        write_eval_result(mutant_folder, name=f'fuzzing_{fuzzing_key}', acc=acc)

    # write_output_db(result_path, db_tuples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir')
    parser.add_argument('model')
    parser.add_argument('fuzzing_key')
    args = parser.parse_args()

    run(Path(args.result_dir), args.model, args.fuzzing_key)