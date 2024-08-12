import argparse
import torch
from pathlib import Path


def do_single(data: torch.tensor):
    output, counts = torch.unique(data, return_counts=True)
    longest = max(counts)

    for lbl, count in zip(output.numpy(), counts.numpy()):
        pct = count / len(data)
        rel_pct = count / longest
        print(f"{lbl:02d} {count: 6d} {pct:.2%} |{int(rel_pct * 100) * '=':<100}|")
    # print(counts.numpy().tolist())


def compare(data_left: torch.tensor, data_right: torch.tensor):
    output_left, counts_left = torch.unique(data_left, return_counts=True)
    output_right, counts_right = torch.unique(data_right, return_counts=True)
    longest = max(max(counts_left), max(counts_right))

    for lbl_l, lbl_r, count_left, count_right in zip(output_left, output_right, counts_left, counts_right):
        assert lbl_l == lbl_r
        if count_left == count_right:
            continue

        same = int(min(count_left, count_right) / longest * 100)
        diff = int(abs(count_left - count_right) / longest * 100)

        bar = ['=' for _ in range(same)]
        bar.append('\033[31m' if count_left > count_right else '\033[32m')
        bar.extend('=' for _ in range(diff))
        bar.append('\033[39m')

        print(f"{lbl_l:02d} {count_left: 6d} {count_right: 6d} |{''.join(bar):<110}|")


def main(data_file_left: Path, data_file_right: Path):
    if data_file_left.name == 'data.link':
        data_file_left = Path(data_file_left.read_text())
    if data_file_right.name == 'data.link':
        data_file_right = Path(data_file_right.read_text())

    data_left = torch.load(data_file_left)
    data_right = torch.load(data_file_right)
    compare(data_left['lbl_train'], data_right['lbl_train'])
    # do_single(data['lbl_train'])
    # do_single(data['lbl_train'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_left')
    parser.add_argument('data_right')
    args = parser.parse_args()

    main(Path(args.data_left), Path(args.data_right))
