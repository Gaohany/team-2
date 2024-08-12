import json
from pathlib import Path

out_folder = Path.cwd() / "data_tmp"
out_folder.mkdir(exist_ok=True, parents=True)

with Path("output.json").open("r") as fp:
    data = [json.loads(line) for line in fp.readlines()]

for i, o in enumerate(data):
    outfile = out_folder / o['dataset']

    bins = [0] * 310

    for data_point in o['dd']:
        if data_point >= 300:
            print(outfile, data_point)
        bins[round(data_point)] += 1

    content_lines = [f'{i}, {c}' for i, c in enumerate(bins)]

    outfile.write_text("\n".join(content_lines))
    print(i, outfile)
