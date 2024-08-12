import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from domain import red_circle_keys

red_circle_keys = [1, 2, 3]

import numpy as np

k = 'avg_classes_distance'
lines = [json.loads(li) for li in Path('presentation/coverage_single.json').read_text().splitlines()]


values = {
    "nc": [],
    # "nbc": [],
    # "kmnc": [],
    "red_circle": [],
    "overlay": [],
    "test": [],
}
# foo = {

# }

rows = []

for dataset in lines:
    name = dataset['dataset']

    if "meta_" in name:
        name = name[5:]

    if name not in values:
        continue

    for i, e in enumerate(dataset[k]):
        if i not in red_circle_keys:
            continue

        values[name].append(e)

        rows.append(
            [i, e, name]
        )

print(rows)

df = pd.DataFrame().from_records(rows, columns=["Class", "Distance", "Dataset"])

sns.set_theme(style="whitegrid", font_scale=1.4, font='sans-serif')

colors_1 = sns.color_palette("colorblind", n_colors=4)

colors = [
    (*colors_1[0], 1.0),
    # (*colors_1[3], 1.0),
    (*colors_1[1], 1.0),
    (*colors_1[2], 1.0),
]

fig = plt.figure(figsize=(16, 9))
g = sns.barplot(
    data=df,
    x="Class", y="Distance", hue="Dataset",  palette=colors,
    hue_order=["nc", "red_circle", "test"],
    saturation=1
)

g.spines['top'].set_visible(False)
g.spines['right'].set_visible(False)
g.spines['left'].set_visible(False)
g.set_ylim(0, 249)
g.yaxis.labelpad = 25
g.legend().borderpad = 10
g.set_title("Average distance per class (red circle signs)", weight='bold')

sns.move_legend(
    g, "upper center", ncol=len(red_circle_keys), title=None, frameon=False,
)

# plt.show()
# g.figure.savefig('red_circle.png', dpi=200, transparent=True)
