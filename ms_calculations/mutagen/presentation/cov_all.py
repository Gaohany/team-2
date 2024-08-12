import json
from pathlib import Path
from matplotlib.patches import Patch

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from domain import red_circle_keys

import numpy as np

k = 'avg_classes_distance'
lines = [json.loads(li) for li in Path('presentation/coverage_by_type.json').read_text().splitlines()]


values = {
#     "nc": [],
#     "nbc": [],
#     "kmnc": [],
#     "meta_red_circle": [],
#     "test": [],
}
# foo = {

# }


def to_group(v):
    if v == 'test':
        return 'test'
    if v in {'nbc', 'nc', 'kmnc', "fuzzing",}:
        return 'fuzzing'

    if v in {'simple', 'medium', "overlay", 'strange', 'shadow', 'meta\nbasic'}:
        return 'meta_basic'

    return 'meta'


rows = []

for dataset in lines:
    # if dataset['dataset'] not in {'test', 'nc', 'nbc', 'kmnc', 'meta_red_circle'}:
    #     continue
    #
    # if dataset['dataset'] in {'nc', 'nbc', 'kmnc'}:
    #     dataset['dataset'] = f"fz_{dataset['dataset']}"

    name = dataset['dataset']
    if "meta_" in name:
        name = name[5:]

    sname = name
    if name == 'fuzzing':
        sname = 'fuzzing\nall'
    if name == 'meta':
        sname = 'meta\nall'
    if name == 'red_circle':
        sname = 'red\ncircle'
    if name == 'blue_circle':
        sname = 'blue\ncircle'
    # if name not in values:
    #     continue
    values[name] = dataset['avg']

    rows.append(('Average',  dataset['avg'], sname, to_group(name)))

df = pd.DataFrame().from_records(rows, columns=["Class", "Distance", "Dataset", "Type"])


# rows = [
#
# ]
sns.set_theme(style="whitegrid", font_scale=1.4, font='sans-serif')

colors_1 = sns.color_palette("colorblind", n_colors=3)
colors_2 = sns.color_palette("dark", n_colors=3, desat=True)

colors = [
    (*colors_1[0], 1.0),
    (*colors_1[1], 1.0),
    # (*colors_1[1], 0.25),
    # (*colors_1[1], 0.25),
    (*colors_1[1], 1.0),
    (*colors_1[2], 1.0),
]

all_colors = [
    (*colors_1[0], 1.0),
    (*colors_1[0], 1.0),
    (*colors_1[0], 1.0),
    (*colors_2[0], 1.0),
    (*colors_1[1], 0.3),
    (*colors_1[1], 0.3),
    (*colors_1[1], 0.3),
    (*colors_1[1], 0.3),
    (*colors_2[1], 0.3),
    (*colors_1[1], 0.3),
    (*colors_1[1], 0.3),
    (*colors_1[1], 0.3),
    (*colors_1[1], 0.3),
    (*colors_1[1], 0.3),
    (*colors_2[1], 0.3),
    (*colors_1[2], 1.0),
]


fig = plt.figure(figsize=(16, 9))
g = sns.barplot(
    data=df,
    x="Dataset", y="Distance", palette=all_colors,
    order=[
        "nc",
        "nbc",
        "kmnc",
        'fuzzing\nall',
        "simple",
        "overlay",
        "strange",
        "shadow",
        "meta\nbasic",
        "medium",
        "triangle",
        "sign",
        "red\ncircle",
        "blue\ncircle",
        'meta\nall',
        "test",
    ],
    hue_order = [
        'fuzzing', 'meta_basic', 'meta', 'test'
    ],
    saturation=1
)
g.axhline(y=41.60488377061001, linestyle='--', zorder=-1, color=colors[3], alpha=0.75)
g.spines['top'].set_visible(False)
g.spines['right'].set_visible(False)
g.spines['left'].set_visible(False)
g.set_ylim(0, 249)
g.yaxis.labelpad = 25
# g.legend().borderpad = 10
g.set_title("Average average class distance per dataset", weight='bold')
# plt.xticks(rotation=-45, fontsize=12)
plt.xticks(fontsize=12)

legend_handles = [
    Patch(color=colors_1[0], label=f'fuzzing'),
    Patch(color=colors_1[1], label=f'metamorphic'),
    Patch(color=colors_1[2], label=f'testing'),
]
plt.legend(handles=legend_handles, ncol=4, bbox_to_anchor=[0.5, 1.02], loc='lower center', fontsize=20, handlelength=.8, frameon=False)


sns.move_legend(
    g, "upper center", ncol=7, title=None, frameon=False,
)

# plt.show()
g.figure.savefig('all_2.png', dpi=200, transparent=True)