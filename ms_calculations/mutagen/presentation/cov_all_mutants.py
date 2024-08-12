import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from domain import red_circle_keys

import numpy as np

k = 'avg_classes_distance'
# lines = [json.loads(li) for li in Path('presentation/mutants_cov_otrain.json').read_text().splitlines()]
lines = [json.loads(li) for li in Path('presentation/mutants_cov.json').read_text().splitlines()]
#

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

    if name in {'simple', 'medium', "overlay", 'strange', 'shadow', 'meta'}:
        return 'meta_aug'

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
        sname = 'fuzzing'
    if name == 'meta':
        sname = 'meta'
    if name == 'red_circle':
        sname = 'red\ncircle'
    if name == 'blue_circle':
        sname = 'blue\ncircle'
    # if name not in values:
    #     continue
    values[name] = dataset['avg']

    rows.append((dataset['sut_name'],  dataset['avg'], sname, to_group(name)))

df = pd.DataFrame().from_records(rows, columns=["SUT", "Distance", "Dataset", "Type"])


# rows = [
#
# ]
sns.set_theme(style="whitegrid", font_scale=1.2, font='sans-serif')

colors_1 = sns.color_palette("colorblind", n_colors=3)
colors_2 = sns.color_palette("pastel", n_colors=3, desat=True)

colors = [
    (*colors_1[0], 1.0),
    (*colors_1[1], 0.25),
    (*colors_1[1], 0.25),
    (*colors_1[2], 1.0),
]

fig = plt.figure(figsize=(16, 9))
g = sns.barplot(
    data=df,
    x="SUT", y="Distance", hue="Dataset",  palette="colorblind",
    # order=[
    #     "nc",
    #     "nbc",
    #     "kmnc",
    #     'fuzzing\nall',
    #     "simple",
    #     "medium",
    #     "overlay",
    #     "strange",
    #     "shadow",
    #     "triangle",
    #     "sign",
    #     "red\ncircle",
    #     "blue\ncircle",
    #     'meta\nall',
    #     "test",
    # ],
    # hue_order = [
    #     'fuzzing', 'meta_aug', 'meta', 'test'
    # ],
    saturation=1
)
# g.axhline(y=41.60488377061001, linestyle='--', zorder=-1, color=colors[3], alpha=0.75)
g.spines['top'].set_visible(False)
g.spines['right'].set_visible(False)
g.spines['left'].set_visible(False)
g.set_ylim(0, 699)
g.yaxis.labelpad = 25
# g.legend().borderpad = 10
g.set_title("Average average class distance for different mutants", weight='bold')
# plt.xticks(rotation=-45, fontsize=12)
plt.xticks(fontsize=12)
sns.move_legend(
    g, "upper center", ncol=7, title=None, frameon=False,
)

plt.show()
g.figure.savefig('m_all_mtrain.png', dpi=200, transparent=True)
