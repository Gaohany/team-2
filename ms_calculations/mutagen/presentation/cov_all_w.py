
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from domain import red_circle_keys

import numpy as np

k = 'avg_classes_distance'
lines = [json.loads(li) for li in Path('presentation/coverage_single.json').read_text().splitlines()]


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


weights = ['1173', '20950', '21276', '13107', '18577', '17405', '3289', '13357', '13073', '13659', '19106', '12272', '20011', '20574', '6876', '5325', '3285', '10136', '2990', '1198', '19753', '2091', '2673', '1466', '1488', '11033', '1196', '2678', '2370', '2962', '4160', '1788', '14011', '5077', '1491', '4456', '1786', '3572', '6846', '1482', '5971', '3278', '10982']
weights = [int(w) for w in weights]

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
    # values[name] = dataset['avg']
    m = np.average(dataset['avg_classes_distance'], weights=weights)
    print(sname, m)
    rows.append(('Average', m, sname, to_group(name)))


df = pd.DataFrame().from_records(rows, columns=["Class", "Distance", "Dataset", "Type"])


# rows = [
#
# ]
sns.set_theme(style="whitegrid", font_scale=1.4, font='sans-serif')

colors_1 = sns.color_palette("colorblind", n_colors=3)
colors_2 = sns.color_palette("pastel", n_colors=3, desat=True)

colors = [
    (*colors_1[0], 1.0),
    (*colors_1[1], 1.0),
    # (*colors_1[1], 0.25),
    # (*colors_1[1], 0.25),
    (*colors_1[1], 1.0),
    (*colors_1[2], 1.0),
]

fig = plt.figure(figsize=(16, 9))
g = sns.barplot(
    data=df,
    x="Dataset", y="Distance", hue="Type",  palette=colors,
    order=[
        "nc",
        "nbc",
        "kmnc",
        # 'fuzzing\nall',
        "simple",
        "medium",
        "overlay",
        "strange",
        "shadow",
        # "meta\nbasic",
        "triangle",
        "sign",
        "red\ncircle",
        "blue\ncircle",
        # 'meta\nall',
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
g.set_title("Weighted mean of average class distance per dataset", weight='bold')
# plt.xticks(rotation=-45, fontsize=12)
plt.xticks(fontsize=12)
sns.move_legend(
    g, "upper center", ncol=7, title=None, frameon=False,
)

# plt.show()
# g.figure.savefig('all_w.png', dpi=200, transparent=True)