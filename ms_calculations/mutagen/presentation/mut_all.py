import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from domain import red_circle_keys

import numpy as np

k = 'avg_classes_distance'
lines = [json.loads(li) for li in Path('presentation/md_deepcrime_all.json').read_text().splitlines()]


values = {}



def to_group(v):
    if v == 'test':
        return 'test'
    if v in {'nbc', 'nc', 'kmnc', "fuzzing\nall",}:
        return 'fuzzing'

    if name in {'simple', "overlay", 'strange', 'shadow', 'meta\nbasic', }:
        return 'meta_basic'

    return 'meta'


rows = []

for dataset in lines:
    name = dataset['dataset']
    if "meta_" in name:
        name = name[5:]

    if name == 'red_circle':
        name = 'red\ncircle'

    # Aggregate score for just simple meta relations
    # if name == 'meta\nall':
    #     dataset['ms'] = 0.7567567567567568

    rows.append(('MS',  dataset['ms'], name, to_group(name)))

rows.append(("MS", 0.0, 'blue\ncircle', 'meta'))

df = pd.DataFrame().from_records(rows, columns=["Class", "Mutation score", "Dataset", "Type"])


sns.set_theme(style="whitegrid", font_scale=1.4, font='sans-serif')

fig = plt.figure(figsize=(16, 9))


colors_1 = sns.color_palette("colorblind", n_colors=3)
colors_2 = sns.color_palette("dark", n_colors=3, desat=True)

colors = [
    (*colors_1[0], 1.0),
    (*colors_1[1], 1.0), #0.25
    (*colors_1[1], 1.0),
    (*colors_1[2], 1.0),
]
# colors = [
#     (*colors_1[0], 1.0),
#     (*colors_1[1], 1.0),
#     (*colors_1[2], 1.0),
# ]

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

g = sns.barplot(
    data=df,
    x="Dataset", y="Mutation score", palette=all_colors,
    order=[
        "nc",
        "nbc",
        "kmnc",
        "fuzzing\nall",
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
        "meta\nall",
        "test",
    ],
    hue_order = [
        'fuzzing', 'meta_basic', 'meta', 'test'
    ],
    saturation=1
)
g.axhline(y=0.4864864864864865, linestyle='--', zorder=-1, color=colors[3], alpha=0.75)
g.spines['top'].set_visible(False)
g.spines['right'].set_visible(False)
g.spines['left'].set_visible(False)
g.set_ylim(0, 1.19)
g.yaxis.labelpad = 25
g.legend().borderpad = 10
g.set_title("DeepCrime mutation score for each dataset", weight='bold')
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
g.figure.savefig('ms_all_2.png', dpi=200, transparent=True)

# print(*colors_1[0])