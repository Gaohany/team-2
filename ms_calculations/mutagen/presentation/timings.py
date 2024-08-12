import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pywaffle import Waffle
from matplotlib.patches import Patch

import math

data = {
    "training": 376400.94080734253,
    "meta_medium":	116438.07885742188,
    "meta_triangle":	111723.81002807617,
    "meta_red_circle":	101443.25218200684,
    "meta_simple":	94249.25729370117,
    "meta_strange":	94023.47581481934,
    "meta_blue_circle":	62778.8571472168,
    "meta_overlay":	53818.81672668457,
    "meta_sign":	26336.18316268921,
    "meta_shadow":	15898.996479034424,
    "nbc":	23.03242015838623  + (7 * 3600 + 4 * 60),
    "kmnc":	22.65690302848816 + (5 * 3600 + 47 * 60),
    "nc":	25.224074840545654 + (1 * 3600 + 10 * 60),
    # "otrain": 2970.8542585372925,
    # "mtrain": 908.7560839653015,
    # "otrain": 2970.8542585372925 + 908.7560839653015,
    # "test": 118.52521777153015,
}

colors = [
(0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
    *sns.color_palette('Blues_d', n_colors=18)[::2],
    *sns.color_palette('Greens_d', n_colors=6)[:3],
    # *sns.color_palette('Oranges_d', n_colors=2),
]


def get_icon(v):
    if v == 'test':
        return 'play'
    if v in {'nbc', 'nc', 'kmnc', "fuzzing\nall",}:
        return 'square'
    if 'train' in v:
        return 'circle'
    return 'diamond'


rows = []
for k, v in data.items():
    if "meta_" in k:
        k = k[5:]
    rows.append((
        v / 3600, k, get_icon(k)
        # "Part", "Time", "Cat",
    ))
    print(k, math.ceil(v / 60))

training_plot = {
    "values": [data['training'] / 3600],
    'title': {'label': f'Total training time: {data["training"]/3600:0.1f}h', 'loc': 'left', 'fontsize': 16}
}

values = [
    v / 3600 for k, v in data.items()
]

fig = plt.figure(
    FigureClass=Waffle,
    values=[r[0] for r in rows],
    colors=colors,
    # icons=[r[2] for r in rows],
    rows=12,  # Outside parameter applied to all subplots, same as below
    # cmap_name="colorblind",  # Change color with cmap
    rounding_rule='ceil',  # Change rounding rule, so value less than 1000 will still have at least 1 block
    figsize=(16, 9),
# title={"label":" ", 'loc': 'left',
#         'fontdict': {
#             'fontsize': 50
#         }},
    legend = {
            'labels': [r[1] for r in rows],
            'loc': 'upper left',
    'bbox_to_anchor': (0, 1.15),
    'ncol': len(data),
    "ncols": 8,
'framealpha': 0,
    "fontsize": 14}
)


def get_group(x):
    if x == 'training':
        return x
    if x in {'nbc', 'nc', 'kmnc'}:
        return 'fuzz'
    if 'train' in x or x == 'test':
        return 'other'
    return 'meta'


sums = {}
cs = 0
for k, v in data.items():
    kk = get_group(k)
    if kk not in sums:
        sums[kk] = 0
    sums[kk] += v
    cs += v

legend_handles = [
    Patch(color=colors[0], label=f'Training: {round(sums["training"]/3600)}h'),
    Patch(color=colors[1], label=f'Metamorphic {round(sums["meta"]/3600)}h'),
    Patch(color=colors[10], label=f'Fuzzing {round(sums["fuzz"]/3600)}h'),
    # Patch(color=colors[13], label=f'Other {sums["other"]/3600:0.2f}h'),
]
plt.legend(handles=legend_handles, ncol=4, bbox_to_anchor=[0.5, 1.02], loc='lower center', fontsize=20, handlelength=.8, frameon=False)
plt.show()

fig.savefig('timins.png', dpi=200, transparent=True)

print(cs / 3600, cs / (3600 * 24))
