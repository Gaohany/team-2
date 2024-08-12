import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pywaffle import Waffle
from matplotlib.patches import Patch

import math

data = {
    "train": 57.5797490850091,
    "mutants": 6.756895784288645,
    "eval": 84.31083480920643,
}
sns.set_theme(style="whitegrid", font_scale=1.4, font='sans-serif')
fig = plt.figure(
    FigureClass=Waffle,
    values=list(data.values()),
    colors=sns.color_palette('colorblind', n_colors=3),
    # icons=[r[2] for r in rows],
    rows=10,  # Outside parameter applied to all subplots, same as below
    # cmap_name="colorblind",  # Change color with cmap
    rounding_rule='ceil',  # Change rounding rule, so value less than 1000 will still have at least 1 block
    figsize=(16, 9),
    legend = {
            'labels': [f"{k}: {v:.1f} GiB" for k, v in data.items()],
            'loc': 'upper left',
                'bbox_to_anchor': (0.04, 1.1),
    "ncols": 8,
'framealpha': 0,
    "fontsize": 20}
)

# plt.show()
fig.savefig('file_sizes.png', dpi=200, transparent=True)
print(sum(data.values()))