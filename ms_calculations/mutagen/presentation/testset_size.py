import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    "nc":	2557,
    "kmnc":	2323,
    "nbc":	2324,
    "simple":	189450,
    "medium":	73529,
    "overlay":	56827,
    "strange":	164190,
    "shadow":	25260,
    "triangle":	874,
    "sign":	188,
    "red\ncircle":	3252,
    "blue\ncircle":	358,
    "test":	12630,
    "mtrain":	353390,
    "otrain":	353390,
}

ata = {
    "nc":	"fuzzing",
    "kmnc":	"fuzzing",
    "nbc":	"fuzzing",
    "simple":	"meta",
    "medium":	"meta",
    "overlay":	"meta",
    "strange":	"meta",
    "shadow":	"meta",
    "triangle": "meta",
    "sign": "meta",
    "red\nircle":	"meta",
    "blue\ncircle": "meta",
    "test":	"test",
    "mtrain":	"train",
    "otrain":	"train",
}

colors_1 = sns.color_palette("colorblind", n_colors=5)
cdict = {
    'fuzzing': colors_1[0],
    'meta': colors_1[1],
    'test': colors_1[2],
    'train': colors_1[3],
}

sns.set_theme(style="whitegrid", font_scale=1.4, font='sans-serif')
fig = plt.figure(figsize=(17, 8))

g = sns.barplot(
    x = data.keys(),
    y = data.values(),
    palette=cdict,
    hue=ata.values(),
    # log_scale=True,
)
g.set(yscale="log")

g.spines['top'].set_visible(False)
g.spines['right'].set_visible(False)
g.spines['left'].set_visible(False)
# g.set_ylim(0, 249)
g.yaxis.labelpad = 25
# g.legend().borderpad = 10
g.set_title("Number of samples evaluated per dataset (non-mutated model)", weight='bold')

sns.move_legend(
    g, "upper center", ncol=len(cdict), title=None, frameon=False,
)

plt.show()
g.figure.savefig('eval_num.png', dpi=200, transparent=True)