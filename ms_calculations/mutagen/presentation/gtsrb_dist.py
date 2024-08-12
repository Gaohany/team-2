import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = [
    1200,
    21300,
    21600,
    13200,
    18900,
    17700,
    3300,
    13500,
    13200,
    13800,
    19200,
    12300,
    20100,
    20700,
    6900,
    5400,
    3300,
    10200,
    11100,
    1200,
    2700,
    2400,
    3000,
    4200,
    1800,
    14100,
    5100,
    1500,
    4500,
    1800,
    3600,
    6900,
    1500,
    5990,
    3300,
    11100,
    3000,
    1200,
    19800,
    2100,
    2700,
    1500,
    1500,
]

data = [d//10 for d in data]

types = [
    'red_circle',#= 0
    'red_circle',#= 1
    'red_circle',#= 2
    'red_circle',#= 3
    'red_circle',#= 4
    'red_circle',#= 5
    'white_circle' ,#= 6
    'red_circle' ,#= 7
    'red_circle' ,#= 8
    'red_circle' ,#= 9
    'red_circle' ,#= 10
    'triangle' ,#= 11
    'other' ,#= 12
    'other' ,#= 13
    'other' ,#= 14
    'red_circle' ,#= 15
    'red_circle' ,#= 16
    'red_circle' ,#= 17
    'triangle' ,#= 18
    'triangle' ,#= 19
    'triangle' ,#= 20
    'triangle' ,#= 21
    'triangle' ,#= 22
    'triangle' ,#= 23
    'triangle' ,#= 24
    'triangle' ,#= 25
    'triangle' ,#= 26
    'triangle' ,#= 27
    'triangle' ,#= 28
    'triangle' ,#= 29
    'triangle' ,#= 30
    'triangle' ,#= 31
    'white_circle' ,#= 32
    'blue_circle' ,#= 33
    'blue_circle' ,#= 34
    'blue_circle' ,#= 35
    'blue_circle' ,#= 36
    'blue_circle' ,#= 37
    'blue_circle' ,#= 38
    'blue_circle' ,#= 39
    'blue_circle' ,#= 40
    'white_circle' ,#= 41
    'white_circle' ,#= 42
]

colors_1 = sns.color_palette("colorblind", n_colors=5)
cdict = {
    'blue_circle': colors_1[0],
    'red_circle': colors_1[1],
    'triangle': colors_1[3],
    'other': colors_1[2],
    'white_circle': colors_1[4],
}

clist = [
    cdict[e] for e in types
]

sns.set_theme(style="whitegrid", font_scale=1.4, font='sans-serif')
fig = plt.figure(figsize=(16, 9))

g = sns.barplot(
    x = range(43),
    y = data,
    palette=cdict,
    hue=types,
)

g.spines['top'].set_visible(False)
g.spines['right'].set_visible(False)
g.spines['left'].set_visible(False)
# g.set_ylim(0, 249)
g.yaxis.labelpad = 25
# g.legend().borderpad = 10
g.set_title("Number of signs per class", weight='bold')

sns.move_legend(
    g, "upper center", ncol=len(cdict), title=None, frameon=False,
)

plt.show()
g.figure.savefig('gtsrb.png', dpi=200, transparent=True)
