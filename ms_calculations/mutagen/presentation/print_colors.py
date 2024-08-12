import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

colors_1 = sns.color_palette("colorblind", n_colors=3)
colors_2 = sns.color_palette("dark", n_colors=3, desat=True)

def foo(c):
    print(f"\definecolor{{}}{{rgb}}{{{','.join(str(a) for a in c)}}}")

for c in colors_1:
    foo(c)

for c in colors_2:
    foo(c)
