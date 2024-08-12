import json
from pathlib import Path
from matplotlib.patches import Patch

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = [json.loads(line) for line in Path("output.json").read_text().splitlines()]

sns.set_theme(style="whitegrid", font_scale=1.4, font='sans-serif')
fig = plt.figure(figsize=(16, 9))

x = [a['dataset'] for a in data]
y = [a['dd'] for a in data]

plt.boxplot(y, labels=x)

# Set labels and title
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Boxplot')

# Show plot
plt.show()