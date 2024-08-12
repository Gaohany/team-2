import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from domain import red_circle_keys

import numpy as np

keys = [
    "AddTrainingNoise_0",
    "ChangeActivationTanH_0",
    "ChangeActivationTanH_1",
    "ChangeActivationTanH_2",
    "ChangeActivationTanH_3",
    "ChangeActivation_0",
    "ChangeActivation_1",
    "ChangeEpochs_0",
    "ChangeKernelSize_0",
    "ChangeKernelSize_1",
    "ChangeLabels_0",
    "ChangeLabels_1",
    "ChangeLayerSize_0",
    "ChangeLoss_0",
    "ChangeMoreLabels_0",
    "ChangeMoreLabels_1",
    "ChangeMoreLabels_2",
    "ChangeMoreLabels_3",
    "ChangeMostLabels_0",
    "ChangeMostLabels_1",
    "ChangeOptimizer_0",
    "ChangeOptimizer_1",
    "ChangeScheduler_0",
    "DecreaseBatchSize_0",
    "DecreaseLearningRate_0",
    "HighOrder1_0",
    "HighOrder2_0",
    "HighOrder3_0",
    "HighOrderData2_0",
    "HighOrderData2_1",
    "HighOrderData_0",
    "IncreaseBatchSize_0",
    "IncreaseLearningRate_0",
    "MakeClassesOverlap_0",
    "RemoveBatchNorm_0",
    "RemoveBatchNorm_1",
    "RemoveBias_0",
    "RemoveBias_1",
    "RemoveBias_2",
    "RemoveCall_0",
    "RemoveSamples_0",
    "RemoveSamples_1",
    "RemoveZeroGrad_0",
]

kv = {k: i for i, k in enumerate(keys)}

k = 'avg_classes_distance'
lines = [json.loads(li) for li in Path('presentation/kill_table.json').read_text().splitlines()]


rows = []

for dataset in lines:
    name = dataset['dataset']
    if "meta_" in name:
        name = name[5:]
    # rows.append((name, dataset['sut_name'], dataset['killed']))
    rows.append({
        "Dataset": name,
        "MutationOp": dataset['sut_name'],
        "Killed": dataset['killed']
    })

# rows.sort(key=lambda x: (x['Dataset'], kv[x['MutationOp']]))

df = pd.DataFrame(rows)

pivot_table = df.pivot_table(index='Dataset', columns='MutationOp', values='Killed', fill_value=False, sort=False)

print(df.pivot_table(index='MutationOp', columns='Dataset', values='Killed', fill_value=False, sort=False).to_latex())

colors = [
    (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
     (0.8352941176470589, 0.3686274509803922, 0.0),
]

# Create the heatmap using Seaborn
plt.figure(figsize=(16, 9))
g = sns.heatmap(
    pivot_table,
    cmap=colors,
    cbar_kws={'label': 'Killed'},
    cbar=False,
    square=True,
    linewidths=.75)
# plt.title('Killed Status Heatmap')

legend_handles = [Patch(color=colors[True], label='Killed'),  # red
                  Patch(color=colors[False], label='Not Killed')]  # green
plt.legend(handles=legend_handles, ncol=2, bbox_to_anchor=[0.5, 1.02], loc='lower center', fontsize=14, handlelength=.8)
plt.xticks(fontsize=12)
# plt.tight_layout()

# plt.show()

# g.spines['top'].set_visible(False)
# g.spines['right'].set_visible(False)
# g.spines['left'].set_visible(False)
# g.set_ylim(0, 249)
# g.yaxis.labelpad = 25
# g.legend().borderpad = 10
# g.set_title("Average average class distance per dataset", weight='bold')

# plt.xticks(fontsize=12)
# sns.move_legend(
#     g, "upper center", ncol=7, title=None, frameon=False,
# )
#
# plt.show()
# g.figure.savefig('killing.png', dpi=200, transparent=True)
#