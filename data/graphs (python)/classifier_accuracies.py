# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 07:37:20 2022

@author: jriendeau
"""
#%% Section 1 - Import required packages

from pathlib import Path 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


#%% Section 2 - Read in and set up dataframe 

#Read in dataframe    
path_main = Path(r'C:\Users\jriendeau\Documents\GitHub\schmitz_r-lymphocyte_activation\figures\SF7')
csv_path = path_main / 'SF7_B_classifier_accuracies.csv'

saveas = os.path.abspath(csv_path).split('.')[0]

all_df = pd.read_csv(csv_path)
df_data = all_df.copy()

#SET LIST
data = df_data.data
labels = list(df_data.labels)


titles = {
   "NADH_t1": r"NAD(P)H $\tau_{1}$",
   "NADH_t2": r"NAD(P)H $\tau_{2}$",
   "NADH_a1": r"NAD(P)H $\alpha_{1}$",
   "FAD_t1": r"FAD $\tau_{1}$",
   "FAD_t2": r"FAD $\tau_{2}$",
   "FAD_a1": r"FAD $\alpha_{1}$",
   "Norm_RR": "Redox Ratio",
   "NADH_tm": r"NAD(P)H $\tau_{m}$",
   "FAD_tm": r"FAD $\tau_{m}$",
   }

for label in labels:
    if label in titles.keys():
        labels[labels.index(label)] = (titles.get(label,''))
        


#%% GRAPH DATA

colors = sns.color_palette('viridis_r', len(labels))

#pie chart
bars = plt.bar(labels, data*100, color = colors)

# access the bar attributes to place the text in the appropriate location
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + 1, f'{yval:0.1f}%')

plt.gcf().autofmt_xdate()

plt.rcParams.update({'font.size': 12})

ax = plt.gca()
ax.tick_params(axis="y", direction="out", reset=True, right=False)
plt.yticks(np.arange(0, 110, 10))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig(saveas+".svg", format='svg', dpi=1200, bbox_inches='tight')

plt.show()











