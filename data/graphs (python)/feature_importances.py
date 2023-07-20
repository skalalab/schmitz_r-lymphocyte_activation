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


#%% Section 2 - Read in and set up dataframe 

#Read in dataframe    
path_main = Path(r'C:\Users\jriendeau\Documents\GitHub\schmitz_r-lymphocyte_activation\figures\SF8')
csv_path = path_main / 'SF8_B_feature_importances.csv'

saveas = os.path.abspath(csv_path).split('.')[0]

all_df = pd.read_csv(csv_path)
df_data = all_df.copy()

#SET LIST
data = df_data.data
labels = df_data.labels

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

#%% GRAPH DATA

colors = sns.color_palette('viridis_r', len(labels))

#pie chart
plt.pie(data, colors = colors)


legends = []
for l in labels:
    legends.append(titles.get(l,''))
    
labels = [f'{s:0.2f}% {l}' for l, s in zip(legends, data)]

plt.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize = 12)

plt.savefig(saveas+".svg", format='svg', dpi=1200, bbox_inches='tight')

plt.show()











