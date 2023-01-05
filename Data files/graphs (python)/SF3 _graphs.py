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
from statannot import add_stat_annotation #https://pypi.org/project/statannot/
import os


#%% Section 2 - Read in and set up dataframe 

#Read in dataframe    
path_main = Path(r'Z:\Rebecca\lymphocyte data\NK cell repeats')
path_save = os.path.join(path_main, "SF3")

csv_path = path_main / 'NK_donors_final_dec02.csv'
all_df = pd.read_csv(csv_path)
df_data = all_df.copy()

### normalize to control
df_data.groupby(['Media','Donor', 'Activation'])['rr.mean'].mean()

# normalize NK cells to donor
for donor in df_data['Donor'].unique():
    pass 
    mean_control = df_data[(df_data['Donor'] == donor) & 
                         (df_data['Media'] == "Control")&
                          (df_data['Activation'] == "CD69-")
                             ]["rr.mean"].mean()
    df_data.loc[(df_data['Donor'] == donor),'rr.mean'] = (df_data.loc[(df_data['Donor'] == donor),'rr.mean'] / mean_control)

order=("Control_CD69-",
        "Control_CD69+",
        "Activated_CD69-",
        "Activated_CD69+")

#SET LIST
x_variable = "media_activation"
y_variables = [
    ### LYMPHOCYTE PAPER VARIABLES#
        "n.t1.mean",
        "n.t2.mean",
        "n.a1.mean",
        "f.t1.mean",
        "f.t2.mean",
        "f.a1.mean",
        "rr.mean",
        "n.tm.mean",
        "f.tm.mean",
        #"npix",
        ]

#%% NAmes fro graph titles
graph_titles = {
   "n.t1.mean": r"NAD(P)H $\tau_{1}$ (ps)",
   "n.t2.mean": r"NAD(P)H $\tau_{2}$ (ps)",
   "n.a1.mean": r"NAD(P)H $\alpha_{1}$ (%)",
   "f.t1.mean": r"FAD $\tau_{1}$ (ps)",
   "f.t2.mean": r"FAD $\tau_{2}$ (ps)",
   "f.a1.mean": r"FAD $\alpha_{1}$ (%)",
   "rr.mean": "Normalized Redeox Ratio\nNAD(P)H/[FAD+NAD(P)H]",
   "n.tm.mean": r"NAD(P)H $\tau_{m}$ (ps)",
   "f.tm.mean": r"FAD $\tau_{m}$ (ps)",
   "npix": r"Cell Size (# pixels)",
   # "Donor": r"Donor",
   }

#%% define function

for y_variable in y_variables:
    
    sns.set(font_scale=1.5)
    sns.set_style(style='white')
    
    sns.swarmplot(
        data=all_df, 
        x=x_variable, 
        y=y_variable,
        s=1.15,
        #hue="Activation",
        #dodge=True,
        palette=["#182BC7", "#CB1515", "#182BC7", "#CB1515"],
        order=order,
        zorder=0)
    
    ax = sns.boxplot(
        data=all_df, 
        x=x_variable, 
        y=y_variable,
        color="black",
        #hue="Activation",
        showfliers=False,
        order=order,
        linewidth=1,
        boxprops={"facecolor": (0, 0, 0, 0)},
        zorder=1)
    
    
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    
    add_stat_annotation(ax, data=all_df, x=x_variable, y=y_variable, order = order,
                        box_pairs=[("Control_CD69-","Control_CD69+",), 
                                    ("Activated_CD69-","Activated_CD69+"), 
                                    ( "Control_CD69-","Activated_CD69+"),                              
                                    ],
                        test='Mann-Whitney', 
                        text_format='star', 
                        loc='outside', 
                        verbose=1)
    
    
                                                      
    ax.set_xticklabels(["CD69-\nControl", 
                        "CD69+\nControl", 
                        "CD69-\nActivated", 
                        "CD69+\nActivated"], fontsize = 15)
    
    #ymax = max(df_data[y_variable].tolist())
   # plt.yticks(np.arange(0, 100, 10))
    plt.ylabel(graph_titles.get(y_variable, ''))

    plt.xlabel("")
    
    plt.legend([],[], frameon=False)
    
    filename = f"{y_variable}.svg"
    plt.savefig(os.path.join(path_save, filename), format='svg', dpi=1200, bbox_inches='tight')
    
    plt.show()





