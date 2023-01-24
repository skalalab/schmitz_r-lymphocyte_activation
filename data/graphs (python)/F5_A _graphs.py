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
path_main = Path(r'C:\Users\jriendeau\Documents\GitHub\schmitz_r-lymphocyte_activation\Data files\UMAPs, boxplots, ROC curves (Python)')
path_save = Path(r'C:\Users\jriendeau\Documents\GitHub\schmitz_r-lymphocyte_activation\figures\F5')

csv_path = path_main / 'JR.csv'
all_df = pd.read_csv(csv_path)
df_data = all_df.copy()

order=("B-Cells: CD69-",
        "B-Cells: CD69+",
        "NK-Cells: CD69-",
        "NK-Cells: CD69+",
        "T-Cells: CD69-",
        "T-Cells: CD69+")

#SET LIST
x_variable = "Type_Activation"
y_variables = [
    ### LYMPHOCYTE PAPER VARIABLES#
        # "NADH_t1",
        # "NADH_t2",
        "NADH_a1",
        # "FAD_t1",
        # "FAD_t2",
        # "FAD_a1",
        "Norm_RR",
        "NADH_tm",
        # "FAD_tm",
        # "Cell_Size_Pix",
        # "Donor
    ]


graph_titles = {
   "NADH_a1": r"NAD(P)H $\alpha_{1}$ (%)", 
   "Norm_RR": "Normalized Redox Ratio\nNAD(P)H/[FAD+NAD(P)H]", 
   "NADH_tm": r"NAD(P)H $\tau_{m}$ (%)", 
   }
   
#GRAPH DATA
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
        palette=["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"],
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
                        box_pairs=[("B-Cells: CD69-","B-Cells: CD69+",), 
                                    ("NK-Cells: CD69-","NK-Cells: CD69+"), 
                                    ( "T-Cells: CD69-","T-Cells: CD69+"),
                                    ("B-Cells: CD69-","T-Cells: CD69-"),
                                    ("B-Cells: CD69-","NK-Cells: CD69-"),
                                    ("NK-Cells: CD69-","T-Cells: CD69-"),
                                    
                                    ],
                        
                        test='Kruskal', 
                        text_format='star', 
                        loc='outside', 
                        verbose=1)
    
    
                                                      
    ax.set_xticklabels(["CD69-\nB", 
                        "CD69+\nB", 
                        "CD69-\nNK", 
                        "CD69+\nNK", 
                        "CD69-\nT", 
                        "CD69+\nT"], fontsize = 15)
    
    plt.ylabel(graph_titles.get(y_variable, ''))
    ax.tick_params(bottom=True, top=False, left=True, right=False)
    plt.xlabel("")
    
    plt.legend([],[], frameon=False)
    
    filename = f"{y_variable}.svg"
    plt.savefig(os.path.join(path_save, filename), format='svg', dpi=1200, bbox_inches='tight')
    
    plt.show()














