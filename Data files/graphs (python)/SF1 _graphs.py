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
path_main = Path(r'Z:\Rebecca\lymphocyte data\B')
path_save = os.path.join(path_main, "SF1")

csv_path = path_main / 'Bcell_cyto_data.csv'
all_df = pd.read_csv(csv_path)
df_data = all_df.copy()

### normalize to control
df_data.groupby(['Media','Donor', 'Activation'])['Norm_RR'].mean()

# normalize NK cells to donor
for donor in df_data['Donor'].unique():
    pass 
    mean_control = df_data[(df_data['Donor'] == donor) & 
                         (df_data['Media'] == "Control")&
                          (df_data['Activation'] == "CD69-")
                             ]["Norm_RR"].mean()
    df_data.loc[(df_data['Donor'] == donor),'Norm_RR'] = (df_data.loc[(df_data['Donor'] == donor),'Norm_RR'] / mean_control)

#SET LIST
x_variable = "media_activation"
y_variables = [
    ### LYMPHOCYTE PAPER VARIABLES#
        "NADH_t1",
        "NADH_t2",
        "NADH_a1",
        "FAD_t1",
        "FAD_t2",
        "FAD_a1",
        "Norm_RR",
        "NADH_tm",
        "FAD_tm",
        #"Cell_Size_Pix",
        ]

#%% NAmes fro graph titles
graph_titles = {
   "NADH_t1": r"NAD(P)H $\tau_{1}$ (ps)",
   "NADH_t2": r"NAD(P)H $\tau_{2}$ (ps)",
   "NADH_a1": r"NAD(P)H $\alpha_{1}$ (%)",
   "FAD_t1": r"FAD $\tau_{1}$ (ps)",
   "FAD_t2": r"FAD $\tau_{2}$ (ps)",
   "FAD_a1": r"FAD $\alpha_{1}$ (%)",
   "Norm_RR": "Normalized Redeox Ratio\nNAD(P)H/[FAD+NAD(P)H]",
   "NADH_tm": r"NAD(P)H $\tau_{m}$ (ps)",
   "FAD_tm": r"FAD $\tau_{m}$ (ps)",
   "Cell_Size_Pix": r"Cell Size (# pixels)",
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
        palette=["#182BC7", "#CB1515", "#182BC7", "#CB1515"],
        order=("Control_CD69-",
                "Control_CD69+",
                "Activated_CD69-",
                "Activated_CD69+"),
        zorder=0)
    
    ax = sns.boxplot(
        data=all_df, 
        x=x_variable, 
        y=y_variable,
        color="black",
        showfliers=False,
        order=("Control_CD69-",
                "Control_CD69+",
                "Activated_CD69-",
                "Activated_CD69+"),
        linewidth=1,
        boxprops={"facecolor": (0, 0, 0, 0)},
        zorder=1)
    
    
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    
    add_stat_annotation(ax, data=all_df, x=x_variable, y=y_variable,
                        box_pairs=[("Control_CD69-","Control_CD69+",), 
                                    ("Activated_CD69-","Activated_CD69+"), 
                                    ( "Control_CD69-","Activated_CD69+"),                              
                                    ],
                        test='Kruskal', 
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
    #plt.savefig(os.path.join(path_save, filename), format='svg', dpi=1200, bbox_inches='tight')
    
    plt.show()





