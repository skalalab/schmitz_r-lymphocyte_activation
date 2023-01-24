#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:54:52 2022

@author: nabiki
"""

from pathlib import Path
import pandas as pd

path_data = Path(r"./Data files/ecg_feature_exports/20221214_all_data_including_new_nk_normalized_donor.csv")

df = pd.read_csv(path_data)

list_omi_parameters = [
                       'Norm_RR',
                       'NADH_tm', 
                       'FAD_tm', 
                       'NADH_a1', 
                       'NADH_t1', 
                       'NADH_t2', 
                       'FAD_a1', 
                       'FAD_t1', 
                       'FAD_t2', 
                       ] # , 'Cell_Size_Pix'


df.groupby(["Cell_Type",'Activation']).count()

#%% B Cells


df_b_cells = df[df['Cell_Type'] == 'B-Cells'][list_omi_parameters + ['Activation' , 'Donor']]

df_b_cells['Activation'] = df_b_cells['Activation'].map({'CD69-':0, 'CD69+':1})

dict_donors = {donor:idx for idx, donor in enumerate(df_b_cells['Donor'].unique(), start=1)}
df_b_cells['Donor'] = df_b_cells['Donor'].map(dict_donors)

df_b_cells.to_csv(path_data.parent / "B_cells.csv", index=False)


#%% NK Cells

df_nk_cells = df[df['Cell_Type'] == 'NK-Cells'][list_omi_parameters + ['Activation' , 'Donor']]

df_nk_cells['Activation'] = df_nk_cells['Activation'].map({'CD69-':0, 'CD69+':1})

dict_donors = {donor:idx for idx, donor in enumerate(df_nk_cells['Donor'].unique(), start=4)}
df_nk_cells['Donor'] = df_nk_cells['Donor'].map(dict_donors)

df_nk_cells.to_csv(path_data.parent / "NK_cells.csv", index=False)


#%% T Cells

df_t_cells = df[df['Cell_Type'] == 'T-cells'][list_omi_parameters + ['Activation' , 'Donor']]

df_t_cells['Activation'] = df_t_cells['Activation'].map({'CD69-':0, 'CD69+':1})

dict_donors = {donor:idx for idx, donor in enumerate(df_t_cells['Donor'].unique(), start=7)}
df_t_cells['Donor'] = df_t_cells['Donor'].map(dict_donors)

df_t_cells.to_csv(path_data.parent / "T_cells.csv", index=False)

#%% Combine DF for all cells 

cell_type_id = {"B-cells": 0,
 "NK-cells" : 1,
 "T-cells" : 3
 }
df_b_cells['Cell_Type'] = cell_type_id["B-cells"]
df_nk_cells['Cell_Type'] = cell_type_id["NK-cells"]
df_t_cells['Cell_Type'] = cell_type_id["T-cells"]

df_all_data = pd.concat([df_b_cells, df_nk_cells, df_t_cells])


df_all_data.to_csv(path_data.parent / "all_data.csv", index=False)




