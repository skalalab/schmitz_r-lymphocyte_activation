# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:17:46 2022

@author: econtrerasguzman
"""

from pathlib import Path
import re
import  tifffile
from skimage.measure import regionprops_table
from skimage.morphology import label
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
import numpy as np

# path_dataset = Path(r"Z:\Rebecca\Immune Cell Projects\Lymphocyte Paper\Paper data\NK cells (Donors 4-6)\20220215 activated nk\Data\Act + Qui masks")
# path_dataset = Path(r"Z:\Rebecca\Immune Cell Projects\Lymphocyte Paper\Paper data\NK cells (Donors 4-6)\20220303 activated nk (from Capitini Lab)\Data\Act + Qui masks")
path_dataset = Path(r"Z:\Rebecca\Immune Cell Projects\Lymphocyte Paper\Paper data\NK cells (Donors 4-6)\20220309 act nk\Data\Act + Qui masks")

list_files = list(map(str,list(path_dataset.glob("*"))))


# activation = 'activated'
activation = 'quiescent'

list_files = list(filter(re.compile(f".*{activation}_cell.*").search, list_files)) 

list_areas = []
for path_mask in list_files:
    pass
    mask = tifffile.imread(path_mask)
    # plt.imshow(mask)
    
    dict_props = regionprops_table(mask, properties=('label','area'))
    list_areas += list(dict_props['area'])
    
plt.title(f"{Path(path_mask).parent.parent.parent.stem} | {activation} \nmean: {np.mean(list_areas):.3f}")
plt.hist(list_areas, histtype='step')
plt.show()

