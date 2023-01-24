
from pathlib import Path
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import pandas as pd
import holoviews as hv
from holoviews import  opts
hv.extension('bokeh')

df = pd.read_csv(r"Data files/UMAPs, boxplots, ROC curves (Python)/NK_donors_final_dec02.csv")

df['Group'] = df['Group'].replace("Control", "Unstimulated")

# df_new = pd.read_csv(r"Data files/UMAPs, boxplots, ROC curves (Python)/NK data donor M 20221101.csv")
# df_new['Group'] = df_new['Group'].replace("Control", "Unstimulated")

df = df.rename(columns={      'n.t1.mean' : 'NADH_t1', 
                              'n.t2.mean' : 'NADH_t2', 
                              'n.a1.mean' : 'NADH_a1', 
                              'n.tm.mean' : 'NADH_tm', 
                              'f.t1.mean' : 'FAD_t1', 
                              'f.t2.mean' : 'FAD_t2',
                              'f.a1.mean' : 'FAD_a1', 
                              'rr.mean' : 'Norm_RR', 
                              'f.tm.mean' : 'FAD_tm', 
                              'npix' : 'Cell_Size_Pix'
                              })


# df = pd.concat([df_original, df_new], axis=0)

print(df.groupby(['Group','Donor']).count())

df.groupby(['Group','Activation','Donor'])['Cell_Size_Pix'].mean()


#%% 40x vs 100x area comparison


# from skimage.measure import regionprops
# from skimage.morphology import label
# import tifffile

# path_40x_100x_data = Path(r"/mnt/Z/Jeremiah/40x_vs_100x_NK_NADH")

# list_masks = list(path_40x_100x_data.glob("*_photons_cytomask.tiff"))

# df_masks_size = pd.DataFrame(columns=['filename', 'roi', 'Group', 'Donor', 'Activation', 'Cell_Size_Pix' ])

# for path_mask in list_masks:
#     pass
    
#     mask = tifffile.imread(path_mask)
#     props = regionprops(mask)
    
#     plt.title(path_mask.name)
#     plt.imshow(mask)
#     plt.show()
#     for roi in props:
#         filename = f'{path_mask.stem}_{roi.label}'
        

#         df_temp = pd.DataFrame({
#             'filename' : [filename], 
#             'roi' : [roi.label], 
#             'Group' : ['Unstimulated' if 'ctrl' in filename else "Activated" ], # dish 
#             'Donor' : ['M-40x' if '40x' in filename else 'M-100x'], 
#             'Activation'  :['unlabeled'], 
#             'Cell_Size_Pix' : [roi.area]
#             })
        
#         df_masks_size = pd.concat([df_masks_size, df_temp])


#%% merge dicts

# df = pd.concat([df,df_masks_size])

#%% SCATTER PLOT BY DONOR BY ACTIVATION

from datetime import datetime

# dish = 'Unstimulated'
dish = 'Activated'

vdim = 'NADH_tm'
# vdim = 'FAD_tm'
# vdim = 'Cell_Size_Pix'
# vdim = 'n.p.mean'
# vdim = 'f.p.mean'


df_subset = df[df['Group']==dish]

list_scatter = []
for donor_id in df_subset['Donor'].unique():
    scatter = hv.Scatter(df_subset[df_subset['Donor'] == donor_id], 
                          kdims='Activation', 
                          vdims=[vdim], 
                          label=donor_id)

    list_scatter.append(scatter)


colors = ['blue','green', 'brown']
for scatter, color in zip(list_scatter, colors):
    scatter.opts(color=color)

title = f"nk cells | {dish}"


overlay = hv.Overlay(list_scatter)
overlay.opts(
    opts.Overlay(
        title=title,
        legend_opts={"click_policy": "hide"},
        legend_position='right',
        show_legend=True,
        ),
    opts.Scatter(
        tools=["hover"],
        # box_muted_alpha=0,
        show_legend=True,
        width=800, 
        height=800,
        jitter=0.4
        ),       
)  

d = datetime.now()
hv.save(overlay, f'./figures/ecg_scatter_{dish}_{vdim}_{d.year}{str(d.month).zfill(2)}{str(d.day).zfill(2)}.html')






