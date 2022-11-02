
from pathlib import Path
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import pandas as pd
import holoviews as hv
from holoviews import  opts
hv.extension('bokeh')


df_original = pd.read_csv(r"Data files/UMAPs, boxplots, ROC curves (Python)/NK data all groups.csv")
df_new = pd.read_csv(r"Data files/UMAPs, boxplots, ROC curves (Python)/NK data donor M 20221101.csv")


df = pd.concat([df_original, df_new], axis=0)
# df = pd.read_csv(r"Data files/UMAPs, boxplots, ROC curves (Python)/NK data 2 groups.csv")

print(df.groupby(['Group','Donor']).count())



#%% SCATTER PLOT BY DONOR BY ACTIVATION

dish = 'Unstimulated'
# dish = 'Activated'

# vdim = 'FAD_tm'
vdim = 'NADH_tm'

df_subset = df[df['Group']==dish]

list_scatter = []
for donor_id in df_subset['Donor'].unique():
    scatter = hv.Scatter(df_subset[df_subset['Donor'] == donor_id], 
                          kdims='Activation', 
                          vdims=[vdim], 
                          label=donor_id)
    # violin = hv.Scatter(df_subset[df_subset['Donor'] == donor_id], 
    #                      kdims=['Group', 'Activation'], 
    #                      vdims=[vdim], 
    #                      label=donor_id)
    
    # violin.opts(opts.Violin(show_legend=True))

    list_scatter.append(scatter)


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
        # aspect="equal",
        width=800, 
        height=800,
        jitter=0.4
        ),       
)  

hv.save(overlay, f'./figures/ecg_scatter_{dish}_{vdim}_incl_new_data.html')
