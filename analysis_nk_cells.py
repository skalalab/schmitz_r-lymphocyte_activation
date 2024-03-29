#%%  Section 1 - Imports

from copy import deepcopy

import umap.umap_ as umap
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.utils as skutils
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from pprint import pprint

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

import holoviews as hv
hv.extension("bokeh")
from holoviews import opts

from bokeh.io import export_svgs

import pandas as pd
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sns.set(rc={'figure.figsize': (15, 15)})
sns.set_style(style='white')
plt.rcParams['svg.fonttype'] = 'none'


from helper import run_analysis_on_classifier, _train_test_split
#%% Section 2 - Loads and normalizes data

path_nk_data = './Data files/UMAPs, boxplots, ROC curves (Python)/NK_cells_dataset.csv'
nk_df = pd.read_csv(path_nk_data)

nk_df = nk_df.rename(columns={'n.t1.mean' : 'NADH_t1', 
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

### normalize to control
nk_df.groupby(['Group','Donor', 'Activation'])['Norm_RR'].mean()

# normalize NK cells to donor
for donor in nk_df['Donor'].unique():
    pass 
    mean_control = nk_df[(nk_df['Donor'] == donor) & 
                         (nk_df['Group'] == "Control")&
                          (nk_df['Activation'] == "CD69-")
                             ]["Norm_RR"].mean()
    nk_df.loc[(nk_df['Donor'] == donor),'Norm_RR'] = (nk_df.loc[(nk_df['Donor'] == donor),'Norm_RR'] / mean_control)

# print("+" * 20)
nk_df.groupby(['Donor','Group', 'Activation'])['Norm_RR'].mean()

    
# keep only Activated CD69+ and Unactivated CD69-
nk_df = nk_df[((nk_df['Group']=='Activated') & (nk_df['Activation']=='CD69+')) | 
               ((nk_df['Group']=='Control') & (nk_df['Activation']=='CD69-'))
               ]


# print(nk_df.groupby(by=['Donor','Group','Activation']).count())
# print(nk_df.groupby(by=['Group','Activation'])['Cell_Size_Pix'].mean())
nk_df.groupby(['Donor','Group', 'Activation']).count()


nk_df.drop(['NADH', 'Group', 'Experiment_Date'], axis=1, inplace=True)

nk_df['Act_Donor'] = nk_df['Activation'] + ' ' + nk_df['Donor']

df_data = nk_df.copy()

# class labels and dict for mapping
classes = ['CD69-', 'CD69+']
dict_classes = {label_int : label_class for label_int, label_class in enumerate(classes)}


#%% Section 3 - Generate graph with all ROCs

# F4 D - ROC curve
# F4 C - accuracies 
list_top_vars = []

dict_accuracies = {}

# plot colors 
colors = ['#fde725', '#a5db36', '#4ac16d','#1f988b','#2a788e','#414487', '#440154']
custom_color = sns.set_palette(sns.color_palette(colors))

list_cols = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'
X_train1, X_test1, y_train1, y_test1 = _train_test_split(nk_df, list_cols, classes)

##%%  ################### 10 features 
clf = RandomForestClassifier(random_state=0).fit(X_train1, y_train1)
fpr, tpr, roc_auc, accuracy, op_point = run_analysis_on_classifier(clf, X_test1, y_test1, dict_classes)
dict_accuracies["all features"] = accuracy

### Figure 4 C
print("F4_C piechart of importance on all features")
forest_importances = pd.Series(clf.feature_importances_*100, index=X_train1.keys()).sort_values(ascending=False)
print(forest_importances)
df_acc = pd.DataFrame(forest_importances)
df_acc.to_csv('./figures/F4/F4_C_feature_importances.csv')

## %%
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label=f'All variables (ROC AUC = {roc_auc:0.2f})' , linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ###################NADH variables + Cell Size
list_cols = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2'] # , 'Cell_Size_Pix'

X_train, X_test, y_train, y_test = _train_test_split(nk_df, list_cols, classes)
clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies["NADH features"] = accuracy



# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label=f'NAD(P)H variables (ROC AUC = {roc_auc:0.2f})'  , linewidth = 5) # + Cell Size
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################### Top 4 (NADH a1, Norm RR, NADH tm, NADH t2)
# list_cols = ['NADH_tm', 'NADH_a1', 'NADH_t2',  'Norm_RR']

list_cols = list(forest_importances.keys()[:4])
list_top_vars.append(f"Top {len(list_cols)} : {list_cols}")

X_train, X_test, y_train, y_test = _train_test_split(nk_df, list_cols, classes)
clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies["top 4"] = accuracy


# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label=f'Top four variables (ROC AUC) = {roc_auc:0.2f})' , linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################### Top 3 (NADH a1, Norm RR, NADH tm)

# list_cols = ['NADH_tm', 'NADH_a1',  'Norm_RR']
list_cols = list(forest_importances.keys()[:3])
list_top_vars.append(f"Top {len(list_cols)} : {list_cols}")


X_train, X_test, y_train, y_test = _train_test_split(nk_df, list_cols, classes)
clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies["top 3"] = accuracy


# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label=f'Top three variables (ROC AUC) = {roc_auc:0.2f})' , linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################### Top 2 (NADH a1, Norm RR)
# list_cols = ['NADH_a1',  'Norm_RR']

list_cols = list(forest_importances.keys()[:2])
list_top_vars.append(f"Top {len(list_cols)} : {list_cols}")


X_train, X_test, y_train, y_test = _train_test_split(nk_df, list_cols, classes)
clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy,  op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies["top 2"] = accuracy


# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label=f'Top two variables (ROC AUC) = {roc_auc:0.2f})', linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################### Top variable (NADH a1)
# list_cols = ['NADH_a1']
list_cols = list(forest_importances.keys()[:1])
list_top_vars.append(f"Top {len(list_cols)} : {list_cols}")


X_train, X_test, y_train, y_test = _train_test_split(nk_df, list_cols, classes)
clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy,  op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies["top 1"] = accuracy



# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label=f'Top variable (ROC AUC) = {roc_auc:0.2})', linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################### Redox + Cell Size

# list_cols = ['Norm_RR'] # 'Cell_Size_Pix',
# X_train, X_test, y_train, y_test = _train_test_split(nk_df, list_cols, classes)
# clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
# fpr, tpr, roc_auc, accuracy,  op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
# dict_accuracies["Norm_RR"] = accuracy


# # Plot of a ROC curve for a specific class
# plt.plot(fpr, tpr, label=f'Norm. Redox Ratio (ROC AUC) = {roc_auc:0.2f})'  , linewidth = 5) # + Cell Size
# plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize = 36)
plt.ylabel('True Positive Rate', fontsize = 36)
plt.xticks(fontsize = 36)
plt.yticks(fontsize = 36)
plt.title('Figure 4. NK Cells', fontsize = 36)
plt.legend(bbox_to_anchor=(-0.1,-0.1), loc="upper left", fontsize = 36)
plt.savefig('./figures/F4/F4_D_RS_nk_ROC.svg',dpi=350, bbox_inches='tight')

plt.show()

pprint(list_top_vars)

print("--- F4 D Accuracies --- ")
pprint(dict_accuracies)
df_acc = pd.DataFrame(dict_accuracies, index=[0])
df_acc.to_csv('./figures/F4/F4_D_accuracies.csv')



#%% Section 4 - donor color coding boxplots: Red/blue, all dots visible

# F3 FGHI

dict_figures = {
            'F' : {
                'y_axis_col_name' : 'Norm_RR',
                "y_label" : r'Normalized Redox Ratio'
                },
            'G' : {
                'y_axis_col_name' : 'NADH_tm',
                "y_label" : r'NAD(P)H $\tau m$'
                },
            'H' : {
                'y_axis_col_name' : 'NADH_a1',
                "y_label" : r'NAD(P)H $\alpha _1$'
                },
            'I' : {
                'y_axis_col_name' : 'FAD_tm',
                "y_label" : r'FAD $\tau m$'
                
                },
            }

for fig_letter, fig_dict in dict_figures.items():
    pass
    df = nk_df.copy()
    
    f, ax = plt.subplots(figsize=(8,20))
    
    #colors = ['#fde725', '#21918c', '#440154']
    #custom_color = sns.set_palette(sns.color_palette(colors))
    # colors = {'CD69- D':'#222255', 
    #           'CD69- E':'#4393C3', 
    #           'CD69- F':'#99DDFF',
    #           'CD69+ D':'#662506', 
    #           'CD69+ E':'#D6604D', 
    #           'CD69+ F':'#F4A582'}
    
    colors = {'CD69- D':'#222255', 
              'CD69- M':'#4393C3', 
              'CD69- N':'#99DDFF',
              'CD69+ D':'#662506', 
              'CD69+ M':'#D6604D', 
              'CD69+ N':'#F4A582'}
    
    
    sns.set_style(style='white')
    
    
    PROPS_BLACK = {'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
                   'medianprops': {'color': 'black'},
                   'whiskerprops': {'color': 'black'},
                   'capprops': {'color': 'black'}}
    
    boxplot_line_width= 5
    boxplot_width = 0.9
    x_axis_col_name = 'Activation'
    y_axis_col_name = dict_figures[fig_letter]['y_axis_col_name'] # NADH_a1'
    
    #graph with indiviual palette
    
    sns.swarmplot(x='Activation', 
                  y=y_axis_col_name, 
                  hue='Act_Donor',  
                  order = ['CD69-', 'CD69+'], 
                  palette = colors, 
                  size = 5.2, 
                  data = df, zorder = 0.5)
    sns.boxplot(x=x_axis_col_name,
                y=y_axis_col_name,
                data=df,
                order = ['CD69-', 'CD69+'],
                linewidth=boxplot_line_width,
                width=boxplot_width,
                showfliers=False,
                zorder=1,
                **PROPS_BLACK)
    
    #get and order legend labels to pass to pyplot
    handles,labels = ax.get_legend_handles_labels()
    
    handles = [handles[1], handles[3], handles[5], handles[0], handles[2], handles[4]]
    labels = [labels[1], labels[3], labels[5], labels[0], labels[2], labels[4]]
    
    
    #customize plot
    plt.xlabel('', size = 1)
    plt.ylabel(dict_figures[fig_letter]['y_label'], size = 40, fontweight='bold')
    #plt.ylabel('Normalized Redox Ratio', size = 40, fontweight='bold')
    plt.xticks(size = 36, fontweight='bold')
    plt.yticks(size = 30, fontweight='bold')
    #plt.legend(handles,labels,loc = 'lower right', fontsize = 36)
    plt.legend([],[], frameon=False)
    
    #add bar for p-value
    x1, x2 = 0, 1   
    # y, h, col = nk_df[y_axis_col_name].max() + 0.5, 0.5, 'k'
    offset = nk_df[y_axis_col_name].max() * 0.01
    y, h, col = nk_df[y_axis_col_name].max() + offset, offset, 'k'
    plt.tight_layout()
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=5, c=col)
    plt.text((x1+x2)*.5, y+h, "****", ha='center', va='bottom', color=col, size = 40)
    
    plt.savefig(f'./figures/F3/F3_{fig_letter}_RS_nk_{fig_dict["y_axis_col_name"]}.svg',dpi=350, bbox_inches='tight')




#%% Section 5 - ROC curves: Random forest, Logistic Regression, SVM curves together

# SF4 F 
# SF4 C,D,E - CONFUSION MATRIX 

class_weight = None 
# class_weight = 'balanced'
 
sns.set(rc={'figure.figsize': (15, 15)})
sns.set_style(style='white')
plt.rcParams['svg.fonttype'] = 'none'

colors = ['#fde725', '#1f988b','#440154']

custom_color = sns.set_palette(sns.color_palette(colors))

# selet data 
list_cols = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'

X_train, X_test, y_train, y_test = _train_test_split(nk_df, list_cols, classes)

##%% ######### RANDON FOREST  - unbalanced 
clf = RandomForestClassifier(random_state=0, class_weight=class_weight).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label=f'Random Forest (ROC AUC = {roc_auc:0.2f})' , linewidth = 7)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

# class_weight = 'balanced'
# ##%% ######### RANDON FOREST  - balanced 
# clf = RandomForestClassifier(random_state=0, class_weight=class_weight).fit(X_train, y_train)
# fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)

# # Plot of a ROC curve for a specific class
# plt.plot(fpr, tpr, label=f'Random Forest (ROC AUC = {roc_auc:0.2f})' , linewidth = 7)
# plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

# plt.show()
## %%

##%% #########  LogisticRegression
clf = LogisticRegression(random_state=0, class_weight=class_weight).fit(X_train, y_train) #JR - use for logistic regression
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test,  y_test, dict_classes)

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label=f'Logistic Regression (ROC AUC = {roc_auc:0.2f})' , linewidth = 7)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ######### SVC
clf = SVC(probability=True, class_weight=class_weight).fit(X_train, y_train) #JR - use for SVM
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)

plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)
# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label=f'Support Vector Machine (ROC AUC = {roc_auc:0.2f})' , linewidth = 7)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize = 36)
plt.ylabel('True Positive Rate', fontsize = 36)
plt.xticks(fontsize = 36)
plt.yticks(fontsize = 36)
plt.title(f'NK Cells | class_weight: {class_weight}', fontsize = 36)
plt.legend(bbox_to_anchor=(-0.1,-0.1), loc="upper left", fontsize = 36)
plt.savefig('./figures/SF4/SF4_F_RS_nk_SVMLR_ROC.svg',dpi=350, bbox_inches='tight')

plt.show()

#%% Section 6 - NK cell activation UMAP

# F4 B

#list of parameters we want to use for the UMAP. I used ten OMI features (Normalized redox ratio, NAD(P)H lifetimes, FAD lifetimes, and cell size)

list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'


df_data = nk_df.copy()

#generate UMAP
data = df_data[list_omi_parameters].values
scaled_data = StandardScaler().fit_transform(data)
reducer = umap.UMAP(
               n_neighbors=15,
               min_dist=0.4,   
               metric='euclidean',
               n_components=2,
               random_state=0
           )
       
fit_umap = reducer.fit(scaled_data)


## additional params for holoviews
#The legend_entries parameter will determine what group we are color-coding by
hover_vdim = "Activation"
legend_entries = "Activation" 

#generate UMAP embedding
df_data = df_data.copy()
df_data["umap_x"] = fit_umap.embedding_[:,0]
df_data["umap_y"] = fit_umap.embedding_[:,1]

kdims = ["umap_x"]
vdims = ["umap_y", hover_vdim]
list_entries = np.unique(df_data[legend_entries])

                    
scatter_umaps = [hv.Scatter(df_data[df_data[legend_entries] == entry], kdims=kdims, 
                            vdims=vdims, label=entry) for entry in list_entries]


#Parameters to control plotting of UMAP in holoviews
colors = ['#B2182B','#000058']
for scatter, color in zip(scatter_umaps, colors):
    scatter.opts(color=color)

overlay = hv.Overlay(scatter_umaps)
overlay.opts(
    opts.Scatter(
        #fontsize={'labels': 14, 'xticks': 12, 'yticks': 12},
        fontscale=2,
        alpha = 0.75,
        size=4,
        tools=["hover"],
        muted_alpha=0,
        aspect="equal",
        width=800, 
        height=800),
    opts.Overlay(
        title='',
        legend_opts={"click_policy": "hide"},
        legend_position='bottom_right'
        )       
    )

#Saves an interactive holoviews plot as a .HTML file
plot = hv.render(overlay)
plot.output_backend = "svg"
export_svgs(plot, filename = './figures/F4/F4_B_NKCell_ActStatus_umap.svg')
# hv.save(overlay, 'NKCell_ActStatus_umap.html')

#%% Section 7 -  Nk cell donor UMAP


# NO FIGURE in paper

#Same structure as Section 6 - see comments above

list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'

df_data = nk_df.copy()

data = df_data[list_omi_parameters].values
scaled_data = StandardScaler().fit_transform(data)
reducer = umap.UMAP(
               n_neighbors=15,
               min_dist=0.4,   
               metric='euclidean',
               n_components=2,
               random_state=0
           )
       
fit_umap = reducer.fit(scaled_data)


## additional params
hover_vdim = "Activation"
legend_entries = "Donor" 

########
df_data = df_data.copy()
df_data["umap_x"] = fit_umap.embedding_[:,0]
df_data["umap_y"] = fit_umap.embedding_[:,1]

kdims = ["umap_x"]
vdims = ["umap_y", hover_vdim]
list_entries = np.unique(df_data[legend_entries])

#
                    
scatter_umaps = [hv.Scatter(df_data[df_data[legend_entries] == entry], kdims=kdims, 
                            vdims=vdims, label=entry) for entry in list_entries]

colors = [ '#440154','#21918c','#fde725']
for scatter, color in zip(scatter_umaps, colors):
    scatter.opts(color=color)

overlay = hv.Overlay(scatter_umaps)
overlay.opts(
    opts.Scatter(
        #fontsize={'labels': 14, 'xticks': 12, 'yticks': 12},
        fontscale=2,
        size=4,
        alpha = 0.75,
        tools=["hover"],
        muted_alpha=0,
        aspect="equal",
        width=800, 
        height=800),
    opts.Overlay(
        title='',
        legend_opts={"click_policy": "hide"},
        legend_position='bottom_right'
        )       
    )

plot = hv.render(overlay)
plot.output_backend = "svg"
# export_svgs(plot, filename = './figures/nk/NF_NKCell_Donor_umap.svg')
# hv.save(overlay, './figures/NKCell_Donor_umap.html')


#%% Section 8 - Nk cell donor + activation UMAP

#Generate column in data frame that has both donor and activation status

# SF4 B
 
df_data = nk_df.copy()

df_data['Donor_Activation'] = df_data['Donor'] + ': ' + df_data['Activation']

#Same structure as Section 6 - see comments above

list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'

data = df_data[list_omi_parameters].values
scaled_data = StandardScaler().fit_transform(data)
reducer = umap.UMAP(
               n_neighbors=15,
               min_dist=0.4,   
               metric='euclidean',
               n_components=2,
               random_state=0
           )
       
fit_umap = reducer.fit(scaled_data)


## additional params
hover_vdim = "Activation"
legend_entries = "Donor_Activation" 

########
df_data = df_data.copy()
df_data["umap_x"] = fit_umap.embedding_[:,0]
df_data["umap_y"] = fit_umap.embedding_[:,1]

kdims = ["umap_x"]
vdims = ["umap_y", hover_vdim]
list_entries = np.unique(df_data[legend_entries])

#
                    
scatter_umaps = [hv.Scatter(df_data[df_data[legend_entries] == entry], kdims=kdims, 
                            vdims=vdims, label=entry) for entry in list_entries]

colors = [ '#414487','#440154', '#22a884', '#2a788e', '#fde725', '#7ad151']
for scatter, color in zip(scatter_umaps, colors):
    scatter.opts(color=color)


overlay = hv.Overlay(scatter_umaps)
overlay.opts(
    opts.Scatter(
        #fontsize={'labels': 14, 'xticks': 12, 'yticks': 12},
        fontscale=1.5,
        size=4,
        alpha = 0.75,
        tools=["hover"],
        muted_alpha=0,
        aspect="equal",
        width=800, 
        height=800),
    opts.Overlay(
        title='',
        legend_opts={"click_policy": "hide"},
        legend_position='bottom_right'
        )       
    )

plot = hv.render(overlay)
plot.output_backend = "svg"
export_svgs(plot, filename = './figures/SF4/SF4_B_NKCell_Donor_ActStatus_umap.svg')
# hv.save(overlay, './figures/nk/SF4_B_NKCell_Donor_ActStatus_umap.html')

#%% Section 9 - UMAP of both groups and activation statuses of NK cells


# NOTE THIS USES ALL THE DATA 
# SF4 A   


#Read in CSV that has data from all 4 combinations of activation/culture condition

# RELOAD ALL THE DATA WITH ALL GROUPS
allgroup_nk_df = pd.read_csv(path_nk_data)

allgroup_nk_df = allgroup_nk_df.rename(columns={'n.t1.mean' : 'NADH_t1', 
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


df_data = allgroup_nk_df.copy()

#generate column in dataframe that combines both culture group and activation status

df_data['Group_Activation'] = df_data['Group'] + ': ' + df_data['Activation']


#Same structure as Section 5 - see comments above
list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'

data = df_data[list_omi_parameters].values
scaled_data = StandardScaler().fit_transform(data)
reducer = umap.UMAP(
               n_neighbors=15,
               min_dist=0.4,   
               metric='euclidean',
               n_components=2,
               random_state=0
           )
       
fit_umap = reducer.fit(scaled_data)


## additional params
hover_vdim = "Activation"
legend_entries = "Group_Activation" 

########
df_data = df_data.copy()
df_data["umap_x"] = fit_umap.embedding_[:,0]
df_data["umap_y"] = fit_umap.embedding_[:,1]

kdims = ["umap_x"]
vdims = ["umap_y", hover_vdim]
list_entries = np.unique(df_data[legend_entries])

                    
scatter_umaps = [hv.Scatter(df_data[df_data[legend_entries] == entry], kdims=kdims, 
                            vdims=vdims, label=entry) for entry in list_entries]

colors = [ '#440154','#31688e', '#35b779', '#fde725']
for scatter, color in zip(scatter_umaps, colors):
    scatter.opts(color=color)


overlay = hv.Overlay(scatter_umaps)
overlay.opts(
    opts.Scatter(
        #fontsize={'labels': 14, 'xticks': 12, 'yticks': 12},
        fontscale=1.5,
        size = 4,
        alpha = 0.5,
        tools=["hover"],
        muted_alpha=0,
        aspect="equal",
        width=800, 
        height=800),
    opts.Overlay(
        title='',
        legend_opts={"click_policy": "hide"},
        legend_position='bottom_right'
        )       
    )

plot = hv.render(overlay)
plot.output_backend = "svg"
export_svgs(plot, filename = './figures/SF4/SF4_A_NKCell_ActStatus_Condition_umap.svg')

# hv.save(overlay, './figures/nk/SF4_A_NKCell_ActStatus_Condition_umap.html')
 

