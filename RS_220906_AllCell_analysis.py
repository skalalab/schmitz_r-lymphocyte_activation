# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:57:16 2022

@author: rschmitz
"""

#%% Section 1 - Import required packages

from copy import deepcopy

import umap.umap_ as umap
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.utils as skutils
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

import holoviews as hv
hv.extension("bokeh")
from holoviews import opts

from bokeh.io import export_svgs
from pprint import pprint

import pandas as pd
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sns.set(rc={'figure.figsize': (15, 15)})
sns.set_style(style='white')
plt.rcParams['svg.fonttype'] = 'none'

from helper import run_analysis_on_classifier, _train_test_split

#%% Section 2 - Define ROC function


# def calculate_roc_rf(rf_df, key='Activation'): 
    
#     # Need to binarize the problem as a 'One vs. all' style approach for ROC classification
#     classes = ['CD69-', 'CD69+']

#     #designate train/test data, random forest classifier
#     X, y = rf_df.iloc[:,:-1], rf_df[[key]]
#     y = label_binarize(y, classes=classes)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
#     y_train = np.ravel(y_train)
#     clf = RandomForestClassifier(random_state=0)
#     y_score = clf.fit(X_train, y_train).predict_proba(X_test)
#     y_pred = clf.fit(X_train, y_train).predict(X_test)


#     # Compute ROC curve and ROC area for each class
    
#     fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
#     roc_auc = auc(fpr, tpr)
    
#     # Plot of a ROC curve for a specific class
#     plt.figure()
#     plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.0])
#     plt.xlabel('False Positive Rate', fontsize = 20)
#     plt.ylabel('True Positive Rate', fontsize = 20)
#     plt.xticks(fontsize = 20)
#     plt.yticks(fontsize = 20)
#     plt.title('')
#     plt.legend(loc="lower right", fontsize = 20)
#     plt.show()
    
#%% Section 3 - Read in and set up dataframe 

#Read in dataframe    

all_df = pd.read_csv('./Data files/UMAPs, boxplots, ROC curves (Python)/AllCellData.csv')

#Add combination variables to data set
all_df.drop(['NADH', 'Group', 'Experiment_Date'], axis=1, inplace=True)
all_df['Type_Activation'] = all_df['Cell_Type'] + ': ' + all_df['Activation']
all_df['Donor_Activation'] = all_df['Cell_Type'] +' '+ all_df['Donor'] + ': ' + all_df['Activation']
all_df['Donor_CellType'] = all_df['Donor'] + ': ' + all_df['Cell_Type'] 

df_data = all_df.copy()

#%% # SF5
classes = ['CD69-', 'CD69+']
dict_classes = {label_int : label_class for label_int, label_class in enumerate(classes)}


#%% Section 4 - All cell activation classifier ROCs - Plot all curves together 

#TODO FIGURE 5

# accuracies in SF5

dict_accuracies = {}

print('All cell activation classifier')

#Generate df with only the OMI variables we want to include in the classifier (***Always keep Activation column last)

colors = ['#fde725', '#a5db36', '#4ac16d','#1f988b','#2a788e','#414487', '#440154']

custom_color = sns.set_palette(sns.color_palette(colors))

#Generate df with only the OMI variables we want to include in the classifier (***Always keep Activation column last)

##%% ################## 10 features

list_cols = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix']


X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['Top_10'] = accuracy

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='All variables (ROC AUC = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## NADH variables + Cell Size

list_cols = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'Cell_Size_Pix']

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['nadh_and_cell_size'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='NAD(P)H variables + Cell Size (ROC AUC = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## Top 4 (NADH a1, Norm RR, Cell Size, NADH t1)

list_cols = ['Norm_RR', 'NADH_a1', 'Cell_Size_Pix',  'NADH_t1']

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['top_4'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='Top four variables (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## Top 3 (NADH a1, Norm RR, Cell Size)
list_cols = ['NADH_a1', 'Norm_RR', 'Cell_Size_Pix']

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['top_3'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='Top three variables (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## #Top 2 (NADH a1, Norm RR)
list_cols = ['NADH_a1',  'Norm_RR']

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['top_2'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='Top two variables (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## Redox + Cell Size
list_cols = ['Cell_Size_Pix',  'Norm_RR']

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['redox_cell_size'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='Norm. Redox Ratio + Cell Size (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)


##%% ################## Top variable (NADH a1)
list_cols = ['NADH_a1']

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['top_1'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='Top variable (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)


plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize = 36)
plt.ylabel('True Positive Rate', fontsize = 36)
plt.xticks(fontsize = 36)
plt.yticks(fontsize = 36)
plt.title('Figure 5. ALL Cells', fontsize = 36)
plt.legend(bbox_to_anchor=(-0.1,-0.1), loc="upper left", fontsize = 36)
plt.savefig('./figures/Figure5_RS_allcell_ROC.svg',dpi=350, bbox_inches='tight')
plt.show()

print("SF5 Accuracies - Random Forest")
pprint(dict_accuracies)

#%% Section 5 - All cell activation classifier - Random forest, Logistic, SVM ROCs - Plot all curves together

#TODO

class_weight = None 
# class_weight = 'balanced'

sns.set(rc={'figure.figsize': (15, 15)})
sns.set_style(style='white')
plt.rcParams['svg.fonttype'] = 'none'

colors = ['#fde725', '#1f988b','#440154']

custom_color = sns.set_palette(sns.color_palette(colors))

list_cols = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix']
X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

clf = RandomForestClassifier(random_state=0, class_weight=class_weight).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)


# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (ROC AUC = %0.2f)' % roc_auc, linewidth = 7)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)


clf = LogisticRegression(random_state=0, class_weight=class_weight).fit(X_train, y_train) #JR - use for logistic regression
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)


plt.plot(fpr, tpr, label='Logistic Regression (ROC AUC = %0.2f)' % roc_auc, linewidth = 7)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)


clf = SVC(probability=True, class_weight=class_weight) #JR - use for SVM # probability=True
csf_fit = clf.fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(csf_fit, X_test, y_test, dict_classes)

plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

# Plot of a ROC curve for a specific class

plt.plot(fpr, tpr, label='Support Vector Machine (ROC AUC = %0.2f)' % roc_auc, linewidth = 7, zorder=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize = 36)
plt.ylabel('True Positive Rate', fontsize = 36)
plt.xticks(fontsize = 36)
plt.yticks(fontsize = 36)
plt.title(f'All Cells | | class_weight: {class_weight}', fontsize = 36)
plt.legend(bbox_to_anchor=(-0.1,-0.1), loc="upper left", fontsize = 36)
plt.savefig('./figures/SF5_RS_allcell_SVMLR_ROC.svg',dpi=350, bbox_inches='tight')

plt.show()

#%% Section 6 - UMAP of activation status

#list of parameters we want to use for the UMAP. I used ten OMI features (Normalized redox ratio, NAD(P)H lifetimes, FAD lifetimes, and cell size)
list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix']


#generate UMAP
data = df_data[list_omi_parameters].values
scaled_data = StandardScaler().fit_transform(data)
reducer = umap.UMAP(
               n_neighbors=15,
               min_dist=1,   
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

#
                    
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
        size = 3,
        alpha = 0.5,
        tools=["hover"],
        muted_alpha=0,
        aspect="equal",
        width=600, 
        height=600),
    opts.Overlay(
        title='',
        legend_opts={"click_policy": "hide"},
        legend_position='bottom_right'
        )       
    )

#Saves an interactive holoviews plot as a .HTML file
plot = hv.render(overlay)
plot.output_backend = "svg"
export_svgs(plot, filename = './figures/AllCell_ActStatus_umap.svg')
# hv.save(overlay, './figures/AllCell_ActStatus_umap.svg')

#%% Section 7 - UMAP of cell type

#Same structure as Section 6 - see comments for more detail 

list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix']

data = df_data[list_omi_parameters].values
scaled_data = StandardScaler().fit_transform(data)
reducer = umap.UMAP(
               n_neighbors=15,
               min_dist=1,   
               metric='euclidean',
               n_components=2,
               random_state=0
           )
       
fit_umap = reducer.fit(scaled_data)


## additional params
hover_vdim = "Activation"
legend_entries = "Cell_Type" 

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
        size = 3,
        alpha = 0.5,
        tools=["hover"],
        muted_alpha=0,
        aspect="equal",
        width=600, 
        height=600),
    opts.Overlay(
        title='',
        legend_opts={"click_policy": "hide"},
        legend_position='bottom_right'
        )       
    )


plot = hv.render(overlay)
plot.output_backend = "svg"
export_svgs(plot, filename = './figures/AllCell_CellType_umap.svg')
# hv.save(overlay, 'AllCell_CellType_umap.svg')

#%% Setion 8 - UMAP of cell type (QUIESCENT ONLY)


#Same structure as Section 6 - see comments for more detail 

list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix']

df_data_qonly = df_data.loc[df_data['Activation'] == 'CD69-']

data = df_data_qonly[list_omi_parameters].values
scaled_data = StandardScaler().fit_transform(data)
reducer = umap.UMAP(
               n_neighbors=15,
               min_dist=1,   
               metric='euclidean',
               n_components=2,
               random_state=0
           )
       
fit_umap = reducer.fit(scaled_data)


## additional params
hover_vdim = "Activation"
legend_entries = "Cell_Type" 

########
df_data_qonly = df_data_qonly.copy()
df_data_qonly["umap_x"] = fit_umap.embedding_[:,0]
df_data_qonly["umap_y"] = fit_umap.embedding_[:,1]

kdims = ["umap_x"]
vdims = ["umap_y", hover_vdim]
list_entries = np.unique(df_data[legend_entries])

#
                    
scatter_umaps = [hv.Scatter(df_data_qonly[df_data_qonly[legend_entries] == entry], kdims=kdims, 
                            vdims=vdims, label=entry) for entry in list_entries]



colors = [ '#440154','#21918c','#fde725']
for scatter, color in zip(scatter_umaps, colors):
    scatter.opts(color=color)

overlay = hv.Overlay(scatter_umaps)
overlay.opts(
    opts.Scatter(
        #fontsize={'labels': 14, 'xticks': 12, 'yticks': 12},
        fontscale=1.5,
        size = 3,
        alpha = 0.5,
        tools=["hover"],
        muted_alpha=0,
        aspect="equal",
        width=600, 
        height=600),
    opts.Overlay(
        title='',
        legend_opts={"click_policy": "hide"},
        legend_position='bottom_right'
        )       
    )

plot = hv.render(overlay)
plot.output_backend = "svg"
export_svgs(plot, filename = './figures/AllCell_CellType_QuiOnly_umap.svg')
# hv.save(overlay, 'AllCell_CellType_QuiOnly_umap.svg')

#%% Section 9 - UMAP of cell type + activation status


#Same structure as Section 6 - see comments for more detail 

list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix']

data = df_data[list_omi_parameters].values
scaled_data = StandardScaler().fit_transform(data)
reducer = umap.UMAP(
               n_neighbors=15,
               min_dist=1,   
               metric='euclidean',
               n_components=2,
               random_state=0
           )
       
fit_umap = reducer.fit(scaled_data)


## additional params
hover_vdim = "Activation"
legend_entries = "Type_Activation" 

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
        size = 3,
        alpha = 0.5,
        tools=["hover"],
        muted_alpha=0,
        aspect="equal",
        width=600, 
        height=600),
    opts.Overlay(
        title='',
        legend_opts={"click_policy": "hide"},
        legend_position='bottom_right'
        )       
    )

plot = hv.render(overlay)
plot.output_backend = "svg"
export_svgs(plot, filename = './figures/AllCell_CellType_ActStatus_umap.svg')
# hv.save(overlay, 'AllCell_CellType_ActStatus_umap.svg')




#%% Section 10 - UMAP of cell type color-coded by donor


#Same structure as Section 6 - see comments for more detail 


list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix']

data = df_data[list_omi_parameters].values
scaled_data = StandardScaler().fit_transform(data)
reducer = umap.UMAP(
               n_neighbors=15,
               min_dist=1,   
               metric='euclidean',
               n_components=2,
               random_state=0
           )
       
fit_umap = reducer.fit(scaled_data)


## additional params
hover_vdim = "Activation"
legend_entries = "Donor_CellType" 

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



colors = ["#440154", "#482173", "#433e85", "#38588c", "#2d708e", "#25858e",  "#1e9b8a",  "#2ab07f", "#52c569", "#86d549", "#c2df23", "#fde725" ]
for scatter, color in zip(scatter_umaps, colors):
    scatter.opts(color=color)

overlay = hv.Overlay(scatter_umaps)
overlay.opts(
    opts.Scatter(
        #fontsize={'labels': 14, 'xticks': 12, 'yticks': 12},
        fontscale=1,
        size = 2,
        alpha = 0.5,
        tools=["hover"],
        muted_alpha=0,
        aspect="equal",
        width=750, 
        height=600),
    opts.Overlay(
        title='',
        legend_opts={"click_policy": "hide"},
        legend_position='right'
        )       
    )

plot = hv.render(overlay)
plot.output_backend = "svg"
export_svgs(plot, filename = './figures/AllCell_CellType_Donor_umap.svg')
# hv.save(overlay, 'AllCell_CellType_Donor_umap.svg')
#%% Section 11 - UMAP of activation status color-coded by donor


#Same structure as Section 6 - see comments for more detail 


list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix']

data = df_data[list_omi_parameters].values
scaled_data = StandardScaler().fit_transform(data)
reducer = umap.UMAP(
               n_neighbors=15,
               min_dist=1,   
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



colors = [ '#440154','#471164','#482071','#472e7c','#443b84','#3f4889','#3a548c','#34608d','#2f6c8e','#2a768e','#26818e','#228b8d','#1f958b','#1fa088','#24aa83','#2fb47c','#42be71','#58c765','#70cf57','#8bd646','#a8db34','#c5e021','#e2e418','#fde725']
for scatter, color in zip(scatter_umaps, colors):
    scatter.opts(color=color)

overlay = hv.Overlay(scatter_umaps)
overlay.opts(
    opts.Scatter(
        #fontsize={'labels': 14, 'xticks': 12, 'yticks': 12},
        fontscale=1,
        size = 2,
        alpha = 0.5,
        tools=["hover"],
        muted_alpha=0,
        aspect="equal",
        width=750, 
        height=600),
    opts.Overlay(
        title='',
        legend_opts={"click_policy": "hide"},
        legend_position='right'
        )       
    )

plot = hv.render(overlay)
plot.output_backend = "svg"
export_svgs(plot, filename = './figures/AllCell_CellType_Donor_ActStatus_umap.svg')

# hv.save(overlay, "figures/" 'AllCell_CellType_Donor_ActStatus_umap.svg')

