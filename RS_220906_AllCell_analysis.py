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


##%% # remove old NK donors and add new ones
all_df = all_df[all_df['Cell_Type'] != 'NK-Cells']

# load new nk cells 
df_nk = pd.read_csv('Data files/UMAPs, boxplots, ROC curves (Python)/NK_donors_final_dec02.csv')
df_nk = df_nk.rename(columns={'n.t1.mean' : 'NADH_t1', 
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

df_nk.groupby(['Donor','Group', 'Activation'])['Norm_RR'].mean()


# normalize NK cells to donor
for donor in df_nk['Donor'].unique():
    pass 
    mean_control = df_nk[(df_nk['Donor'] == donor) & 
                         (df_nk['Group'] == "Control")&
                          (df_nk['Activation'] == "CD69-")
                             ]["Norm_RR"].mean()
    df_nk.loc[(df_nk['Donor'] == donor),'Norm_RR'] = (df_nk.loc[(df_nk['Donor'] == donor),'Norm_RR'] / mean_control)

# print("+" * 20)
df_nk.groupby(['Donor','Group', 'Activation'])['Norm_RR'].mean()


# keep only Activated CD69+ and Unactivated CD69-
df_nk = df_nk[((df_nk['Group']=='Activated') & (df_nk['Activation']=='CD69+')) | 
               ((df_nk['Group']=='Control') & (df_nk['Activation']=='CD69-'))
               ]

df_nk.groupby(['Donor','Group', 'Activation'])['Cell_Type'].count()

## Concat dicts
df_concat = pd.concat([all_df,df_nk])
df_concat['Donor'].unique()
df_concat['Cell_Type'].unique()

all_df = df_concat

##%%%

#Add combination variables to data set
all_df.drop(['NADH', 'Group', 'Experiment_Date'], axis=1, inplace=True)
all_df['Type_Activation'] = all_df['Cell_Type'] + ': ' + all_df['Activation']
all_df['Donor_Activation'] = all_df['Cell_Type'] +' '+ all_df['Donor'] + ': ' + all_df['Activation']
all_df['Donor_CellType'] = all_df['Donor'] + ': ' + all_df['Cell_Type'] 

df_data = all_df.copy()

##%% # SF5
classes = ['CD69-', 'CD69+']
dict_classes = {label_int : label_class for label_int, label_class in enumerate(classes)}


print(all_df.groupby(by=['Cell_Type','Donor','Activation',])['Cell_Size_Pix'].count())
print("*" * 20)
print(all_df.groupby(by=['Cell_Type','Donor','Activation'])['Norm_RR'].mean())



#%% Section 4 - All cell activation classifier ROCs - Plot all curves together 

#TODO FIGURE 5 D

# SF5 B accuracies 

print('All cell activation classifier')

#Generate df with only the OMI variables we want to include in the classifier (***Always keep Activation column last)

colors = ['#fde725', '#a5db36', '#4ac16d','#1f988b','#2a788e','#414487', '#440154']

custom_color = sns.set_palette(sns.color_palette(colors))

#Generate df with only the OMI variables we want to include in the classifier (***Always keep Activation column last)

list_top_vars = []

dict_accuracies = {}


##%% ################## 10 features

list_cols = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] #, 'Cell_Size_Pix'

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['all feautres'] = accuracy

print("Figure 5D piechart of importance on all features")
forest_importances = pd.Series(clf.feature_importances_*100, index=X_train.keys()).sort_values(ascending=False)
print(forest_importances)


# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='All variables (ROC AUC = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## NADH variables + Cell Size

list_cols = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2'] # , 'Cell_Size_Pix'

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['nadh features'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='NAD(P)H variables (ROC AUC = %0.2f)' % roc_auc, linewidth = 5) # + Cell Size 
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## Top 4 (NADH a1, Norm RR, Cell Size, NADH t1) ####################################### 4

# list_cols = ['Norm_RR', 'NADH_a1', 'Cell_Size_Pix',  'NADH_t1']
list_cols = list(forest_importances.keys()[:4])
list_top_vars.append(f"Top {len(list_cols)} : {list_cols}")


X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['top_4'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='Top four variables (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## Top 3 (NADH a1, Norm RR, Cell Size) ####################################### 3
# list_cols = ['NADH_a1', 'Norm_RR', 'Cell_Size_Pix']
list_cols = list(forest_importances.keys()[:3])
list_top_vars.append(f"Top {len(list_cols)} : {list_cols}")

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['top_3'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='Top three variables (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## #Top 2 (NADH a1, Norm RR)  ####################################### 2
# list_cols = ['NADH_a1',  'Norm_RR']
list_cols = list(forest_importances.keys()[:2])
list_top_vars.append(f"Top {len(list_cols)} : {list_cols}")

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['top_2'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='Top two variables (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## Top variable (NADH a1) ####################################### 1
# list_cols = ['NADH_a1']
list_cols = list(forest_importances.keys()[:1])
list_top_vars.append(f"Top {len(list_cols)} : {list_cols}")

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['top_1'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='Top variable (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## Redox + Cell Size
list_cols = [ 'Norm_RR'] #, 'Cell_Size_Pix'

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['redox'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='Norm. Redox Ratio (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5) #  + Cell Size 
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)


plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize = 36)
plt.ylabel('True Positive Rate', fontsize = 36)
plt.xticks(fontsize = 36)
plt.yticks(fontsize = 36)
plt.title('Figure 5. ALL Cells', fontsize = 36)
plt.legend(bbox_to_anchor=(-0.1,-0.1), loc="upper left", fontsize = 36)
plt.savefig('./figures/all/F5_D_RS_allcell_ROC.svg',dpi=350, bbox_inches='tight')
plt.show()

pprint(list_top_vars)
print('*' * 20)
print("SF 5 B : all cell activation accuracies")
pprint(dict_accuracies)

#%% Section 5 - All cell activation classifier - Random forest, Logistic, SVM ROCs - Plot all curves together

#TODO SF5 D, CONFUSION MATRICES E,F,G

class_weight = None 
# class_weight = 'balanced'

sns.set(rc={'figure.figsize': (15, 15)})
sns.set_style(style='white')
plt.rcParams['svg.fonttype'] = 'none'

colors = ['#fde725', '#1f988b','#440154']

custom_color = sns.set_palette(sns.color_palette(colors))

list_cols = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] #, 'Cell_Size_Pix'
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
plt.savefig('./figures/all/SF5_D_RS_allcell_SVMLR_ROC.svg',dpi=350, bbox_inches='tight')

plt.show()

#%% Section 6 - UMAP of activation status


# SF5 A
 
#list of parameters we want to use for the UMAP. I used ten OMI features (Normalized redox ratio, NAD(P)H lifetimes, FAD lifetimes, and cell size)
list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'


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
        width=750, 
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
export_svgs(plot, filename = './figures/all/SF5_A_AllCell_ActStatus_umap.svg')
# hv.save(overlay, './figures/AllCell_ActStatus_umap.svg')

#%% Section 7 - UMAP of cell type


# SF6 A 

#Same structure as Section 6 - see comments for more detail 

list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'

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
        width=750, 
        height=600),
    opts.Overlay(
        title='',
        legend_opts={"click_policy": "hide"},
        legend_position='bottom_right'
        )       
    )


plot = hv.render(overlay)
plot.output_backend = "svg"
export_svgs(plot, filename = './figures/all/SF6_A_AllCell_CellType_umap.svg')
# hv.save(overlay, 'AllCell_CellType_umap.svg')

#%% Setion 8 - UMAP of cell type (QUIESCENT ONLY)


# SF7_A
#Same structure as Section 6 - see comments for more detail 

list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'

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
        width=750, 
        height=600),
    opts.Overlay(
        title='',
        legend_opts={"click_policy": "hide"},
        legend_position='bottom_right'
        )       
    )

plot = hv.render(overlay)
plot.output_backend = "svg"
export_svgs(plot, filename = './figures/all/SF7_A_AllCell_CellType_QuiOnly_umap.svg')
# hv.save(overlay, 'AllCell_CellType_QuiOnly_umap.svg')

#%% Section 9 - UMAP of cell type + activation status


# F5_C

#Same structure as Section 6 - see comments for more detail 

list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'

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
        width=750, 
        height=600),
    opts.Overlay(
        title='',
        legend_opts={"click_policy": "hide"},
        legend_position='bottom_right'
        )       
    )

plot = hv.render(overlay)
plot.output_backend = "svg"
export_svgs(plot, filename = './figures/all/F5_C_AllCell_CellType_ActStatus_umap.svg')
# hv.save(overlay, 'AllCell_CellType_ActStatus_umap.svg')




#%% Section 10 - UMAP of cell type color-coded by donor


# NOT a FIGURE in paper 

#Same structure as Section 6 - see comments for more detail 


list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'

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
export_svgs(plot, filename = './figures/all/NF_AllCell_CellType_Donor_umap.svg')
# hv.save(overlay, 'AllCell_CellType_Donor_umap.svg')
#%% Section 11 - UMAP of activation status color-coded by donor


# SF8_A

#Same structure as Section 6 - see comments for more detail 


list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'

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
        height=750),
    opts.Overlay(
        title='',
        legend_opts={"click_policy": "hide"},
        legend_position='right'
        )       
    )

plot = hv.render(overlay)
plot.output_backend = "svg"
export_svgs(plot, filename = './figures/all/SF8_A_AllCell_CellType_Donor_ActStatus_umap.svg')

# hv.save(overlay, "figures/" 'AllCell_CellType_Donor_ActStatus_umap.svg')


#%% MERGED FROM THE PREVIOUS SCRIPT 



# import umap.umap_ as umap
# import numpy as np 
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

# from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.preprocessing import label_binarize
# from sklearn.model_selection import train_test_split

# import holoviews as hv
# hv.extension("bokeh")
# from holoviews import opts
# from holoviews.plotting import list_cmaps



#%% Section 2 Set-up for classifiers + ROC curves


def calculate_roc_rf(rf_df, key='Activation'): 
    
    # Need to binarize the problem as a 'One vs. all' style approach for ROC classification
    classes = ['CD69-', 'CD69+']

    #designate train/test data, random forest classifier
    X, y = rf_df.iloc[:,:-1], rf_df[[key]]
    y = label_binarize(y, classes=classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    y_train = np.ravel(y_train)
    clf = RandomForestClassifier(random_state=0)
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    y_pred = clf.fit(X_train, y_train).predict(X_test)


    # Compute ROC curve and ROC area for each class
    
    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize = 20)
    plt.ylabel('True Positive Rate', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.title('')
    plt.legend(loc="lower right", fontsize = 20)
    plt.show()
    
#%% Section 3 - Read in and set up dataframe 

#Read in dataframe    

# all_df = pd.read_csv('Z:/0-Projects and Experiments/RS - lymphocyte activation/data/AllCellData.csv')
# all_df = pd.read_csv('Data files/UMAPs, boxplots, ROC curves (Python)/AllCellData.csv')

# #Add combination variables to data set
# all_df.drop(['NADH', 'Group', 'Experiment_Date'], axis=1, inplace=True)
# all_df['Type_Activation'] = all_df['Cell_Type'] + ': ' + all_df['Activation']
# all_df['Donor_Activation'] = all_df['Cell_Type'] +' '+ all_df['Donor'] + ': ' + all_df['Activation']
# all_df['Donor_CellType'] = all_df['Donor'] + ': ' + all_df['Cell_Type'] 

# df_data = all_df.copy()

#%% Section 4 - All cell activation classifier


# NO FIGURE in paper
print('All cell activation classifier')

#List of OMI variables we want in the classifier (**Make sure Activation is last item in list)
list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR',  'Activation'] # , 'Cell_Size_Pix'

   
#Make copy of main data frame, pull out OMI variables we want in classifier
all_df_edit = all_df.copy()
all_df_edit = all_df_edit[list_omi_parameters]
classes = ['CD69-', 'CD69+']


#Split training/testing data, random forest classifier
X, y = all_df_edit.iloc[:,:-1], all_df_edit[['Activation']]
y = label_binarize(y, classes=classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
y_train = np.ravel(y_train)
clf = RandomForestClassifier(random_state=0)
y_score = clf.fit(X_train, y_train).predict_proba(X_test)
y_pred = clf.fit(X_train, y_train).predict(X_test)

#Calculate and display confusion matrix
factor = pd.factorize(all_df_edit[['Activation']].squeeze())
definitions = factor[1]
reversefactor = dict(zip(range(5), definitions))
y_test_rf = np.vectorize(reversefactor.get)(y_test)
y_pred_rf = np.vectorize(reversefactor.get)(y_pred)
print(pd.crosstab(np.ravel(y_test_rf), y_pred_rf, rownames=['Actual Condition'], colnames=['Predicted Condition']))

#Print features with weight in classifier
for col, feature in zip(np.flip(all_df_edit.columns[np.argsort(clf.feature_importances_)]), np.flip(np.argsort(clf.feature_importances_))):
    print(col, clf.feature_importances_[feature])

#Generate ROC curve
omi_params_umap = all_df_edit.copy()
calculate_roc_rf(omi_params_umap)    

#Print metrics to assess classifier performance
print('Accuracy score =', accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))



#%% Section 5 - Cell Type Classifier

# SF6_C

print('All cell data cell type classifier')

#List of OMI variables we want in the classifier - do NOT have to list variable with classes ('Cell_Type'), just OMI variables
#TODO SF6 -confusion matrix and pie chart 
list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'

# F5 E accuracies

# list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'

# # # # # # # ##### Top variables
# list_omi_parameters = ['FAD_tm']
# list_omi_parameters = ['FAD_tm', 'FAD_t1']
# list_omi_parameters = ['FAD_tm', 'FAD_t1', 'NADH_tm']
# list_omi_parameters = ['FAD_tm', 'FAD_t1', 'NADH_tm', 'FAD_a1']
# # # # # # # #####
# list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2'] # , 'Cell_Size_Pix'
# list_omi_parameters = ['Norm_RR', ] #'Cell_Size_Pix'
# list_omi_parameters = ['NADH_a1']
   
#Make copy of main data frame, pull out OMI variables we want in classifier
all_df_edit = all_df.copy()
all_df_edit = all_df_edit[list_omi_parameters]

from sklearn.preprocessing import StandardScaler

#extracts classes from variable of interest (here it's cell type)
classes = all_df.Cell_Type.unique()

factor = pd.factorize(all_df.Cell_Type)

#make new column with labels for each class
all_df['CT_LABELS'] = factor[0]
definitions = factor[1]
X, y = all_df_edit, all_df['CT_LABELS'].values

#Designate train/test data, random forest classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
clf = RandomForestClassifier(random_state=0, class_weight=None)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


#Generate and print confusion matrix
reversefactor = dict(zip(range(len(classes)), definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
print("SF6_C   | T B and NK cells")
# cm_table = pd.crosstab(y_test, y_pred, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100
# print(cm_table)
cm_table = pd.crosstab(y_test, y_pred, rownames=['Actual Condition'], colnames=['Predicted Condition'])
print(cm_table)

print("+"*20)
print("Figure SF6_C piechart of importance on all features")
forest_importances = pd.Series(clf.feature_importances_*100, index=list_omi_parameters).sort_values(ascending=False)
print(forest_importances)
print("+"*20)

#Print metrics for classifier assessment
print('Accuracy score =', accuracy_score(y_test, y_pred))
# print(classification_report(y_test,y_pred))

#%% Section 6 - Cell type classifer (QUIESCENT ONLY)
from sklearn.preprocessing import StandardScaler

#TODO SF7 D confusion matrix and pie chart

list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'

#List of OMI variables we want to include in the classifier. No variable with classes is needed - that is extracted later
# SF 7 B accuracies
# list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] #, 'Cell_Size_Pix'

# # # # # # # ##### Top variables
# list_omi_parameters = ['FAD_tm']
# list_omi_parameters = ['FAD_tm', 'FAD_t1']
# list_omi_parameters = ['FAD_tm', 'FAD_t1','NADH_tm']
# list_omi_parameters = ['FAD_tm', 'FAD_t1','NADH_tm', 'NADH_a1']
# # # # # # # ##### Top variables

# list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2'] #, 'Cell_Size_Pix'
# list_omi_parameters = ['Norm_RR'] # , 'Cell_Size_Pix'

#Create subset of dataset that only contains CD69- control cells 
all_df_qonly = all_df.loc[all_df['Activation']=='CD69-']
all_df_edit = all_df_qonly[list_omi_parameters]


print('All cell data cell type classifier - Quiescent cells only')

#Same classifier code structure as Section 5 - see Section 5 comments for details
classes = all_df_qonly.Cell_Type.unique()
factor = pd.factorize(all_df_qonly.Cell_Type)
all_df_qonly['CT_LABELS'] = factor[0]
definitions = factor[1]

X, y = all_df_edit, all_df_qonly['CT_LABELS'].values

#####
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clf = RandomForestClassifier(random_state=0, class_weight=None)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

reversefactor = dict(zip(range(len(classes)), definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)

print("+"*20)
print("Figure SF7_C piechart of importance on all features")
forest_importances = pd.Series(clf.feature_importances_*100, index=list_omi_parameters).sort_values(ascending=False)
print(forest_importances)
print("+"*20)

print("SF7_D Quiescent")
# cm_table = pd.crosstab(y_test, y_pred, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100
# print(cm_table)
print("-"*20)
cm_table = pd.crosstab(y_test, y_pred, rownames=['Actual Condition'], colnames=['Predicted Condition'])
print(cm_table)

# for col, feature in zip(np.flip(all_df_edit.columns[np.argsort(clf.feature_importances_)]), np.flip(np.argsort(clf.feature_importances_))):
#     print(col, clf.feature_importances_[feature])

print('Accuracy score =', accuracy_score(y_test, y_pred))
# print(classification_report(y_test,y_pred))

#%% Section 7 - Cell type + activation classifier

#TODO SF8 C

print('All cell data cell type + activation classifier')

#Same classifier code structure as Section 5 - see Section 5 comments for details


list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] #, 'Cell_Size_Pix'

# # # # ## Figure 5 F
# list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'

# # # # # # ##### Top variables
# list_omi_parameters = ['NADH_a1']
# list_omi_parameters = ['NADH_a1', 'NADH_t1']
# list_omi_parameters = ['NADH_a1', 'NADH_t1','NADH_tm'] # , 'Cell_Size_Pix'
# list_omi_parameters = ['NADH_a1', 'NADH_t1', 'NADH_tm', 'FAD_tm'] # , 'Cell_Size_Pix'
# # # # # # # ##### Top variables


# list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2'] # , 'Cell_Size_Pix'
# list_omi_parameters = ['Norm_RR'] # , 'Cell_Size_Pix'


all_df_edit = all_df.copy()
all_df_edit = all_df_edit[list_omi_parameters]

from sklearn.preprocessing import StandardScaler

classes = all_df.Type_Activation.unique()

factor = pd.factorize(all_df.Type_Activation)

all_df['CT_LABELS'] = factor[0]

definitions = factor[1]

X, y = all_df_edit, all_df['CT_LABELS'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clf = RandomForestClassifier(random_state=0, class_weight=None)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

reversefactor = dict(zip(range(len(classes)), definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)


print("+"*20)
print("Figure SF8_B piechart of importance on all features")
forest_importances = pd.Series(clf.feature_importances_*100, index=list_omi_parameters).sort_values(ascending=False)
print(forest_importances)
print("+"*20)

print("SF8_C")
# cm_table1 = pd.crosstab(y_test, y_pred, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100
# print(cm_table1)
print("-" * 30)
cm_table2 = pd.crosstab(y_test, y_pred, rownames=['Actual Condition'], colnames=['Predicted Condition'])
print(cm_table2)
# for col, feature in zip(np.flip(all_df_edit.columns[np.argsort(clf.feature_importances_)]), np.flip(np.argsort(clf.feature_importances_))):
#     print(col, clf.feature_importances_[feature])

print('Accuracy score =', accuracy_score(y_test, y_pred))
# print(classification_report(y_test,y_pred))

