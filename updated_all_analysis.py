# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:57:16 2022

@author: rschmitz
"""

#%% Section 1 - Import required packages

import umap.umap_ as umap
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

import holoviews as hv
hv.extension("bokeh")
from holoviews import opts
from holoviews.plotting import list_cmaps

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
all_df = pd.read_csv('Data files/UMAPs, boxplots, ROC curves (Python)/AllCellData.csv')

#Add combination variables to data set

all_df.drop(['NADH', 'Group', 'Experiment_Date'], axis=1, inplace=True)
all_df['Type_Activation'] = all_df['Cell_Type'] + ': ' + all_df['Activation']
all_df['Donor_Activation'] = all_df['Cell_Type'] +' '+ all_df['Donor'] + ': ' + all_df['Activation']
all_df['Donor_CellType'] = all_df['Donor'] + ': ' + all_df['Cell_Type'] 

df_data = all_df.copy()

#%% Section 4 - All cell activation classifier


print('All cell activation classifier')

#List of OMI variables we want in the classifier (**Make sure Activation is last item in list)
list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix', 'Activation']

   
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
print(pd.crosstab(np.ravel(y_test_rf), y_pred_rf, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100)

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

#TODO

print('All cell data cell type classifier')


#List of OMI variables we want in the classifier - do NOT have to list variable with classes ('Cell_Type'), just OMI variables
list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix']

   
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
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#Generate and print confusion matrix
reversefactor = dict(zip(range(len(classes)), definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
print("S6 T B and NK cells")
cm_table = pd.crosstab(y_test, y_pred, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100
print(cm_table)
cm_table = pd.crosstab(y_test, y_pred, rownames=['Actual Condition'], colnames=['Predicted Condition'])
print(cm_table)

#List weights of each feature in classifier
# for col, feature in zip(np.flip(all_df_edit.columns[np.argsort(clf.feature_importances_)]), np.flip(np.argsort(clf.feature_importances_))):
#     print(col, clf.feature_importances_[feature])

#Print metrics for classifier assessment
print('Accuracy score =', accuracy_score(y_test, y_pred))
# print(classification_report(y_test,y_pred))

#%% Section 6 - Cell type classifer (QUIESCENT ONLY)

#TODO

#List of OMI variables we want to include in the classifier. No variable with classes is needed - that is extracted later
list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix']

#Create subset of dataset that only contains CD69- control cells 
   
all_df_qonly = all_df.loc[all_df['Activation']=='CD69-']
all_df_edit = all_df_qonly[list_omi_parameters]

print('All cell data cell type classifier - Quiescent cells only')

#Same classifier code structure as Section 5 - see Section 5 comments for details

from sklearn.preprocessing import StandardScaler

classes = all_df_qonly.Cell_Type.unique()

factor = pd.factorize(all_df_qonly.Cell_Type)

all_df_qonly['CT_LABELS'] = factor[0]

definitions = factor[1]

X, y = all_df_edit, all_df_qonly['CT_LABELS'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clf = RandomForestClassifier(random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

reversefactor = dict(zip(range(len(classes)), definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)

print("SF 7 Quiescent")
cm_table = pd.crosstab(y_test, y_pred, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100
print(cm_table)
print("-"*20)
cm_table = pd.crosstab(y_test, y_pred, rownames=['Actual Condition'], colnames=['Predicted Condition'])
print(cm_table)



# for col, feature in zip(np.flip(all_df_edit.columns[np.argsort(clf.feature_importances_)]), np.flip(np.argsort(clf.feature_importances_))):
#     print(col, clf.feature_importances_[feature])

print('Accuracy score =', accuracy_score(y_test, y_pred))
# print(classification_report(y_test,y_pred))



#%% Section 7 - Cell type + activation classifier

#TODO

print('All cell data cell type + activation classifier')

#Same classifier code structure as Section 5 - see Section 5 comments for details


list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix']

   
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

clf = RandomForestClassifier(random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

reversefactor = dict(zip(range(len(classes)), definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)

print("SF8")
cm_table1 = pd.crosstab(y_test, y_pred, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100
print(cm_table1)
print("-" * 30)
cm_table2 = pd.crosstab(y_test, y_pred, rownames=['Actual Condition'], colnames=['Predicted Condition'])
print(cm_table2)
# for col, feature in zip(np.flip(all_df_edit.columns[np.argsort(clf.feature_importances_)]), np.flip(np.argsort(clf.feature_importances_))):
#     print(col, clf.feature_importances_[feature])

print('Accuracy score =', accuracy_score(y_test, y_pred))
# print(classification_report(y_test,y_pred))



#%% Section 8 - UMAP of activation status

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
hv.save(overlay, 'all_act_umap.html')

#%% Section 9 - UMAP of cell type

#Same structure as Section 8 - see comments for more detail 

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

hv.save(overlay, 'all_CT_umap.html')

#%% Setion 10 - UMAP of cell type (QUIESCENT ONLY)


#Same structure as Section 8 - see comments for more detail 

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

hv.save(overlay, 'all_CT_QUI_umap.html')

#%% Section 11 - UMAP of cell type + activation status


#Same structure as Section 8 - see comments for more detail 

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

hv.save(overlay, 'all_CTA_umap.html')




#%% Section 12 - UMAP of cell type color-coded by donor


#Same structure as Section 8 - see comments for more detail 


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

hv.save(overlay, 'all_D_umap.html')
#%% Section 13 - UMAP of activation status color-coded by donor


#Same structure as Section 8 - see comments for more detail 


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

hv.save(overlay, 'all_DA_umap.html')