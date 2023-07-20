#%% Section 1 - Import needed modules

# from copy import deepcopy

import umap.umap_ as umap
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from bokeh.io import export_svgs

from pprint import pprint

import holoviews as hv
hv.extension("bokeh")
from holoviews import opts
# from holoviews.plotting import list_cmaps

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

from helper import run_analysis_on_classifier, _train_test_split

#%% Section 2 - Read in B-cell data

bcell_df = pd.read_csv('data/UMAPs, boxplots, ROC curves (Python)/Bcell_cyto_2group.csv')
bcell_df.drop(['NADH', 'Experiment_Date'], axis=1, inplace=True)

bcell_df['Act_Donor'] = bcell_df['Activation'] + ' ' + bcell_df['Donor']
bcell_df.groupby(by=['Donor','Group','Activation'])['Cell_Type'].count()

df_data = bcell_df.copy()

classes = ['CD69-', 'CD69+']
dict_classes = {label_int : label_class for label_int, label_class in enumerate(classes)}
#%% Section 3 - ROC curves for different variable combos plotted together

# F2 D
# F2 C -- feature importances
# F2 D/ST1 - accuracies
     
#B-cell random forest classifiers

sns.set(rc={'figure.figsize': (15, 15)})
sns.set_style(style='white')
plt.rcParams['svg.fonttype'] = 'none'

dict_accuracies = {}
#Generate df with only the OMI variables we want to include in the classifier (***Always keep Activation column last)

colors = ['#fde725', '#a5db36', '#4ac16d','#1f988b','#2a788e','#414487', '#440154']
custom_color = sns.set_palette(sns.color_palette(colors))

##%% ################## 10 features
#Generate df with only the OMI variables we want to include in the classifier (***Always keep Activation column last)
list_cols = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'
X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)


clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['all features'] = accuracy

### Figure 4 C
print("F2_C piechart of importance on all features")
forest_importances = pd.Series(clf.feature_importances_*100, index=X_train.keys()).sort_values(ascending=False)
print(forest_importances)

df_acc = pd.DataFrame(forest_importances)
df_acc.to_csv('./figures/F2/F2_C_feature_importances.csv')

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='All variables (ROC AUC = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## NADH variables + Cell Size
list_cols = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2'] # , 'Cell_Size_Pix'

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)
clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['nadh_features'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='NAD(P)H variables (ROC AUC = %0.2f)' % roc_auc, linewidth = 5) # + Cell Size
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## Top 4 (NADH a1, NADH tm, FAD t2, NADH t1)

# list_cols = ['NADH_tm', 'NADH_a1', 'FAD_t2',  'NADH_t1']
list_cols = list(forest_importances.keys()[:4])
X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)
clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['top_4'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='Top four variables (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## Top 3 (NADH a1, NADH tm, FAD t2)

# list_cols = ['NADH_tm', 'NADH_a1',  'FAD_t2']
list_cols = list(forest_importances.keys()[:3])

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)
clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['top_3'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='Top three variables (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## Top 2 (NADH a1, NADH tm)
# list_cols = ['NADH_a1',  'NADH_tm']
list_cols = list(forest_importances.keys()[:2])

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)
clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['top_2'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='Top two variables (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## Top variable (NADH a1)

# list_cols = ['NADH_a1']
list_cols = list(forest_importances.keys()[:1])

X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)
clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)
dict_accuracies['top_1'] = accuracy

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='Top variable (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## Redox + Cell Size

list_cols = ['Norm_RR'] # 'Cell_Size_Pix',  

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
plt.title('Figure 2. Bcells', fontsize = 36)
plt.legend(bbox_to_anchor=(-0.1,-0.1), loc="upper left", fontsize = 36)
plt.savefig('./figures/F2/F2_C_RS_bcell_ROC.svg',dpi=350, bbox_inches='tight')
plt.show()

print("F2 D : B Cell Accuracies - Random Forest")
pprint(dict_accuracies)
df_acc = pd.DataFrame(dict_accuracies, index=[0])
df_acc.to_csv('./figures/F2/F2_D_accuracies.csv')

 #%% Section 4 - Box-and-whisker/swarm plots with red and blue color coding

df = bcell_df.copy()

f, ax = plt.subplots(figsize=(8,20))

#colors = ['#fde725', '#21918c', '#440154']
#custom_color = sns.set_palette(sns.color_palette(colors))
colors = {'CD69- A':'#222255', 
          'CD69- B':'#4393C3', 
          'CD69- C':'#99DDFF',
          'CD69+ A':'#662506', 
          'CD69+ B':'#D6604D', 
          'CD69+ C':'#F4A582'}
sns.set_style(style='white')


PROPS_BLACK = {'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
               'medianprops': {'color': 'black'},
               'whiskerprops': {'color': 'black'},
               'capprops': {'color': 'black'}}

boxplot_line_width= 5
boxplot_width = 0.9
x_axis_col_name = 'Activation'
y_axis_col_name = 'Norm_RR'

#graph with indiviual palette

sns.swarmplot(x='Activation', y='Norm_RR', hue='Act_Donor',  order = ['CD69-', 'CD69+'], palette = colors, size = 5.2, data = df, zorder = 0.5)
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
#plt.ylabel(r'FAD $\tau _m$', size = 40, fontweight='bold')
plt.ylabel('Normalized Redox Ratio', size = 40, fontweight='bold')
plt.ylim(0.1, 1.65)
plt.xticks(size = 36, fontweight='bold')
plt.yticks(size = 30, fontweight='bold')
plt.legend(handles,labels,loc = 'lower right', fontsize = 36)
#plt.legend([],[], frameon=False)


#add bar for p-value
x1, x2 = 0, 1   
y, h, col = bcell_df['Norm_RR'].max() + .05, .05, 'k'
plt.tight_layout()
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=5, c=col)
plt.text((x1+x2)*.5, y+h, "****", ha='center', va='bottom', color=col, size = 40)

# plt.savefig('./figures/b/bcell_rr.svg',dpi=350)


#%% Section 5 - ROC curves: Random forest, Logistic Regression, SVM 

# SF2 F
# SF2 CDE - confusion matrices

sns.set(rc={'figure.figsize': (15, 15)})
sns.set_style(style='white')
plt.rcParams['svg.fonttype'] = 'none'

colors = ['#fde725', '#1f988b','#440154']
custom_color = sns.set_palette(sns.color_palette(colors))

class_weight = None 
# class_weight = 'balanced'

list_cols = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'
X_train, X_test, y_train, y_test = _train_test_split(df_data, list_cols, classes)

##%% ################## RANDOM FOREST
clf = RandomForestClassifier(random_state=0, class_weight=class_weight).fit(X_train, y_train)
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (ROC AUC = %0.2f)' % roc_auc, linewidth = 7)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## LOGISTIC REGRESSION
clf = LogisticRegression(random_state=0, class_weight=class_weight).fit(X_train, y_train) #JR - use for logistic regression
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)


plt.plot(fpr, tpr, label='Logistic Regression (ROC AUC = %0.2f)' % roc_auc, linewidth = 7)
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

##%% ################## SVC
clf = SVC(probability=True, class_weight=class_weight).fit(X_train, y_train) #JR - use for SVM
fpr, tpr, roc_auc, accuracy, op_point  = run_analysis_on_classifier(clf, X_test, y_test, dict_classes)

# Plot of a ROC curve for a specific class
plt.scatter(op_point[0],op_point[1], c='k', s= 500, zorder=2)

plt.plot(fpr, tpr, label='Support Vector Machine (ROC AUC = %0.2f)' % roc_auc, linewidth = 7)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize = 36)
plt.ylabel('True Positive Rate', fontsize = 36)
plt.xticks(fontsize = 36)
plt.yticks(fontsize = 36)
plt.title(f'SF2. B Cells | class_weight: {class_weight}', fontsize = 36)
plt.legend(bbox_to_anchor=(-0.1,-0.1), loc="upper left", fontsize = 36)
plt.savefig('./figures/SF2/SF2_F_RS_bcell_SVMLR_ROC.svg',dpi=350, bbox_inches='tight')

plt.show()

#%% Section 6 - B-cell UMAP - by CD69+ activation status

#list of parameters we want to use for the UMAP. I used ten OMI features (Normalized redox ratio, NAD(P)H lifetimes, FAD lifetimes, and cell size)

# F2 B Activation

list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR'] # , 'Cell_Size_Pix'


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
        size=4,
        alpha = 0.75,
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
export_svgs(plot, filename = './figures/F2/F2_B_BCell_ActStatus_umap.svg')
# hv.save(overlay, 'BCell_ActStatus_umap.html')


#%%  Section 7 - B-cell UMAP - By donor

# NF not in figure
#Same structure as Section 7 - see comments above

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
legend_entries = "Donor" 

########
df_data = df_data.copy()
df_data["umap_x"] = fit_umap.embedding_[:,0]
df_data["umap_y"] = fit_umap.embedding_[:,1]

kdims = ["umap_x"]
vdims = ["umap_y", hover_vdim]
list_entries = np.unique(df_data[legend_entries])


                    
scatter_umaps = [hv.Scatter(df_data[df_data[legend_entries] == entry], kdims=kdims, 
                            vdims=vdims, label=entry) for entry in list_entries]

#formatting for holoviews plot
colors = [ '#440154','#21918c','#fde725']
for scatter, color in zip(scatter_umaps, colors):
    scatter.opts(color=color)

overlay = hv.Overlay(scatter_umaps)
overlay.opts(
    opts.Scatter(
        #fontsize={'labels': 14, 'xticks': 12, 'yticks': 12},
        fontscale=2,
        size=4,
        tools=["hover"],
        muted_alpha=0,
        alpha = 0.75,
        aspect="equal",
        width=600, 
        height=600),
    opts.Overlay(
        title='',
        legend_opts={"click_policy": "hide"},
        legend_position='bottom_right'
        )       
    )

#saves UMAP as interactive HMTL
plot = hv.render(overlay)
plot.output_backend = "svg"
# export_svgs(plot, filename = './figures/b/NF_BCell_Donor_umap.svg')
# hv.save(overlay, 'BCell_Donor_umap.html')


#%%  Section 8 - B-cell UMAP - by donor and activation 

# SF2 B

#Generate column in data frame that has both donor and activation status
df_data['Donor_Activation'] = df_data['Donor'] + ': ' + df_data['Activation']

#Same structure as Section 7 - see comments above
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
export_svgs(plot, filename = './figures/SF2/SF2_B_BCell_Donor_ActStatus_umap.svg')
# hv.save(overlay, 'BCell_Donor_ActStatus_umap.html')


#%%  Section 9 - B-cell data UMAP with all groups (Activated + Quiescent, CD69+ and CD69-)

# SF2 A

#Read in CSV that has data from all 4 combinations of activation/culture condition
# allgroup_b_df = pd.read_csv('Z:/0-Projects and Experiments/RS - lymphocyte activation/data/B-cells (Donors 1-3)/Bcell_cyto_data.csv')
allgroup_b_df = pd.read_csv('./Data files/UMAPs, boxplots, ROC curves (Python)/Bcell_cyto_data.csv')


df_data = allgroup_b_df.copy()

#generate column in dataframe that combines both culture group and activation status

df_data['Group_Activation'] = df_data['Group'] + ': ' + df_data['Activation']


#Same structure as Section 7 - see comments above
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

#
                    
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
        size=4,
        alpha = 0.75,
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
export_svgs(plot, filename = './figures/SF2/SF2_A_BCell_ActStatus_Condition_umap.svg')

# hv.save(overlay, 'BCell_ActStatus_Condition_umap.html')

