#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:45:58 2022

@author: rschmitz
"""

#%%  Section 1 

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

import pandas as pd
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sns.set(rc={'figure.figsize': (15, 15)})
sns.set_style(style='white')
plt.rcParams['svg.fonttype'] = 'none'

#%% Section 2

nk_df = pd.read_csv('./Data files/UMAPs, boxplots, ROC curves (Python)/NK data 2 groups.csv')
nk_df.drop(['NADH', 'Group', 'Experiment_Date'], axis=1, inplace=True)

nk_df['Act_Donor'] = nk_df['Activation'] + ' ' + nk_df['Donor']

df_data = nk_df.copy()

#%% Section 3 - Generate graph with all ROCs

colors = ['#fde725', '#a5db36', '#4ac16d','#1f988b','#2a788e','#414487', '#440154']

custom_color = sns.set_palette(sns.color_palette(colors))

#Generate df with only the OMI variables we want to include in the classifier (***Always keep Activation column last)


nk_rf = nk_df[['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix', 'Activation']]

classes = ['CD69-', 'CD69+']

#designate train/test data, random forest classifier
X, y = nk_rf.iloc[:,:-1], nk_rf[['Activation']]
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
plt.plot(fpr, tpr, label='All variables (ROC AUC = %0.2f)' % roc_auc, linewidth = 5)

#generate and print confusion matrix and feature weights
factor = pd.factorize(nk_rf[['Activation']].squeeze())
definitions = factor[1]
reversefactor = dict(zip(range(5), definitions))
y_test_rf = np.vectorize(reversefactor.get)(y_test)
y_pred_rf = np.vectorize(reversefactor.get)(y_pred)
print(pd.crosstab(np.ravel(y_test_rf), y_pred_rf, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100)
for col, feature in zip(np.flip(nk_rf.columns[np.argsort(clf.feature_importances_)]), np.flip(np.argsort(clf.feature_importances_))):
    print(col, clf.feature_importances_[feature])



#print metrics to assess classifier performance
print('Accuracy score =', accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))



#NADH variables + Cell Size

nk_rf = nk_df[['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'Cell_Size_Pix', 'Activation']]

classes = ['CD69-', 'CD69+']

#designate train/test data, random forest classifier
X, y = nk_rf.iloc[:,:-1], nk_rf[['Activation']]
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
plt.plot(fpr, tpr, label='NAD(P)H variables + Cell Size (ROC AUC = %0.2f)' % roc_auc, linewidth = 5)


#generate and print confusion matrix and feature weights
factor = pd.factorize(nk_rf[['Activation']].squeeze())
definitions = factor[1]
reversefactor = dict(zip(range(5), definitions))
y_test_rf = np.vectorize(reversefactor.get)(y_test)
y_pred_rf = np.vectorize(reversefactor.get)(y_pred)
print(pd.crosstab(np.ravel(y_test_rf), y_pred_rf, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100)
for col, feature in zip(np.flip(nk_rf.columns[np.argsort(clf.feature_importances_)]), np.flip(np.argsort(clf.feature_importances_))):
    print(col, clf.feature_importances_[feature])


#print metrics to assess classifier performance
print('Accuracy score =', accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))








#Top 4 (NADH a1, Norm RR, NADH tm, NADH t2)


nk_rf = nk_df[['NADH_tm', 'NADH_a1', 'NADH_t2',  'Norm_RR', 'Activation']]

classes = ['CD69-', 'CD69+']

#designate train/test data, random forest classifier
X, y = nk_rf.iloc[:,:-1], nk_rf[['Activation']]
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
plt.plot(fpr, tpr, label='Top four variables (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)


#generate and print confusion matrix and feature weights
factor = pd.factorize(nk_rf[['Activation']].squeeze())
definitions = factor[1]
reversefactor = dict(zip(range(5), definitions))
y_test_rf = np.vectorize(reversefactor.get)(y_test)
y_pred_rf = np.vectorize(reversefactor.get)(y_pred)
print(pd.crosstab(np.ravel(y_test_rf), y_pred_rf, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100)
for col, feature in zip(np.flip(nk_rf.columns[np.argsort(clf.feature_importances_)]), np.flip(np.argsort(clf.feature_importances_))):
    print(col, clf.feature_importances_[feature])


#print metrics to assess classifier performance
print('Accuracy score =', accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))





#Top 3 (NADH a1, Norm RR, NADH tm)


nk_rf = nk_df[['NADH_tm', 'NADH_a1',  'Norm_RR', 'Activation']]

classes = ['CD69-', 'CD69+']

#designate train/test data, random forest classifier
X, y = nk_rf.iloc[:,:-1], nk_rf[['Activation']]
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
plt.plot(fpr, tpr, label='Top three variables (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)


#generate and print confusion matrix and feature weights
factor = pd.factorize(nk_rf[['Activation']].squeeze())
definitions = factor[1]
reversefactor = dict(zip(range(5), definitions))
y_test_rf = np.vectorize(reversefactor.get)(y_test)
y_pred_rf = np.vectorize(reversefactor.get)(y_pred)
print(pd.crosstab(np.ravel(y_test_rf), y_pred_rf, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100)
for col, feature in zip(np.flip(nk_rf.columns[np.argsort(clf.feature_importances_)]), np.flip(np.argsort(clf.feature_importances_))):
    print(col, clf.feature_importances_[feature])


#print metrics to assess classifier performance
print('Accuracy score =', accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))





#Top 2 (NADH a1, Norm RR)


nk_rf = nk_df[['NADH_a1',  'Norm_RR', 'Activation']]

classes = ['CD69-', 'CD69+']

#designate train/test data, random forest classifier
X, y = nk_rf.iloc[:,:-1], nk_rf[['Activation']]
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
plt.plot(fpr, tpr, label='Top two variables (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)


#generate and print confusion matrix and feature weights
factor = pd.factorize(nk_rf[['Activation']].squeeze())
definitions = factor[1]
reversefactor = dict(zip(range(5), definitions))
y_test_rf = np.vectorize(reversefactor.get)(y_test)
y_pred_rf = np.vectorize(reversefactor.get)(y_pred)
print(pd.crosstab(np.ravel(y_test_rf), y_pred_rf, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100)
for col, feature in zip(np.flip(nk_rf.columns[np.argsort(clf.feature_importances_)]), np.flip(np.argsort(clf.feature_importances_))):
    print(col, clf.feature_importances_[feature])


#print metrics to assess classifier performance
print('Accuracy score =', accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))




#Top variable (NADH a1)


nk_rf = nk_df[['NADH_a1', 'Activation']]

classes = ['CD69-', 'CD69+']

#designate train/test data, random forest classifier
X, y = nk_rf.iloc[:,:-1], nk_rf[['Activation']]
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
plt.plot(fpr, tpr, label='Top variable (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)


#generate and print confusion matrix and feature weights
factor = pd.factorize(nk_rf[['Activation']].squeeze())
definitions = factor[1]
reversefactor = dict(zip(range(5), definitions))
y_test_rf = np.vectorize(reversefactor.get)(y_test)
y_pred_rf = np.vectorize(reversefactor.get)(y_pred)
print(pd.crosstab(np.ravel(y_test_rf), y_pred_rf, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100)
for col, feature in zip(np.flip(nk_rf.columns[np.argsort(clf.feature_importances_)]), np.flip(np.argsort(clf.feature_importances_))):
    print(col, clf.feature_importances_[feature])


#print metrics to assess classifier performance
print('Accuracy score =', accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))




#Redox + Cell Size


nk_rf = nk_df[['Cell_Size_Pix',  'Norm_RR', 'Activation']]

classes = ['CD69-', 'CD69+']

#designate train/test data, random forest classifier
X, y = nk_rf.iloc[:,:-1], nk_rf[['Activation']]
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
plt.plot(fpr, tpr, label='Norm. Redox Ratio + Cell Size (ROC AUC) = %0.2f)' % roc_auc, linewidth = 5)


#generate and print confusion matrix and feature weights
factor = pd.factorize(nk_rf[['Activation']].squeeze())
definitions = factor[1]
reversefactor = dict(zip(range(5), definitions))
y_test_rf = np.vectorize(reversefactor.get)(y_test)
y_pred_rf = np.vectorize(reversefactor.get)(y_pred)
print(pd.crosstab(np.ravel(y_test_rf), y_pred_rf, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100)
for col, feature in zip(np.flip(nk_rf.columns[np.argsort(clf.feature_importances_)]), np.flip(np.argsort(clf.feature_importances_))):
    print(col, clf.feature_importances_[feature])


#print metrics to assess classifier performance
print('Accuracy score =', accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))




plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize = 36)
plt.ylabel('True Positive Rate', fontsize = 36)
plt.xticks(fontsize = 36)
plt.yticks(fontsize = 36)
plt.title('')
plt.legend(bbox_to_anchor=(-0.1,-0.1), loc="upper left", fontsize = 36)
plt.savefig('./figures/RS_nk_ROC.svg',dpi=350, bbox_inches='tight')

plt.show()


#%% Section 4 - donor color coding boxplots: Red/blue, all dots visible

df = nk_df.copy()

f, ax = plt.subplots(figsize=(8,20))

#colors = ['#fde725', '#21918c', '#440154']
#custom_color = sns.set_palette(sns.color_palette(colors))
colors = {'CD69- D':'#222255', 
          'CD69- E':'#4393C3', 
          'CD69- F':'#99DDFF',
          'CD69+ D':'#662506', 
          'CD69+ E':'#D6604D', 
          'CD69+ F':'#F4A582'}
sns.set_style(style='white')


PROPS_BLACK = {'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
               'medianprops': {'color': 'black'},
               'whiskerprops': {'color': 'black'},
               'capprops': {'color': 'black'}}

boxplot_line_width= 5
boxplot_width = 0.9
x_axis_col_name = 'Activation'
y_axis_col_name = 'NADH_a1'

#graph with indiviual palette

sns.swarmplot(x='Activation', y='NADH_a1', hue='Act_Donor',  order = ['CD69-', 'CD69+'], palette = colors, size = 5.2, data = df, zorder = 0.5)
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
plt.ylabel(r'NAD(P)H $\alpha _1$', size = 40, fontweight='bold')
#plt.ylabel('Normalized Redox Ratio', size = 40, fontweight='bold')
plt.xticks(size = 36, fontweight='bold')
plt.yticks(size = 30, fontweight='bold')
#plt.legend(handles,labels,loc = 'lower right', fontsize = 36)
plt.legend([],[], frameon=False)


#add bar for p-value
x1, x2 = 0, 1   
y, h, col = nk_df['NADH_a1'].max() + 0.5, 0.5, 'k'
plt.tight_layout()
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=5, c=col)
plt.text((x1+x2)*.5, y+h, "****", ha='center', va='bottom', color=col, size = 40)

plt.savefig('./figures/RS_nk_na1.svg',dpi=350, bbox_inches='tight')




#%% Section 5 - ROC curves: Random forest, Logistic Regression, SVM curves together

sns.set(rc={'figure.figsize': (15, 15)})
sns.set_style(style='white')
plt.rcParams['svg.fonttype'] = 'none'

colors = ['#fde725', '#1f988b','#440154']

custom_color = sns.set_palette(sns.color_palette(colors))


all_rf = nk_df[['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix', 'Activation']]

classes = ['CD69-', 'CD69+']

#designate train/test data, random forest classifier
X, y = all_rf.iloc[:,:-1], all_rf[['Activation']]
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
plt.plot(fpr, tpr, label='Random Forest (ROC AUC = %0.2f)' % roc_auc, linewidth = 7)

#generate and print confusion matrix and feature weights
factor = pd.factorize(all_rf[['Activation']].squeeze())
definitions = factor[1]
reversefactor = dict(zip(range(5), definitions))
y_test_rf = np.vectorize(reversefactor.get)(y_test)
y_pred_rf = np.vectorize(reversefactor.get)(y_pred)
print(pd.crosstab(np.ravel(y_test_rf), y_pred_rf, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100)
for col, feature in zip(np.flip(all_rf.columns[np.argsort(clf.feature_importances_)]), np.flip(np.argsort(clf.feature_importances_))):
    print(col, clf.feature_importances_[feature])



#print metrics to assess classifier performance
print('Accuracy score =', accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))


all_rf = nk_df[['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix', 'Activation']]

classes = ['CD69-', 'CD69+']

#designate train/test data, random forest classifier
X, y = all_rf.iloc[:,:-1], all_rf[['Activation']]
y = label_binarize(y, classes=classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
y_train = np.ravel(y_train)

clf = LogisticRegression(random_state=0) #JR - use for logistic regression


y_score = clf.fit(X_train, y_train).predict_proba(X_test)
y_pred = clf.fit(X_train, y_train).predict(X_test)

# Compute ROC curve and ROC area for each class



fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
roc_auc = auc(fpr, tpr)

# Plot of a ROC curve for a specific class

plt.plot(fpr, tpr, label='Logistic Regression (ROC AUC = %0.2f)' % roc_auc, linewidth = 7)

#generate and print confusion matrix and feature weights
factor = pd.factorize(all_rf[['Activation']].squeeze())
definitions = factor[1]
reversefactor = dict(zip(range(5), definitions))
y_test_rf = np.vectorize(reversefactor.get)(y_test)
y_pred_rf = np.vectorize(reversefactor.get)(y_pred)
print(pd.crosstab(np.ravel(y_test_rf), y_pred_rf, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100)




#print metrics to assess classifier performance
print('Accuracy score =', accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))



all_rf = nk_df[['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix', 'Activation']]

classes = ['CD69-', 'CD69+']

#designate train/test data, random forest classifier
X, y = all_rf.iloc[:,:-1], all_rf[['Activation']]
y = label_binarize(y, classes=classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
y_train = np.ravel(y_train)

clf = SVC(probability=True) #JR - use for SVM


y_score = clf.fit(X_train, y_train).predict_proba(X_test)
y_pred = clf.fit(X_train, y_train).predict(X_test)

# Compute ROC curve and ROC area for each class



fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
roc_auc = auc(fpr, tpr)

# Plot of a ROC curve for a specific class

plt.plot(fpr, tpr, label='Support Vector Machine (ROC AUC = %0.2f)' % roc_auc, linewidth = 7)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize = 36)
plt.ylabel('True Positive Rate', fontsize = 36)
plt.xticks(fontsize = 36)
plt.yticks(fontsize = 36)
plt.title('')
plt.legend(bbox_to_anchor=(-0.1,-0.1), loc="upper left", fontsize = 36)
plt.savefig('./figures/RS_nk_SVMLR_ROC.svg',dpi=350, bbox_inches='tight')

plt.show()



#generate and print confusion matrix and feature weights
factor = pd.factorize(all_rf[['Activation']].squeeze())
definitions = factor[1]
reversefactor = dict(zip(range(5), definitions))
y_test_rf = np.vectorize(reversefactor.get)(y_test)
y_pred_rf = np.vectorize(reversefactor.get)(y_pred)
print(pd.crosstab(np.ravel(y_test_rf), y_pred_rf, rownames=['Actual Condition'], colnames=['Predicted Condition'], normalize='columns')*100)




#print metrics to assess classifier performance
print('Accuracy score =', accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))


#%% Section 6 - NK cell activation UMAP

#list of parameters we want to use for the UMAP. I used ten OMI features (Normalized redox ratio, NAD(P)H lifetimes, FAD lifetimes, and cell size)

list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix']


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
        alpha = 0.75,
        size=4,
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
export_svgs(plot, filename = './figures/NKCell_ActStatus_umap.svg')
# hv.save(overlay, 'NKCell_ActStatus_umap.html')

#%% Section 7 -  Nk cell donor UMAP


#Same structure as Section 6 - see comments above

list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix']

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
export_svgs(plot, filename = './figures/NKCell_Donor_umap.svg')
# hv.save(overlay, 'NKCell_Donor_umap.html')


#%% Section 8 - Nk cell donor + activation UMAP

#Generate column in data frame that has both donor and activation status

df_data['Donor_Activation'] = df_data['Donor'] + ': ' + df_data['Activation']

#Same structure as Section 6 - see comments above

list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix']

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
export_svgs(plot, filename = './figures/NKCell_Donor_ActStatus_umap.svg')
# hv.save(overlay, 'NKCell_Donor_ActStatus_umap.html')

#%% Section 9 - UMAP of both groups and activation statuses of NK cells


#Read in CSV that has data from all 4 combinations of activation/culture condition

# allgroup_nk_df = pd.read_csv('Z:/0-Projects and Experiments/RS - lymphocyte activation/data/NK cells (Donors 4-6)/NK data all groups.csv')
allgroup_nk_df = pd.read_csv('./Data files/UMAPs, boxplots, ROC curves (Python)/NK data all groups.csv')

df_data = allgroup_nk_df.copy()

#generate column in dataframe that combines both culture group and activation status

df_data['Group_Activation'] = df_data['Group'] + ': ' + df_data['Activation']


#Same structure as Section 5 - see comments above
list_omi_parameters = ['NADH_tm', 'NADH_a1', 'NADH_t1', 'NADH_t2', 'FAD_tm', 'FAD_a1', 'FAD_t1', 'FAD_t2', 'Norm_RR', 'Cell_Size_Pix']

data = df_data[list_omi_parameters].values
scaled_data = StandardScaler().fit_transform(data)
reducer = umap.UMAP(
               n_neighbors=15,
               min_dist=0.5,   
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
        size = 4,
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
export_svgs(plot, filename = './figures/NKCell_ActStatus_Condition_umap.svg')

# hv.save(overlay, 'NKCell_ActStatus_Condition_umap.html')




#%% Graphing code from Peter -- probably don't need, but keeping just in case



def generate_sub_dataframes(df,
                            x_axis_col_name,
                            y_axis_col_name,
                            replicate_id_col_name,
                            replicate_id_list,
                            replicate_color_list,
                            condition_list):
    df_dict = dict()
    color_counter = 0
    if x_axis_col_name not in df.columns:
        raise ValueError(f'Please double check x axis column name! Acceptable column names: {list(df.columns)}')
    if replicate_id_col_name not in df.columns:
        raise ValueError(f'Please double check replicate column name! Acceptable column names: {list(df.columns)}')
    if y_axis_col_name not in df.columns:
        raise ValueError(f'Please double check y axis column name! Acceptable column names: {list(df.columns)}')
    for condition in condition_list:
        curr_condition_df = df[df[x_axis_col_name] == condition]
        if len(curr_condition_df) != 0:
            curr_replicate_df_dict = dict()
            for replicate in replicate_id_list:
                curr_replicate_df = curr_condition_df[curr_condition_df[replicate_id_col_name] == replicate]
                if len(curr_replicate_df) != 0:
                    curr_temp_dict = {'replicate': replicate,
                                      'color': replicate_color_list[color_counter],
                                      'df': curr_replicate_df}
                    curr_replicate_df_dict[replicate] = curr_temp_dict
                color_counter += 1
            df_dict[condition] = {'condition': condition,
                                  'condition_df': curr_condition_df,
                                  'replicate_df_dict': curr_replicate_df_dict}
    return df_dict


def generate_figure(df,
                    condition_list,
                    x_axis_col_name,
                    y_axis_col_name,
                    fig_title,
                    figsize,
                    fig_title_size, 
                    fig_title_style,
                    y_axis_label,
                    y_axis_label_size,
                    y_axis_label_style,
                    x_axis_label,
                    x_axis_label_size, 
                    x_axis_label_style,
                    x_axis_data_labels,
                    x_axis_data_label_size, 
                    x_axis_data_label_style,
                    swarmplot_size,
                    axis_line_width,
                    y_axis_tick_thickness,
                    y_axis_tick_label_size, 
                    y_axis_tick_label_style,
                    boxplot_line_width,
                    boxplot_width,
                    dotplot_replicate_alpha,
                    save_fig,
                    save_fig_filename, 
                    y_axis_ticks=None):
    PROPS_BLACK = {'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
                   'medianprops': {'color': 'black'},
                   'whiskerprops': {'color': 'black'},
                   'capprops': {'color': 'black'}}
    fig, axs = plt.subplots(ncols=len(condition_list),figsize=figsize, sharey=True, sharex=True)
    fig.suptitle(fig_title, fontsize=fig_title_size, weight=fig_title_style)
    fig.supxlabel(x_axis_label, fontsize=x_axis_label_size, weight=x_axis_label_style)
    for ix, (condition, curr_df) in enumerate(df.items()):
        
        for (replicate, replicate_df) in curr_df['replicate_df_dict'].items():
            sns.swarmplot(x=x_axis_col_name,
                          y=y_axis_col_name,
                          data=replicate_df['df'],
                          color=replicate_df['color'],
                          size=swarmplot_size,
                          ax=axs[ix],
                          alpha=dotplot_replicate_alpha, zorder=0.5)
        sns.boxplot(x=x_axis_col_name,
                    y=y_axis_col_name,
                    data=curr_df['condition_df'],
                    ax=axs[ix],
                    linewidth=boxplot_line_width,
                    width=boxplot_width,
                    showfliers=False,
                    zorder=1,
                    **PROPS_BLACK)
    for ix, ax in enumerate(axs):
        if y_axis_ticks: 
            ticks = [round(i, 2) for i in np.arange(y_axis_ticks[0], y_axis_ticks[1], round(y_axis_ticks[1]/y_axis_ticks[2], 3))]
            ticks.append(y_axis_ticks[1])
            plt.ylim(y_axis_ticks[0], y_axis_ticks[1])
        else:
            ticks = ax.get_yticks()
        if ix == 0:
            sns.despine(left=False, bottom=False, right=True, ax=ax)
            ax.spines['left'].set_linewidth(axis_line_width)
            ax.spines['bottom'].set_linewidth(axis_line_width)
            ax.set_yticks(ticks)
            ax.yaxis.set_tick_params(width=y_axis_tick_thickness)
            ax.set_yticklabels([str(round(i, 2)) for i in ticks], fontsize=y_axis_tick_label_size, weight=y_axis_tick_label_style)
            ax.set_xticklabels([str(i) for i in ax.get_xticks()], fontsize=0)
            ax.set_xlabel(x_axis_data_labels[ix], fontsize=x_axis_data_label_size, weight=x_axis_data_label_style)
            ax.set_ylabel(y_axis_label, fontsize=y_axis_label_size, weight=y_axis_label_style)
            ax.tick_params(left=True, length=10)
        else:
            sns.despine(left=True, bottom=False, right=True, ax=ax)
            ax.spines['bottom'].set_linewidth(axis_line_width)
            ax.set_xlabel(x_axis_data_labels[ix], fontsize=x_axis_data_label_size, weight=x_axis_data_label_style)
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(round(i, 2)) for i in ticks], fontsize=0, weight='bold')
            ax.set_ylabel(y_axis_label, fontsize=0, weight='bold')
            ax.set_xticklabels([str(i) for i in ax.get_xticks()], fontsize=0)
    
    if save_fig:
        fig.savefig(save_fig_filename, transparent=True)

sns.set(rc={'figure.figsize': (10, 15)})

colors = ['#fde725', '#21918c', '#440154']

custom_color = sns.set_palette(sns.color_palette(colors))



plt.rcParams['ytick.left'] = False


PROPS_BLACK = {'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
                'medianprops': {'color': 'black'},
                'whiskerprops': {'color': 'black'},
                'capprops': {'color': 'black'}}
fig_title = ''
figsize = (5,10)
fig_title_size = 25 
fig_title_style = 'normal'
y_axis_label = "Normalized Redox Ratio"
y_axis_label_size = 45
y_axis_label_style = 'normal'
x_axis_label = 'Activation'
x_axis_label_size = 45 
x_axis_label_style = 'normal'
x_axis_data_labels = ['CD69+','CD69-']
x_axis_data_label_size = 45
x_axis_data_label_style= 'normal'
boxplot_line_width= 5
axis_line_width = 4
y_axis_tick_thickness = 0.5
swarmplot_size = 3.5
dotplot_replicate_alpha = 0.8
boxplot_width = 0.9
save_fig = True
save_fig_filename = './figures/NK_redox.svg'
x_axis_col_name = 'Activation'
y_axis_col_name = 'Norm_RR'
replicate_id_col_name = 'Donor'
replicate_list = ['F','E','D']
condition_list = ['CD69+','CD69-']
y_axis_tick_label_size = 45
y_axis_tick_label_style = 'normal'
# Type None for auto scaling
y_axis_ticks = None
replicate_color_list = [ (68/255,1/255,84/255),
                          (33/255,145/255,140/255),
                          (253/255,231/255,137/255),
                          (68/255,1/255,84/255),
                          (33/255,145/255,140/255),
                          (253/255,231/255,137/255)]
assert(len(x_axis_data_labels) == len(condition_list))
assert(len(replicate_color_list) == (len(condition_list)*len(replicate_list)))
test_dataframe_dict = generate_sub_dataframes(df,
                                              x_axis_col_name,
                                              y_axis_col_name,
                                              replicate_id_col_name,
                                              replicate_list,
                                              replicate_color_list,
                                              condition_list)

#graph where each donor is overlaid on the others

generate_figure(test_dataframe_dict,
                condition_list,
                x_axis_col_name,
                y_axis_col_name,
                fig_title,
                figsize,
                fig_title_size, 
                fig_title_style,
                y_axis_label,
                y_axis_label_size,
                y_axis_label_style,
                x_axis_label,
                x_axis_label_size, 
                x_axis_label_style,
                x_axis_data_labels,
                x_axis_data_label_size, 
                x_axis_data_label_style,
                swarmplot_size,
                axis_line_width,
                y_axis_tick_thickness,
                y_axis_tick_label_size, 
                y_axis_tick_label_style, 
                boxplot_line_width,
                boxplot_width,
                dotplot_replicate_alpha,
                save_fig,
                save_fig_filename,
                y_axis_ticks)


