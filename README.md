# schmitz_r-lymphocyte_activation

Optical metabolic imaging successfully classifies primary human lymphocyte activation

Contents:
•	updated_b_analysis.py
•	updated_nk_analysis.py
•	updated_all_analysis.py
•	b_heatmap.R
•	nk_heatmap.R
•	allcell_hmap.R

Dataset list:
•	NK data 2 groups.csv
•	NK data all groups.csv
•	NK_heatmap.csv
•	Bcell_cyto_2group.csv
•	Bcell_cyto_data.csv
•	B_heatmap.csv
•	AllCellData.csv
•	AllCellData_hmap.csv

Software Requirements:
Python 3.8:
•	holoviews 1.14.8
•	numpy 1.20.3
•	pandas 1.3.3
•	scikit-learn 0.24.2
•	umap-learn 0.5.3
R 4.1.0:
•	RStudio 1.4.1717
•	circlize 0.4.14
•	ComplexHeatmap 2.10.0
•	ggplot2 3.3.5
•	RColorBrewer 1.1-3
Instructions:
updated_B_analysis.py
Generates UMAPs and random forest classifiers for B-cell data.
Required files: 
•	Bcell_cyto_2group.csv
•	Bcell_cyto_data.csv
Steps:
1.	Load file into Python IDE of choice (this code was written in Spyder 4.2.5)
2.	Run section 1 to import required packages
3.	Run section 2 to set-up for ROC curve generation 
4.	Run section 3 to read in Bcell_cyto_2group.csv, which contains single-cell OMI data from the CD69- control and CD69+ activated B-cells. 
5.	B-cell activation status classification:
a.	In section 4, change line 82 to contain the desired OMI parameters used in the classifier. The code is written to have all 10 features, but can be changed to have any combination of features. 
b.	Make sure that ‘Activation’ is the last parameter in the list in line 82.
c.	Run Section 4, which will produce a confusion matrix, list of feature weights, an ROC curve, and metrics for assessing the performance of the classifier (accuracy, recall, precision, f1 score).
6.	B-cell OMI UMAPs: 

