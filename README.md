# schmitz_r-lymphocyte_activation

#Optical metabolic imaging successfully classifies primary human lymphocyte activation

#Contents:
* updated_b_analysis.py
* updated_nk_analysis.py
* updated_all_analysis.py
* b_heatmap.R
* nk_heatmap.R
* allcell_hmap.R

#Dataset list:
* **NK data 2 groups.csv** - Single-cell OMI data for CD69+ activated and CD69- control NK cells.
* **NK data all groups.csv** - Single-cell OMI data for NK cells in all four combinations of activation status (CD69-/CD69+) and culture condition (activated/control)
* **NK_heatmap.csv** - Contents of NK data 2 groups.csv, but formatted for ease of use in R with the ComplexHeatmap package.
    * Activation: 0 = CD69- control, 1 = CD69+ activated
    * Donor: 4 = Donor D, 5 = Donor E, 6 = Donor F
* **Bcell_cyto_2group.csv** - Single-cell OMI data for CD69+ activated and CD69- control B-cells.
* **Bcell_cyto_data.csv** - Single-cell OMI data for B-cells in all four combinations of activation status (CD69-/CD69+) and culture condition (activated/control)
* **B_heatmap.csv** - Contents of Bcell_cyto_2group.csv, but formatted for ease of use in R with the ComplexHeatmap package.
    * Activation: 0 = CD69- control, 1 = CD69+ activated
    * Donor: 1 = Donor A, 2 = Donor B, 3 = Donor C
* **AllCellData.csv** - Single-cell OMI data for CD69+ activated and CD69- control cells from all three cell types (NK, B, T)
* **AllCellData_hmap.csv** - Contents of AllCellData.csv, but formatted for ease of use in R with the ComplexHeatmap package.
    * Activation: 0 = CD69- control, 1 = CD69+ activated
    * Cell Type: 0 = B-cells, 1 = NK cells, 2 = T-cells
    * Donor: 1-6 same as above, 7 = Donor G, 8 = Donor H, 9 = Donor I, 10 = Donor J, 11 = Donor K, 12 = Donor L

#Software Requirements:
_Python 3.8:_
* holoviews 1.14.8
* numpy 1.20.3
* pandas 1.3.3
* scikit-learn 0.24.2
* umap-learn 0.5.3

_R 4.1.0:_
* RStudio 1.4.1717
* circlize 0.4.14
* ComplexHeatmap 2.10.0
* ggplot2 3.3.5
* RColorBrewer 1.1-3


#Instructions:

**updated_B_analysis.py**
_Generates UMAPs and random forest classifiers for B-cell data._

Required files: 
* Bcell_cyto_2group.csv
* Bcell_cyto_data.csv

Steps:
1.	Load script into Python IDE of choice (this code was written in Spyder 4.2.5)
2.	Run section 1 to import required packages
3.	Run section 2 to set-up for ROC curve generation 
4.	Run section 3 to read in Bcell_cyto_2group.csv, which contains single-cell OMI data from the CD69- control and CD69+ activated B-cells. 
5.	B-cell activation status classification:
    * In section 4, change line 82 to contain the desired OMI parameters used in the classifier. The code is written to have all 10 features, but can be changed to have any combination of features. 
    * Make sure that ‘Activation’ is the last parameter in the list in line 82.
    * Run Section 4, which will produce a confusion matrix, list of feature weights, an ROC curve, and metrics for assessing the performance of the classifier (accuracy, recall, precision, f1 score).
6.	B-cell OMI UMAPs of CD69- control and CD69+ activated B-cells: 
    * Sections 5-7 produce UMAPs from Bcell_cyto_2group.csv. All three sections use the same overall structure to generate the UMAP.
        * Section 5: UMAP color-coded by activation
        * Section 6: UMAP color-coded by donor
        * Section 7: UMAP color-coded by both activation and donor.
    * The UMAPs used in the paper figures can be generated by running this code as-is. 
    * To alter the parameters of the UMAP or plotting in Section 5:
        * UMAP input parameters such as n_neighbors, min_dist, etc. can be altered in lines 126 - 132.
        * Factor used to determine color-coding is set by legend_entries in line 140.
    * Similar alterations can be done in Sections 6 & 7 by locating and changing the same sections of code. 
7.  B-cell OMI UMAPs of all four combinations of activation status (CD69-/CD69+) and culture condition (activated/control):
    * Run Section 8 as-is to produce the UMAP used in the paper figures
    * To alter the parameters of the UMAP or plotting:
        * UMAP input parameters such as n_neighbors, min_dist, etc. can be altered in lines 342 - 348.
        * Factor used to determine color-coding is set by legend_entries in line 355.

**updated_nk_analysis.py**
_Generates UMAPs and random forest classifiers for  NK cell data._

Required files: 
* NK data 2 groups.csv
* NK data all groups.csv

Steps:
1.	Load script into Python IDE of choice (this code was written in Spyder 4.2.5)
2.	Run section 1 to import required packages
3.	Run section 2 to set-up for ROC curve generation 
4.	Run section 3 to read in NK data 2 groups.csv, which contains single-cell OMI data from the CD69- control and CD69+ activated NK cells. 
5.	NK cell activation status classification:
    * In section 4, change line 75 to contain the desired OMI parameters used in the classifier. The code is written to have all 10 features, but can be changed to have any combination of features. 
    * Make sure that ‘Activation’ is the last parameter in the list in line 75.
    * Run Section 4, which will produce a confusion matrix, list of feature weights, an ROC curve, and metrics for assessing the performance of the classifier (accuracy, recall, precision, f1 score).
6.	NK cell OMI UMAPs of CD69- control and CD69+ activated NK cells: 
    * Sections 5-7 produce UMAPs from NK data 2 groups.csv. All three sections use the same overall structure to generate the UMAP.
        * Section 5: UMAP color-coded by activation
        * Section 6: UMAP color-coded by donor
        * Section 7: UMAP color-coded by both activation and donor.
    * The UMAPs used in the paper figures can be generated by running this code as-is. 
    * To alter the parameters of the UMAP or plotting in Section 5:
        * UMAP input parameters such as n_neighbors, min_dist, etc. can be altered in lines 118 - 126.
        * Factor used to determine color-coding is set by legend_entries in line 140.
    * Similar alterations can be done in Sections 6 & 7 by locating and changing the same sections of code. 
7.  NK cell OMI UMAPs of all four combinations of activation status (CD69-/CD69+) and culture condition (activated/control):
    * Run Section 8 as-is to produce the UMAP used in the paper figures
    * To alter the parameters of the UMAP or plotting:
        * UMAP input parameters such as n_neighbors, min_dist, etc. can be altered in lines 336-342.
        * Factor used to determine color-coding is set by legend_entries in line 349.

**updated_all_analysis.py**
_Generates UMAPS and random forest classifiers for data from all three cell types (B-cells, NK cells, and T-cells)_

Required files:
* AllCellData.csv
  
Steps:
1.	Load script into Python IDE of choice (this code was written in Spyder 4.2.5)
2.	Run section 1 to import required packages
3.	Run section 2 to set-up for ROC curve generation 
4.	Run section 3 to read in AllCellData.csv, which contains single-cell OMI data for all three cell types (T-cells, NK cells, and B-cells)
5.	Activation status classifier:
    * In section 4, change line 85 to contain the desired OMI parameters used in the classifier. The code is written to have all 10 features, but can be changed to have any combination of features. 
    * Make sure that ‘Activation’ is the last parameter in the list in line 85.
    * Run Section 4, which will produce a confusion matrix, list of feature weights, an ROC curve, and metrics for assessing the performance of the classifier (accuracy, recall, precision, f1 score).
6.  Cell type and cell type + activation classifiers:
    * Sections 5 - 7 use the same structure to run and assess random forest classifiers. This classifiers are focused on different classes within the single cell OMI data of CD69- control and CD69+ activated lymphocytes:
        * Section 5: Classification of cell type (B-cell, T-cell, NK cell)
        * Section 6: Classification of cell type (based on only CD69- control cells)
        * Section 7: Classification of both cell type and activation status (B-cell, T-cell, NK cell and CD69+/CD69-)
    * These classifiers do not generate ROC curves because ROC curve generation requires "one vs. all" classification where the class outcomes are binary. However, the other metrics to assess classifier performance are still produced.
    * In line 132/179/238, change the list to include whichever OMI parameters are desired in the classifier. 
        * The parameter encoding class (previously 'Activation') does NOT need to be added to this list, unlike in the activation status classifier. This code will generate the classes separately from this list. 
    * Run Section 5/6/7, which will produce a confusion matrix, feature weights, and metrics to assess classifier performance (accuracy/precision/recall/f1 score). 
7.	UMAPs of OMI data from all cell types: 
    * Sections 8 - 13 produce UMAPs from AllCellData.csv. All sections use the same overall structure to generate the UMAP.
        * Section 8: UMAP color-coded by activation status
        * Section 9: UMAP color-coded by cell type 
        * Section 10: UMAP color-coded by cell type (CD69- control cells only)
        * Section 11: UMAP color-coded by cell type and activation status
        * Section 12: UMAP color-coded by cell type and donor
        * Section 13: UMAP color-coded by activation status and donor
    * The UMAPs used in the paper figures can be generated by running this code as-is. 
    * To alter the parameters of the UMAP or plotting in Section 8:
        * UMAP input parameters such as n_neighbors, min_dist, etc. can be altered in lines 294 - 300.
        * Factor used to determine color-coding is set by legend_entries in line 308.
        * Colors used in color-coding and other plotting parameters can be changed in lines 327 - 348.
    * Similar alterations can be done in Sections 9 - 13 by locating and changing the same sections of code. 




**B_heatmap.R**
_Generates a heatmap with hierarchical clustering of single-cell OMI data from CD69+ activated and CD69- control B-cells_

Required files:
* b_heatmap.R
  
Steps: 
1. Load script into RStudio.
2. Install any R packages you have not previously installed by using RStudio's built-in package installation feature or the install.packages() function. 
   * If you are having trouble installing the ComplexHeatmap package, follow the installation instructions from Bioconductor (https://www.bioconductor.org/packages/release/bioc/html/ComplexHeatmap.html)
3. To replicate the figure from the manuscript, run the entire script as-is. 
   * The mean of all CD69- control cells from all three cell types is used to calculate a Z-score for each single cell
   * Hierarchical clustering of single cells is determined based on OMI parameters using Ward's method.
4. The heatmap dimensions may be altered and the figure saved as the desired file type by using RStudio's Export feature.


**NK_heatmap.R**
_Generates a heatmap with hierarchical clustering of single-cell OMI data from CD69+ activated and CD69- control NK cells_

Required files:
* NK_heatmap.R
  
Steps: 
1. Load script into RStudio.
2. Install any R packages you have not previously installed by using RStudio's built-in package installation feature or the install.packages() function. 
   * If you are having trouble installing the ComplexHeatmap package, follow the installation instructions from Bioconductor (https://www.bioconductor.org/packages/release/bioc/html/ComplexHeatmap.html)
3. To replicate the figure from the manuscript, run the entire script as-is. 
   * The mean of all CD69- control cells from all three cell types is used to calculate a Z-score for each single cell
   * Hierarchical clustering of single cells is determined based on OMI parameters using Ward's method.
4. The heatmap dimensions may be altered and the figure saved as the desired file type by using RStudio's Export feature.

**allcell_hmap.R**
_Generates a heatmap with hierarchical clustering of groups of CD69+ activated and CD69- control cells from all three cell types (B, T, NK) by donor_

Required files:
* AllCellData_hmap.R
  
Steps: 
1. Load script into RStudio.
2. Install any R packages you have not previously installed by using RStudio's built-in package installation feature or the install.packages() function. 
   * If you are having trouble installing the ComplexHeatmap package, follow the installation instructions from Bioconductor (https://www.bioconductor.org/packages/release/bioc/html/ComplexHeatmap.html)
3. To replicate the figure from the manuscript, run the entire script as-is. 
   * The mean of all CD69- control cells from all three cell types is used to calculate a Z-score for each group (grouped by activation and donor)
   * Hierarchical clustering of groups is determined based on OMI parameters using Ward's method.
4. The heatmap dimensions may be altered and the figure saved as the desired file type by using RStudio's Export feature.