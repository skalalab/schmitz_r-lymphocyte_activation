# Autofluorescence lifetime imaging classifies human lymphocyte activation and subtype
 

## Abstract
New non-destructive tools are needed to reliably assess lymphocyte function for immune profiling and adoptive cell therapy. Optical metabolic imaging (OMI) is a label-free method that measures the autofluorescence intensity and lifetime of metabolic cofactors NAD(P)H and FAD to quantify metabolism at a single-cell level. Here, we investigate whether OMI can resolve metabolic changes between human quiescent versus IL4/CD40 activated B cells and IL12/IL15/IL18 activated memory-like NK cells. We found that quiescent B and NK cells were more oxidized compared to activated cells. Additionally, the NAD(P)H mean fluorescence lifetime decreased and the fraction of unbound NAD(P)H increased in the activated B and NK cells compared to quiescent cells. Machine learning classified B cells and NK cells according to activation state (CD69+) based on OMI parameters with up to 93.4% and 92.6% accuracy, respectively. Leveraging our previously published OMI data from activated and quiescent T cells, we found that the NAD(P)H mean fluorescence lifetime increased in NK cells compared to T cells, and further increased in B cells compared to NK cells. Random forest models based on OMI classified lymphocytes according to subtype (B, NK, T cell) with 97.8% accuracy, and according to activation state (quiescent or activated) and subtype (B, NK, T cell) with 90.0% accuracy. Our results show that autofluorescence lifetime imaging can accurately assess lymphocyte activation and subtype in a label-free, non-destructive manner.

[Article on BioRxiv](https://www.biorxiv.org/content/10.1101/2023.01.23.525260v1)

## Contents

Various software was used to perform the analysis and generate the figures in the paper.

* Figures made in python
  * analysis_b_cells.py
  * analysis_nk_cells.py
  * analysis_all_cells.py
* Figures made in R
  * b_heatmap.R
  * nk_heatmap.R
  * allcell_hmap.R
* Figures made in Prism
  * NK graphs.pzfx
  * B cell graphs.pzfx
  * B cell graphs.pzfx

---

## Datasets (Single-cell OMI variables)

* B Cell
  * _data/UMAPs, boxplots, ROC curves (Python)_
    * **Bcell_cyto_2group.csv** -  CD69+ activated and CD69- control
    * **Bcell_cyto_data.csv** - all four combinations of activation status (CD69-/CD69+) and culture condition (activated/control)
  * _data/Heatmaps (R)_
    * **B_heatmap.csv** - Contents of _Bcell_cyto_2group.csv_, but formatted for ease of use in R with the ComplexHeatmap package.
      * Activation
        * 0 = CD69- control
        * 1 = CD69+ activated
      * Donor
        * 1 = Donor A
        * 2 = Donor B
        * 3 = Donor C
* NK Dataset
  * _data/UMAPs, boxplots, ROC curves (Python)_
    * **NK_cells_dataset.csv** - all four combinations of activation status (CD69-/CD69+) and culture condition (activated/control)
  * _data/Heatmaps (R)_
    * **NK_heatmap.csv** - Contents of _NK data 2 groups.csv_, but formatted for ease of use in R with the ComplexHeatmap package.
      * Activation
        * 0 = CD69- control
        * 1 = CD69+ activated
      * Donor
        * 4 = Donor D
        * 5 = Donor E
        * 6 = Donor F

* All Data
  * _data/UMAPs, boxplots, ROC curves (Python)_
    * **all_data.csv** - CD69+ activated and CD69- control cells from all three cell types (NK, B, T)
  * _data/Heatmaps (R)_
    * **AllCellData_hmap.csv** - Contents of _AllCellData.csv_, but formatted for ease of use in R with the ComplexHeatmap package.
      * Activation
        * 0 = CD69- control
        * 1 = CD69+ activated
      * Cell Type
        * 0 = B-cells
        * 1 = NK cells
        * 2 = T-cells
      * Donor
        * 1-6 same as above
        * 7 = Donor G
        * 8 = Donor H
        * 9 = Donor I

---

### Software Requirements

_Python (3.8):_

* numpy (1.20.3)
* pandas (1.3.3)
* scikit-learn (0.24.2)
* umap-learn (0.5.3)
  
For plots

* selenium (4.4.3)
* geckodriver (0.30.0)
* holoviews (1.14.8)
* matplotlib (3.5.1)

_R 4.1.0:_

* RStudio (1.4.1717)
* circlize (0.4.14)
* ComplexHeatmap (2.10.0)
* ggplot2 (3.3.5)
* RColorBrewer (1.1-3)

---

### Instructions

**analysis_b_cells.py**
_Generates figures and outputs used in other figures._

Required files:

* Bcell_cyto_2group.csv
* Bcell_cyto_data.csv

Steps:

1. Load script into you Python IDE of choice
2. Run section 1 to import required packages
3. Run section 2 to read in Bcell_cyto_2group.csv, which contains single-cell OMI data from the CD69- control and CD69+ activated B-cells.

* **Section 3:** Random forest activation status classifiers for various combination of features
* **Section 4:** Box and whisker plot of Normlized Redox Ratio
* **Section 5:** Random forest, Logistic Regression and SVM classifiers
* **UMAP's**
  * **Section 6:** UMAP color-coded by activation
  * **Section 7:** UMAP color-coded by donor
  * **Section 8:** UMAP color-coded by both activation and donor.
  * **Section 9:** UMAPs of all four combinations of activation status (CD69-/CD69+) and culture condition (activated/control):

---

**analysis_nk_cells.py**
_Generates figures and outputs used in other figures._

Required files:

* NK_cells_dataset.csv

Steps:

1. Load script into Python IDE of choice
2. **Section 1:** Imports required packages
3. **Section 2:**  Reads in _NK_cells_dataset.csv_, which contains all NK cell data.

* **Section 3:** Random forest activation status classifiers for various combination of features
* **Section 4:** Box and whisker plots for various OMI parameters
* **Section 5:** Random forest, Logistic Regression and SVM classifiers
* **UMAP's**
  * **Section 6:** UMAP color-coded by activation
  * **Section 7:** UMAP color-coded by donor
  * **Section 8:** UMAP color-coded by both activation and donor.
  * **Section 8:** NK cell OMI UMAPs of all four combinations of activation status (CD69-/CD69+) and culture condition (activated/control)

---

**analysis_all_cells.py**
_Generates figures and outputs used in figures using B-cells, NK cells, and T-cell data_

Required files:

* all_data.csv
  
Steps:

1. Load script into Python IDE of choice
2. **Section 1:** Import required packages
3. **Section 2:** Read in _all_data.csv_, which contains single-cell OMI data for T, NK and B-cells.

* **Section 3:** Random forest activation status classifiers for various combination of features
* **Section 4:** Random forest, Logistic Regression and SVM classifiers for cell type (B-cell, T-cell, NK cell)
* **Section 5:** UMAP of activation status for all cell types
* **Section 6:** UMAP colored by cell type
* **Section 7:** UMAP colored by cell type (quiescet only)
* **Section 8:** UMAP colored by cell type and activation status
* **Section 9:** UMAP color-coded by cell type, donor and activation status
* **Section 10:** Classifier of cell type
* **Section 11:** Classifier of both cell type (quiescent only)
* **Section 12:** Classifier of both cell type and activation status

---

**B_heatmap.R**
_Generates a heatmap with hierarchical clustering of single-cell OMI data from CD69+ activated and CD69- control B-cells_

Required files

* b_heatmap.R
  
Steps

1. Load script into RStudio.
2. Install any R packages you have not previously installed by using RStudio's built-in package installation feature or the install.packages() function.
   * If you are having trouble installing the ComplexHeatmap package, follow the [installation instructions from Bioconductor](https://www.bioconductor.org/packages/release/bioc/html/ComplexHeatmap.html)
3. To replicate the figure from the manuscript, run the entire script as-is.
   * The mean of all CD69- control cells from all three cell types is used to calculate a Z-score for each single cell
   * Hierarchical clustering of single cells is determined based on OMI parameters using Ward's method.
4. The heatmap dimensions may be altered and the figure saved as the desired file type by using RStudio's Export feature.

---

**NK_heatmap.R**
_Generates a heatmap with hierarchical clustering of single-cell OMI data from CD69+ activated and CD69- control NK cells_

Required files

* NK_heatmap.R
  
Steps

1. Load script into RStudio.
2. Install any R packages you have not previously installed by using RStudio's built-in package installation feature or the install.packages() function.
   * If you are having trouble installing the ComplexHeatmap package, follow the [installation instructions from Bioconductor](https://www.bioconductor.org/packages/release/bioc/html/ComplexHeatmap.html)
3. To replicate the figure from the manuscript, run the entire script as-is.
   * The mean of all CD69- control cells from all three cell types is used to calculate a Z-score for each single cell
   * Hierarchical clustering of single cells is determined based on OMI parameters using Ward's method.
4. The heatmap dimensions may be altered and the figure saved as the desired file type by using RStudio's Export feature.

---

**allcell_hmap.R**
_Generates a heatmap with hierarchical clustering of groups of CD69+ activated and CD69- control cells from all three cell types (B, T, NK) by donor_

Required files

* AllCellData_hmap.R
  
Steps

1. Load script into RStudio.
2. Install any R packages you have not previously installed by using RStudio's built-in package installation feature or the install.packages() function.
   * If you are having trouble installing the ComplexHeatmap package, follow the [installation instructions from Bioconductor](https://www.bioconductor.org/packages/release/bioc/html/ComplexHeatmap.html)
3. To replicate the figure from the manuscript, run the entire script as-is.
   * The mean of all CD69- control cells from all three cell types is used to calculate a Z-score for each group (grouped by activation and donor)
   * Hierarchical clustering of groups is determined based on OMI parameters using Ward's method.
4. The heatmap dimensions may be altered and the figure saved as the desired file type by using RStudio's Export feature.
