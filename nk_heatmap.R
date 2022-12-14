#Single-cell heatmap across donors/cell type - NK cells
#R. Schmitz, adapted from heatmap code made by Alex Walsh

#load required packages

library(ggplot2)
library(RColorBrewer)
#invisible(source("https://bioconductor.org/biocLite.R"))
#invisible(biocLite("ComplexHeatmap"))
invisible(library(ComplexHeatmap))
#source_url("https://raw.githubusercontent.com/obigriffith/biostar-tutorials/master/Heatmaps/heatmap.3.R")
library(circlize)


#Make sure order of columns in CSV matches order in this list
labCol_test <- c("Norm RR", 
                 expression(paste("NAD(P)H ",tau[m])),
                 expression(paste("FAD ",tau[m])),
                 expression(paste("NAD(P)H ",alpha[1])),
                 expression(paste("NAD(P)H ",tau[1])), 
                 expression(paste("NAD(P)H ",tau[2])),
                 expression(paste("FAD ",alpha[1])),
                 expression(paste("FAD ",tau[1])),
                 expression(paste("FAD ",tau[2])))

#Read in single cell NK cell data
# mydata <- read.csv("Z:/Rebecca/Immune Cell Projects/Lymphocyte Paper/Paper data/NK cells (Donors 4-6)/NK_heatmap.csv", header=T)
mydata <- read.csv("./Data files/ecg_feature_exports/NK_cells.csv", header=T)
all_mean <- mydata

#Calculate mean and sd of each OMI variable in CD69- control cells to use for calculating Z-scores
control_mean <- subset(aggregate(all_mean, list(all_mean$Activation), mean),Group.1==0)
control_sd <- subset(aggregate(all_mean, list(all_mean$Activation), sd),Group.1==0)
cn<- c("Norm_RR", "Na1", "Ntm", "Nt2", "Ft1","Ftm","Ft2","Nt1","Fa1")

#Calculated Z-score for each OMI variable vs. mean of control cells (single cell level)
all_mean_s <- as.matrix(scale(all_mean[, c(1:9)], center=as.matrix(control_mean[, c(2:10)]), scale = as.matrix(control_sd[, c(2:10)])))
row_col <- cbind(all_mean$Activation, all_mean$Donor)
colnames(row_col) <- c('Activation', 'Donor')
all_mean2 <- cbind(data.frame(all_mean_s),row_col)
all_mean2$Activation <- ifelse(all_mean2$Activation == 0, "CD69-","CD69+")

#Dendrogram -- hierarchical clustering of groups (rows) and OMI variables (columns)
col.hc <- hclust(dist(t(all_mean_s)), method="ward.D2")
col.dd <- as.dendrogram(col.hc)
weights.dd <- ifelse(colnames(all_mean_s)=="Ntm",1,100)
col.dd.reordered <- reorder(col.dd,wts = weights.dd,agglo.FUN = mean)
row.hc <- hclust(dist(all_mean_s,  method = "euclidean"), method = "ward.D")
row.dd <- as.dendrogram(row.hc)
col_palette2 <- colorRamp2(c(-3,0,3), c("cyan", "black", "magenta"))


#Match numbers to donors
all_mean2$Donor <- ifelse(all_mean2$Donor == 4, "D",ifelse(all_mean2$Donor == 5,"E", 'F'))

#construct and display heatmap of data
df <- all_mean2[,c("Activation", "Donor")]
ha = HeatmapAnnotation(df = df, col = list(Activation = c("CD69-" = "#000058", "CD69+" = "#B2182B"),  Donor = c("F" = "#440154", 'E' = "#21918c", 'D' = "#fde725")),     which = "row", width = unit(1.25, "cm"), annotation_name_gp= gpar(fontsize = 18), annotation_legend_param = list(Activation = list(title = "Activation", title_gp = gpar(fontsize = 16), labels_gp = gpar(fontsize = 14)), Donor = list(title = "Donor", title_gp = gpar(fontsize = 16), labels_gp = gpar(fontsize = 14))))
hb = HeatmapAnnotation(text = anno_text(labCol_test, rot=90, gp = gpar(fontsize=18)), height = unit(3.65,"cm"))
h1 = Heatmap(all_mean_s,bottom_annotation = hb, cluster_rows = row.dd, show_column_names = FALSE, show_row_names = FALSE, cluster_columns  = col.dd.reordered, col=col_palette2, name = "Z score", row_dend_side = "left",heatmap_legend_param = list(title_gp = gpar(fontsize = 16), labels_gp = gpar(fontsize = 14)), row_dend_width = unit(2,"cm"), column_dend_height = unit(1,"cm"))

h1+ha

