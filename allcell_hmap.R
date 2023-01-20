#Heatmap across donors/cell type by group (Activation status)
#R. Schmitz, adapted from heatmap code made by Alex Walsh

#load required packages

library(ggplot2)
library(RColorBrewer)
#invisible(source("https://bioconductor.org/biocLite.R"))
#invisible(biocLite("ComplexHeatmap"))
invisible(library(ComplexHeatmap))
#source_url("https://raw.githubusercontent.com/obigriffith/biostar-tutorials/master/Heatmaps/heatmap.3.R")
library(circlize)


#Make sure orders of columns on CSV match order of these labels
labCol_test <- c("Norm RR",
                 expression(paste("NAD(P)H ",tau[m])),
                 expression(paste("FAD ",tau[m])),
                 expression(paste("NAD(P)H ",alpha[1])),
                 expression(paste("NAD(P)H ",tau[1])),
                 expression(paste("NAD(P)H ",tau[2])),
                 expression(paste("FAD ",alpha[1])), 
                 expression(paste("FAD ",tau[1])), 
                 expression(paste("FAD ",tau[2])))

#reads in organized data
# mydata <- read.csv("./Data files/Heatmaps (R)/AllCellData_hmap.csv", header=T)
mydata <- read.csv("./Data files/Heatmaps (R)/2023120_AllCellData_hmap.csv", header=T)


all_cell <- mydata
#computes means for each group within donor
all_cell_mean <- aggregate(all_cell, list(all_cell$Donor,
                                    all_cell$Activation, all_cell$Cell_Type), mean)
#computes mean + sd of all CD69- control cells
control_sd <- subset(aggregate(all_cell, list(all_cell$Activation), sd),Group.1==0)
control_mean <- subset(aggregate(all_cell, list(all_cell$Activation), mean),Group.1==0)

#Calculate z-scores vs. the CD69- control mean
all_mean <- rbind(all_cell_mean)
all_mean_s <- as.matrix(scale(all_mean[, c(4:12)], center=as.matrix(control_mean[, c(2:10)]), scale=as.matrix(control_sd[, c(2:10)])))

#Dendrogram -- hierarchical clustering of groups (rows) and OMi variables (columns)
col.hc <- hclust(dist(t(all_mean_s)), method="ward.D2")
col.dd <- as.dendrogram(col.hc)
weights.dd <- c(500, 10, 100, 1000, 1, 1, 10, 10, 10) # 10000 cell size weight 4th place from left
col.dd.reordered1 <- reorder(col.dd,wts = weights.dd,agglo.FUN = mean)
col_palette2 <- colorRamp2(c(-3,0,3), c("cyan", "black", "magenta"))
row.hc <- hclust(dist(all_mean_s), method = "ward.D2")
row.dd <- as.dendrogram(row.hc)
weights.row <- c(1, 1, 1, 1, 1, 30, 100, 100, 100, 100, 100, 100, 10, 1, 1, 1, 1, 1, 200, 200, 200, 50, 50, 200)
row.dd.reordered <- reorder(row.dd, wts=weights.row, agglo.FUN = mean)

#Match numbers to donors/activation status
all_mean$Activation <- ifelse(all_mean$Activation == 0, "CD69-","CD69+")
all_mean$Donor <- ifelse(all_mean$Donor == 1, "A",ifelse(all_mean$Donor == 2,"B", ifelse(all_mean$Donor == 3, "C", ifelse(all_mean$Donor == 4,"D", ifelse(all_mean$Donor == 5,"E",ifelse(all_mean$Donor == 6,"F",ifelse(all_mean$Donor == 7,"G",ifelse(all_mean$Donor == 8,"H",ifelse(all_mean$Donor == 9,"I",ifelse(all_mean$Donor == 10,"J",ifelse(all_mean$Donor == 11,"K","L")))))))))))
all_mean$Cell_Type<- ifelse(all_mean$Cell_Type==0, "B",ifelse(all_mean$Cell_Type==1, "NK","T"))
df <- all_mean[,c("Activation", "Cell_Type", "Donor")]

#construct and display heatmap of data
#ha = HeatmapAnnotation(df = df, annotation_width = unit(2, "cm"), col = list(Activation = c("CD69-" = "#000058", "CD69+" = "#B2182B"), Cell_Type = c("B" = "#440154", 'NK' = "#21918c", 'T' = "#fde725"), Donor = c('A' = "#D9CCE3", 'B' = "#BA8DB4", 'C' = "#994F88", 'D' = "#4EB265", 'E' = "#90C987", 'F' = "#CAE0AB", 'G' = "#F7F056", 'H' = "#F7CB45", 'I' = "#EE8026", 'J' = "#DC050C", 'K' = "#A5170E", 'L' = "#72190E")), annotation_name_gp= gpar(fontsize = 18),which = "row", width = unit(2, "cm"), annotation_legend_param = list(Activation = list(title = "Activation", title_gp = gpar(fontsize = 16), labels_gp = gpar(fontsize = 14)), Cell_Type = list(title = "Cell Type", title_gp = gpar(fontsize = 16), labels_gp = gpar(fontsize = 14)), Donor = list(title = "Donor", title_gp = gpar(fontsize = 16), labels_gp = gpar(fontsize = 14))))
ha = HeatmapAnnotation(df = df, annotation_width = unit(2, "cm"), col = list(Activation = c("CD69-" = "#000058", "CD69+" = "#B2182B"), Cell_Type = c("B" = "#440154", 'NK' = "#21918c", 'T' = "#fde725"), Donor = c('A' = "#440154", 'B' = "#482173", 'C' = "#433e85", 'D' = "#38588c", 'E' = "#2d708e", 'F' = "#25858e", 'G' = "#1e9b8a", 'H' = "#2ab07f", 'I' = "#52c569", 'J' = "#86d549", 'K' = "#c2df23", 'L' = "#fde725")), annotation_name_gp= gpar(fontsize = 18),which = "row", width = unit(2, "cm"), annotation_legend_param = list(Activation = list(title = "Activation", title_gp = gpar(fontsize = 16), labels_gp = gpar(fontsize = 14)), Cell_Type = list(title = "Cell Type", title_gp = gpar(fontsize = 16), labels_gp = gpar(fontsize = 14)), Donor = list(title = "Donor", title_gp = gpar(fontsize = 16), labels_gp = gpar(fontsize = 14))))
hb = HeatmapAnnotation(text = anno_text(labCol_test, rot=90, gp = gpar(fontsize=18)), height = unit(3.65,"cm"))
h1 = Heatmap(all_mean_s, column_dend_reorder = weights.dd, cluster_rows=row.dd, bottom_annotation = hb, show_column_names = FALSE,  col=col_palette2, name = "Z score", row_dend_side = "left", heatmap_legend_param = list(title_gp = gpar(fontsize = 16), labels_gp = gpar(fontsize = 14)), row_dend_width = unit(2,"cm"), column_dend_height = unit(1,"cm"))
h1+ha


