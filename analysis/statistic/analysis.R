library(effsize) # for Vargha and Delaney A^12
library(stringr) # for string padding via str_pad

dirs <- commandArgs(trailingOnly = TRUE)

files <- character()

# find all the relavant files
for(dir in dirs) { files <- c(files, list.files(path=dir, pattern="*.metrics", full.names=TRUE))}

build_exp_desc <- function(filepath) {
  fpath <- strsplit(filepath, "/")
  exp_name <- fpath[[1]][length(fpath[[1]])-2]
  exp_conf <- sub("_\\d{2}\\.metrics\\.V2", "", fpath[[1]][length(fpath[[1]])])
  exp_conf <- sub(".*_g", "g", exp_conf)
  exp <- paste(exp_name, exp_conf, sep="_")

  return(exp)
}

# function for reading the metrics from one file
read_metrics <- function(file) {
  # mangle together the experiment name from the filename

  # read the values
  vals <- read.table(file, sep=":")
  # rownames(vals) <- vals$V1
  vals[1] <- NULL
  # vals <- t(vals)
  # rownames(vals) <- c(exp)
  return(vals)
}

metric_list <- sapply(files, read_metrics)
metric_df <- data.frame(matrix(unlist(metric_list), ncol=9, byrow=TRUE))
colnames(metric_df) <- c('EP', 'HV', 'GD', 'IGD', 'IGDPLUS', 'SPREAD', 'GSPREAD', 'ER', 'RT')
metric_df$exp <- factor(apply(array(names(metric_list)), 1, build_exp_desc))

# extract the names
metric_names <- colnames(metric_df)
metric_names <- metric_names[metric_names != "exp"]
exp_names <- levels(metric_df$exp)

### Testing for Normality

# separate the single experiments and metrics into lists
# then calculate the shapiro-wilk normality test
normality_list <- lapply(exp_names, function(exp) lapply(metric_names , function (metr) {
    values <- metric_df[metric_df$exp == exp, ][[metr]]
    if(!identical(max(values), min(values))) { 
      shapiro.test(values)$p.value
    }else{
      1.0
    } 
  }))
normality_mat <- matrix(unlist(unlist(normality_list)), ncol=9, byrow=TRUE)
colnames(normality_mat) <- metric_names
rownames(normality_mat) <- exp_names

### Testing for Differences

# form groups for the research questions
# different mcea alg
metric_df$algor <- sub("_.*_dt[0-9]", "", metric_df$exp)

## ANOVA

# different mcea alg
aov_mcea_list <- lapply(metric_names, function(metric) { aov(as.formula(paste(metric, " ~ algor")), metric_df) })
pval_mcea_list <- lapply(aov_mcea_list, function(aov_res) { summary(aov_res)[[1]][["Pr(>F)"]][1] })
pval_mcea_arr <- matrix(unlist(pval_mcea_list))
colnames(pval_mcea_arr) <- "aov_mcea_alg"
rownames(pval_mcea_arr) <- metric_names

## POST HOC TEST: TukeyHSD

# different mcea alg
tuk_res_list <- lapply(aov_mcea_list, function(aov) TukeyHSD(aov)[[1]][, 4])
tuk_res_mat <- matrix(unlist(tuk_res_list), ncol=3, byrow=TRUE)
rownames(tuk_res_mat) <- metric_names
colnames(tuk_res_mat) <- names(tuk_res_list[[1]])

### Testing effect size

# get pairwise combinations of testsettings
mcea_com <- combn(levels(factor(metric_df$algor)), 2)
VD_res <- matrix(unlist(lapply(metric_names, function(metric) {
  apply(mcea_com, 2, function(comb) {
    selection <- metric_df$algor %in% comb
    VD.A(metric_df[[metric]][selection], metric_df$algor[selection])$estimate
  })
})), ncol=3, byrow=TRUE)
rownames(VD_res) <- metric_names
colnames(VD_res) <- apply(mcea_com, 2, function(algs) paste(algs[1], algs[2]))

### write the results

# Normality
rownames(normality_mat) <- str_pad(rownames(normality_mat), 55, side="right")
colnames(normality_mat) <- str_pad(colnames(normality_mat), 8, side="left")
colnames(normality_mat)[1] <- paste(str_pad("Experiment:", 55, side="right"), colnames(normality_mat)[1])
fileConn<-file("normality.txt", "w")
writeLines("Normality results: (p < 0.05: no normality)", fileConn)
write.table(formatC(normality_mat, digits=3, width=8), file=fileConn, append=TRUE, sep='\t', quote=FALSE)
close(fileConn)

# ANOVA
rownames(pval_mcea_arr) <- str_pad(rownames(pval_mcea_arr), 7, side="right")
colnames(pval_mcea_arr) <- str_pad(colnames(pval_mcea_arr), 14, side="left")
colnames(pval_mcea_arr)[1] <- paste(str_pad("Metric:", 7, side="right"), colnames(pval_mcea_arr)[1])
fileConn<-file("anova_pval_mcea_alg.txt", "w")
writeLines("ANOVA p-values: (p < 0.05: no equal means)", fileConn)
write.table(formatC(pval_mcea_arr, digits=3, width=8), file=fileConn, append=TRUE, sep='\t', quote=FALSE)
close(fileConn)

# TukeyHSD
rownames(tuk_res_mat) <- str_pad(rownames(tuk_res_mat), 7, side="right")
colnames(tuk_res_mat) <- str_pad(colnames(tuk_res_mat), 13, side="left")
colnames(tuk_res_mat)[1] <- paste(str_pad("Metric:", 7, side="right"), colnames(tuk_res_mat)[1])
fileConn<-file("tukey_pval_mcea_alg.txt", "w")
writeLines("TukeyHSD p-values: (p < 0.05: no equal means)", fileConn)
write.table(formatC(tuk_res_mat, digits=3, width=8), file=fileConn, append=TRUE, sep='\t', quote=FALSE)
close(fileConn)

# effect size
rownames(VD_res) <- str_pad(rownames(VD_res), 7, side="right")
colnames(VD_res) <- str_pad(colnames(VD_res), 13, side="left")
colnames(VD_res)[1] <- paste(str_pad("Metric:", 7, side="right"), colnames(VD_res)[1])
fileConn<-file("VD_res_mcea_alg.txt", "w")
writeLines("Vargha and Delaney’s Aˆ12: (probability, that a run A is worse than B)", fileConn)
write.table(formatC(VD_res, digits=3, width=8), file=fileConn, append=TRUE, sep='\t', quote=FALSE)
close(fileConn)
