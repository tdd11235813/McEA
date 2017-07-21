library(effsize) # for Vargha and Delaney A^12

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
pval_mcea_arr <- data.frame(unlist(pval_mcea_list))
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

print(VD_res)
