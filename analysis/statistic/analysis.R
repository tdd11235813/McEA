dirs <- commandArgs(trailingOnly = TRUE)

files <- character()

# find all the relavant files
for(dir in dirs) { files <- c(files, list.files(path=dir, pattern="*.metrics", full.names=TRUE))}

build_exp_desc <- function(filepath) {
  fpath <- strsplit(filepath, "/")
  exp_name <- fpath[[1]][length(fpath[[1]])-3]
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
metric_df <- data.frame(matrix(unlist(metric_list), ncol=8, byrow=TRUE))
colnames(metric_df) <- c('EP', 'HV', 'GD', 'IGD', 'IGD+', 'SPREAD', 'GSPREAD', 'ER')
metric_df$exp <- apply(array(names(metric_list)), 1, build_exp_desc)

# extract the names
metric_names <- colnames(metric_df)
metric_names <- metric_names[metric_names != "exp"]
metric_names <- metric_names[metric_names != "ER"]
exp_names <- levels(factor(metric_df$exp))

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
normality_mat <- matrix(unlist(unlist(normality_list)), ncol=7, byrow=TRUE)
colnames(normality_mat) <- metric_names
rownames(normality_mat) <- exp_names

print(normality_mat > 0.05)
