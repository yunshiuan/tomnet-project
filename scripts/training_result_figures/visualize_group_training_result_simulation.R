##################
# Authour: Chuang, Yun-Shiuan
# Date: 2020/03/07
# This script is for visualizing the model accuracy
# across all subjects in one line plot.
# Note:
# - based on visualize_group_training_result_human.R
# - axises:
#   - x axis: SD of the social rewards
#   -	y axis: accuracy
#   -	2 bars: testing accuracy and random rates

#################
library(stringr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)

# Constants -----------------------------------------------
# - parameter
# TYPE = "test_on_simulation_data"
TYPE <- "simulation"
VERSION <- "v26"
PATTERN_ACCURACY <- "_train_test_and_validation_accuracy.csv"
NUM_AUGMENTATION <- 1 # no augmentation for simulation data
# - path
# the root path of the project
PATH_ROOT <- str_extract(
  string = getwd(),
  pattern = ".*tomnet-project"
)
EXCLUSION_SUBJ <- c("")
PATTENR_SUBJ = "S\\d+b"
VAR_TRUE_U = "candidate"
VAR_MEAN = "mean"
VAR_SD = "SD"
VAR_SK = "skewness"
# EXCLUSION_SUBJ <- paste0(
#   "S0",
#   c( # does not have ground-truth score
#     "52",
#     # less than 100 trajectories
#     "26", "35", "43", "52", "55", "58",
#     # does not act according to the score
#     "69"
#   )
# )
# the least training files the subject should have for the "thresholded version"
# - thresholding ensure the estimate of accuracy is precise enough
IMG_SIZE_RATIO <- 0.55
IMG_WIDTH <- 8 * IMG_SIZE_RATIO
IMG_HEIGHT <- 6 * IMG_SIZE_RATIO
# Manually set the root path of the repo if running the script via RStudio
# - this could ensure the script to run even if the directory is restructured
if (interactive()) {
  # Should be manually adjusted to correspond to either 'test_on_human_data' or 'test_on_simulated_data'
  PATH_RESULT_ROOT <- file.path(
    PATH_ROOT, "models", "working_model",
    paste0("test_on_", TYPE, "_data")
  )
} else {
  cat(paste0(
    "Please enter the path of where the training_result is located in (without quotation mark), \n",
    "e.g., /Users/tomnet-project/models/working_model/test_on_simulation_data \n"
  ))
  PATH_RESULT_ROOT <- readLines("stdin", n = 1)
}

# get the version number if running from the terminal
if (!interactive()) {
  cat(paste0("Please enter the version number (e.g., v12) \n"))
  VERSION <- readLines("stdin", n = 1)
}
PATH_TRAINING_RESULT <- file.path(PATH_RESULT_ROOT, "training_result", "caches", VERSION)
PATH_FIGURE_OUTPUT <- file.path(PATH_RESULT_ROOT, "training_result", "figures", VERSION)
PATH_VALUE_U <- file.path(
  PATH_ROOT, "data",
  "data_simulation", "simulation_data_on_server",
  "36agents")
# - file
# the file that contain the random rate for each subject
FILE_RANDOM_RATE <- file.path(
  PATH_ROOT, "data",
  paste0("data_", TYPE), "simulation_data_on_server", "data",
  paste0("data_", TYPE), "S004-S033",
  "processed",
  "summary_count_targets_2020-03-08.csv"
)
FILE_OUTPUT <- file.path(
  PATH_FIGURE_OUTPUT,"all_training_results"
  )

# Processing data -----------------------------------------------
# - get the random rate of accuracy for each agent
df_random_rate <- read.csv(FILE_RANDOM_RATE, header = T, stringsAsFactors = F)

# - get the accuracy for each agent's model
list_errors <-
  list.files(
    path = PATH_TRAINING_RESULT,
    pattern = PATTERN_ACCURACY,
    recursive = T, full.names = T
  )
list_df_error <-
  lapply(list_errors, function(error_csv) {
    df_error <- read.csv(error_csv, header = T, stringsAsFactors = F)
    # add the subject name as a column
    this_subj_name <-
      str_extract(string = error_csv, pattern = "(?<=/)S\\d+.*(?=/train)")
    df_error <-
      df_error %>%
      mutate(subj_name = this_subj_name) %>%
      rename(accuracy = accurary)
  })
# - merge all error dfs into one error df
df_error_all <- bind_rows(list_df_error)

# - joint 'df_error_all' and 'df_random_rate'
df_error_all <-
  df_random_rate %>%
  select(-X)%>%
  group_by(subj_name) %>%
  summarise(
    # get the random rate for each agent
    random_rate = unique(random_rate)
  ) %>%
  right_join(df_error_all, by = "subj_name")

# - get the total number of processed files for training (before data augmentation) for each subject
df_error_all <-
  df_error_all %>%
  filter(mode == "train_proportion") %>%
  mutate(
    total_processed_training_files =
      as.numeric(str_extract(
        string = matches,
        pattern = "(?<=/)\\d+$"
      )) / NUM_AUGMENTATION
  ) %>%
  select(subj_name, total_processed_training_files) %>%
  right_join(df_error_all, by = "subj_name") %>%
  as.data.frame()
# - convert the variable type and level name for ploting
df_error_all <-
  df_error_all %>%
  mutate(
    mode = str_extract(
      string = mode,
      pattern = "^\\w+(?=_)"
    ),
    accuracy = as.numeric(str_extract(
      string = accuracy,
      pattern = ".*(?=%)"
    ))
  )
# - make random rate as one of the 'mode'
df_error_all <-
  df_error_all %>%
  select(-matches) %>%
  pivot_wider(
    names_from = mode,
    values_from = accuracy
  ) %>%
  pivot_longer(
    cols = c("train", "vali", "test", "random_rate"),
    names_to = "mode",
    values_to = "accuracy"
  )
df_error_all <-
  df_error_all %>%
  filter(!subj_name %in% EXCLUSION_SUBJ)
# Get the true preference of simulation data ---------------
df_files_true_u <- data.frame(
  file =
    list.files(
      path = PATH_VALUE_U,
      pattern = paste0(PATTENR_SUBJ,".*.csv"),
      full.names = T
    ),
  stringsAsFactors = F
)

df_files_true_u <-
  df_files_true_u %>%
  mutate(
    subj_name =
      str_extract(
        string = file,
        pattern = PATTENR_SUBJ
      )
  )
# get the distribution of the true u ----------------------------
# - read in all the preference files
list_df_true_u <- c()
for (file_index in 1:nrow(df_files_true_u)) {
  csv <- df_files_true_u$file[file_index]
  subj_name <- df_files_true_u$subj_name[file_index]
  df_preference <- read.csv(csv, header = T, stringsAsFactors = F)
  df_preference <-
    df_preference %>%
    rename(true_u =!!(VAR_TRUE_U))%>%
    mutate(
      target_id = as.numeric(str_extract(string = X, pattern = "\\d$")),
      subj_name = subj_name
    )
  list_df_true_u[[file_index]] <- df_preference
}
df_all_true_u <- bind_rows(list_df_true_u)

# Merge df_error_all with list_df_true_u--------------------------
df_all = 
  df_all_true_u%>%
    group_by(subj_name)%>%
    summarise(
      mean = unique(mean),
      sd = unique(SD),
      sk = unique(skewness)
      )%>%
    as.data.frame()%>%
    right_join(df_error_all,by = "subj_name")
# Plot ------------------------------------------------------------
for (file_type in c(".png", ".pdf")) {
  df_plot <- df_all 
  # set the scale
  x_scale_minor_breaks <- seq(0, 4000, 100)
  x_scale_breaks <- c(100, 500, 1000, 2000, 3000, 4000)

  df_plot <-
    df_plot %>%
    # log transform the scale to make the dots more visible
    # mutate(log_total_processed_training_files = log10(total_processed_training_files)) %>%
    filter(mode %in% c("random_rate", "test")) %>%
    mutate(
      mode = factor(mode,
        levels = c("test", "random_rate"),
        labels = c("Test", "Random Rate")
      )
      # subj_label = paste0(
      #   "A",
      #   as.numeric(str_extract(string = subj_name, pattern = "(?<=S0)\\d+"))-3
      # )
    )
  # convert to bar plot
  df_plot = 
    df_plot%>%
      group_by(sd,mode)%>%
      summarise(
        mean_accuracy = mean(accuracy),
        sd_accuracy = sd(accuracy),
        n = n(),
        se = sd_accuracy/sqrt(n)
      )%>%
    mutate(
      sd_level = factor(sd,
                        levels = c(0.1,1.1,2.1))
    )
  df_plot %>%
    ggplot(aes(x = sd_level, y = mean_accuracy, group = mode,fill = mode)) +
    geom_col(position = "dodge")+
    geom_errorbar(aes(ymax = mean_accuracy+se,
                      ymin = mean_accuracy-se),  
                  width = 0.3,
                  position = position_dodge(width = 0.9))+
    scale_fill_discrete(NULL) +
    coord_cartesian(ylim=c(30,85))+
    # geom_errorbar(aes(ymax = mean_accuracy+se,
    #                   ymin = mean_accuracy-se),
    #               width = 0.1)+    
    # geom_point(aes(color = mode,shape = mode),size = 3)+
    # scale_shape_discrete(NULL) +
    # scale_color_discrete(NULL) +
    theme_bw() +
    labs(
      x = "SD of Social Support Values Across 4 Targets",
      y = "Accuracy"
    ) +
    ggsave(
      filename = paste0(FILE_OUTPUT, file_type),
      width = IMG_WIDTH,
      height = IMG_HEIGHT
    )
}
