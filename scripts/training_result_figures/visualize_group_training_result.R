##################
# Authour: Chuang, Yun-Shiuan
# Date: 2020/03/04
# This script is for visualizing the model accuracy
# across all subjects in one line plot.
# Note:
# - (1) axises:
#   - x axis: total unique trajectories (# of processed files)
#   - y axis: accuracy
#   - 3 lines: training, validation, and testing accuracy
# - (2) This only works for the traing version which have multiple models trained on
#       multiple agents, e.g., human, v12.
#################
library(stringr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)

# Constants -----------------------------------------------
# - parameter
# TYPE = "test_on_simulation_data"
TYPE <- "human"
VERSION <- "v12"
PATTERN_ACCURACY <- "_train_test_and_validation_accuracy.csv"
NUM_AUGMENTATION <- 8
# - path
# the root path of the project
PATH_ROOT <- str_extract(
  string = getwd(),
  pattern = ".*tomnet-project"
)
EXCLUSION_SUBJ <- paste0(
  "S0",
  c( # does not have ground-truth score
    "52",
    # less than 100 trajectories
    "26", "35", "43", "52", "55", "58",
    # does not act according to the score
    "69"
  )
)
# the least training files the subject should have for the "thresholded version"
# - thresholding ensure the estimate of accuracy is precise enough
THRESHOLD_NUM_FILES <- 100
IMG_SIZE_RATIO <- 0.8
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

# - file
# the file that contain the random rate for each subject
FILE_RANDOM_RATE <- file.path(
  PATH_ROOT, "data",
  paste0("data_", TYPE), "processed",
  "summary_count_targets_2020-03-03.csv"
)
FILE_OUTPUT <- list(
  no_threshold = file.path(PATH_FIGURE_OUTPUT, "all_training_results.pdf"),
  with_threshold = file.path(
    PATH_FIGURE_OUTPUT,
    paste0("all_training_results_traj", THRESHOLD_NUM_FILES, ".pdf")
  )
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
# Plot ------------------------------------------------------------

for (threshold in c("with_threshold")) {
  if (threshold == "with_threshold") {
    # filter out those below the threshold
    df_plot <-
      df_error_all %>%
      filter(total_processed_training_files >= THRESHOLD_NUM_FILES)
    # set the scale
    x_scale_minor_breaks <- seq(0, 4000, 100)
    x_scale_breaks <- c(200, 500, 1000, 2000, 3000, 4000)
  } else {
    df_plot <- df_error_all
    # set the scale
    x_scale_minor_breaks <- seq(0, 4000, 100)
    x_scale_breaks <- c(100, 500, 1000, 2000, 3000, 4000)
  }
  df_plot = 
    df_plot %>%
      # log transform the scale to make the dots more visible
      # mutate(log_total_processed_training_files = log10(total_processed_training_files)) %>%
      filter(mode %in% c("random_rate", "test")) %>%
      mutate(
        mode = factor(mode,
          levels = c("test", "random_rate"),
          labels = c("Test", "Random Rate")
        ),
        subj_label = paste0("S",
                            str_extract(string = subj_name, pattern = "(?<=S0)\\d+$"))
      ) 
  df_plot %>%
    ggplot(aes(x = total_processed_training_files, y = accuracy, group = mode)) +
    geom_point(aes(shape = mode,color = mode)) +
    geom_line(aes(linetype = mode,color = mode)) +
    geom_text_repel(data = df_plot%>%
                      filter(mode =="Test"),
                    aes(label = subj_label),
                    size = 2) +
    coord_trans(x = "log10") +
    scale_x_continuous(
      minor_breaks = x_scale_minor_breaks,
      breaks = x_scale_breaks
    ) +
    scale_y_continuous(breaks = seq(30, 80, by = 10), limits = c(30, 85)) +
    scale_color_discrete(NULL) +
    scale_shape_discrete(NULL) +
    scale_linetype_discrete(NULL) +
    theme_bw() +
    labs(
      x = "Trajectories in the Training Set",
      y = "Accuracy"
    ) +
    ggsave(
      filename = FILE_OUTPUT[[threshold]],
      width = IMG_WIDTH,
      height = IMG_HEIGHT
    )
}
