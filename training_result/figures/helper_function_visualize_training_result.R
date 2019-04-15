##################
# Authour: Chuang, Yun-Shiuan
# These are the helper functions for visualizing the traning performaces
# that are store in cache directories.
#################
library(ggplot2)
library(dplyr)
library(tidyr)

visualize_one_traning_performace <- function(file_error_csv, file_name_output_figure) {
  # This funciton is to visualize one training result that is stored
  # in error.csv file in the cache directory.
  # - param file_error_csv: the "error_csv" file stored in the cache directory.
  # - param file_name_output_figure: the full file name of the output figure.
  # Should include the file type suffix (e.g., .pdf, .png)
  # - param path_output_figure: the path of the output figure.
  # - return None: It write the plot to a output file.

  # local constant
  LABEL_Y_AXIS <- "Error Value"
  VALUE_Y_AXIS_LIMIT <- c(0, 1)
  LABEL_ERROR_TYPE <- c("Train Error", "Validation Error")
  VALUE_PLOT_WIDTH <- 8
  VALUE_PLOT_HEIGHT <- 6
  df_error_csv <- read.csv(file_error_csv, header = T)
  df_error_csv %>%
    gather(error_type, error_value, train_error, validation_error) %>%
    ggplot(aes(x = step, y = error_value, color = error_type)) +
    geom_line() +
    labs(y = LABEL_Y_AXIS) +
    scale_y_continuous(limits = VALUE_Y_AXIS_LIMIT) +
    scale_color_discrete(labels = LABEL_ERROR_TYPE) +
    ggsave(
      file = file_name_output_figure, path = path_figure_output,
      width = VALUE_PLOT_WIDTH, height = VALUE_PLOT_HEIGHT,
      units = "in"
    )
}

visualize_all_traning_performace <- function(path_training_result, path_output_figure) {
  # This funciton is to output figures from all cache results.
  # It checks the existence of the figures and skip if the figures already exist.
  # - param path_training_result: the root path where all the cache directories exist.
  # - param path_output_figure: the root path where all the figures should be created.
  # - return None: output all the plots to a output directoris
  list_error_csv <- list.files(
    path = path_training_result,
    recursive = T,
    pattern = "_error.csv"
  )
}
