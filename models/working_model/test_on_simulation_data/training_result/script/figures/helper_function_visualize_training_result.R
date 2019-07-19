##################
# Authour: Chuang, Yun-Shiuan
# These are the helper functions for visualizing the traning performaces
# that are stored in cache directories.
#################
library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)

visualize_one_traning_performace <- function(file_error_csv,
                                             file_name_figure_output,
                                             path_figure_output,
                                             overwrite = FALSE) {
  # This funciton is to visualize one training result that is stored
  # in error.csv file in the cache directory.
  # - param file_error_csv: the "error_csv" file stored in the cache directory.
  # - param file_name_figure_output: the file name of the output figure.
  # - param path_figure_output: the path where the output figure should be created
  # Should include the file type suffix (e.g., .pdf, .png)
  # - param path_output_figure: the path of the output figure.
  # - param overwrite: whether to overwrite if the figure file already exists
  # - return None: It write the plot to a output file.

  # check if the figure file already exists
  if (!overwrite) { # skip it if overwrite == F
    figure_full_name <- file.path(path_figure_output, file_name_figure_output)
    if (file.exists(figure_full_name)) {
      return(warning("The figure file already exists.", call. = F))
    }
  }
  # local constant
  LABEL_Y_AXIS <- "Error Value"
  VALUE_Y_AXIS_LIMIT <- c(0, 1)
  LABEL_ERROR_TYPE <- c("Train Error", "Validation Error")
  VALUE_PLOT_WIDTH <- 8 * 0.5
  VALUE_PLOT_HEIGHT <- 5 * 0.5
  df_error_csv <- read.csv(file_error_csv, header = T)
  df_error_csv %>%
    gather(error_type, error_value, train_error, validation_error) %>%
    ggplot(aes(x = step, y = error_value, color = error_type)) +
    geom_line() +
    labs(y = LABEL_Y_AXIS) +
    scale_y_continuous(limits = VALUE_Y_AXIS_LIMIT) +
    scale_color_discrete(labels = LABEL_ERROR_TYPE) +
    ggsave(
      file = file_name_figure_output, path = path_figure_output,
      width = VALUE_PLOT_WIDTH, height = VALUE_PLOT_HEIGHT,
      units = "in"
    )
}

visualize_all_traning_performace <- function(path_training_result,
                                             path_figure_output,
                                             overwrite = FALSE) {
  # This funciton is to output figures from all cache results.
  # It checks the existence of the figures and skip if the figures already exist.
  # - param path_training_result: the root path where all the cache directories exist.
  # - param path_figure_output: the root path where all the figures should be created.
  # - param overwrite: whether to overwrite if the figure file already exists
  # - return None: output all the plots to a output directoris
  list_error_csv <- list.files(
    path = path_training_result,
    recursive = T,
    pattern = "_error.csv",
    full.names = T
  )
  list_warning_message = c()
  list_error_message = c()
  result <- lapply(list_error_csv, function(csv_file) {
    tryCatch({
      train_type <- str_extract(
        string = csv_file,
        pattern = "(?<=cache_).*(?=/train)"
      )
      file_name_figure_output <- paste0(train_type, ".pdf")

      # Make a valid file name
      file_name_figure_output <- gsub(
        pattern = "/",
        x = file_name_figure_output,
        replacement = "__"
      )
      visualize_one_traning_performace(
        file_error_csv = csv_file,
        file_name_figure_output = file_name_figure_output,
        path_figure_output = path_figure_output
      )
      return(paste0("Finish: ", file_name_figure_output))
    }, warning = function(msg) {
      #print(paste0("Warning with ", file_name_figure_output, ": ", msg))
      warning_message = paste0("Warning with ", file_name_figure_output, ": ", msg$message)
      #list_warning_message = append(list_warning_message,warning_message)
      return(warning_message)
    }, error = function(msg) {
      #print(paste0("Error with ", file_name_figure_output, ": ", msg))
      error_message = paste0("Error with ", file_name_figure_output, ": ", msg$message)
      #list_error_message = append(list_error_message,error_message)
      return(error_message)
    })
  })
  # Print the summary
  result = unlist(result)
  list_finish = grep(pattern = "Finish",x = result,value = T)
  list_skip = grep(pattern = "already exists",x = result,value = T)
  summary =  paste0("Output figure: ",length(list_finish),"\n",
                    "Skip: ", length(list_skip),"\n",
                    "Total: ", length(result),"\n\n",
                    "Output details:","\n",
                    paste(paste0("  ",list_finish),collapse = "\n")
                    )
  cat(summary)
}
