##################
# Authour: Chuang, Yun-Shiuan
# Date: 2020/02/24
# This script is for counting the total playing time based on the modifed time of the raw data.
# Note:
# - Total playing time = sum of the playing duration of each file.
# - File are separated into fragments if the interval of modified time
# between two consecutive files is greater than a certain threshold (e.g., 5 mins).
# - The modified time of the file is the end time of each file (the completion of the trajectory)
# - The duration of the first file of each segment is uncomputable because only the
# end time of each file is recorded.
# - Use imputation (filling by grand mean of duration) to fill in the duration of
# first file of each segment
# CAUTION! Currently, the script will only work under 1 condition.
#(1) The raw data is "fresh". This is, they are just downloaded from the server so that 
# the modifed time will correctly indication the trajectory completion time. Any modification, e.g.,
# move, rename will change the time stamp.
library(stringr)
library(dplyr)
library(tidyr)
# Constants-----------------------------------------------
# Parameter
LIST_SUBJ <- paste0("S0", c(63))
# LIST_SUBJ <- paste0("S0", c(24, 30, 33, 35, 40, 50, 51, 52))

# the threshold that decides whether
# - two consecutive files belong to the same playing fragment
FRAGEMENT_INTERVAL <- 600
# the minimun time to complete the trajectory
# - this is necessary when there are duplicated files as in S063
MIN_DURATION = 1

# Path
PATH_ROOT <- str_extract(getwd(), pattern = ".*tomnet-project")
PATH_HUMAN_DATA <- file.path(PATH_ROOT, "data", "data_human")
PATH_DATA_INPUT <- file.path(PATH_HUMAN_DATA, "raw", LIST_SUBJ)
PATH_COUNT_TIME_OUTPUT <- file.path(PATH_HUMAN_DATA, "raw")

# File
FILE_COUNT_TIME_OUTPUT <- file.path(PATH_COUNT_TIME_OUTPUT, "playing_time.csv")

# Helper functions
get_modified_time <- function(file_full_path) {
  return(file.info(file_full_path)$mtime)
}

collect_total_duration <- c()
collect_mean_duration <- c()
collect_total_files <- c()
# save the result
# count playing time-------------------------------------------------
for (subj_index in 1:length(PATH_DATA_INPUT)) {
  # local constants --------------------------------
  subj_path_data_input <- PATH_DATA_INPUT[subj_index]
  subj_name <- LIST_SUBJ[subj_index]

  cat(paste0("Start processing ", subj_name, ".\n"))

  # if (!dir.exists(subj_txt_output)) {
  #   dir.create(subj_txt_output)
  # }
  # list all txt files
  txt_raw_files <- list.files(
    path = subj_path_data_input, recursive = F, pattern = ".*txt", full.names = T
  )

  # sort the files so that they follow the number order
  traj_id <- as.numeric(str_extract(
    pattern = "(?<=_)\\d+(?=.txt)",
    string = txt_raw_files
  ))
  txt_raw_files <- txt_raw_files[order(traj_id)]

  # store the files in a data frame
  df_files <- data.frame(
    file = txt_raw_files,
    stringsAsFactors = F
  )
  # get the time stamp of each file
  df_files <-
    df_files %>%
    mutate(
      time = get_modified_time(file)
    )
  # separate files into fragments
  # - intialize as the invalid value -1
  df_files$fragment <- -1
  df_files$duration <- -1 # playing time for this file
  segment_index <- 1
  for (txt_index in 1:nrow(df_files)) {
    this_file_time <- df_files$time[txt_index]

    # special case for the first file (of the first segment)
    if (txt_index == 1) {
      df_files$duration[txt_index] <- NA # uncomputable for the first file of the segment
      # the start of the first fragment
      df_files$fragment[txt_index] <- segment_index
      # save for the next file to compared with
      previous_file_time <- this_file_time
      next
    }
    # compare the time difference between this file and the previous file
    this_file_time <- df_files$time[txt_index]
    time_diff <-
      as.numeric(difftime(this_file_time, previous_file_time, units = "secs"))

    # save the duration and segment index for this file if still in the same segment
    if (time_diff <= FRAGEMENT_INTERVAL) {
      # save the duration of this file
      df_files$duration[txt_index] <- time_diff
      df_files$fragment[txt_index] <- segment_index
      # save for the next file to compared with
      previous_file_time <- this_file_time
    } else {
      # start a new segment if the interval is larger than the threshold
      segment_index <- segment_index + 1

      df_files$duration[txt_index] <- NA # uncomputable for the first file of the segment
      # the start of the first fragment
      df_files$fragment[txt_index] <- segment_index
      # save for the next file to compared with
      previous_file_time <- this_file_time
      next
    }
  }
  # filter out duplicate files by thresholding 
  df_files = 
  df_files%>%
    filter(duration>MIN_DURATION)
  
  # impute the duration of the first file of each segment
  # - by filling the grand mean of duration
  mean_duration <- mean(df_files$duration, na.rm = T)
  df_files <-
    df_files %>%
    mutate(duration = replace_na(duration, mean_duration))
  # compute the total duration
  total_duration <- sum(df_files$duration)
  mean_duration <- total_duration / length(df_files$duration)

  # assert: all duration and fragement values should not be -1
  stopifnot(!any(df_files$duration == -1))
  stopifnot(!any(df_files$fragment == -1))
  cat(
    "Finish:", subj_name, "\n",
    "Total playing time = ", total_duration, "\n",
    "Total files = ", nrow(df_files), "\n",
    "Average playing time = ", mean_duration, "\n"
  )
  # save the result
  collect_total_duration <- append(collect_total_duration, total_duration)
  collect_mean_duration <- append(collect_mean_duration, mean_duration)
  collect_total_files <- append(collect_total_files, nrow(df_files))
}
# 
# # write the result as csv file
# df_playing_time <- data.frame(
#   subj = LIST_SUBJ,
#   total_duration = collect_total_duration,
#   mean_duration = collect_mean_duration,
#   total_files = collect_total_files,
#   stringsAsFactors = F
# )
