##################
# Authour: Chuang, Yun-Shiuan
# Date: 2020/03/03
#- count the number of targets in each of the processed trajectory.
#- purpose:
# - derive the "random rate" for each human/agent. Because some trajectories have less than 4 targets, the random rate would be larger than 25%.
# - also, in order for the function 'filter_full_targets_trajectories.R' to check if filtering has been done for an agent, it needs to know the number of trajectories that have exact 4 targets.
#- note:
# - output 1: the agent-level csv that records the number of targets in each trajectories.
# - output 2: the group-level csv that recors the avergage number of targetr for each agent
#################
library(stringr)
# Constants-----------------------------------------------
# Parameter
LIST_SUBJ <- paste0(
  "S0",
  c(
    24, 26, 28,
    30, 31, 33,
    35, 36, 37,
    40, 43, 45,
    46, 50, 51,
    52, 53, 55,
    58, 59, 60,
    61, 62, 63,
    65, 66, 67,
    69
  )
)

# Path
PATH_ROOT <- str_extract(getwd(), pattern = ".*tomnet-project")
PATH_DATA_ROOT <- file.path(PATH_ROOT, "data", "data_human")
PATH_DATA_INPUT <- file.path(PATH_DATA_ROOT, "processed", LIST_SUBJ)

PATH_OUTPUT <- file.path(PATH_DATA_ROOT, "processed")
PATH_OUTPUT_AGENT <- file.path(PATH_DATA_ROOT, "processed", LIST_SUBJ)
# File
# - agent-level csv
FILE_COUNT_AGENT_OUTPUT <- file.path(
  PATH_OUTPUT_AGENT,
  paste0("count_targets_", Sys.Date(), ".csv")
)
# - group-level csv
FILE_COUNT_GROUP_OUTPUT <- file.path(
  PATH_OUTPUT,
  paste0("summary_count_targets_", Sys.Date(), ".csv")
)
# count the number of targets per agent -------------------------------------------------
# - collect the df_count_per_agent per agent
collect_df_count_per_agent = c()

for (subj_index in 1:length(PATH_DATA_INPUT)) {
  # local constants --------------------------------
  subj_path_data_input <- PATH_DATA_INPUT[subj_index]
  subj_name <- LIST_SUBJ[subj_index]

  # list all txt files which we want to count the targets
  txt_raw_files <- list.files(
    path = subj_path_data_input, recursive = F, pattern = ".*txt"
  )

  cat(paste0("Start processing ", subj_name, ".\n"))
  # exception handling --------------------------------
  # check if there is any files in the dir
  if ((length(txt_raw_files) == 0)) {
    warning(paste0(subj_name, " has no input files."))
    next
  }
  # start processing and output txt --------------------------------
  list_num <-
    lapply(txt_raw_files, FUN = function(txt_file_name) {
      txt_full_file_name <- file.path(subj_path_data_input, txt_file_name)
      df_txt <- read.delim(txt_full_file_name, header = F, stringsAsFactors = F)
      num_targets <-
        sum(grepl(x = df_txt$V1, pattern = "C")) +
        sum(grepl(x = df_txt$V1, pattern = "D")) +
        sum(grepl(x = df_txt$V1, pattern = "E")) +
        sum(grepl(x = df_txt$V1, pattern = "F"))
      return(num_targets)
    })
  df_count_per_agent = data.frame(
    file = txt_raw_files,
    num_targets = unlist(list_num))
  # write the agent-level csv
  output_file <- FILE_COUNT_AGENT_OUTPUT[subj_index]
  write.csv(
    x = df_count_per_agent,
    file = output_file
  )
  collect_df_count_per_agent[[subj_index]] = df_count_per_agent
}

# the group-level summary --------------------------------
