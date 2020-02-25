##################
# Authour: Chuang, Yun-Shiuan
# This script is for filtering in the mazes that
# have exact the maximun of the number of targets and collect this
# mazes into an output directory.
#################
library(stringr)
# Constants-----------------------------------------------
# Parameter
LIST_SUBJ <- paste0("S0", c(24, 30, 33, 35, 50, 51, 52))
MAX_TARGETS <- 4

# Path
PATH_ROOT <- str_extract(getwd(), pattern = ".*tomnet-project")
PATH_DATA_ROOT <- file.path(PATH_ROOT, "data", "data_human")
PATH_DATA_INPUT <- file.path(PATH_DATA_ROOT, "processed", LIST_SUBJ)
PATH_TXT_OUTPUT <- file.path(PATH_DATA_ROOT, "filtered", LIST_SUBJ)
# File
# Convert-------------------------------------------------
for (subj_index in 1:length(PATH_DATA_INPUT)) {
  # local constants --------------------------------
  subj_path_data_input <- PATH_DATA_INPUT[subj_index]
  subj_txt_output <- PATH_TXT_OUTPUT[subj_index]
  subj_name <- LIST_SUBJ[subj_index]

  if (!dir.exists(subj_txt_output)) {
    dir.create(subj_txt_output)
  }
  # list all txt files (unfiltered)
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
  # check if the files have already been filtered
  txt_processed_files <- list.files(
    path = subj_txt_output, recursive = F, pattern = ".*txt"
  )

  # skip if already filtered
  if ((length(txt_raw_files) >= length(txt_processed_files)) & (length(txt_processed_files) > 0)) {
    warning(paste0(
      subj_name,
      " has already been processed.", "\n",
      "#input files = ", length(txt_raw_files), "\n",
      "#filtered files = ", length(txt_processed_files), "\n"
    ))
    next
  }

  # start processing and output txt --------------------------------
  lapply(txt_raw_files, FUN = function(txt_file_name) {
    txt_full_file_name <- file.path(subj_path_data_input, txt_file_name)
    df_txt <- read.delim(txt_full_file_name, header = F, stringsAsFactors = F)
    num_targets <-
      sum(grepl(x = df_txt$V1, pattern = "C")) +
      sum(grepl(x = df_txt$V1, pattern = "D")) +
      sum(grepl(x = df_txt$V1, pattern = "E")) +
      sum(grepl(x = df_txt$V1, pattern = "F"))
    # if the number of targets in the maze is the maximunm, copy it to the 'filtered' dir
    if(num_targets==MAX_TARGETS){
        output_file <- file.path(subj_txt_output, txt_file_name)
        write.table(
            x = df_txt$V1,
            file = output_file,
            col.names = F, row.names = F,
            quote = FALSE
        )        
    }

  })
}
