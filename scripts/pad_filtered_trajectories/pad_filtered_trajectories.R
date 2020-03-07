##################
# Authour: Chuang, Yun-Shiuan
# Date:2020/03/06
# This script is for padding the filtered data 
# (output of 'filter_full_targets_trajectories') to be equal to or above a specific number. 
# This is necessary for the human who has too less filtered data, e.g., 
# S026 only has 19 filtered data, but the commented_preference_predictor.py 
# needs at least 96.
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
MIN_FILES <- 96

# Path
PATH_ROOT <- str_extract(getwd(), pattern = ".*tomnet-project")
PATH_DATA_ROOT <- file.path(PATH_ROOT, "data", "data_human")
PATH_DATA_INPUT <- file.path(PATH_DATA_ROOT, "filtered", LIST_SUBJ)
PATH_TXT_OUTPUT <- file.path(PATH_DATA_ROOT, "filtered", LIST_SUBJ)

# File

# Convert-------------------------------------------------
for (subj_index in 1:length(PATH_DATA_INPUT)) {
  # local constants --------------------------------
  subj_path_data_input <- PATH_DATA_INPUT[subj_index]
  subj_txt_output <- PATH_TXT_OUTPUT[subj_index]
  this_subj_name <- LIST_SUBJ[subj_index]

  if (!dir.exists(subj_txt_output)) {
    dir.create(subj_txt_output)
  }
  # list all txt files (current filtered data)
  txt_raw_files <- list.files(
    path = subj_path_data_input, recursive = F, pattern = ".*txt",full.names = T
  )
  
  # sort the files so that they follow the number order
  traj_id <- as.numeric(str_extract(
    pattern = "(?<=_)\\d+(?=.txt)",
    string = txt_raw_files
  ))
  txt_raw_files <- txt_raw_files[order(traj_id)]
  
  cat(paste0("Start processing ", this_subj_name, ".\n"))
  # exception handling --------------------------------
  # check if there is any files in the dir
  if ((length(txt_raw_files) == 0)) {
    warning(paste0(this_subj_name, " has no input files."))
    next
  }
  # check if the files have already been filtered

  # skip if already filtered
if ((length(txt_raw_files) >= MIN_FILES)) {
    warning(paste0(
      this_subj_name,
      " has already been processed.", "\n",
      "#filtered files = ", length(txt_raw_files), "\n"
    ))
    next
  }

  # start processing and output txt --------------------------------
  num_files = length(txt_raw_files)
  file_index = 0
  pad_round = 0
  while(num_files < MIN_FILES){
    # update the file index
    file_index = file_index+1
    if(file_index > length(txt_raw_files)){
      file_index = file_index %% length(txt_raw_files)
    }
    
    # update the padding round index
    if(file_index == 1){
      pad_round= pad_round+1
    }
    
    file_to_copy = txt_raw_files[file_index]
    file_name = paste0(
      str_extract(string = file_to_copy,pattern = ".*(?=.txt)"),
      "_pad",pad_round,
      ".txt")
    file.copy(file_to_copy, file_name)
    # file.copy(file_to_copy, subj_path_data_input)
    num_files = num_files + 1
    # cat(paste0(num_files,":",file_name,"\n"))
  }
}
