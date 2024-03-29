##################
# Authour: Chuang, Yun-Shiuan
# This script is for converting human data to the format of simulated data.
# Note:
# (1) difference betweem the raw human txt and the target format (as in simulated txt)
# (i) there are commas at the start of each line
# (ii) there is no 'S'. So should take the first position as the position of 'S'.
# (iii) there is no 'Maze:' at the first line
# (iv) there is 'unmoved' step which should be ignored. E.g., S030_1189.txt
# (v) it is A, B, C, D instead of C, D, E, F.
# (vi) do not process if the starting point and the ending point is the same
# (don't put it to the processed data dir) E.g., S030_3660.txt
# (2) Because of 1-vi, the number of raw txts and the processed txts might not be the same.
# The processed files should always be equal to or less than the raw txt files.

#################
library(stringr)
# Constants-----------------------------------------------
# Parameter
# LIST_SUBJ <- paste0("S0", c(40))
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
# LIST_SUBJ <- paste0(
#   "S0",
#   c(
#     24, 53, 
#     58, 59, 
#     61, 62, 63,
#     67, 69
#   )
# )

# whether skip the subjects that already have some processed files
SKIP_PROCESSED <- TRUE
# MAZE_HEIGHT = 14 #including upper and lower wall
MAZE_UPPER_WALL_ROW_INDEX <- 1
MAZE_LOWER_WALL_ROW_INDEX <- 14
MAZE_LEFT_WALL_COL_INDEX <- 1
MAZE_RIGHT_WALL_COL_INDEX <- 14
SYMBOL_AGENT <- "S"
PADDING_FIRST_ROW <- "Maze:"

# Path
PATH_ROOT <- file.path(getwd(), "..", "..")
PATH_HUMAN_DATA <- file.path(PATH_ROOT, "data", "data_human")
# PATH_ROOT <- "/Users/vimchiz/bitbucket_local/observer_model_group/benchmark/test_on_human_data/data"
# PATH_ROOT <- "/home/.bml/Data/Bank6/Robohon_YunShiuan/tomnet-project/data/data_human"
PATH_DATA_INPUT <- file.path(PATH_HUMAN_DATA, "raw", LIST_SUBJ)
PATH_TXT_OUTPUT <- file.path(PATH_HUMAN_DATA, "processed", LIST_SUBJ)
# File
# Helper functions
parse_coordinate <- function(step_string) {
  coordinate_x <- as.numeric(str_extract(
    string = step_string,
    pattern = "(?<=\\()\\d+"
  ))
  coordinate_y <- as.numeric(str_extract(
    string = step_string,
    pattern = "\\d+(?=\\))"
  ))
  return(list(coordinate_x = coordinate_x, coordinate_y = coordinate_y))
}

# Convert-------------------------------------------------
for (subj_index in 1:length(PATH_DATA_INPUT)) {
  # local constants --------------------------------
  subj_path_data_input <- PATH_DATA_INPUT[subj_index]
  subj_txt_output <- PATH_TXT_OUTPUT[subj_index]
  subj_name <- LIST_SUBJ[subj_index]

  if (!dir.exists(subj_txt_output)) {
    dir.create(subj_txt_output)
  }
  # list all txt files
  txt_raw_files <- list.files(
    path = subj_path_data_input, recursive = F, pattern = ".*txt"
  )

  cat(paste0("Start processing ", subj_name, ".\n"))
  # exception handling --------------------------------
  # check if there is any files in the dir
  if ((length(txt_raw_files) == 0)) {
    cat(paste0(subj_name, " has no input files.\n"))
    next
  }
  # check if the files have already been processed
  txt_processed_files <- list.files(
    path = subj_txt_output, recursive = F, pattern = ".*txt"
  )
  # sort the files so that they follow the number order
  traj_id <- as.numeric(str_extract(
    pattern = "(?<=_)\\d+(?=.txt)",
    string = txt_raw_files
  ))
  txt_raw_files <- txt_raw_files[order(traj_id)]
  # skip if already processed
  if (SKIP_PROCESSED & (length(txt_raw_files) >= length(txt_processed_files)) & (length(txt_processed_files) > 0)) {
    cat(paste0(
      subj_name,
      " has already been processed.", "\n",
      "#raw files = ", length(txt_raw_files), "\n",
      "#processed files = ", length(txt_processed_files), "\n"
    ))
    next
  }

  # start processing and output txt --------------------------------
  for (txt_index in 1:length(txt_raw_files)) {
    txt_file_name <- txt_raw_files[txt_index]
    txt_full_file_name <- file.path(subj_path_data_input, txt_file_name)

    # read in the current txt file
    df_txt <- read.delim(txt_full_file_name, header = F, stringsAsFactors = F)

    # skip if this file is a duplicate of the previous one
    # - special case for the first file
    if (txt_index == 1) {
      previous_txt_file_name <- txt_raw_files[txt_index]
      previous_txt_full_file_name <- file.path(subj_path_data_input, previous_txt_file_name)
      previous_df_txt <- read.delim(previous_txt_full_file_name, header = F, stringsAsFactors = F)
      skip_duplicate <- FALSE
      count_skip_duplicate <- 0
      count_processed <- 0
    } else if (!skip_duplicate) {
      # - get the previous non-skipped file (start updating on the second file)
      previous_txt_full_file_name <- file.path(subj_path_data_input, previous_txt_file_name)
      previous_df_txt <- read.delim(previous_txt_full_file_name, header = F, stringsAsFactors = F)
    }
    # - see if they share the same maze and the same starting point
    #   (start checking on the second file)
    if (txt_index != 1) {
      if (all(df_txt[1:MAZE_LOWER_WALL_ROW_INDEX + 1, ] == previous_df_txt[1:MAZE_LOWER_WALL_ROW_INDEX + 1, ])) {
        cat(paste0("Skip: ", txt_file_name, " is a duplicate of ", previous_txt_file_name), ".\n")
        count_skip_duplicate <- count_skip_duplicate + 1
        skip_duplicate <- TRUE
        # skip this duplicated file
        next
      } else {
        # cat(paste0("Does not skip ", txt_file_name, "\n"))
        skip_duplicate <- FALSE
        count_processed <- count_processed + 1
        # save for next iteraction
        previous_txt_file_name <- txt_file_name
      }
    }


    # Remove the commas at each line (except the first line)
    for (row_index in 1:nrow(df_txt)) {
      if (row_index != 1) {
        df_txt[row_index, ] <- substring(df_txt[row_index, ], first = 2)
      }
    }

    # Put "S" into the maze
    tryCatch({
      initial_coordinate <- df_txt[MAZE_LOWER_WALL_ROW_INDEX + 1, ]
      coordinate <- parse_coordinate(initial_coordinate)
      initial_coordinate_x <- coordinate$coordinate_x
      initial_coordinate_y <- coordinate$coordinate_y
      # Replace the starting point "" by "S"
      row_to_be_replaced <- unlist(str_split(
        string = df_txt[MAZE_UPPER_WALL_ROW_INDEX + initial_coordinate_y, ],
        pattern = ""
      ))
      row_to_be_replaced[MAZE_LEFT_WALL_COL_INDEX + initial_coordinate_x] <- SYMBOL_AGENT
      row_to_be_replaced <- paste0(row_to_be_replaced, collapse = "")
      df_txt[MAZE_UPPER_WALL_ROW_INDEX + initial_coordinate_y, ] <- row_to_be_replaced
    },
    error = function(msg) {
      print(paste0(msg, ": ", txt_file_name))
    }
    )
    # Convert C, D, E, F to A, B, C, D
    for (row_index in (MAZE_UPPER_WALL_ROW_INDEX + 1):(MAZE_LOWER_WALL_ROW_INDEX - 1)) {
      df_txt$V1[row_index] <-
        df_txt$V1[row_index] %>%
        gsub(x = ., pattern = "D", replacement = "F") %>%
        gsub(x = ., pattern = "C", replacement = "E") %>%
        gsub(x = ., pattern = "B", replacement = "D") %>%
        gsub(x = ., pattern = "A", replacement = "C")
    }
    # Get the position of each target
    # - for truncating the steps (see the break condition in the while loop below)
    collect_row_index <- c()
    collect_col_index <- c()
    collect_target_name <- c()
    for (row_index in (MAZE_UPPER_WALL_ROW_INDEX + 1):(MAZE_LOWER_WALL_ROW_INDEX - 1)) {
      # try to get column coordinate of any target
      target_col_index <- (df_txt$V1[row_index] %>%
        str_locate_all(pattern = "[CDEF]", string = .))[[1]][, "start"]

      target_name <- strsplit(df_txt$V1[row_index], "")[[1]][target_col_index]
      # adjust the wall position
      target_col_index <- target_col_index - 1
      row_index <- row_index - 1

      # collect the target coordinate
      collect_col_index <- append(collect_col_index, target_col_index)
      collect_row_index <- append(
        collect_row_index,
        rep(row_index,
          times = length(target_col_index)
        )
      )
      collect_target_name <- append(
        collect_target_name, target_name,
      )
    }

    df_target_coordinates <- data.frame(
      row = collect_row_index,
      col = collect_col_index,
      name = collect_target_name
    )

    # Ignore unmoved steps
    step_starting_line <- MAZE_LOWER_WALL_ROW_INDEX + 1
    line_index <- step_starting_line
    processed_steps <- c() # to collect the filtered out steps
    # iterate through all the steps
    while (line_index <= nrow(df_txt)) {
      this_step_string <- df_txt$V1[line_index]
      # no need to check repetition for the first step
      if (line_index == step_starting_line) {
        processed_steps <- append(processed_steps, this_step_string)

        line_index <- line_index + 1
        previous_step_string <- this_step_string # for the next step to use
        next # skip this iteration
      }

      # check repetition
      # previous_step_string = df_txt$V1[line_index-1]
      if (this_step_string != previous_step_string) {
        processed_steps <- append(processed_steps, this_step_string)
      }


      # break the while loop if the agent has alreay reached one of the target (truncate the steps)
      # - this is necessary for handling the bug where the agent keep moving after reaching a target
      this_step_col <-
        as.numeric(str_extract(string = this_step_string, pattern = "(?<=())\\d+(?=,)"))
      this_step_row <-
        as.numeric(str_extract(string = this_step_string, pattern = "(?<=, )\\d+(?=\\))"))

      if (nrow(subset(df_target_coordinates, row == this_step_row & col == this_step_col)) > 0) {
        # cat(
        #   paste0("Break at row:", this_step_row," col:", this_step_col," line: ",line_index,"\n")
        #   )
        break
      }
      # for the next step to use
      previous_step_string <- this_step_string
      line_index <- line_index + 1
    }

    # skip the file if the starting point and the ending point is the same
    if (processed_steps[1] == processed_steps[length(processed_steps)]) {
      # Do nothing
    } else {
      # create the output string: replace the steps by the processed one
      output_string <- append(
        x = df_txt$V1[MAZE_UPPER_WALL_ROW_INDEX:MAZE_LOWER_WALL_ROW_INDEX],
        values = processed_steps
      )

      # Add 'Maze:' at the first line
      output_string <- append(
        x = output_string,
        values = PADDING_FIRST_ROW,
        after = 0
      )

      output_file <- file.path(subj_txt_output, txt_file_name)
      write.table(
        x = output_string,
        file = output_file,
        col.names = F, row.names = F,
        quote = FALSE
      )
    }
  }
  cat(
    "Finish:", subj_name, "\n",
    "Skip = ", count_skip_duplicate, "\n",
    "Processed =", count_processed, "\n"
  )
}
