##################
# Authour: Chuang, Yun-Shiuan
# This script is for converting human data to the format of simulated data.
# Note:
# (i) there are commas at the start of each line
# (ii) there is no 'S'. So should take the first position as the position of 'S'.
# (iii) there is no 'Maze:' at the first line
# (iv) there is 'unmoved' step which should be ignored. E.g., S030_1189.txt
# (v) it is A, B, C, D instead of C, D, E, F.

#################
library(stringr)
# Constants-----------------------------------------------
# Parameter
# MAZE_HEIGHT = 14 #including upper and lower wall
MAZE_UPPER_WALL_ROW_INDEX <- 1
MAZE_LOWER_WALL_ROW_INDEX <- 14
MAZE_LEFT_WALL_COL_INDEX <- 1
MAZE_RIGHT_WALL_COL_INDEX <- 14
SYMBOL_AGENT <- "S"
PADDING_FIRST_ROW <- "Maze:"

# Path
#PATH_ROOT <- "/Users/vimchiz/bitbucket_local/observer_model_group/benchmark/test_on_human_data/data"
PATH_ROOT <- "/home/.bml/Data/Bank6/Robohon_YunShiuan/observer_model_group/benchmark/test_on_human_data/data"
PATH_DATA_INPUT <- file.path(PATH_ROOT, "raw", "S030")
PATH_TXT_OUTPUT <- file.path(PATH_ROOT, "processed", "S030")
# File
# Helper functions
parse_coordinate <- function(step_string) {
  coordinate_x <- as.numeric(str_extract(
    string = step_string,
    pattern = "(?<=\\()\\d"
  ))
  coordinate_y <- as.numeric(str_extract(
    string = step_string,
    pattern = "\\d(?=\\))"
  ))
  return(list(coordinate_x = coordinate_x, coordinate_y = coordinate_y))
}
# Convert-------------------------------------------------
# list all txt files
txt_files <- list.files(
  path = PATH_DATA_INPUT, recursive = F, pattern = ".*txt"
)

lapply(txt_files, FUN = function(txt_file_name) {
  txt_full_file_name <- file.path(PATH_DATA_INPUT, txt_file_name)
  df_txt <- read.delim(txt_full_file_name, header = F, stringsAsFactors = F)

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
    previous_step_string <- this_step_string # for the next step to use
    line_index <- line_index + 1
  }

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

  output_file <- file.path(PATH_TXT_OUTPUT, txt_file_name)
  write.table(
    x = output_string,
    file = output_file,
    col.names = F, row.names = F,
    quote = FALSE
  )
})
