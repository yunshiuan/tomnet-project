##################
# Authour: Chuang, Yun-Shiuan
# This script is for converting human data to the format of simulated data.
# Note:
# (1) augment the trajectory data by reflection and rotation of the x and y axis.
# Each maze results in 8 agumented data (4 rotation x 2 reflection)
# (2) Rotation: 0, 90, 180, 270 degrees
# (3) Reflection: swap the x and y axis (could be done by transposing)
#################
library(stringr)
library(dplyr)
# Constants-----------------------------------------------
# Parameter
# LIST_SUBJ <- paste0("S0", c(52))
LIST_SUBJ <- paste0("S0", c(24, 30, 33, 35, 50, 51, 52))
MAZE_UPPER_WALL_ROW_INDEX <- 2
MAZE_LOWER_WALL_ROW_INDEX <- 15
MAZE_HEIGHT <- 12
MAZE_WIDTH <- 12
PADDING_FIRST_ROW <- "Maze:"


ROTATE_TIMES <- c(0, 1, 2, 3) # how many times of 90 degrees to be rotated
TRANSPOSE_TIMES <- c(0, 1) # how many times of transposing

# Path
PATH_ROOT <- str_extract(getwd(), pattern = ".*tomnet-project")
PATH_DATA_ROOT <- file.path(PATH_ROOT, "data", "data_human")
PATH_DATA_INPUT <- file.path(PATH_DATA_ROOT, "processed", LIST_SUBJ)
PATH_TXT_OUTPUT <- file.path(PATH_DATA_ROOT, "augmented", LIST_SUBJ)

# File
# Helper functions
# - parse a coordinate string into numbers
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
# - convert coordinate numbers into a string
convert_coordinate_to_string <- function(x, y) {
  return(paste0("(", x, ", ", y, ")"))
}
# rotate a step coordinate 90 degrees clockwise
rotate_step_90 <- function(x, y, times) {
  # make sure times is a non-negative integer
  assertthat::assert_that(times %% 1 == 0 & times >= 0)
  # how many times of 90 degrees
  times <- times %% 4
  if (times == 0) {
    new_x <- x
    new_y <- y
  } else if (times == 1) {
    new_x <- MAZE_HEIGHT - y + 1
    new_y <- x
  } else if (times == 2) {
    new_x <- MAZE_WIDTH - x + 1
    new_y <- MAZE_HEIGHT - y + 1
  } else if (times == 3) {
    new_x <- y
    new_y <- MAZE_WIDTH - x + 1
  }
  return(list(new_x = new_x, new_y = new_y))
}
# transpose a step coordinate
transpose_step <- function(x, y, times) {
  # make sure times is a non-negative integer
  assertthat::assert_that(times %% 1 == 0 & times >= 0)
  # how many times of tansposing
  if (times == 1) {
    new_x <- y
    new_y <- x
  } else {
    new_x <- x
    new_y <- y
  }
  return(list(new_x = new_x, new_y = new_y))
}
# rotate a matrix 90 degrees clockwise
rotate_maze_90 <- function(x) t(apply(x, 2, rev))

# convert a string dataframe (with one column) to a matrix
df_to_matrix <- function(df) {
  matrix <- c()
  for (row_index in 1:nrow(df)) {
    row_vector <- str_split(string = df[row_index, 1], pattern = "", simplify = T)
    matrix <- append(matrix, row_vector)
  }
  matrix <- matrix(data = matrix, nrow = nrow(df), byrow = T)
  return(matrix)
}
# convert a string matrix bach to a string dataframe (with one column)
matrix_to_df <- function(matrix) {
  df <- c()
  for (row_index in 1:nrow(matrix)) {
    row_string <- paste(matrix[row_index, ], collapse = "")
    df <- append(df, row_string)
  }
  df <- data.frame(V1 = df, stringsAsFactors = F)
  return(df)
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
  # check if the files have already been augmented
  txt_processed_files <- list.files(
    path = subj_txt_output, recursive = F, pattern = ".*txt"
  )

  # skip if already augmented
  if (((length(txt_raw_files) * 8) == length(txt_processed_files)) & (length(txt_processed_files) > 0)) {
    warning(paste0(
      subj_name,
      " has already been processed.", "\n",
      "#input files = ", length(txt_raw_files), "\n",
      "#augmented files = ", length(txt_processed_files), "\n"
    ))
    next
  }

  # start processing and output txt --------------------------------
  lapply(txt_raw_files, FUN = function(txt_file_name) {
    txt_full_file_name <- file.path(subj_path_data_input, txt_file_name)
    df_txt <- read.delim(txt_full_file_name, header = F, stringsAsFactors = F)

    # separate the maze and the steps
    df_maze <- data.frame(
      V1 = df_txt[MAZE_UPPER_WALL_ROW_INDEX:MAZE_LOWER_WALL_ROW_INDEX, ]
    )
    df_steps <- data.frame(
      V1 = df_txt[(MAZE_LOWER_WALL_ROW_INDEX + 1):nrow(df_txt), ]
    )

    # Start augmentation and write out the augmented files

    # - rotate the maze by 0/90/180/360 degrees (and also update the steps accordingly)
    for (rotate_time in ROTATE_TIMES) {
      # - transpose the maze or not (and also update the steps accordingly)
      for (transpose_time in TRANSPOSE_TIMES) {
        # transform the maze
        matrix_maze <- df_to_matrix(df_maze)
        count_rotate <- 0
        count_transpose <- 0
        while (count_rotate < rotate_time) {
          matrix_maze <- rotate_maze_90(matrix_maze)
          count_rotate <- count_rotate + 1
        }
        while (count_transpose < transpose_time) {
          matrix_maze <- t(matrix_maze)
          count_transpose <- count_transpose + 1
        }

        # transform the steps
        transformed_step_string <-
          df_steps %>%
          mutate(
            # parse the string coordinate
            step_x = parse_coordinate(step_string = V1)$coordinate_x,
            step_y = parse_coordinate(step_string = V1)$coordinate_y,
            # transform
            # - rotate
            rotated_step_x = rotate_step_90(
              x = step_x,
              y = step_y,
              times = rotate_time
            )$new_x,
            rotated_step_y = rotate_step_90(
              x = step_x,
              y = step_y,
              times = rotate_time
            )$new_y,
            # - transpose
            transformed_step_x = transpose_step(
              x = rotated_step_x,
              y = rotated_step_y,
              times = transpose_time
            )$new_x,
            transformed_step_y = transpose_step(
              x = rotated_step_x,
              y = rotated_step_y,
              times = transpose_time
            )$new_y,
            # - convert back to string
            transformed_V1 = convert_coordinate_to_string(
              x = transformed_step_x,
              y = transformed_step_y
            )
          ) %>%
          pull(transformed_V1)

        # convert to an augmented df and write it out
        df_augmented <- matrix_to_df(matrix_maze)
        
        # add 'Maze:' at the first line
        
        df_augmented =         
        data.frame(V1 = PADDING_FIRST_ROW,stringsAsFactors = F)%>%
          bind_rows(df_augmented)
        
        # add steps after the maze
        df_augmented <-
          df_augmented %>%
          bind_rows(data.frame(V1 = transformed_step_string, stringsAsFactors = F))

        txt_subj <- str_extract(string = txt_file_name, pattern = "S\\d+(?=_)")
        txt_number <- str_extract(string = txt_file_name, pattern = "(?<=_)\\d+(?=\\.txt)")
        txt_augmented <- paste0("r", count_rotate, "t", count_transpose)
        output_txt_file_name <- paste0(txt_subj, "_", txt_number, "_", txt_augmented, ".txt")

        output_file <- file.path(subj_txt_output, output_txt_file_name)
        write.table(
          x = df_augmented$V1,
          file = output_file,
          col.names = F, row.names = F,
          quote = FALSE
        )
      }
    }
  })
}
