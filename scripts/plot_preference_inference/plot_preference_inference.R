##################
# Authour: Chuang, Yun-Shiuan
# Date: 2020/03/06
# - plot the preference inference result at the group level
# Note:
# - data matrix
# - (number of agents/models x number of target). Cell_ij is the true preference of agent_i towards target_j.
# - the true preference value
# - for simulation data: the 'u' value
# - for human: the score of each target that human players see on the screen, which is the score of the social support questionnaire * 23/9. The score of the social support questionnaire is stored on the server http://gibms.mc.ntu.edu.tw/phpmyadmin. See 'models\working_model\test_on_human_data\README.txt' for access details.)
#
# - inferred matrix
# - (number of agents/models x number of target). Cell_ij is the true preference of agent_i towards target_j.
# - the inferred preference value
# - the inferred value in "proportion_prediction_and_ground_truth_labels.csv" at
# - Two versions of query states
# - 'Sxxx\prediction\Query_Stest_subset96\' (blank query state)
# 				- 'Sxxx\prediction\Query_Straj_subset96\' (the first shot of trajectories that have 4 targtes)
# 			- Two versions of preference measure (the column of the csv file)
# 				- prediction_proportion: the proportion of the predicted targets
# 				- avg_prediction_probability: the average softmax probability
#################
library(stringr)
library(dplyr)
library(tidyr)
library(ggplot2)
# Constants -----------------------------------------------
# - parameter
# AGENT_TYPE = "test_on_simulation_data"
AGENT_TYPE <- "human"
VERSION <- "v12"
PATTERN_PREFERENCE <- "proportion_prediction_and_ground_truth_labels.csv"
LIST_QUERY_TYPES <- list(
  qtest = "Query_Stest_subset96",
  qtraj = "Query_Straj_subset96"
)
LIST_INFERENCE_TYPES <- c("prediction_proportion", "avg_prediction_probability")
EXCLUSION_SUBJ <- "S052"
# - path
# the root path of the project
PATH_ROOT <- str_extract(
  string = getwd(),
  pattern = ".*tomnet-project"
)
# the path for file that contain the raw 'u' value for the simulation data
if (AGENT_TYPE == "simulation") {
  PATH_VALUE_U <- file.path(
    PATH_ROOT, "data",
    "data_simulation", "simulation_data_on_server",
    "36agents"
  )
} else if (AGENT_TYPE == "human") {
  PATH_VALUE_U <- file.path(
    PATH_ROOT, "data",
    "data_simulation", "simulation_data_on_server",
    "agents_from_human"
  )
}
PATH_PREFERENCE_PREDICTION <- file.path(
  PATH_ROOT, "models",
  "working_model",
  paste0("test_on_", AGENT_TYPE, "_data"),
  "training_result",
  "caches", VERSION
)
# - file
# Processing data -----------------------------------------------
# Get the predicted preference of simulation data ---------------
# - get the list of the predicted preference of simulation data
df_files_predicted_u <-
  data.frame(
    file = list.files(
      path = PATH_PREFERENCE_PREDICTION, pattern = PATTERN_PREFERENCE,
      full.names = T, recursive = T
    ),
    stringsAsFactors = F
  )

df_files_predicted_u <-
  df_files_predicted_u %>%
  mutate(
    query_type =
      case_when(
        grepl(
          pattern = LIST_QUERY_TYPES$qtest,
          x = file
        ) ~ "qtest",
        grepl(
          pattern = LIST_QUERY_TYPES$qtraj,
          x = file
        ) ~ "qtraj"
      ),
    subj_name =
      str_extract(
        string = file,
        pattern = "S\\d+(?=/prediction)"
      )
  )
# - read in all the preference files
list_df_predicted_u <- c()
for (file_index in 1:nrow(df_files_predicted_u)) {
  csv <- df_files_predicted_u$file[file_index]
  subj_name <- df_files_predicted_u$subj_name[file_index]
  query_type <- df_files_predicted_u$query_type[file_index]
  df_preference <- read.csv(csv, header = T, stringsAsFactors = F)
  df_preference <-
    df_preference %>%
    mutate(
      subj_name = subj_name,
      query_type = query_type,
      target_id = targets + 1 # convert to 1-based
    )
  list_df_predicted_u[[file_index]] <- df_preference
}
df_all_predicted_u <- bind_rows(list_df_predicted_u)

# Get the true preference of simulation data ---------------
df_files_true_u <- data.frame(
  file =
    list.files(
      path = PATH_VALUE_U,
      pattern = "S\\d+.*.csv",
      full.names = T
    ),
  stringsAsFactors = F
)

df_files_true_u <-
  df_files_true_u %>%
  mutate(
    subj_name =
      str_extract(
        string = file,
        pattern = "S\\d+"
      )
  )

# - read in all the preference files
list_df_true_u <- c()
for (file_index in 1:nrow(df_files_true_u)) {
  csv <- df_files_true_u$file[file_index]
  subj_name <- df_files_true_u$subj_name[file_index]
  df_preference <- read.csv(csv, header = T, stringsAsFactors = F)
  df_preference <-
    df_preference %>%
    mutate(
      target_id = as.numeric(str_extract(string = X, pattern = "\\d$")),
      true_u = V1,
      subj_name = subj_name
    )
  list_df_true_u[[file_index]] <- df_preference
}
df_all_true_u <- bind_rows(list_df_true_u)
# Combine the predicted u and the true u into one df ------------------------------
df_all <-
  df_all_predicted_u %>%
  select(subj_name, target_id, query_type, !!(LIST_INFERENCE_TYPES)) %>%
  filter(subj_name != EXCLUSION_SUBJ) %>%
  # rename(type = query_type) %>%
  left_join(df_all_true_u %>%
    select(subj_name, target_id, true_u, mu, sd, sk)
    # mutate(
    #   type = "ground_truth"
    # )
    ,
  by = c("subj_name", "target_id"))

# Visualize the preference matrix ------------------------------
