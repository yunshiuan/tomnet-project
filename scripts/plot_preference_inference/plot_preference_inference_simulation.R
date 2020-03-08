##################
# Authour: Chuang, Yun-Shiuan
# Date: 2020/03/07
# - plot the preference inference result at the group level for simulation data.
# Note:
# - Based on plot_preference_inference_human.R.
#################
library(stringr)
library(dplyr)
library(tidyr)
library(ggplot2)
# Constants -----------------------------------------------
# - parameter
AGENT_TYPE <- "simulation"
VERSION <- "v26"
PATTERN_PREFERENCE <- "proportion_prediction_and_ground_truth_labels.csv"
LIST_QUERY_TYPES <- list(
  qtest = "Query_Stest_subset96",
  qtraj = "Query_Straj_subset96"
)
LIST_INFERENCE_TYPES <- c("prediction_proportion", "avg_prediction_probability")
EXCLUSION_SUBJ <- paste0(
  "S0",
  c( # does not have ground-truth score
    "52",
    # less than 100 trajectories
    "26", "35", "43", "52", "55", "58",
    # does not act according to the score
    "69"
  )
)
IMG_SIZE_RATIO <- 0.8
IMG_WIDTH <- 8 * IMG_SIZE_RATIO
IMG_HEIGHT <- 11 * IMG_SIZE_RATIO
SCALE_MIN <- 1
SCALE_MAX <- 4
# The color bar for bright blue to dark blue
COLOR_LOW <- "#132B43"
COLOR_HIGH <- "#56B1F7"
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
PATH_RESULT_ROOT <- file.path(
  PATH_ROOT, "models",
  "working_model",
  paste0("test_on_", AGENT_TYPE, "_data"),
  "training_result"
)
PATH_PREFERENCE_PREDICTION <- file.path(
  PATH_RESULT_ROOT,
  "caches", VERSION
)
PATH_FIGURE_OUTPUT <-
  file.path(PATH_RESULT_ROOT, "figures", VERSION)

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
  filter(!subj_name %in% EXCLUSION_SUBJ) %>%
  # rename(type = query_type) %>%
  left_join(df_all_true_u %>%
    select(subj_name, target_id, true_u, mu, sd, sk)
  # mutate(
  #   type = "ground_truth"
  # )
  ,
  by = c("subj_name", "target_id")
  )

df_all <-
  df_all %>%
  pivot_longer(
    cols = LIST_INFERENCE_TYPES,
    values_to = "predicted_u",
    names_to = "inference_type"
  ) %>%
  as.data.frame()
# Rank Transform the preference score --------------------------
df_all <-
  df_all %>%
  group_by(subj_name, query_type, inference_type) %>%
  mutate(
    true_u_rank = rank(-true_u, ties.method = "min"),
    predicted_u_rank = rank(-predicted_u, ties.method = "min"),
    sd_true_u_rank = sd(true_u_rank),
    sd_predicted_u_rank = sd(predicted_u_rank)
  ) %>%
  as.data.frame()
# Sort the subject by the SD of the transformed score-----------
num_rep <- length(LIST_QUERY_TYPES) * length(LIST_INFERENCE_TYPES)
num_target <- length(unique(df_all$target_id))
num_subj <- length(unique(df_all$subj_name))
df_all_plot <-
  df_all %>%
  arrange(desc(sd)) %>%
  mutate(
    # order the levels of subject name by the current rank order of 'sd_true_u_rank'
    subj_name_labeled = paste0(
      "S", str_extract(string = subj_name, pattern = "(?<=S0)\\d+$"),
      "\n(", round(sd, 2), ")"
    ),
    subj_name_labeled = factor(subj_name_labeled, levels = rev(unique(subj_name_labeled))),
    # # for the arrange below to break ties in 'true_u_rank'
    # query_type = as.factor(query_type),
    # inference_type = as.factor(inference_type)
  ) %>%
  arrange(desc(subj_name_labeled), true_u_rank, desc(target_id)) %>%
  mutate(
    # order the levels of targte id by the rank order of 'true_u_rank'
    target_id_reordered = rep(
      rep(1:num_rep, each = num_target),
      time = num_subj
    )
  )


# Visualize the preference matrix ------------------------------
# the predicted preference matrix
for (file_type in c(".png", ".pdf")) {
  df_all_plot %>%
    filter(inference_type == "avg_prediction_probability", query_type == "qtest") %>%
    ggplot(aes(x = target_id_reordered, y = subj_name_labeled)) +
    # geom_raster(aes(fill = predicted_u_rank))+
    geom_tile(aes(fill = predicted_u_rank)) +
    scale_fill_gradient(
      name = "Rank by \nPredicted Preference",
      limits = c(SCALE_MIN, SCALE_MAX), # Set the limits of the color bar
      low = COLOR_LOW, high = COLOR_HIGH,
      guide = guide_colourbar(reverse = T)
    ) + # Set the color of the color bar
    # coord_fixed(ratio = 1, xlim = NULL, ylim = NULL, expand = TRUE)+
    # facet_grid(inference_type~query_type)+
    # scale_fill_continuous("Rank-Transformed \nPredicted Preference")+
    labs(
      x = "Target ID",
      y = "Subject ID"
    ) +
    ggsave(
      filename = file.path(
        PATH_FIGURE_OUTPUT,
        paste0("Predicted Preference Matrix", file_type)
      ),
      width = IMG_WIDTH,
      height = IMG_HEIGHT
    )

  # facet_grid(inference_type~query_type)

  # the true preference matrix
  df_all_plot %>%
    ggplot(aes(x = target_id_reordered, y = subj_name_labeled)) +
    # geom_raster(aes(fill = true_u_rank))+
    geom_tile(aes(fill = true_u_rank)) +
    scale_fill_gradient(
      name = "Rank by \nGround-Truth Preference",
      limits = c(SCALE_MIN, SCALE_MAX), # Set the limits of the color bar
      low = COLOR_LOW, high = COLOR_HIGH,
      guide = guide_colourbar(reverse = T)
    ) + # Set the color of the color bar
    # coord_fixed(ratio = 1, xlim = NULL, ylim = NULL, expand = TRUE)+
    # guides(guide_colourbar(reverse = T))+
    # scale_fill_continuous("Rank-Transformed \nPredicted Preference")+
    labs(
      x = "Target ID",
      y = "Subject ID"
    ) +
    ggsave(
      filename = file.path(
        PATH_FIGURE_OUTPUT,
        paste0("Ground-Truth Preference Matrix", file_type)
      ),
      width = IMG_WIDTH,
      height = IMG_HEIGHT
    )
}
