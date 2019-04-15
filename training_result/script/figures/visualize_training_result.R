##################
# Authour: Chuang, Yun-Shiuan
# This script is for visualizing the traning performaces
# that are stored in cache directories.
#################

# Constants
#path
path_root <- readline("Please enter the root path of the repository.")
PATH_ROOT = "/Users/vimchiz/bitbucket_local/observer_model_group"
PATH_TRAINING_RESULT = file.path(PATH_ROOT,"training_result","caches")
PATH_FIGURE_OUTPUT = file.path(PATH_ROOT,"training_result","figures")
#file
FILE_HELPER_FUNCTION = file.path(PATH_ROOT,
                                 "training_result","script","figures",
                                 "helper_function_visualize_training_result.R")
source(FILE_HELPER_FUNCTION)

# Visualize
visualize_all_traning_performace(path_training_result = PATH_TRAINING_RESULT,
                                 path_figure_output = PATH_FIGURE_OUTPUT)
