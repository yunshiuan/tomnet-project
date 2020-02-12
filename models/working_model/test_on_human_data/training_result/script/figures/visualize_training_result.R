##################
# Authour: Chuang, Yun-Shiuan
# This script is for visualizing the traning performaces
# that are stored in cache directories.
#################

# Constants
#path
# Manually set the root path of the repo if running the script via RStudio
if(interactive()){
  # Should be manually adjusted to correspond to either 'test_on_human_data' or 'test_on_simulated_data'
  PATH_ROOT = file.path(getwd(),"..","..","..")
}else{
  cat(paste0("Please enter the path of where the training_result is located in (without quotation mark), \n",
             "e.g., /Users/tomnet-project/models/working_model/test_on_human_data \n"))
  PATH_ROOT  <- readLines("stdin",n=1);
}

PATH_TRAINING_RESULT = file.path(PATH_ROOT,"training_result","caches")
PATH_FIGURE_OUTPUT = file.path(PATH_ROOT,"training_result","figures")
#file
FILE_HELPER_FUNCTION = file.path(PATH_ROOT,
                                 "training_result","script","figures",
                                 "helper_function_visualize_training_result.R")
source(FILE_HELPER_FUNCTION)

# Visualize
result = visualize_all_traning_performace(path_training_result = PATH_TRAINING_RESULT,
                                          path_figure_output = PATH_FIGURE_OUTPUT)
