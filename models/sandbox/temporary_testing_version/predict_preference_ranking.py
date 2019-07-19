import os
import sys
import time
import math
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import resnet as rn
import data_handler as dh
import argparse
import itertools
import pdb



# Constants
NUM_RESIDUAL_BLOCKS = 5
HEIGHT = 12
WIDTH = 12
DEPTH = 11
BATCH_SIZE_TRAIN = 96
BATCH_SIZE_VAL = BATCH_SIZE_TRAIN
BATCH_SIZE_TEST = BATCH_SIZE_TRAIN
TRAIN_STEPS = 200000
EPOCH_SIZE = 78600
DECAY_STEP_0 = 10000
DECAY_STEP_1 = 15000
NUM_CLASS = 4

ckpt_file = 'training_result/caches/cache_S002a_v6_commit_495618_epoch80000_tuning_batch96_train_step_40M_INIT_LR_10-51/train/model.ckpt-3999999'

dir_testing_maze_txt = '/Users/vimchiz/bitbucket_local/observer_model_group/benchmark/temporary_testing_version/data_for_making_preference_predictions'




tf.reset_default_graph()
predict_preference_ranking(ckpt_file, dir_testing_maze_txt)	    
