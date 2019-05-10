This folder contains temporary testing files.
The codes root from Edwinn's codes with step-by-step modification.

#########################################
Current training session (commit ???)
Time: 2019/05/10
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v3_commit_???_epoch80000_tuning_batch96_train_step_10M_INIT_LR_10-5

Modify:
(1) Remove regulizization for LSTM (keep_prob = 1.0)

#########################################
Finished training session (commit d42c3b)
Time: 2019/05/09
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v2_commit_d42c3b_epoch80000_tuning_batch96_train_step_10M_INIT_LR_10-5

Add:
(1) Add in regulizization for LSTM (keep_prob = 0.8)
(2) Print out testing performance to a csv (main_model_tuning.match_estimation())

#########################################
Finished training session (commit 31cb84)
Time: 2019/05/08
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v1_commit_31cb84_epoch80000_tuning_batch96_train_step_4000000_INIT_LR_0.000011
Note: epoch_size = 80000, batch_size = 96,LR = 10^-5, steps = 40M
Add:
(1) To test Edwinn's codes with the addition of a 3x3 conv layer before resnet.

