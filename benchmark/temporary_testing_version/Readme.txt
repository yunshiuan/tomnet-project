This folder contains temporary testing files.
The codes root from Edwinn's codes with step-by-step modification.

#########################################
Current training session (v6, commit ???)
Time: 2019/05/10
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v6_commit_???_epoch80000_tuning_batch96_train_step_40M_INIT_LR_10-5

Modify:
(1) Use 40M steps, so as to compare with v1 fairly

Note:

#########################################

Finished training session (v5, commit c8e358)
Time: 2019/05/10
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v5_commit_c8e358_epoch80000_tuning_batch96_train_step_2M_INIT_LR_10-5

Add:
(1) Resume to v2, with keep_prob = 0.8

Note:
(1) Not sure how well the performance is. Should try more steps.
#########################################
Finished training session (f4, commit 71297d)
Time: 2019/05/10
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v4_commit_71297d_epoch80000_tuning_batch96_train_step_10M_INIT_LR_10-5

Add:
(1) Take LSTM result from the final state instead of results from all time steps

Note: 
(1) Bad result, not learning at all.
#########################################
Finished training session (v3, commit 0f83ad)
Time: 2019/05/10
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v3_commit_0f83ad_epoch80000_tuning_batch96_train_step_10M_INIT_LR_10-5

Modify:
(1) (modified) Remove regulizization for LSTM (keep_prob = 1.0)

Note:
(1) Bad result, but validation error is not going down.
#########################################
Finished training session (v2, commit d42c3b)
Time: 2019/05/09
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v2_commit_d42c3b_epoch80000_tuning_batch96_train_step_10M_INIT_LR_10-5

Add:
(1) (modified) Add in regulizization for LSTM (keep_prob = 0.8)
(2) Print out testing performance to a csv (main_model_tuning.match_estimation())

Note:
(1) Learning great.

#########################################
Finished training session (v1, commit 31cb84)
Time: 2019/05/08
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v1_commit_31cb84_epoch80000_tuning_batch96_train_step_4000000_INIT_LR_0.000011
Note: epoch_size = 80000, batch_size = 96,LR = 10^-5, steps = 40M
Add:
(1) Add a 3x3 conv layer before resnet to scale up the number of filters to 32.

Note:
(1) Learning great.
