##################################################################
This folder contains temporary testing files.
The codes root from Edwinn's codes with step-by-step modification.
Data path:
'/../S002a'
##################################################################

Future training session (v14, commit ???)[At benchmark/temporary]
Time: 2019/05/27
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v14_commit_???_epoch78600_tuning_batch96_train_step_2M_INIT_LR_10-5

(1) Try to identify the differences other than the LSTM issue.
(2) See if setting epoch size to 80000 would make a difference.

Note:

#########################################

Current training session (v13, commit 62cae2)[At benchmark/temporary]
Time: 2019/05/27
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v13_commit_62cae2_epoch78600_tuning_batch96_train_step_2M_INIT_LR_10-5

(1) Resume to v8 , with the LSTM issue fixed.

Note:
(1) Is is still NOT learning! It should not only because of the LSTM issue.
There should be something else.
(2) 
30.79%	vali_proportion
30.79%	vali_match_estimation
31.89%	test_proportion
31.89%	test_match_estimation


#########################################

Finished training session (v12, commit 62c0e8[At benchmark/temporary]
Time: 2019/05/27
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v12_commit_62c0e8_epoch80000_tuning_batch96_train_step_2M_INIT_LR_10-5

(1) Try an older version of v7 and see if it works. If not, then I should probably
refresh the S002a data.

Note:
(1) It actually works! WOW! What's happened between the newer v7 (as in v11)
and older v7 (as in v12) ?
-> I found the reason! It is because v8~v11 is based on v4 version 
(commit 71297d, where the final state is extracted from LSTM
 instead of the real v7)

(2)
1.
45.23%	vali_proportion
45.27%	vali_match_estimation
BUG	test_proportion
30.54%	test_match_estimation
2.
52.07%	vali_proportion
52.0%	vali_match_estimation
BUG	test_proportion
30.12%	test_match_estimation
3.
accurary	mode
46.8%	vali_proportion
46.9%	vali_match_estimation
BUG	test_proportion
28.81%	test_match_estimation

#########################################
Finished training session (v11, commit d2650a[At benchmark/temporary]
Time: 2019/05/10
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v10_commit_0b68f9_epoch80000_tuning_batch96_train_step_2M_INIT_LR_10-5

(1) Use the identical version as v7. See what has happened with v8, v9, v10.
(2) Try 4 times and see if it is because of the random intialization.

Note:
(1) It is not learing. It is not because of the random initialization.
Try an older version of v7!.
(2)
1:
accurary	mode
30.79%	vali_proportion
30.79%	vali_match_estimation
31.89%	test_proportion
31.89%	test_match_estimation

2:
accurary	mode
30.79%	vali_proportion
30.79%	vali_match_estimation
31.89%	test_proportion
31.89%	test_match_estimation

3:
accurary	mode
30.79%	vali_proportion
30.79%	vali_match_estimation
31.89%	test_proportion
31.89%	test_match_estimation

4:
accurary	mode
30.79%	vali_proportion
30.79%	vali_match_estimation
31.89%	test_proportion
31.89%	test_match_estimation

#########################################
Finished training session (v10, commit 0b68f9[At benchmark/temporary]
Time: 2019/05/10
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v10_commit_0b68f9_epoch80000_tuning_batch96_train_step_2M_INIT_LR_10-5

(1) Follow the set up of v8.

Modify:
(1) Use epoch size 80000 to see if the different is a result of the epoch size.

Note:
(1) It is not learing... WHY!!??? 
(2)
accurary	mode
30.79%	vali_proportion
30.79%	vali_match_estimation
31.89%	test_proportion
31.89%	test_match_estimation

#########################################
Finished training session (v9, commit f1e33a)[At benchmark/temporary]
Time: 2019/05/10
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v9_commit_f1e33a_epoch78600_tuning_batch96_train_step_2M_INIT_LR_10-5

(1) Follow the set up of v7.
Modify:
(1) Set offset of validation batch as 0 instead of a random number in order
to stabilize the result across trials.

Note:
(1) It is not learing... WHY!!???
(2)
accurary	mode
30.79%	vali_proportion
30.79%	vali_match_estimation
31.89%	test_proportion
31.89%	test_match_estimation


#########################################
Finished training session (v8, commit 7d8cff)[At benchmark/temporary]
Time: 2019/05/10
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v8_commit_7d8cff_epoch78600_tuning_batch96_train_step_2M_INIT_LR_10-5

Modify:
(1) Set everything same as v7, but use epoch size = 78600 instead of 80000.

Note:
(1) It is not learing... WHY!!???
(2) 
accurary	mode
30.79%	vali_proportion
30.79%	vali_match_estimation
31.89%	test_proportion
31.89%	test_match_estimation


#########################################
Finished training session (v7, commit 6f14c6)[At benchmark/temporary]
Time: 2019/05/10
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v7_commit_6f14c6_epoch80000_tuning_batch96_train_step_2M_INIT_LR_10-5

Add:
(1) Add a full validation set performance metric (instead of validation batch performance)
(2) Add another version of test accuracy (proportion accuracy). This version of accuracy is defined by proportion.

Note:
(1) It is learning.
(2) 
vali: match_estimation()
Accuracy: 43.15%
vali: proportion_accuracy()
Accuracy: 43.31%
test: match_estimation()
Accuracy: BUG QAQ
test: proportion_accuracy()
Accuracy: 45.22%

#########################################
Current training session (v6, commit 495618) [At benchmark/temporary]
Time: 2019/05/10
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v6_commit_495618_epoch80000_tuning_batch96_train_step_40M_INIT_LR_10-5

Modify:
(1) Use 40M steps, so as to compare with v1 fairly

Note:
(1) Learning great.
(2) Test accuracy = 59.81%

#########################################

Finished training session (v5, commit c8e358) [At benchmark/temporary]
Time: 2019/05/10
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v5_commit_c8e358_epoch80000_tuning_batch96_train_step_2M_INIT_LR_10-5

Add:
(1) Resume to v2, with keep_prob = 0.8
(2) Test accuracy = 43.31%

Note:
(1) Not sure how well the performance is. Should try more steps.
-> According to v6 results, it performes well.
#########################################
Finished training session (v4, commit 71297d) [At benchmark/temporary]
Time: 2019/05/10
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v4_commit_71297d_epoch80000_tuning_batch96_train_step_10M_INIT_LR_10-5

Add:
(1) Take LSTM result from the final state instead of results from all time steps

Note: 
(1) Bad result, not learning at all.
(2)
accurary	mode
30.79%	vali_proportion
30.79%	vali_match_estimation
31.89%	test_proportion
31.89%	test_match_estimation

#########################################
Finished training session (v3, commit 0f83ad) [At benchmark/temporary]
Time: 2019/05/10
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v3_commit_0f83ad_epoch80000_tuning_batch96_train_step_10M_INIT_LR_10-5

Modify:
(1) (modified) Remove regulizization for LSTM (keep_prob = 1.0)

Note:
(1) Bad result, but validation error is not going down.
#########################################
Finished training session (v2, commit d42c3b) [At benchmark/temporary]
Time: 2019/05/09
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v2_commit_d42c3b_epoch80000_tuning_batch96_train_step_10M_INIT_LR_10-5

Add:
(1) (modified) Add in regulizization for LSTM (keep_prob = 0.8)
(2) Print out testing performance to a csv (main_model_tuning.match_estimation())

Note:
(1) Learning great.

#########################################
Finished training session (v1, commit 31cb84) [At benchmark/temporary]
Time: 2019/05/08
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v1_commit_31cb84_epoch80000_tuning_batch96_train_step_4000000_INIT_LR_0.000011
Note: epoch_size = 80000, batch_size = 96,LR = 10^-5, steps = 40M
Add:
(1) Add a 3x3 conv layer before resnet to scale up the number of filters to 32.

Note:
(1) Learning great.









