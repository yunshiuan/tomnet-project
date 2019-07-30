#########################################
This folder contains details for each training version.
Data path:
'/S002a'
------------------------------------------

Note 
(1) bracket along the version indicates where the scripts are for the version.
E.g., the version with the bracket '[At sandbox/temporary]'
is run by the temporary scripts.
(2) '[At sandbox/temporary]' and '[At working_model]' share the same version naming
sequence.
---------------------

Directory description:
-test_on_human_data
This folder contains analysis on human data.
See README in it for detail.

-temporary_testing_version
This folder contains temporary testing files.
The codes root from Edwinn's codes with step-by-step modification.
See README in it for detail.

#########################################
Future training session
Time: 2019/06/03
Future training session (v?, commit ???)[At working_model]
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v?_commit_???_epoch8000_tuning_batch16_train_step_0.5M_INIT_LR_10-5

(1) Train the model with variable sequence lengths:
https://danijar.com/variable-sequence-lengths-in-tensorflow/
(2) Don't use dropout for preference inference.
(3) Reconsider the data format of 'data_preference_predictions'. Maybe I should try more combination?

########################################

Finished training session (v22, commit c293fa)[At working_model]
Time: 2019/07/23
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002b_v22_commit_c293fa
Info: file10000_tuning_batch16_train_step_10K_INIT_LR_10-4

(1) Follow v21 but with a new data set S020b (each maze has equal number of potential targets.)
(2) Resume weight decay back to 2 x 10-5.

Note:
(1) Overfitting is reduced but still exists.
(2)
accurary	mode
97.66%	train_proportion
77.62%	vali_proportion
80.85%	test_proportion

(3)
Traj_S002a_Query_S002a_subset96
avg_prediction_probability	ground_truth_label_count	prediction_count	accuracy_data_set
0.2	19	19	92.71
0.22	24	21	92.71
0.23	22	22	92.71
0.36	31	34	92.71

prediction_proportion	avg_prediction_probability	ground_truth_label_count	prediction_count
0	0.01	0	0
0.04	0.15	0	4
0	0.06	0	0
0.96	0.77	0	92

########################################

Finished training session (v21, commit 9f3e1a)[At working_model]
Time: 2019/07/23
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v21_commit_9f3e1a_file10000_tuning_batch16_train_step_10K_INIT_LR_10-4

(1) Follow v20 but with greater regularization term 
(weight decay = 1 x 10-3, increased from 2 x 10-5)
(2) Aims to tackle the overfitting issues.
Note:
(1) It is still overfitting.
(2)
accurary	mode
95.11%	train_proportion
63.41%	vali_proportion
61.49%	test_proportion
(3)
Traj_S002a_Query_S002a_subset96:
avg_prediction_probability	ground_truth_label_count	prediction_count	accuracy_data_set
0.13	10	12	85.42
0.3	28	28	85.42
0.25	26	25	85.42
0.32	32	31	85.42

Traj_S002a_Query_Stest_subset96:
prediction_proportion	avg_prediction_probability	ground_truth_label_count	prediction_count
0.5	0.3	0	48
0.44	0.35	0	42
0	0.07	0	0
0.06	0.28	0	6

########################################

Finished training session (v20, commit 207536)[At working_model]
Time: 2019/07/23
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v20_commit_207536_file10000_tuning_batch16_train_step_10K_INIT_LR_10-4

(1) Follow v19 but with less steps.
(2) Add the train_proportion metric.
Note:
(1) It is overfitting.
(2)
accurary	mode
95.12%	train_proportion
68.25%	vali_proportion
65.52%	test_proportion
(3)
Traj_S002a_Query_S002a_subset96
avg_prediction_probability	ground_truth_label_count	prediction_count	accuracy_data_set
0.1	10	8	92.71
0.28	28	27	92.71
0.28	26	27	92.71
0.35	32	34	92.71

Traj_S002a_1000_Query_Stest_subset96:
prediction_proportion	avg_prediction_probability	ground_truth_label_count	prediction_count
0.05	0.11	0	5
0	0.07	0	0
0.25	0.25	0	24
0.7	0.57	0	67

########################################

Finshed training session (v19, commit 36462b)[At working_model]
Time: 2019/07/23
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v19_commit_36462b_file10000_tuning_batch16_train_step_0.2M_INIT_LR_10-5

(1) Follow v17 (with prednet) but with more trainig steps.
Note:
(1) Seems to be overfitting.
(2)
accurary	mode
67.54% 	vali_proportion
64.72% 	test_proportion

########################################

Finished training session (v18, commit 0b7e45)[At working_model]
Time: 2019/07/23
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v18_commit_0b7e45_file1000_tuning_batch16_train_step_1K_INIT_LR_10-5

(1) Follow v17 but without prednet.
Note:
(1)
accurary	mode
84.38%	vali_proportion
82.29%	test_proportion

########################################

Finished training session (v17, commit 48fa87)[At working_model]
Time: 2019/07/23
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v17_commit_48fa87_file10000_tuning_batch16_train_step_10K_INIT_LR_10-5

(1) Follow v15 but with the bugs fixed.
Note:
(1)
accurary	mode
64.31% vali_proportion
62.7% test_proportion

########################################

Finished training session (v16, commit 926291)[At working_model]
Time: 2019/07/17
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v16_commit_926291_file1000_tuning_batch16_train_step_1K_INIT_LR_10-5

(1) Make sure the model also works without prenet.

Note:
(1) 
accurary	mode
89.58%	vali_proportion
90.62%	test_proportion


########################################
Finished training session (v15, commit acc400)[At working_model]
Time: 2019/07/17
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v15_commit_acc400_file1000_tuning_batch16_train_step_1K_INIT_LR_10-5

(1) Train with model with both char net and pred net.
(2) [WARNING!] The final assessment of the model is buggy. 
The buses were fixed in the commit 512909 and cb38ac.
Note:
(1) [Buggy!]
accurary	mode
96.98%	vali_proportion
97.48%	test_proportion
#########################################

Finished training session (v14, commit 95c693)[At working_model]
Time: 2019/07/17
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v14_commit_95c693_epoch8000_tuning_batch16_train_step_1K_INIT_LR_10-5

(1) Test if the OOP version of charnet works.
(2) Use only 1000 steps to speed up the process.

Note:
(1)
accurary	mode
99.7%	vali_proportion
99.7%	test_proportion
#########################################
Finished training session (v13, commit 9ebcd4)[At working_model]
Time: 2019/07/12
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v13_commit_9ebcd4_epoch80000_tuning_batch16_train_step_0.5M_INIT_LR_10-5


(1) Only use the final state from LSTM instead of using the whole sequence outputs.
Note:
(1)
accurary	mode
99.9%	vali_proportion
100.0%	test_proportion

#########################################
Finished training session (v12, commit 29151f)[At working_model]
Time: 2019/06/03
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v12_commit_29151f_epoch80000_tuning_batch16_train_step_0.5M_INIT_LR_10-5

(1) Following v11, use full size = 80000. If the machine could not handle,
consider refactor the data_handler() function.

Note:
(1) Perfect results!
(2)
accurary	mode
99.8%	vali_proportion
99.9%	test_proportion

#########################################
Finished training session (v11, commit ce0992)[At working_model]
Time: 2019/05/28
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v11_commit_ce0992_epoch8000_tuning_batch16_train_step_0.5M_INIT_LR_10-5

(1) Fix the seq_len in LSTM from 0 to the correct length (i.e.,10).

Note:
(1) Because our machine could not handle size = 80000, I first use size = 8000
to see if it will work.
(2)
accurary	mode
51.04%	vali_proportion
54.17%	test_proportion

#########################################
Finshed training session (v10, commit 5951c9)[At working_model]
Time: 2019/05/16
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v10_commit_5951c9_epoch80000_tuning_batch16_train_step_2M_INIT_LR_10-5

Add:
(1) Take LSTM result from all time steps and feed them into a linear layer 
    instead of taking the result from the final state alone. (like what Edwinn
    has done)
Note:
(1) It is not learning.    
vali: proportion_accuracy()
Matches: 305/992
Accuracy: 30.75%
test: proportion_accuracy()
Matches: 315/992
Accuracy: 31.75%

#########################################
Finished training session (v9, commit 2a3d5e)[At working_model]
Time: 2019/05/16
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v9_commit_2a3d5e_epoch80000_tuning_batch16_train_step_2M_INIT_LR_10-4


Modify:
(1) Increase LR to 10-4 (as in the paper).
(2) Do not use decay step (Adam itself is already a automatic decaying optimizer)

Note:
(1) Note learning.
(2)
vali: proportion_accuracy()
Matches: 305/992
Accuracy: 30.75%
test: proportion_accuracy()
Matches: 315/992
Accuracy: 31.75%
#########################################
Finished training session (v8, commit 478c9f)[At working_model]
Time: 2019/05/16
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v8_commit_478c9f_epoch80000_tuning_batch16_train_step_2M_INIT_LR_10-5


Modify:
(1) Add parent function (i) generate_batch() and (ii) evaluate_whole_data_set
(2) Adapt the test() function for data format (batch_size, trajectory_size, width, height,  depth)

Remove:
(1) Remove the match_estimation() metric. Deprecate the function.

Note:
(1) Not learning at all.
vali: proportion_accuracy()
Matches: 305/992
Accuracy: 30.75%
test: proportion_accuracy()
Matches: 315/992
Accuracy: 31.75%

