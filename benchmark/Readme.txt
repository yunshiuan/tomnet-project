#########################################
This folder contains details for each training version.
Data path:
'/S002a'
------------------------------------------

Note 
(1) bracket along the version indicates where the scripts are for the version.
E.g., the version with the bracket '[At benchmark/temporary]'
is run by the temporary scripts.
(2) '[At benchmark/temporary]' and '[At benchmark]' share the same version naming
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
Future training session (v14, commit ???)[At benchmark]
Time: 2019/??
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v14_commit_???_epoch8000_tuning_batch16_train_step_0.5M_INIT_LR_10-5

(1) Train the model with variable sequence lengths:
https://danijar.com/variable-sequence-lengths-in-tensorflow/

#########################################
Future training session (v13, commit 9ebcd4)[At benchmark]
Time: 2019/06/03
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v13_commit_9ebcd4_epoch8000_tuning_batch16_train_step_0.5M_INIT_LR_10-5


(1) Only use the final state from LSTM instead of using the whole sequence outputs.
Note:
(1)
#########################################
Finished training session (v12, commit 29151f)[At benchmark]
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
Finished training session (v11, commit ce0992)[At benchmark]
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
Finshed training session (v10, commit 5951c9)[At benchmark]
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
Finished training session (v9, commit 2a3d5e)[At benchmark]
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
Finished training session (v8, commit 478c9f)[At benchmark]
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

