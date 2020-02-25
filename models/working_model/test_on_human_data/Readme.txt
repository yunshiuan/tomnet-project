#########################################
Folder description:
(1) This folder contains analysis on human data.
(2) All python scripts are located in sandbox/temporary_testing_version or working_model.
(3) This folder only contains training results as well as a Rscript to convert the
format of human data.
Data source:
(1) Sign in to server: ssh gibmsadmin@140.112.122.160 (pw: See https://gitlab.com/brain-and-mind-lab/notes-for-bml/bmlab-wiki-home/wikis/server#gibms_server%E6%95%99%E5%AD%B8%E6%96%87%E4%BB%B6)
(2) Data path: '/var/www/html/bmlab/ai-robo/game/data'
(3) Note that the txt file is very similar to the simulated txt, except for
	- This is handled by '/scripts/convert_human_data_format/convert_human_data_format.R' to ensure the format of human data to be the same as the simulated data. See README there for details.
(4) Use FileZilla SFTP to log in and download files

Note:
To derive the scores that participants see:
(1) http://gibms.mc.ntu.edu.tw/phpmyadmin
(2) See this for account name and password: https://gitlab.com/brain-and-mind-lab/notes-for-bml/bmlab-wiki-home/wikis/server#gibms_server%E6%95%99%E5%AD%B8%E6%96%87%E4%BB%B6
(3) bmlab/ai_social_game
(4) Note that the scores that participants see = round(averaged score * (23/9)). The column 'agent1_value' shown is rounded, but the original value to derive the score is not rounded. Rounding occurs after *(23/9).
data path:
'\data\data_human'

#########################################
Finished training session (v11, commit c2f0ff) 
Time: 2020/02/19
Author: Chuang, Yun-Shiuan
Output file name: 
/v11

(1) Same as human/v10 except that 
	(1)the 'augmented data' were used for training and testing.
	(2) Batch size changes back to 16 from 10 because all subjects now have enough data.
	(3) S030 is also included.

Note
(1) All models have higher accuracy with the augmented data.
(2)  
S030:
accurary	mode
82.54%	train_proportion
79.33%	vali_proportion
79.45%	test_proportion

(3) 
S024:
ground_truth_label_count	prediction_count	accuracy_data_set


S024_Stest_subset96:
prediction_proportion	avg_prediction_probability	ground_truth_label_count	prediction_count
#########################################
Finished training session (v10, commit 35e29c) 
Time: 2020/02/11
Author: Chuang, Yun-Shiuan
Output file name: 
/v10

(1) Same as human/v9 except that the following subjects are used for training as well, ['S024', 'S033', 'S035', 'S050', 'S051', 'S052'].
(2) Batch size changes to 10 from 16 because some subject only has about 100 files. Batch size = 16 only works for number of files greater than 160 since the validation and the test set only have 1/10 of the total files. Having 160 files ensures there is at least one batch for dev/test set.

Note
(1) All models trained by each set of the new human data is overfitting.
(2)  The only model with above-random accuracy: S024 (920 trajs). Other subjects have too few trajectories (just above 100).
accurary	mode
99.86%	train_proportion
52.22%	vali_proportion
42.22%	test_proportion

(3) 
S024:
ground_truth_label_count	prediction_count	accuracy_data_set


S024_Stest_subset96:
prediction_proportion	avg_prediction_probability	ground_truth_label_count	prediction_count

-----------------------------------------

Finished training session (v9, commit 78092b) 
Time: 2019/07/25
Author: Chuang, Yun-Shiuan
Output file name: /cache_S030_v9_commit_78092b_file9830_tuning_batch16_train_step_10K_INIT_LR_10-4

(1) Use the model as in 'working_model (simulation, v20').
Note
(1) A bit overfitting.
(2)
accurary	mode
97.51%	train_proportion
75.0%	vali_proportion
73.26%	test_proportion
(3) 
S030_subset1000:
ground_truth_label_count	prediction_count	accuracy_data_set
393	407	92.24
189	188	92.24
221	206	92.24
189	191	92.24

Traj_S030_Query_Stest_subset96:
prediction_proportion	avg_prediction_probability	ground_truth_label_count	prediction_count
0.71	0.59	0	68
0.01	0.08	0	1
0.28	0.29	0	27
0	0.04	0	0
-----------------------------------------
Finished training session (v8, commit 0c7df5) [At working_model/human_data]
Time: 2019/06/03
Author: Chuang, Yun-Shiuan
Output file name: /cache_S030_v8_commit_0c7df5_epoch78600_tuning_batch16_train_step_0.5M_INIT_LR_10-5

(1) Use the scripts in 'working_model' (v12).

Note
(1)
accurary	mode
99.9%	vali_proportion
99.69%	test_proportion

-----------------------------------------

Finished training session (v7, commit 0050d9) [At working_model/human_data]
Time: 2019/05/22
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v7_commit_0050d9_epoch78600_tuning_batch16_train_step_0.5M_INIT_LR_10-4

(1) Remove the scripts in this folder.
(2) Use the scripts in 'sandbox/temporary' and set the output to here. 
This is because the human data format is identical to 
the simulated data (thanks to the R script), 
so there is no need to separate two set of scripts.

Note
(1)
51.52%	vali_proportion
51.32%	vali_match_estimation
48.46%	test_proportion
48.3%	test_match_estimation

-----------------------------------------

Finished training session (v6, commit f912f8) [At working_model/human_data]
Time: 2019/05/27
Author: Chuang, Yun-Shiuan
Output file name: /cache_S030_v6_commit_f912f8_epoch78600_tuning_batch96_train_step_2M_INIT_LR_10-5

(1) Fix LSTM layer.
(2) Use the same set up as v3 with 2M steps and epoch = 78600.

Note:
(1) Is is still NOT learning! It should not only because of the LSTM issue.
There should be something else.
(2)
40.46%	vali_proportion
40.46%	vali_match_estimation
38.01%	test_proportion
38.01%	test_match_estimation

-----------------------------------------
Finished training session (v5, commit a207fa) [At working_model/human_data]
Time: 2019/05/23
Author: Chuang, Yun-Shiuan
Output file name: /cache_S030_v5_commit_a207fa_epoch78600_tuning_batch96_train_step_40M_INIT_LR_10-5

(1) Use the same set up as v3 with 40M steps
Note:
(1) Not learning.
(2)
accurary	mode
40.46%	vali_proportion
40.46%	vali_match_estimation
38.01%	test_proportion
38.01%	test_match_estimation

-----------------------------------------
Current training session (v4, commit cecb7a) [At working_model/human_data]
Time: 2019/05/23
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v4_commit_cecb7a_epoch78600_tuning_batch96_train_step_2M_INIT_LR_10-5

(1) Use the same set up as v3 but use the data S002a (with epoch = 78600)

-----------------------------------------
Finished training session (v3, commit dd21c9) [At working_model/human_data]
Time: 2019/05/22
Author: Chuang, Yun-Shiuan
Output file name: /cache_S030_v3_commit_dd21c9_epoch78600_tuning_batch96_train_step_2M_INIT_LR_10-5

(1) Copy codes again directly from sandbox/temporary (v7, commit c8e358).
(2) Preprocess the txt files by R and convert them to the format of simulated data,
 so no need to adjust the codes at all for human data.

Note:
(1) The validation error by batch is not going down but final validation performace is
not bad though.
(2) Performace:
vali: match_estimation()
Accuracy: 40.46%
vali: proportion_accuracy()
Accuracy: 40.46%
test: match_estimation()
Accuracy: 38.01%
test: proportion_accuracy()
Accuracy: 38.01%
-----------------------------------------
Finished training session (v2, commit 4a00dc) [At working_model/human_data]
Time: 2019/05/22
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v2_commit_4a00dc_epoch80000_tuning_batch96_train_step_2M_INIT_LR_10-5

(1) Test the functions (v1) on simulated data and check whether the codes are working.
Note:
(1) Not learning at all. Terminated manully.
-----------------------------------------

Finished training session (v1, commit 95ab2d) [At working_model/human_data]
Time: 2019/05/21
Author: Chuang, Yun-Shiuan
Output file name: /cache_S030_v1_commit_95ab2d_epoch8000_tuning_batch96_train_step_40M_INIT_LR_10-5

Description:
(1) Use the model architecture (v7, commit c8e358) [At sandbox/temporary]
(2) Use the commented_data_handler.py (v10, commit 5951c9) [At working_model]. I modify the function parse_trajectory() to handle human data, which slightly differs from the simulated data in term of format (see above at 'Folder description').

Note:
(1) Not learning at all. Terminated manully.
(2) Mistakenly set epoch size = 8000.
#########################################

