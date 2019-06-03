#########################################
Folder description:
(1) This folder contains analysis on human data.
(2) All python scripts are located in benchmark/temporary_testing_version or benchmark.
(3) This folder only contains training results as well as a Rscript to convert the
format of human data.
Data source:
(1) Sign in to server: ssh gibmsadmin@140.112.122.160 (pw: See https://gitlab.com/brain-and-mind-lab/notes-for-bml/bmlab-wiki-home/wikis/server#gibms_server%E6%95%99%E5%AD%B8%E6%96%87%E4%BB%B6)
(2) Data path: '/var/www/html/bmlab/ai-robo/game/data'
(3) Note that the txt file is very similar to the simulated txt, except for
	(i) there are commas at the start of each line
	(ii) there is no 'S'. So should take the first position as the position of 'S'.
	(iii) there is no 'Maze:' at the first line
	(iv) there is 'unmoved' step which should be ignored
	(v) it is A, B, C, D instead of C, D, E, F.
	(vi) do not process if the starting point and the ending point is the same
	     (don't put it to the processed data dir)

(4) Use FileZilla SFTP to log in and download files

Note:
To derive the scores that participants see:
(1) http://gibms.mc.ntu.edu.tw/phpmyadmin
(2) See this for account name and password: https://gitlab.com/brain-and-mind-lab/notes-for-bml/bmlab-wiki-home/wikis/server#gibms_server%E6%95%99%E5%AD%B8%E6%96%87%E4%BB%B6
(3) bmlab/ai_social_game
(4) Note that the scores that participants see = round(averaged score * (23/9)). The column 'agent1_value' shown is rounded, but the original value to derive the score is not rounded. Rounding occurs after *(23/9).
data path:
'/S030'

#########################################

-----------------------------------------

Future training session (v7, commit 0050d9) [At benchmark/human_data]
Time: 2019/05/22
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v7_commit_0050d9_epoch78600_tuning_batch16_train_step_0.5M_INIT_LR_10-5

(1) Add predict_ranking() function to the commented_data_handler.py for making predictions on target preference ranking.
(2) Remove the scripts in this folder. Use the scripts in 'temporary'
and set the output to here. This is because the human data format is identical to 
the simulated data (thanks to the R script), so there is no need to separate two set of scripts.

Note
(1)
51.52%	vali_proportion
51.32%	vali_match_estimation
48.46%	test_proportion
48.3%	test_match_estimation

-----------------------------------------

Current training session (v6, commit f912f8) [At benchmark/human_data]
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
Finished training session (v5, commit a207fa) [At benchmark/human_data]
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
Current training session (v4, commit cecb7a) [At benchmark/human_data]
Time: 2019/05/23
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v4_commit_cecb7a_epoch78600_tuning_batch96_train_step_2M_INIT_LR_10-5

(1) Use the same set up as v3 but use the data S002a (with epoch = 78600)

-----------------------------------------
Finished training session (v3, commit dd21c9) [At benchmark/human_data]
Time: 2019/05/22
Author: Chuang, Yun-Shiuan
Output file name: /cache_S030_v3_commit_dd21c9_epoch78600_tuning_batch96_train_step_2M_INIT_LR_10-5

(1) Copy codes again directly from benchmark/temporary (v7, commit c8e358).
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
Finished training session (v2, commit 4a00dc) [At benchmark/human_data]
Time: 2019/05/22
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v2_commit_4a00dc_epoch80000_tuning_batch96_train_step_2M_INIT_LR_10-5

(1) Test the functions (v1) on simulated data and check whether the codes are working.
Note:
(1) Not learning at all. Terminated manully.
-----------------------------------------

Finished training session (v1, commit 95ab2d) [At benchmark/human_data]
Time: 2019/05/21
Author: Chuang, Yun-Shiuan
Output file name: /cache_S030_v1_commit_95ab2d_epoch8000_tuning_batch96_train_step_40M_INIT_LR_10-5

Description:
(1) Use the model architecture (v7, commit c8e358) [At benchmark/temporary]
(2) Use the commented_data_handler.py (v10, commit 5951c9) [At benchmark]. I modify the function parse_trajectory() to handle human data, which slightly differs from the simulated data in term of format (see above at 'Folder description').

Note:
(1) Not learning at all. Terminated manully.
(2) Mistakenly set epoch size = 8000.
#########################################

