#########################################
Folder description:
This folder contains analysis on human data.

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

data path:
'/S030'

#########################################

-----------------------------------------

Future training session (v?, commit ???) [At benchmark/human_data]
Time: 2019/05/22
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v?_commit_???_epoch80000_tuning_batch16_train_step_2M_INIT_LR_10-5

(1) Add predict_ranking() function to the commented_data_handler.py for making predictions on target preference ranking.
-----------------------------------------
Current training session (v5, commit ???) [At benchmark/human_data]
Time: 2019/05/23
Author: Chuang, Yun-Shiuan
Output file name: /cache_S002a_v5_commit_???_epoch78600_tuning_batch96_train_step_40M_INIT_LR_10-5

(1) Use the same set up as v3 with 40M steps

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

