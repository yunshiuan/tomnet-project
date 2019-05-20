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
(4) Use FileZilla SFTP to log in and download files.illi

data path:
'/S030'

#########################################

Future training session (v1, commit ???) [At benchmark/human_data]
Time: 2019/05/17
Author: Chuang, Yun-Shiuan

(1) Add predict_ranking() function to the commented_data_handler.py for making predictions on target preference ranking.

=========================================
Current training session (v1, commit ???) [At benchmark/human_data]
Time: 2019/05/17
Author: Chuang, Yun-Shiuan

Description:
(1) Use the model architecture (v5, commit c8e358) [At benchmark/temporary]
(2) Use the commented_data_handler.py [At benchmark]. I modify the function parse_trajectory() to handle human data, which slightly differs from the simulated data in term of format (see above at 'Folder description').

Note:
#########################################

