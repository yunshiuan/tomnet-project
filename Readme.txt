 ##################################################################

This is the file to record the discription of each file/directory.

##################################################################
#########################################
Current training session
Time:
Author:
Output file name:
Note:
#########################################

1. Main Codes for AI_Robohon training:

main_model.py
create_tfrecords.py
data_handler.py
resnet.py 

############################################

2. Training S002 (4 agents + one subject)

(1) S002: 10,000 trajectories in grid world for S002
(2) S002a: 1,000 trajectories in grid world for S002 
(3) cache_S002a_80000steps: Training S002a with 80000steps

To check why the error would sometimes go to 0.25 in cache_S002a_80000steps: I try reducing training steps

(4) cache_S002a_10000steps: Training S002a with 10000steps
(5) cache_S002a_40000steps: Training S002a with 40000steps

############################################

3. Training S003 (6 agents + one subject)

To check why the error would sometimes go to 0.25 in cache_S002a_80000steps: I try increase complexity

(1) cache_S003a: Training S003a with 80000steps 


############################################

4. Training S004 (15 agents + one subject)

To check why the error would sometimes go to 0.25 in cache_S002a_80000steps: I try increase complexity

(1) cache_S004a: Training S004a with 80000steps, but the output is weird 
    (the output on 20190222 is only 30000 and re-training from 27700 steps???)

(2) cache_S004a_2: Re-training S004a with 80000steps to see whether the output above is weird.











