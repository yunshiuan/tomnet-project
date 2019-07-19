 ##################################################################

This is the file to record the discription of each file/directory.

##################################################################

#########################################
Finished training session

Time: 20190507
Author: Chuang, Yun-Shiuan (commit 092117)
Output file name: output_S002a_epoch80000_tuning_batch96_train_step_40M_INIT_LR=0.00001
Note: try using epoch size 80000 (correspongs to 10000 files)


#########################################
Finished training session
Time: 20190417
Author: HsinYi Hung
Output file name: cache_S002a_epoch8000_tuning_batch96__train_step_100000_1 & cache_S002a_epoch8000_tuning_batch96__train_step_4000000_1 
Note: try training step=10000 and 4000000.


#########################################
Finished training session
Time: 20190415
Author: HsinYi Hung
Output file name: cache_S002a_epoch8000_train_step_10000_tuning_batch96_XX
Note: Run the model epoch8000_train_step_10000_tuning_batch96 9 more times to see whether the results are stable or not.



#########################################
Training session
Time: 20190412
Author: HsinYi Hung
Output file name: cache_S002a_epoch8000_train_step_10000_tuning_batchXXX 
Note: Find out epoch size named by Edwinn should be training set size. So change epoch to 8000.

#########################################
Training session
Time: 20190408
Author: HsinYi Hung
Output file name: cache_S002a_tuning_train_step_20000_batchXX
Note: Tune batch size with specific training step=20000.


#########################################
Training session
Time: 20190408
Author: HsinYi Hung
Output file name: cache_S002a_batchXX_tuning_train_step_XXX
Note: Tune training step with specific batch size.




#########################################
Training session: tuning batch size
Time: 20190402
Author: HsinYi Hung
Output file name: cache_S002a_tuning_batch_sizeXXX
Note: In order to write a script for looping the training, I revise the "resnet.py". 



#########################################
Training session
Time: 20190402
Author: HsinYi Hung
Output file name: cache_S002a_test_tuning
Note: Add a file main_model_tuning.py and test whether the ouput is same as main_model.py

#########################################

Training session: S002a with 10000 training files and 10000 training steps.
Time: 20190330
Author: HsinYi Hung
Output file name:cache_S002a_10000files
Note:
cache_S002a_10000steps: Training S002a with 10000files to see whether the performance would be better than 1000 training files.

It seems that Edwinn's code can only read the file directory like "S002a". If I change to "S002a_10000files", there would be an error in terms of the maze size..It's weird...So the training files should all put in SXXXa.



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

(6) cache_S002a_10000files: Training S002a with 10000files to see whether the performance would be better than 1000 training files. 
**Note** It seems that Edwinn's code can only read the file directory like "S002a". If I change to "S002a_10000files", there would be an error in terms of the maze size..It's weird...So the training files should all put in SXXXa. 


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











