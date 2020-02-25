--------------------------
simulation_data_generator
--------------------------
the script that generate the simulation trajectories in 'tomnet-project\data\data_simulation'.

--------------------------
training_result_figures
--------------------------
the script that generated the training result plot, e.g., 'tomnet-project\models\working_model\test_on_simulation_data\training_result\figures'.

--------------------------
convert_human_data_format
--------------------------
Convert human raw data to processed data that are ready for training. The processed human data match the format of simulated data.
- Note that the txt file is very similar to the simulated txt, except for
	(i) there are commas at the start of each line
	(ii) there is no 'S'. So should take the first position as the position of 'S'.
	(iii) there is no 'Maze:' at the first line
	(iv) there is 'unmoved' step which should be ignored
	(v) it is A, B, C, D instead of C, D, E, F.
	(vi) do not process if the starting point and the ending point is the same
	     (don't put it to the processed data dir)
  (vii) delete consecutive files that are from identical maze (e.g., S053_980.txt & S053_981.txt). Keep the first file and delete the rest of the duplicated files. This seems to only be an issue for those who play with computer.
	(viii) handle the case where the player has already reached a target but continued to move afterwards. Truncate the trajectory to the step where the player reached the first target. This seems to only be an issue when issue vii happens.
	
--------------------------
filter_full_target_trajectories
--------------------------
- feed in the processed trajectories, and only keep the mazes with all targets. E.g. for 'tomnet-project\data\data_human\processed\S030', there are 1~4 targets in different mazes. This script will filter in the mazed with exact 4 targets and output the results at 'tomnet-project\data\data_human\filtered'.
- This is to faciliate the process of preference prediction. It is impossible for a
model to infer the target by a qeury state if the trajectory the model saw
contained only a subset of the targets.
- input:
	- simulated data: No need for now because we could generate data that all have exactly 4 targets, e.g., S002b, S003b
	- human data: 'processed'
- output:
	- simulated data: No need for now
	- human data: 'filtered'
--------------------------
augment_traj_data
--------------------------
- augment the trajectory data by reflection and rotation of the x and y axis. Each maze results in 8 agumented data (4 rotation x 2 reflection)
- the augmented data is based on the processed data format which is good for training.
- the augmented data is the input for the training. The purpose is to increase the training set size for human data.
- input:
	- simulated data: No need for now
	- human data: 'processed'
- output:
	- simulated data: No need for now
	- human data: 'augmented'
	
--------------------------
count_playing_time
--------------------------	
- count the total playing time based on the modifed time of the raw data.
# - Total playing time = sum of the playing duration of each file.
# - File are separated into fragments if the interval of modified time 
# between two consecutive files is greater than a certain threshold (e.g., 5 mins).
# - The modified time of the file is the end time of each file (the completion of the trajectory)
# - The duration of the first file of each segment is uncomputable because only the 
# end time of each file is recorded.
# - Use imputation (filling by grand mean of duration) to fill in the duration of 
# first file of each segment
