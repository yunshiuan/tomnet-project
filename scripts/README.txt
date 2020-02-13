--------------------------
simulation_data_generator
--------------------------
the script that generate the simulation trajectories in 'tomnet-project\data\data_simulation'.

--------------------------
training_result_figures
--------------------------
the script that generated the training result plot, e.g., 'tomnet-project\models\working_model\test_on_simulation_data\training_result\figures'.

--------------------------
filter_full_target_trajectories
--------------------------
- feed in the processed trajectories, and only keep the mazes with all targets. E.g. for 'tomnet-project\data\data_human\processed\S030', there are 1~4 targets in different mazes. This script will filter in the mazed with exact 4 targets and output the results at 'tomnet-project\data\data_human\filtered'.
- This is to faciliate the process of preference prediction. It is impossible for a
model to infer the target by a qeury state if the trajectory the model saw
contained only a subset of the targets.