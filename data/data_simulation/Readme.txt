Version:
S002:
- 38 potential targets in the maze.
S002a:
- 1 ~ 4 potential targets in the maze.
- This is the version that all results in the 'working_model' is trained on, because 
38 targets are just way too hard for the model to learn. Plus, 4 goals are in consistence with the ToMNET paper.
- Based on 'S002a_familyonly.csv'.

S002b:
- Like S002a but all mazed have exact 4 goals.
- This is to make the task easier for model to learn. It is impossible for a
model to infer the target by a qeury state if the trajectory the model saw
contained only 1 goal.
- Based on 'S002b_familyonly.csv'.
 
