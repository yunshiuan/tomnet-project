from os import path
import sys
sys.path.append('/Users/vimchiz/github_local/observer_model/') 
import numpy as np
import test_data_handler as tdh



#File Constants
PATH_ROOT='/Users/vimchiz/bitbucket_local/observer_model_group/benchmark/Social Network Simulation Data (24 agents)'
FILE_TEST = path.join(PATH_ROOT,'S008_28552.txt')
#FILE_TEST = path.join(PATH_ROOT,'S001_18.txt')
#FILE_TEST = path.join(PATH_ROOT,'S012_16728.txt') #Exceed MAX_TRAJECTORY_SIZE

# =============================================================================
# #Test the parse_trajectory() function
# =============================================================================

data_handler = tdh.DataHandler(dir)
parsed_result = data_handler.parse_trajectory(FILE_TEST)
output = parsed_result[0]
agent_locations = parsed_result[2]
lines = parsed_result[3]
possible_actions = parsed_result[4]

#Check result
np_output = np.array(output)
np_agent_locations = np.array(agent_locations).astype(int)
np_lines = np.array(lines)
n_action = len(np_agent_locations)

#lines
print(*lines)
#print('Agent location: \n' + 
#      np.array2string(np_agent_locations))

for step in range(n_action):
    action_location_x = np_agent_locations[step][0] - 1
    action_location_y = np_agent_locations[step][1] - 1
    starting_step = np_output.shape[0] - n_action 
    action_array = np_output[starting_step + step,
                             action_location_y, #the order of y and x, see line 112
                             action_location_x,
                             [40,41,42,43,44]]
    action_index = np.where(action_array)[0][0]
    action_name = possible_actions[action_index]
    print('The step ' + str(step+1) + ' is ' + action_name)
    
# =============================================================================
# Not used: Dig into the source codes within the function parse_trajectory()
# =============================================================================
##Function constants
#MAZE_WIDTH = 12
#MAZE_HEIGHT = 12
#MAZE_DEPTH = 45
#MAX_TRAJECTORY_SIZE = 10
#
##Initilization
#steps = []
#output = np.zeros((MAZE_WIDTH, MAZE_HEIGHT, MAZE_DEPTH, MAX_TRAJECTORY_SIZE))
#label = ''
#filename = FILE_TEST
#fp = open(filename)
##with open(filename) as fp:
#
#lines = list(fp)
#maze = lines[2:14]
#
##Parse maze to 2d array, remove walls.
#i=0
#while i < 12:
#    maze[i]= list(maze[i])
#    maze[i].pop(0)
#    maze[i].pop(len(maze[i])-1)
#    maze[i].pop(len(maze[i])-1)
#    i+=1
#
##Original maze (without walls)
#np_maze = np.array(maze)
#
##Plane for obstacles
#np_obstacles = np.where(np_maze == '#', 1, 0).astype(np.int8)
#
##Plane for agent's initial position
#np_agent = np.where(np_maze == 'S', 1, 0).astype(np.int8)
#
##Planes for each possible goal
#targets = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m']
#np_targets = np.repeat(np_maze[:, :, np.newaxis], len(targets), axis=2)
#for target, i in zip(targets, range(len(targets))):
#    np_targets[:,:,i] = np.where(np_maze == target, 1, 0)
#np_targets = np_targets.astype(int)
#
##Parse trajectory into 2d array
#trajectory = lines[15:]
#agent_locations = []
#for i in trajectory:
#    i = i[1:len(i)-2]
#    tmp = i.split(",")
#    try:
#        agent_locations.append([tmp[0],tmp[1]])
#    except:pass
#possible_actions=['right', 'left', 'up', 'down', 'goal']
#for i in range(len(agent_locations) - 1):
#    
#    #Create a 12x12sx4 tensor for each t(x,a)
#    np_actions = np.zeros((12,12,len(possible_actions)), dtype=np.int8)
#    #Determine the type of action
#    if agent_locations[i][0] > agent_locations[i+1][0]:
#        layer = 'right'
#    elif agent_locations[i][0] < agent_locations[i+1][0]:
#        layer = 'left'
#    elif agent_locations[i][1] > agent_locations[i+1][1]:
#        layer = 'up'
#    elif agent_locations[i][1] < agent_locations[i+1][1]:
#        layer = 'down'                
#    #Assign a value of 1 to that location
#    np_actions[int(agent_locations[i][1])-1, int(agent_locations[i][0])-1, possible_actions.index(layer)] = 1
#    np_tensor = np.dstack((np_obstacles,np_targets,np_agent,np_actions))
#    steps.append(np_tensor)
#    output = np.array(steps)
#
##The last tensor of every series will be the position of the goal.
#np_actions = np.zeros((12,12,len(possible_actions)), dtype=np.int8)
#np_actions[int(agent_locations[-1][1])-1, int(agent_locations[-1][0])-1, possible_actions.index('goal')] = 1
#np_tensor = np.dstack((np_obstacles,np_targets,np_agent,np_actions))
#
##Make the label from the letter in the final position of the agent in one hot encoding style
#goal = np_maze[int(agent_locations[-1][1])-1][int(agent_locations[-1][0])-1]
#char_to_int = dict((c, i) for i, c in enumerate(targets))
#integer_encoded = char_to_int[goal]
##label = [0 for _ in range(len(targets))]
##label[integer_encoded] = 1
#
##Return label as a number
#label = int(integer_encoded)
#
##Put everything together in a 4-dim tensor            
#steps.append(np_tensor)
#output = np.array(steps)
#
#pad_size = int(MAX_TRAJECTORY_SIZE - output.shape[0])
#
##Zeroes pre-padding to max length
#if pad_size > 0:
#    np_pad = np.zeros((12,12,45), dtype=np.int8)
#    for i in range(pad_size):
#        output = np.insert(output, 0, np_pad, axis=0)
#
##Truncating trajectory to max length
#elif pad_size < 0:
#    for i in range(abs(pad_size)):
#        output = np.delete(output, 0, axis=0)
#fp.close()
#
##
## np_output[-3,5,5,[40,41,42,43,44]] #down
## np_output[-2,6,5,[40,41,42,43,44]] #left (Should be right!)
## np_output[-1,6,6,[40,41,42,43,44]] #goal
#    
#    