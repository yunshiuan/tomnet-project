#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is for generating simulation data.

@author: Hsinyi; Chuang, Yun-Shiuang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Dependency imports

#from ai_safety_gridworlds.environments.shared import safety_game

import numpy as np
import csv
import random 
import pdb
import os
import pandas as pd

from queue import *

# pdb.set_trace()
# --------------------------------------------------------------      
# Constant block
# --------------------------------------------------------------      
# Read suject's network
# - param
FNAME = 'S002_test.csv'
RANDOM_NUM_GOALS = False # If true, the number of goals will vary across mazes
VERSION_NAME = 'S002a_4goals'
TARGET_ORDER = np.array(['C','D','E','F'])
# - dir
DIR_ROOT = os.getcwd()
DIR_TXT_OUTPUT = os.path.join(DIR_ROOT, '..','data','data_simulation',\
                              VERSION_NAME)
if not os.path.exists(DIR_TXT_OUTPUT):
  os.makedirs(DIR_TXT_OUTPUT)

# - file
FILE_CSV_SUMMARY = os.path.join(DIR_TXT_OUTPUT, 'summary_simulation.csv')
#FNAME='/bml/Data/Bank5/AI/AI_simulation/S002_familyonly.csv'
#FNAME='/bml/Data/Bank5/AI/ai-safety-gridworlds-master/Robohon/Subj_social_network_paramerter_simulation/S001.csv'
  
# ---------------------------------------------
# A data frame to collect parameters maze by maze
# ---------------------------------------------
df_collect_final_target_predictions = pd.DataFrame()

with open(FNAME, encoding='utf-8') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    agents_list=list(readCSV)

simulation_time=0
simulation_total = 100
while simulation_time <= simulation_total:
    # pdb.set_trace()
    # Number of agents in the environment 
    number_agents=len(agents_list) - 2 #exclude the header and the acting agent
    if RANDOM_NUM_GOALS:
      if number_agents<=11:
          n_chosen_agents = random.choice(list(range(1,number_agents+1)))
      else :
          pdb.set_trace()
          n_chosen_agents = random.choice(list(range(1,10))) # do not choose more than 9 goals
    else:
      n_chosen_agents = number_agents
              
    #Grid world environment


    GAME_ART = [
    ['##############',
     '#            #',
     '#            #',
     '#            #',  # Environment.
     '#            #',
     '#            #',
     '#            #',
     '#            #',
     '#            #',
     '#            #',
     '#            #',
     '#            #',
     '#            #',
     '##############'],
     ]

    Subj_CHR = 'S'
    Chosen_agents_label=['A','B','C','D', 'E', 'F']
    #Chosen_agents_label=['A','B','C','D','E','F','G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
      
    #pdb.set_trace()
    c = list(range(0+2, number_agents+2)) 
    Chosen_agents_index=random.sample(c, n_chosen_agents)
    Chosen_agents_index=np.asarray(Chosen_agents_index)
    Chosen_agents=[]
    
    for i in range(n_chosen_agents):
        Chosen_agents.append(agents_list[Chosen_agents_index[i]])
    

    Distance=np.zeros((n_chosen_agents, 2))

    y_s=random.choice(range(1,13))
    x_s=random.choice(range(1,13))
    GAME_ART[0][y_s]=GAME_ART[0][y_s][0:x_s]+'S'+GAME_ART[0][y_s][x_s+1:]

    n_barrier=np.random.choice(range(1,50))


    x_random=random.sample(range(1,13), (n_chosen_agents))
    y_random=random.sample(range(1,13), (n_chosen_agents))

    for i in range(n_chosen_agents):
        while (x_random[i]==x_s) and (y_random[i]==y_s):
            x_random[i]=random.choice(range(1,13))
            y_random[i]=random.choice(range(1,13))
            

    for i in range(n_chosen_agents):
        GAME_ART[0][y_random[i]]=GAME_ART[0][y_random[i]][0:x_random[i]]+Chosen_agents_label[Chosen_agents_index[i]]+GAME_ART[0][y_random[i]][x_random[i]+1:]


    Barrier=[]

    for i in range(n_barrier):
        x_barrier=random.choice(range(1,13))
        y_barrier=random.choice(range(1,13))
        Barrier.append(tuple((x_barrier, y_barrier)))
        if GAME_ART[0][y_barrier][x_barrier]==' ':
            GAME_ART[0][y_barrier]=GAME_ART[0][y_barrier][0:x_barrier]+'#'+GAME_ART[0][y_barrier][x_barrier+1:]

    for i in range(14):
        Barrier.append(tuple((i,0)))
        Barrier.append(tuple((i,13)))
    
    for i in range(14):
        Barrier.append(tuple((0,i)))
        Barrier.append(tuple((13,i)))

    AGENT_S_CHR = 'S'
    GOAL_CHR1='A'
    GOAL_CHR2='B'
    GOAL_CHR3='C'
    GOAL_CHR4='D'
    GOAL_CHR5='E'
    GOAL_CHR6='F'
    GOAL_CHR7='G'
    GOAL_CHR8='H'
    GOAL_CHR9='I'
    GOAL_CHR10='J'



    GAME_BG_COLOURS = {
            AGENT_S_CHR: (999, 0, 0),
            GOAL_CHR1: (431, 274, 823),
            GOAL_CHR2: (431, 274, 823),
            GOAL_CHR3: (431, 274, 823),
            GOAL_CHR4: (431, 274, 823),
            GOAL_CHR5: (431, 274, 823),
            GOAL_CHR6: (431, 274, 823),
            GOAL_CHR7: (431, 274, 823),
            GOAL_CHR8: (431, 274, 823),
            GOAL_CHR9: (431, 274, 823),
            GOAL_CHR10: (431, 274, 823),
            }
   # GAME_BG_COLOURS.update(safety_game.GAME_BG_COLOURS)

    #GAME_FG_COLOURS = dict.fromkeys(list(GAME_BG_COLOURS.keys()), (0, 0, 0))
    #GAME_FG_COLOURS.update(safety_game.GAME_FG_COLOURS)

    # pdb.set_trace()
    Subj_parameter=agents_list[1]

    D_eff=np.zeros(n_chosen_agents)
    for i in range (n_chosen_agents):
        personality_weight=np.sqrt(((int(Chosen_agents[i][6])-int(Subj_parameter[6]))/45)**2+((int(Chosen_agents[i][7])-int(Subj_parameter[7]))/50)**2+((int(Chosen_agents[i][8])-int(Subj_parameter[8])/50)**2))
        association=np.sqrt(((int(Chosen_agents[i][3]))/26)**2+((int(Chosen_agents[i][4]))/5)**2+((int(Chosen_agents[i][5]))/3)**2)
        D_eff[i]= association/personality_weight


    D_eff=D_eff/sum(D_eff)





    class Field(object):

        def __init__(self, len, start, finish, barriers):
            self._len = len
            self.__start = start
            self.finish = finish
            self._barriers = barriers
            self.__field = None
            self._build()

        def __call__(self, *args, **kwargs):
            self._show()

        def __getitem__(self, item):
            return self.__field[item]

        def _build(self):
            self.__field = [[0 for i in range(self._len)] for i in range(self._len)]
            for b in self._barriers:
                self[b[0]][b[1]] = -1
            self[self.__start[0]][self.__start[1]] = 1

        def emit(self):
            q = Queue()
            q.put(self.__start)
            while not q.empty():
                index = q.get()
                l = (index[0]-1, index[1])
                r = (index[0]+1, index[1])
                u = (index[0], index[1]-1)
                d = (index[0], index[1]+1)

                if l[0] >= 0 and self[l[0]][l[1]] == 0:
                    self[l[0]][l[1]] += self[index[0]][index[1]] + 1
                    q.put(l)
                if r[0] < self._len and self[r[0]][r[1]] == 0:
                    self[r[0]][r[1]] += self[index[0]][index[1]] + 1
                    q.put(r)
                if u[1] >= 0 and self[u[0]][u[1]] == 0:
                    self[u[0]][u[1]] += self[index[0]][index[1]] + 1
                    q.put(u)
                if d[1] < self._len and self[d[0]][d[1]] == 0:
                    self[d[0]][d[1]] += self[index[0]][index[1]] + 1
                    q.put(d)

        def get_path(self):
            if self[self.finish[0]][self.finish[1]] == 0 or \
                    self[self.finish[0]][self.finish[1]] == -1:
                raise

            path = []
            item = self.finish
            while not path.append(item) and item != self.__start:
                l = (item[0]-1, item[1])
                if l[0] >= 0 and self[l[0]][l[1]] == self[item[0]][item[1]] - 1:
                    item = l
                    continue
                r = (item[0]+1, item[1])
                if r[0] < self._len and self[r[0]][r[1]] == self[item[0]][item[1]] - 1:
                    item = r
                    continue
                u = (item[0], item[1]-1)
                if u[1] >= 0 and self[u[0]][u[1]] == self[item[0]][item[1]] - 1:
                    item = u
                    continue
                d = (item[0], item[1]+1)
                if d[1] < self._len and self[d[0]][d[1]] == self[item[0]][item[1]] - 1:
                    item = d
                    continue
            return reversed(path)

        def update(self):
            self.__field = [[0 for i in range(self._len)] for i in range(self._len)]

        def _show(self):
                print('')
                #    print('Maze:')
                #    for i in range(14):
                    #        print(GAME_ART[0][i])
                    #        for j in i:
                        #            print(j)
                        #        print(i)

    # pdb.set_trace()
    Path=np.zeros(n_chosen_agents)
    
    if __name__ == '__main__':
        for i in range(n_chosen_agents):

            field = Field(len=14, start=tuple((x_s, y_s)), finish=tuple((x_random[i], y_random[i])),
                  barriers=Barrier)
            field.emit()
            field()
            try:
                path = field.get_path()
                count_path=0
        
                for p in path:
                    count_path=count_path+1
                Path[i]=count_path
            except:
                Path[i]=0
        
        tem=np.where(Path==0)
        if len(tem[0])>0:
            continue
        else:
            # Note that the order of agents is not sequential
            # because 'random.sample(c, n_chosen_agents)'
            agent_order_index = Chosen_agents_index.astype(int) - 2
            agent_order_name = np.array(Chosen_agents_label)[agent_order_index+2]
            # the location of C, D, E, F
            agent_CDEF_index =  agent_order_index.argsort()
            
            simulation_time=simulation_time+1
            Path=Path/sum(Path)
            # pdb.set_trace()
            energy_cost=-Path*np.log2(Path)
            social_reward=-D_eff*np.log2(D_eff)
    
            subj_get=social_reward-energy_cost
            determine=np.where(subj_get==np.nanmax(subj_get))
            index_determine=determine[0][0]
            # pdb.set_trace()
            
            # Save the parameters of this maze
            path_ordered = Path[agent_CDEF_index]
            energy_cost_ordered = energy_cost[agent_CDEF_index]
            social_reward_ordered = social_reward[agent_CDEF_index]
            total_score_ordered = subj_get[agent_CDEF_index]
            final_target_index = np.where((total_score_ordered == np.nanmax(total_score_ordered)))
            final_target_name = TARGET_ORDER[final_target_index]
            df_final_target_predictions = pd.DataFrame(data = {'target_name': TARGET_ORDER,\
                                                               'social_reward': social_reward_ordered,\
                                                               'path': path_ordered,\
                                                               'energy_cost': energy_cost_ordered,\
                                                               'total_score': total_score_ordered,\
                                                               'final_target': np.repeat(final_target_name,4),\
                                                               'simulation_index': np.repeat(str(simulation_time),4)})
            
            # pdb.set_trace()
            if simulation_time > 0:
              df_collect_final_target_predictions = df_collect_final_target_predictions.append(df_final_target_predictions)
              
            if simulation_time == simulation_total:
              df_collect_final_target_predictions.to_csv(FILE_CSV_SUMMARY)              
            
        if __name__ == '__main__':   
            field = Field(len=14, start=tuple((x_s, y_s)), finish=tuple((x_random[index_determine], y_random[index_determine])),
                  barriers=Barrier)
            field.emit()
            field()
            try:
                output_file  = os.path.join(DIR_TXT_OUTPUT, \
                                            "S002_"+str(simulation_time)+".txt")
                text_file = open(output_file, "w")
                text_file.write('Maze:\n')
                
                for i in range(14):
                   text_file.write(GAME_ART[0][i])
                   text_file.write('\n')
                path = field.get_path()
                
                text_file.write(''.join((str(p)+'\n') for p in path))
                text_file.write('\n')
                text_file.close()
            except:
                print('Path not found')



