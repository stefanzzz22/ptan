from os import path, listdir
import numpy as np
from scipy import stats as st
from lib.agc.util import preprocess, get_action_name
import math
import cv2
import sys

class AtariDataset():

    TRAJS_SUBDIR = 'trajectories'
    SCREENS_SUBDIR = 'screens'

    def __init__(self, data_path):
        
        '''
            Loads the dataset trajectories into memory. 
            data_path is the root of the dataset (the folder, which contains
            the 'screens' and 'trajectories' folders. 
        '''

        self.trajs_path = path.join(data_path, AtariDataset.TRAJS_SUBDIR)       
        self.screens_path = path.join(data_path, AtariDataset.SCREENS_SUBDIR)
    
        #check that the we have the trajs where expected
        assert path.exists(self.trajs_path)
        
        self.trajectories = self.load_trajectories()

        # compute the stats after loading
        self.stats = {}
        for g in self.trajectories.keys():
            self.stats[g] = {}
            nb_games = self.trajectories[g].keys()

            total_frames = sum([len(self.trajectories[g][traj]) for traj in self.trajectories[g]])
            final_scores = [self.trajectories[g][traj][-1]['score'] for traj in self.trajectories[g]]

            self.stats[g]['total_replays'] = len(nb_games)
            self.stats[g]['total_frames'] = total_frames
            self.stats[g]['max_score'] = np.max(final_scores)
            self.stats[g]['min_score'] = np.min(final_scores)
            self.stats[g]['avg_score'] = np.mean(final_scores)
            self.stats[g]['stddev'] = np.std(final_scores)
            self.stats[g]['sem'] = st.sem(final_scores)


    def load_trajectories(self):

        trajectories = {}
        for game in listdir(self.trajs_path):
            trajectories[game] = {}
            game_dir = path.join(self.trajs_path, game)
            for traj in listdir(game_dir):
                curr_traj = []
                with open(path.join(game_dir, traj)) as f:
                    for i,line in enumerate(f):
                        #first line is the metadata, second is the header
                        if i > 1:
                            #TODO will fix the spacing and True/False/integer in the next replay session
                            #frame,reward,score,terminal, action
                    
                            curr_data = line.rstrip('\n').replace(" ","").split(',')
                            curr_trans = {}
                            curr_trans['frame']    = int(curr_data[0])
                            curr_trans['reward']   = int(curr_data[1])
                            curr_trans['score']    = int(curr_data[2])
                            curr_trans['terminal'] = int(eval(curr_data[3]))
                            curr_trans['action']   = int(curr_data[4])
                            curr_traj.append(curr_trans)
                trajectories[game][int(traj.split('.txt')[0])] = curr_traj
        return trajectories
                   

    def compile_data(self, game, score_lb=0, score_ub=math.inf, max_nb_transitions=None):
        data = []
        game_trajectories = self.trajectories[game]
        game_screens_path = path.join(self.screens_path, game)

        shuffled_trajs = np.array(list(game_trajectories.keys()))
        np.random.shuffle(shuffled_trajs)

        for t in shuffled_trajs:
            st_dir   = path.join(game_screens_path, str(t))
            cur_traj = game_trajectories[t]
            cur_traj_len = len(listdir(st_dir))

            # cut off trajectories with final score beyound the limit
            if not score_lb <= cur_traj[-1]['score'] <= score_ub:
                continue

            print("Training data frames imported: {} out of {}".format(len(data), max_nb_transitions))
            sys.stdout.flush()

            #we're here if the trajectory is within lb/ub
            for pid in range(cur_traj_len):

                #screens are numbered from 1, transitions from 0
                #TODO change screen numbering from zero during next data replay
                state = preprocess(cv2.imread(path.join(st_dir, str(pid) + '.png'), cv2.IMREAD_GRAYSCALE))

                data.append({'action': get_action_name(cur_traj[pid]['action']),
                             'state':  state,
                             'reward': cur_traj[pid]['reward'],
                             'terminal': cur_traj[pid]['terminal'] == 1
                            })

                # if nb_transitions is None, we want the whole dataset limited only by lb and ub
                if max_nb_transitions and len(data) == max_nb_transitions:
                    return data

        #we're here if we need all the data
        return data
     
