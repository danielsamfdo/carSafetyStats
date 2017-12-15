from collections import defaultdict
from datetime import datetime
from Fourier import *
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

def decision(probability):
    return random.random() < probability
class QLearning():
    def __init__(self, env, order=2, alpha=0.05, gamma=1, epsilon=0.1,eps_cnt=10 ):
        self.total = order*order*order
        self.W = np.array([0.0] * (self.total * len(env.possible_actions)) )
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env=env
        self.f = FirstOrderFourier(order)
        self.episode_count = 0
        self.episode_stop_count = eps_cnt
        self.time_steps = np.array([0.0] * self.episode_stop_count)
        self.episodes = np.array([0.0] * self.episode_stop_count)
        return
    
    def selectEGreedy(self, state, Egreedy=True):
        feature = self.phi(state.ped_x,state.pos_x,state.vel)
        if(Egreedy and decision(self.epsilon)):
            # Return some value, exploring
            return random.choice(self.env.possible_actions)
        maxVal = None
        maxAct = None
        # assert len(self.W) == len(np.tile(feature,len(self.env.possible_actions)))
        y = self.W*np.tile(feature,len(self.env.possible_actions))
        result = np.array([0 for x in range(len(self.env.possible_actions))])
        # result = np.add.reduceat(y, [0,self.total,2*self.total])
        for i in range(len(self.env.possible_actions)):
            result[i] = np.sum(y[i*self.total:(i+1)*self.total])

        # assert result[1]==np.sum(y[self.total:2*self.total])
        # assert result[2]==np.sum(y[2*self.total:])
        if(not Egreedy): # For Delat Update
            return np.max(result)
        # assert (np.argmax(result) - 1) in [-1,0,1]
        return np.argmax(result)
        
    def phi(self,ped_pos, s,v):
        state = (ped_pos,s,v)
        return self.f.getFeature(state)
    
    def run(self, env):
        self.episode_count = 0
        self.episodes_values = {}
        while(self.episode_count < self.episode_stop_count):
            values = []
            total_rew = 0
            # if(self.episode_count%10==0):
                # print(self.episode_count)
            env.reset_state()
            s = copy.deepcopy(env.state)
            values.append(tuple([s.ped_x,s.pos_x,s.vel,0]))
            while(not env.terminated):
                a = self.selectEGreedy(s)
                next_state,r = env.next_step(a)
                values.append(tuple([next_state.ped_x,next_state.pos_x,next_state.vel,r]))
                phi_v = self.phi(s.ped_x,s.pos_x,s.vel)
                start_a = a
                total_rew+= r
                delta_t_multiplier = self.selectEGreedy(next_state, Egreedy=False) 
                if(env.terminated):
                    delta_t_multiplier = 0
                delta_t = (r + ( self.gamma * delta_t_multiplier) - (np.dot(self.W[start_a*self.total:start_a*self.total+self.total],phi_v)))
                self.W[start_a*self.total:start_a*self.total+self.total] += (self.alpha* delta_t * phi_v)
                s = copy.deepcopy(next_state)
            self.episodes[self.episode_count] = total_rew
            self.time_steps[self.episode_count] = env.time
            self.episode_count+=1
            self.episodes_values[self.episode_count] = values
