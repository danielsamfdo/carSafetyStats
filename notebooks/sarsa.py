import random
import numpy as np
import copy
from collections import defaultdict
from datetime import datetime
from Fourier import *

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random

def decision(probability):
    return random.random() < probability
class SARSA():
    def __init__(self, env, order=3, alpha=0.05, gamma=1, epsilon=0.01, eps_cnt=30 ):
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
        self.debug = False
        return
    
    def selectEGreedy(self, state):
        feature = self.phi(state.ped_x,state.pos_x,state.vel)
        if(decision(self.epsilon)):
            # Return some value, exploring
            if(self.debug):
                print("Taking Random")
            return random.choice(self.env.possible_actions)
        if(self.debug):
                print("Taking Greedy")
        maxVal = None
        maxAct = None
        y = self.W*np.tile(feature,len(self.env.possible_actions))
        # assert len(self.W) == len(np.tile(feature,3))
        # result = np.add.reduceat(y, [0,self.total,2*self.total])
        result = np.array([0 for x in range(len(self.env.possible_actions))])
        # result = np.add.reduceat(y, [0,self.total,2*self.total])
        for i in range(len(self.env.possible_actions)):
            result[i] = np.sum(y[i*self.total:(i+1)*self.total])
        val = random.choice(np.where(result == result.max())[0])
        return val
        # assert (val - 1) in [-1,0,1]
        # return np.argmax(result) - 1
        
    def phi(self,ped_pos, s,v):
        state = (ped_pos,s,v)
        return self.f.getFeature(state)
    
    def run(self, env):
        self.episode_count = 0
        self.episodes_values = {}
        while(self.episode_count < self.episode_stop_count):
            env.reset_state()
            total_rew = 0
            s = copy.deepcopy(env.state)
            a = self.selectEGreedy(s)
            values = []
            values.append(tuple([s.ped_x,s.pos_x,s.vel,0]))
            while(not env.terminated):
                next_state,r = env.next_step(a)
                values.append(tuple([next_state.ped_x,next_state.pos_x,next_state.vel,0]))
                total_rew += r
                a_prime = self.selectEGreedy(next_state)
                phi_v = self.phi(s.ped_x,s.pos_x,s.vel)
                start = a_prime
                start_a = a

                delta_t_multiplier = np.dot(self.W[start*self.total : start*self.total+self.total],self.phi(next_state.ped_x,next_state.pos_x,next_state.vel))
                if(env.terminated):
                    delta_t_multiplier = 0
                delta_t = (r + (self.gamma * (delta_t_multiplier)) - (np.dot(self.W[start_a*self.total:start_a*self.total+self.total],phi_v)))
                self.W[start_a*self.total:start_a*self.total+self.total] += (self.alpha* delta_t * phi_v)
                if(self.debug==True):
                    print(next_state,a_prime)
                a = copy.deepcopy(a_prime)
                s = copy.deepcopy(next_state)
            self.episodes[self.episode_count] = total_rew
            self.time_steps[self.episode_count] = env.time
            self.episode_count+=1
            self.episodes_values[self.episode_count] = values
