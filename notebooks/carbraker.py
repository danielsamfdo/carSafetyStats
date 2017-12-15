from collections import namedtuple

features = ['ped_x','ped_y','pos_x','pos_y', 'vel']
State = namedtuple('State', features, verbose=True)

class CarBrakerEnv():
    def __init__(self,sampler):
        self.sampler = sampler
        self.reset_state()
        
    def reset_state(self):
        self.state = self.sampler.random_sample()
        self.SP = self.state['SP']
        state_val = []
        self.ped_x = self.state['ped_x']
#         assert self.ped_x != 75
#         assert self.state['vel'] != 5
        self.ped_y = self.state['ped_y']
        self.end_pos_x = 125.0
        for name in features:
            state_val.append(self.state[name])
        
        self.state_tuple = State._make(state_val)
        self.state = self.state_tuple
        self.state_tuple = self.state_tuple._replace(ped_x = -1)
        self.state_tuple = self.state_tuple._replace(ped_y = -1)
        self.vel = self.state_tuple.vel
        self.possible_actions = [0,1,2] # 0 for not braking, 1 for accelerating, 2 for braking to the warning
        self.time = 0
        self.warning_signalled = 0
        self.terminated = False
        return
    
    def next_step(self, action):
        self.time+=1
        self.reward = -1
        if(action == 0):
            self.state_tuple = self.state_tuple._replace(pos_x = self.state_tuple.pos_x + self.state_tuple.vel)
        elif(action == 1):
            if(self.state_tuple.vel == 0):
                self.state_tuple = self.state_tuple._replace(vel = self.vel)
            elif(1.2*self.state_tuple.vel >=30):
                self.state_tuple = self.state_tuple._replace(vel = 30)
            elif(self.state_tuple.vel > 0):
                self.state_tuple = self.state_tuple._replace(vel = 1.2*self.state_tuple.vel)
            self.state_tuple = self.state_tuple._replace(pos_x = self.state_tuple.pos_x + self.state_tuple.vel)
        elif(action == 2):
            self.state_tuple = self.state_tuple._replace(vel = 0)
            
        if(abs(self.ped_x - self.state_tuple.pos_x)<=40 and self.warning_signalled==0):
            self.state_tuple = self.state_tuple._replace(ped_x = self.ped_x)
            self.state_tuple = self.state_tuple._replace(ped_y = self.ped_y)
            self.warning_signalled = 1
            
        if((action == 0 or action==1) and self.state_tuple.ped_x > 0 and (self.state_tuple.ped_x<=self.state_tuple.pos_x)):
            self.reward = -3000
            self.terminated = True
        elif(action == 2 and self.state_tuple.ped_x > 0):
            self.reward = 10
            self.state_tuple = self.state_tuple._replace(ped_x = -1)
            self.state_tuple = self.state_tuple._replace(ped_y = -1)
        if(self.state_tuple.pos_x >= self.end_pos_x or (self.state_tuple.ped_x > 0 and action==0 and (self.state_tuple.ped_x<=self.state_tuple.pos_x))):
            self.terminated = True
        if(self.time == 200):
            self.terminated = True
        return (self.state_tuple), self.reward
    
