import numpy as np
class FirstOrderFourier():
    def __init__(self, order):
        self.order = order
        c = np.arange(order)
        #v1 - ped x, #v2 - posx, #v3 - vel
        self.c = np.array([ np.array([v1,v2,v3]) for v1 in c for v2 in c for v3 in c ])
        self.length = len(self.c)
        return
    
    def normalize(self,ped_pos, pos, vel):
        if(pos>125):
            pos = 125
        # print ped_pos,pos,vel
        # if(ped_pos>0):
            # print ped_pos,(ped_pos + 1)/76.0
        ped_pos = (ped_pos + 1)/126.0
        pos = (pos)/(125.0)
        vel = (vel)/30
        # print ped_pos,pos,vel
        assert ped_pos>=0 and ped_pos<=1
        assert pos>=0 and pos<=1
        assert vel>=0 and vel<=1
        return ped_pos,pos,vel
    
    
    def getFeature(self,s):
        v = np.ones(self.order*self.order*self.order) # 3 for len of actions
        ped_pos, pos_v, vel_v = s
        ped_pos, pos_v, vel_v = self.normalize(ped_pos, pos_v,vel_v)
        v = np.cos(np.sum(self.c * np.array([ped_pos, pos_v, vel_v]),axis=1) * np.pi)
        return v

        