import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class PanTiltEnv(gym.Env):

    def __init__(self, num_bins=(10,2), n_hist=5, episode_range=5, disc=(0.2,5),
            fov=(100,85), cyl_rad=2.5):

        self.min_pan = -135
        self.max_pan = 135
        self.min_tilt = 0
        self.max_tilt = 45

        self.n_hist = n_hist
        self.fov = fov
        self.disc = disc
        self.num_bins = num_bins
        self.cyl_rad = cyl_rad
        self.episode_range = episode_range

        self.occ_arr = np.zeros((int(2*episode_range/disc[0]),int(360/disc[1])))
        self.action_space = spaces.Discrete(num_bins[0]*num_bins[1])

        self.actions = sorted([(x,y) for x in np.linspace(self.min_pan,self.max_pan,num_bins[0]) for y in np.linspace(self.min_tilt,self.max_tilt,num_bins[1])],key=lambda x: x[1])

        self.low = np.append(np.repeat([self.min_pan,self.min_tilt],n_hist+1,0).flatten(),0)
        self.high = np.append(np.repeat([self.max_pan,self.max_tilt],n_hist+1,0).flatten(),episode_range)

        self.state = np.zeros(2*(self.n_hist+1)+1)
        self.state[-1] = episode_range

        self.observation_space = spaces.Box(self.low,self.high,dtype=np.float32)

    def reset(self):
        self.state = np.zeros(2*(self.n_hist+1)+1)
        self.state[-1] = self.episode_range
        return self.state

    def step(self,action):
        a = self.actions[action]
        x = self.state[-1]+self.disc[0]

        self.state = np.insert(self.state,0,a)
        self.state = self.state[:-2]
        self.state[-1] = x

        reward = self.fill_fov(x,a[0],a[1],self.cyl_rad)

        return self.state,reward,(x>=self.episode_range),{}

    def fill_fov(self,x,theta,phi,r):
        '''
        Given a pan tilt pointing, fill in the occupancy array via tons of raycasts
        '''
        n = 0
        reward = 0
        for dtheta in np.linspace(theta-self.fov[0]/2,theta+self.fov[0]/2,int(self.fov[0]/self.disc[1])):
            for dphi in np.linspace(phi-self.fov[1]/2,phi+self.fov[1]/2,int(self.fov[1]/self.disc[1])):
                reward += self.raycast_cylinder(x,theta+dtheta,phi+dphi,r)
                n+=1

        return 1.0*reward/n

    def raycast_cylinder(self,x,theta,phi,r):
        '''
        Find the intersection of a ray located along a cylinders inner axis with 
        a radius of r. theta = pan angle, phi = tilt angle, r = radius
        '''
        theta *= np.pi/180
        phi *= np.pi/180
        t = 1.0/np.sqrt((np.cos(phi)*np.sin(theta))**2+np.sin(phi)**2)

        pos = [x + r*t*np.cos(phi)*np.cos(theta),r*t*np.cos(phi)*np.sin(theta), r*t*np.sin(phi)]

        xind = int(pos[0]/(2*self.episode_range) * (self.occ_arr.shape[0]-1))
        yind = int(((np.arctan2(pos[1],pos[0])*180/np.pi)%360)/360 * (self.occ_arr.shape[1]-1))

        if xind >= self.occ_arr.shape[0]: return 0 

        new_pixel = self.occ_arr[xind,yind] == 0
        self.occ_arr[xind,yind] = 1
        return new_pixel
