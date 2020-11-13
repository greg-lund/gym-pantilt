import numpy as np
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R
import quaternion
import gym
from gym import spaces
from gym.utils import seeding

class PanTiltEnv(gym.Env):

    def __init__(self):

        # Discretization
        self.disc = (0.05,1)
        self.raycast_disc = 0.75 
        self.n_hist = 5

        # Bounds for pan-tilt
        self.min_pan = -135
        self.max_pan = 135
        self.min_tilt = 0
        self.max_tilt = 90
        self.num_bins = (10,3)

        # Discount Factors
        self.free_space_discount = 0.3
        self.pan_discount = 1.0
        self.tilt_discount = 1.0

        # Cylinder shape
        self.cyl_len = 18
        self.start_x = 6
        self.episode_range = 6
        self.cyl_rad = 2.5

        # Robot parameters
        self.fov = (100,85)
        self.robot_pos = -self.cyl_rad/2

        # Gym spaces and setup
        self.occ_arr = np.zeros((int(self.cyl_len/self.disc[0]),int(360/self.disc[1])))
        self.action_space = spaces.Discrete(self.num_bins[0]*self.num_bins[1])

        self.actions = sorted([(x,y) for x in np.linspace(self.min_pan,self.max_pan,self.num_bins[0]) for y in np.linspace(self.min_tilt,self.max_tilt,self.num_bins[1])],key=lambda x: x[1])

        self.low = np.append(np.repeat([self.min_pan,self.min_tilt],self.n_hist+1,0).flatten(),self.start_x)
        self.high = np.append(np.repeat([self.max_pan,self.max_tilt],self.n_hist+1,0).flatten(),self.start_x+self.episode_range)

        self.state = np.zeros(2*(self.n_hist+1)+1)
        self.state[-1] = self.start_x

        self.observation_space = spaces.Box(np.float32(self.low),np.float32(self.high))

    def reset(self):
        self.state = np.zeros(2*(self.n_hist+1)+1)
        self.state[-1] = self.start_x
        self.occ_arr[:] = 0
        return self.state

    def step(self,action):
        a = self.actions[action]
        x = self.state[-1]+self.disc[0]

        self.state = np.insert(self.state,0,a)
        self.state = self.state[:-2]
        self.state[-1] = x

        # Calculate reward
        new_pixels, free_space_pixels, total_pixels = self.fill_fov(x,a[0],a[1])
        pan_distance = abs(self.state[1][0]-self.state[0][0])
        tilt_distance = abs(self.state[1][1]-self.state[0][1])

        reward = new_pixels/total_pixels + self.free_space_discount*free_space_pixels/total_pixels 
        - self.pan_discount*pan_distance**2/(self.max_pan-self.min_pan)**2 - self.tilt_discount*tilt_distance**2/(self.max_tilt-self.min_tilt)**2

        return self.state,reward,(x>=self.cyl_len),{}

    def fill_fov(self,x,theta,phi):
        '''
        Given a pan tilt pointing, fill in the occupancy array via tons of raycasts
        '''

        theta *= np.pi/180
        phi *= np.pi/180

        # Get basis vectors for our camera plane
        v =  np.array([1,0,0])
        e1 = np.array([0,0,1])
        e2 = np.array([0,-1,0])

        r1 = R.from_quat([e1[0]*np.sin(theta/2),e1[1]*np.sin(theta/2),e1[2]*np.sin(theta/2),np.cos(theta/2)])
        v = r1.apply(v)
        e2 = r1.apply(e2)
        r2 = R.from_quat([e2[0]*np.sin(phi/2),e2[1]*np.sin(phi/2),e2[2]*np.sin(phi/2),np.cos(phi/2)])
        v = r2.apply(v)
        e1 = r2.apply(e1)

        corners = np.zeros((4,3))
        reward,n = 0,0
        for dtheta in np.linspace(-self.fov[0]*np.pi/360,self.fov[0]*np.pi/360,2):
            for dphi in np.linspace(-self.fov[1]*np.pi/360,self.fov[1]*np.pi/360,2):
                rr1 = R.from_quat([e1[0]*np.sin(dtheta/2),e1[1]*np.sin(dtheta/2),e1[2]*np.sin(dtheta/2),np.cos(dtheta/2)])
                vec = rr1.apply(v)
                e22 = rr1.apply(e2)
                rr2 = R.from_quat([e22[0]*np.sin(dphi/2),e22[1]*np.sin(dphi/2),e22[2]*np.sin(dphi/2),np.cos(dphi/2)])
                vec = rr2.apply(vec)

                corners[n,:] = vec
                n+=1
        
        n1 = int(self.fov[0]/self.raycast_disc)
        n2 = int(self.fov[1]/self.raycast_disc)

        theta_lin = np.linspace(corners[0],corners[1],int(self.fov[0]/self.raycast_disc))
        phi_lin = np.linspace(corners[0],corners[2],int(self.fov[1]/self.raycast_disc))
        vecs = np.zeros((n1*n2,3))
        n = 0
        casts = np.zeros(2)
        for t in theta_lin:
            for p in phi_lin:
                v = t + (p-phi_lin[0])
                casts += self.raycast_cylinder(x,v)

        return [casts[0],casts[1],len(theta_lin)*len(phi_lin)]

    def raycast_cylinder(self,x,vec):
        '''
        Find the intersection of a ray located along a cylinders inner axis with 
        a radius of r. vec is the normal from the camera
        Return [a,b] a = 1 if new pixel, b = 1 if free space
        '''

        # Shooting down barrel
        if vec[1] == 0 and vec[2] == 0: return [0,1]

        # Parameterized intersection with cylinder
        t1 = (-2*self.robot_pos*vec[2] + np.sqrt((2*self.robot_pos*vec[2])**2 - 4*(vec[1]**2+vec[2]**2)*(self.robot_pos**2-self.cyl_rad**2))) / (2*(vec[1]**2+vec[2]**2))
        t2 = (-2*self.robot_pos*vec[2] - np.sqrt((2*self.robot_pos*vec[2])**2 - 4*(vec[1]**2+vec[2]**2)*(self.robot_pos**2-self.cyl_rad**2))) / (2*(vec[1]**2+vec[2]**2))
        t = max(t1,t2)
        #t = np.roots([vec[1]**2+vec[2]**2,2*self.robot_pos*vec[2],self.robot_pos**2-self.cyl_rad**2]).max()

        # If outside of our sensor range lets return free space
        if t < 0 or t > 6 or np.isnan(t) or np.isinf(t): return [0,1]

        pos = [x + t*vec[0],t*vec[1], self.robot_pos+t*vec[2]]

        xind = int(pos[0]/(self.cyl_len) * (self.occ_arr.shape[0]-1))
        yind = int(((np.arctan2(pos[1],pos[2])*180/np.pi)%360)/360 * (self.occ_arr.shape[1]-1))

        if xind >= self.occ_arr.shape[0] or xind < 0:
            return [0,1] 

        new_pixel = self.occ_arr[xind,yind] == 0
        self.occ_arr[xind,yind] = 1
        return [new_pixel,0]
