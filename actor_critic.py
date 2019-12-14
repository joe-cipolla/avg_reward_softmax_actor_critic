# import libraries
import numpy as np
import matplotlib.pyplot as plt

import os
from tdqm import tqdm

from rl_glue import RLGlue
from pendulum_env import PendulumEnvironment
from agent import BaseAgent
import plot_script
import tiles3 as tc


# helper functions
class PendulumTileCoder:
    def __init__(self, iht_size=4096, num_tilings=32, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the tiles are the same

        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """

        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.iht = tc.IHT(iht_size)

    def get_tiles(self, angle, ang_vel):
        """
        Takes in an angle and angular velocity from the pendulum environment
        and returns a numpy array of active tiles.

        Arguments:
        angle -- float, the angle of the pendulum between -np.pi and np.pi
        ang_vel -- float, the angular velocity of the agent between -2*np.pi and 2*np.pi

        returns:
        tiles -- np.array, active tiles

        """

        ### Set the max and min of angle and ang_vel to scale the input
        ANGLE_MIN = -np.pi
        ANGLE_MAX = np.pi
        ANG_VEL_MIN = -2*np.pi
        ANG_VEL_MAX = 2*np.pi

        ### Use the ranges above and self.num_tiles to set angle_scale and ang_vel_scale
        angle_scale = self.num_tiles / (ANGLE_MAX - ANGLE_MIN)
        ang_vel_scale = self.num_tiles / (ANG_VEL_MAX - ANG_VEL_MIN)

        # Get tiles by calling tc.tileswrap method
        # wrapwidths specify which dimension to wrap over and its wrapwidth
        tiles = tc.tileswrap(self.iht, self.num_tilings, [angle * angle_scale, ang_vel * ang_vel_scale], wrapwidths=[self.num_tiles, False])

        return np.array(tiles)






#### Unit Tests

## Test Code for PendulumTileCoder ##
# tile coder should also work for other num. tilings and num. tiles
test_obs = [[-np.pi, 0], [-np.pi, 0.5], [np.pi, 0], [np.pi, -0.5], [0, 1]]

pdtc = PendulumTileCoder(iht_size=4096, num_tilings=8, num_tiles=4)

result=[]
for obs in test_obs:
    angle, ang_vel = obs
    tiles = pdtc.get_tiles(angle=angle, ang_vel=ang_vel)
    result.append(tiles)

for tiles in result:
    print(tiles)
