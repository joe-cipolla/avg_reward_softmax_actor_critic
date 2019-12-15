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


def compute_softmax_prob(actor_w, tiles):
    """
    Computes softmax probability for all actions

    Args:
    actor_w - np.array, an array of actor weights
    tiles - np.array, an array of active tiles

    Returns:
    softmax_prob - np.array, an array of size equal to num. actions, and sums to 1.
    """

    # First compute the list of state-action preferences
    state_action_preferences = [actor_w[a][tiles].sum() for a in actor_w]

    # Set the constant c by finding the maximum of state-action preferences (use np.max) (1 line)
    c = np.max(state_action_preferences)

    # Compute the numerator by subtracting c from state-action preferences and exponentiating it (use np.exp) (1 line)
    numerator = [np.exp(a - c) for a in state_action_preferences]

    # Next compute the denominator by summing the values in the numerator (use np.sum) (1 line)
    denominator = np.sum(numerator)


    # Create a probability array by dividing each element in numerator array by denominator (1 line)
    # We will store this probability array in self.softmax_prob as it will be useful later when updating the Actor
    softmax_prob = [i/denominator for i in numerator]

    return softmax_prob


class ActorCriticSoftmaxAgent(BaseAgent):
    def __init__(self):
        self.rand_generator = None

        self.actor_step_size = None
        self.critic_step_size = None
        self.avg_reward_step_size = None

        self.tc = None

        self.avg_reward = None
        self.critic_w = None
        self.actor_w = None

        self.actions = None

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD(0) state aggregation agent.

        Assume agent_info dict contains:
        {
            "iht_size": int
            "num_tilings": int,
            "num_tiles": int,
            "actor_step_size": float,
            "critic_step_size": float,
            "avg_reward_step_size": float,
            "num_actions": int,
            "seed": int
        }
        """

        # set random seed for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        iht_size = agent_info.get("iht_size")
        num_tilings = agent_info.get("num_tilings")
        num_tiles = agent_info.get("num_tiles")

        # initialize self.tc to the tile coder we created
        self.tc = PendulumTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

        # set step-size accordingly (we normally divide actor and critic step-size by num. tilings (p.217-218 of textbook))
        self.actor_step_size = agent_info.get("actor_step_size")/num_tilings
        self.critic_step_size = agent_info.get("critic_step_size")/num_tilings
        self.avg_reward_step_size = agent_info.get("avg_reward_step_size")

        self.actions = list(range(agent_info.get("num_actions")))

        # Set initial values of average reward, actor weights, and critic weights
        # We initialize actor weights to three times the iht_size.
        # Recall this is because we need to have one set of weights for each of the three actions.
        self.avg_reward = 0.0
        self.actor_w = np.zeros((len(self.actions), iht_size))
        self.critic_w = np.zeros(iht_size)

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None

    def agent_policy(self, active_tiles):
        """ policy of the agent
        Args:
            active_tiles (Numpy array): active tiles returned by tile coder

        Returns:
            The action selected according to the policy
        """

        # compute softmax probability
        softmax_prob = compute_softmax_prob(self.actor_w, active_tiles)

        # Sample action from the softmax probability array
        # self.rand_generator.choice() selects an element from the array with the specified probability
        chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob)

        # save softmax_prob as it will be useful later when updating the Actor
        self.softmax_prob = softmax_prob

        return chosen_action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """

        angle, ang_vel = state

        ### Use self.tc to get active_tiles using angle and ang_vel (2 lines)
        # set current_action by calling self.agent_policy with active_tiles
        # active_tiles = ?
        # current_action = ?

        ### START CODE HERE ###
        active_tiles = self.tc.get_tiles(angle, ang_vel)
        current_action = self.agent_policy(active_tiles)
        ### END CODE HERE ###

        self.last_action = current_action
        self.prev_tiles = np.copy(active_tiles)

        return self.last_action


    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the environment's step based on
                                where the agent ended up after the
                                last step.
        Returns:
            The action the agent is taking.
        """

        angle, ang_vel = state

        ### Use self.tc to get active_tiles using angle and ang_vel (1 line)
        # active_tiles = ?
        ### START CODE HERE ###

        ### END CODE HERE ###

        ### Compute delta using Equation (1) (1 line)
        # delta = ?
        ### START CODE HERE ###

        ### END CODE HERE ###

        ### update average reward using Equation (2) (1 line)
        # self.avg_reward += ?
        ### START CODE HERE ###

        ### END CODE HERE ###

        # update critic weights using Equation (3) and (5) (1 line)
        # self.critic_w[self.prev_tiles] += ?
        ### START CODE HERE ###

        ### END CODE HERE ###

        # update actor weights using Equation (4) and (6)
        # We use self.softmax_prob saved from the previous timestep
        # We leave it as an exercise to verify that the code below corresponds to the equation.
        for a in self.actions:
            if a == self.last_action:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (1 - self.softmax_prob[a])
            else:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (0 - self.softmax_prob[a])

        ### set current_action by calling self.agent_policy with active_tiles (1 line)
        # current_action = ?
        ### START CODE HERE ###

        ### END CODE HERE ###

        self.prev_tiles = active_tiles
        self.last_action = current_action

        return self.last_action


    def agent_message(self, message):
        if message == 'get avg reward':
            return self.avg_reward



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


## Test Code for compute_softmax_prob() ##
# set tile-coder
iht_size = 4096
num_tilings = 8
num_tiles = 8
test_tc = PendulumTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

num_actions = 3
actions = list(range(num_actions))
actor_w = np.zeros((len(actions), iht_size))

# setting actor weights such that state-action preferences are always [-1, 1, 2]
actor_w[0] = -1./num_tilings
actor_w[1] = 1./num_tilings
actor_w[2] = 2./num_tilings

# obtain active_tiles from state
state = [-np.pi, 0.]
angle, ang_vel = state
active_tiles = test_tc.get_tiles(angle, ang_vel)

# compute softmax probability
softmax_prob = compute_softmax_prob(actor_w, active_tiles)
print('softmax probability: {}'.format(softmax_prob))


## Test Code for agent_start()##

agent_info = {"iht_size": 4096,
              "num_tilings": 8,
              "num_tiles": 8,
              "actor_step_size": 1e-1,
              "critic_step_size": 1e-0,
              "avg_reward_step_size": 1e-2,
              "num_actions": 3,
              "seed": 99}

test_agent = ActorCriticSoftmaxAgent()
test_agent.agent_init(agent_info)

state = [-np.pi, 0.]

test_agent.agent_start(state)

print("agent active_tiles: {}".format(test_agent.prev_tiles))
print("agent selected action: {}".format(test_agent.last_action))
