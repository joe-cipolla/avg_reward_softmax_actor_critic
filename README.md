# avg_reward_softmax_actor_critic
This is an implementation of the Average Reward Softmax Actor-Critic reinforcement learning
policy parameterization using Tile Coding in the Pendulum Swing-Up environment.

&nbsp;

## Pendulum Swing-Up Environment
![Pendulum Swing-Up](https://miro.medium.com/max/1200/1*jLj9SYWI7e6RElIsI3DFjg.gif)

This environment has a single pendulum that can swing 360 degrees. The pendulum is actuated by applying torque on its pivot point.
The goal is to get the pendulum to balance up right from its resting position of hanging down towards the bottom with no velocity.
The pendulum can move freely, subject only to gravity and the action applied to the agent.

The state consists of two dimensions: the current angle (-pi to pi) and the current angular velocity (-2pi to 2pi).

The action is the angular acceleration, with discrete values [-1, 0, 1], applied to the pendulum.

The goal is to swing-up the pendulum and maintain its upright angle as long as possible.
The reward is the negative absolute angel from the vertical position.
Since the action options in the environment are not strong enough to allow the agent to move the pendulum directly to the vertical position,
the agent must first learn to move the pendulum away from the desired position in order to build up enough momentum to then swing it towards
the desired position.

&nbsp;

## Actor-Critic Agent
![Actor-Critic Agent](https://sergioskar.github.io/assets/img/posts/ac.jpg)
The agent consists of two parts, an Actor and a Critic. The Actor learns a parameterized policy wile the Critic learns a state-value function.
Since we are using discrete actions in the environment, the Actor will use a Softmax policy (formula below) with exponentiated action preferences.
The Actor learns the sample-based estimate for the gradient of the average reward objective.
The Critic learns using the average reward version of the semi-gradient TD (0) algorithm.

## Softmax Probability
![Softmax Formula](https://miro.medium.com/max/900/1*tmz_nlcdNyCN0LXr123EqA.png)

&nbsp;

## Tile Coding
The Tile Coding Function provides good generalization and discrimination, consisting of multiple overlapping tilings, where each tiling is a partitioning of the space into tiles.
![Tile Coding](https://www.researchgate.net/profile/Florin_Leon/publication/265110533/figure/fig2/AS:392030699180047@1470478810724/Tile-coding-example.png)

Tile coding is used for function approximation because of the potentially infinite number of states that can occur in the two continuous series of velocity and position (while not exactly infinite at these bounded settings, still large enough to be too expensive to compute).
