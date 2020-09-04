# Reinforcement Learning Algorithms, Implemented in Pytorch

Hey there! This repository holds some reinforcement learning algorithms that I will be implementing in Pytorch. Some examples of things I will implement:

## Notes

Some of the algorithms (A2C, PPO, SAC) use mujoco-py for the Inverted Pendulum environment.
You can check out MuJoCo's [website](https://www.roboti.us/license.html) in order to get a license.
If you're a student you should a yearly free license.
Pybullet is a good alternative if you can't get the license.

## List of things to implement

1. Basic Policy Gradient
2. Advantage Actor Critic (A2C)
3. Proximal Policy Optimization (PPO)
3. Deep Deterministic Policy Gradient (DDPG)
4. Twin Delayed DDPG (TD3)
5. Soft Actor Critic (SAC)

## Additional things to add to help performance:

- Batch sizes (right now it's just single episode online learning which has high variance)
- Normalize rewards

## Basic Policy Gradient

Combined with GAE ([Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)), 
I tried to keep this policy gradient implementation simple.
    
### [A2C](https://arxiv.org/abs/1602.01783)

The secret to getting this one working? Realizing that storing things in numpy arrays gets rid of gradient history.

TODO: Add entropy term in the loss (subtract to encourage exploration)! (You can just use .entropy() on the distribution - multiply this by a constant)
    
### [PPO](https://arxiv.org/abs/1707.06347)

While trying to rewrite the actor critic part of this, I realized the importance of discount factors in
infinite MDPs. They help the actor critic converge.

Additional notes: There might be a balance issue between policy loss compared to value fn loss, especially in low dimensional action spaces such as inverted pendulum. Would this be an issue?

More notes: Because of this issue, the value function often times never really converges - even when maxxing out the episode limit,
the value loss is still ridiculously high. Why is this?

The problem with the value function for an infinite horizon is that it's heavily dependent on the future of the episode
because of the structure of rewards-to-go, and also because of the constant value of the inverted pendulum environment.

TLDR: The value fn essentially just becomes a predictor of the episode length... But this conflicts with the fact that you want the episode length to go farther

The main way the value fn can get to a low value - and typically the easiest: Just have short episodes!

### [DDPG](https://arxiv.org/abs/1509.02971)

New stuff: Replay buffer, deterministic policy, very DQN esque

Note: WIP

### [TD3](https://arxiv.org/abs/1802.09477)

Just DDPG with bells and whistles

Note: WIP

### [SAC](https://arxiv.org/abs/1801.01290)

Emphasis on entropy term inside loss (scaled by temperature term alpha - set to constant in this implementation)

Note: WIP
