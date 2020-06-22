import torch
import torch.nn as nn

from utils.utils import make_mlp

class ActorNetwork:
    ...

class CriticNetwork:
    ...


class A2CAgent:
    """
    Attempt 1 of A2C algorithm..

    notes:
    By running different exploration policies in diff threads, the overall changes made to the params by multiple actor learnings applying online updates in parallel
    are less correlated in time compared to single agent applyijng online updates... so no need for a replay memory

    No replay memory
    Use onpolicy RL because no expperience replay

    For advantage function, you only need a value fn estimator.
    adv = r_(t+1) + V(S_(t+1)) - V(S_t)

    Update the actor and critic networks based on mod fn from counter.
    No need to polyak average for naive implementation.

    take action a based on policy using Q estimate



    """

    def __init__(self):
        """
        What needs to be here:
        initialize the actor and critic networks.

        """

    def choose_action(self):
        ...

    def train(self):
        ...


