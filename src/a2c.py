import torch

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




    """

    def __init__(self):
        ...

    def choose_action(self):
        ...

    def train(self):
        ...


