from typing import List
from .config import GAMMA, TAU

class RolloutBuffer:
    """Trajectory storage for PPO updates."""
    def __init__(self):
        self.actions   : List = []
        self.states    : List = []
        self.logprobs  : List = []
        self.rewards   : List = []
        self.dones     : List = []
        self.values    : List = []
        self.next_value      = None

    def clear(self):
        """Reset all buffers."""
        self.__init__()

def compute_gae(next_value, rewards, masks, values):
    """
    Compute Generalized Advantage Estimate (GAE).
    Returns list of discounted returns.
    """
    values = values + [next_value]
    gae    = 0.0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * values[step + 1] * masks[step] - values[step]
        gae   = delta + GAMMA * TAU * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns
