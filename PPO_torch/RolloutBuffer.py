class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.next_value = None

    def clear(self):
        self.__init__()