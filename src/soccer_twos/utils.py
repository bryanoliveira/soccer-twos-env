import gym


class DummyEnv(gym.Env):
    """
    Dummy environment for testing purposes.
    """

    def __init__(self, observation_space, action_space, *_):
        self.observation_space = observation_space
        self.action_space = action_space
