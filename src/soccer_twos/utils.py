import inspect
import logging

import gym

from soccer_twos.agent_interface import AgentInterface


class DummyEnv(gym.Env):
    """
    Dummy environment for testing purposes.
    """

    def __init__(self, observation_space, action_space, *_):
        self.observation_space = observation_space
        self.action_space = action_space


def get_agent_class(module):
    for class_name, class_type in inspect.getmembers(module, inspect.isclass):
        if class_name != "AgentInterface" and issubclass(class_type, AgentInterface):
            logging.info(f"Found agent {class_name} in module {module.__name__}")
            return class_type

    raise ValueError(
        "No AgentInterface subclass found in module {}".format(module.__name__)
    )
