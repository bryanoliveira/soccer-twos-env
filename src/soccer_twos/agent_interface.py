import abc
from typing import Dict

import numpy as np


class AgentInterface(abc.ABC):
    def __init__(self):
        self.name = "UNNAMED AGENT"

    @abc.abstractmethod
    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        raise NotImplementedError
