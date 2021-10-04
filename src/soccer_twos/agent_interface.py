import abc
from typing import Any, Dict, Tuple

import numpy as np


class AgentInterface(abc.ABC):
    @abc.abstractmethod
    def act(self, observation: Dict[int, np.ndarray]) -> np.ndarray:
        """The act method is called when the agent is asked to act.
        Args:
            observation: the observation of the environment.
        Returns:
            action: np.array representing the action to be taken.
        """
        raise NotImplementedError
