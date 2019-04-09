from abc import ABC
from abc import abstractmethod


class RLAgent(ABC):

    def __init__(self):
        """Base class for all RL agents.

        """
        pass

    @abstractmethod
    def step(self, **kwargs):
        """Processes updates (e.g. next_state, reward, done etc) caused by action execution

        Args:
            **kwargs: environment and agent info and other params

        Returns:

        """
        pass

    @abstractmethod
    def act(self, **kwargs):
        """Returns actions for given state as per current policy

        Args:
            **kwargs: state, epsilon etc

        Returns:
            action selected by agent

        """
        pass

    @abstractmethod
    def learn(self, **kwargs):
        """Update model using collected/online experience.

        Args:
            **kwargs: experiences, gamma etc

        Returns:

        """
        pass