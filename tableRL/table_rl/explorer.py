from abc import ABCMeta, abstractmethod


class Explorer(object, metaclass=ABCMeta):
    """Abstract explorer."""

    @abstractmethod
    def select_action(self, action_values=None):
        """Select an action.

        Args:
          action_values: np.ndarray of action-values
        """
        raise NotImplementedError()

    def observe(self, obs):
        """Select an action.

        Args:
          obs: Q-values
          action_value = vector of action values
        """
        raise NotImplementedError()
