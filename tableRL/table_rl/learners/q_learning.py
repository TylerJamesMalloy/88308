from table_rl import learner
import numpy as np


class QLearning(learner.Learner):
    """Class that implements Q-Learning."""

    def __init__(self,
                 num_states,
                 num_actions,
                 learning_rate,
                 explorer,
                 discount=0.99,
                 initial_val=0.):
        self.explorer = explorer
        self.learning_rate = learning_rate
        self.q = np.full((num_states, num_actions), initial_val, dtype=float)
        self.discount = discount

    def update_q(self, obs, action, reward, terminated, next_obs):
        target = reward if terminated else reward + self.discount * np.max(self.q[next_obs])
        estimate = self.q[obs, action]
        self.q[obs, action] = estimate + self.learning_rate * (target - estimate)
    
    def act(self, obs: int, train: bool) -> int:
        """Returns an integer 
        """
        self.current_obs = obs
        q_values = self.q[obs]
        action = self.explorer.select_action(q_values) if train else np.argmax(q_values)
        self.last_action = action
        return action
        
    def observe(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly.

        Returns:
            None
        """
        self.update_q(self.current_obs, self.last_action, reward, terminated, obs)
        self.explorer.observe(self.current_obs)
        if terminated or truncated:
            self.current_obs = None
            self.last_action = None

