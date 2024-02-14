import pytest
import numpy as np
import math
import table_rl
import table_rl.dp.dp as dp
from table_rl.learners.double_q_learning import DoubleQLearning
from table_rl.learners.double_q_learning import select_a_greedy_action

class TestDoubleQLearning:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.env = table_rl.envs.BasicEnv(discount=0.9)
        self.num_states = 3
        self.num_actions = 2 # L and R
        self.T = self.env.T
        self.R = self.env.R
        self.discount = 0.9

        self.policy = np.array([[0.3, 0.7],
                           [0.2, 0.8],
                           [0.4, 0.6]])


    def test_double_q_learning_loop(self):
        explorer = table_rl.explorers.ConstantEpsilonGreedy(0.1, self.T.shape[1])

        agent = table_rl.learners.DoubleQLearning(self.T.shape[0],
                          self.T.shape[1],
                          0.02,
                          explorer,
                          discount=self.discount,
                          initial_val=0.)

        observation, info = self.env.reset()

        for _ in range(300000):
            action = agent.act(observation, True)
            observation, reward, terminated, truncated, info = self.env.step(action)
            agent.observe(observation, reward, terminated, truncated)
            if terminated or truncated:
                observation, info = self.env.reset()

        hand_confirmed_opt = np.array([[0.79345359, 0.8708637 ], 
                                       [0.80539959, 0.97837773],
                                       [0.,         0.        ]])
        np.testing.assert_almost_equal(hand_confirmed_opt, agent.q1, decimal=2)
        np.testing.assert_almost_equal(hand_confirmed_opt, agent.q2, decimal=2)

    def test_double_q_learning_update(self):
        explorer = table_rl.explorers.ConstantEpsilonGreedy(0.1, self.T.shape[1])
        agent = table_rl.learners.DoubleQLearning(self.T.shape[0],
                          self.T.shape[1],
                          0.1,
                          explorer,
                          discount=self.discount,
                          initial_val=0.)
        agent.q1 = np.array([[1,2], [4,3], [5,6]], dtype=float)
        agent.q2 = np.array([[7,8], [9,10], [11,12]], dtype=float)
        agent.current_obs = 0
        agent.last_action = 0
        mock_next_state = 1
        mock_reward = 2.0
        possible_next_q1_value = 1 + 0.1 * (mock_reward + self.discount * 9  - 1)
        possible_next_q2_value = 7 + 0.1 * (mock_reward + self.discount * 3 - 7)
        agent.observe(1, mock_reward, False, False)
        q1_chosen = agent.q1[0,0] == possible_next_q1_value
        q2_chosen = agent.q2[0,0] == possible_next_q2_value
        assert q1_chosen or q2_chosen


    def test_select_a_greedy_action(self):
        action = select_a_greedy_action(np.array([0., 1., 3., 2.]))
        assert action == 2
        actions = []
        for _ in range(100):
            action = select_a_greedy_action(np.array([0., 1., 3., 3.]))
            actions.append(action)
        assert 2 in actions
        assert 3 in actions
        assert 1 not in actions


