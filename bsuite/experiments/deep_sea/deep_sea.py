# pylint: disable=g-bad-file-header
# Copyright 2019 The bsuite Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Python implementation of 'Deep Sea' exploration environment.

This environment is designed as a stylized version of the 'exploration chain':
  - The observation is an N x N grid, with a falling block starting in top left.
  - Each timestep the agent can move 'left' or 'right', which are mapped to
    discrete actions 0 and 1 on a state-dependent level.
  - There is a large reward of +1 in the bottom right state, but this can be
    hard for many exploration algorithms to find.

The stochastic version of this domain only transitions to the right with
probability (1 - 1/N) and adds N(0,1) noise to the 'end' states of the chain.
Logging notes 'bad episodes', which are ones where the agent deviates from the
optimal trajectory by taking a bad action, this is *almost* equivalent to the
total regret, but ignores the (small) effects of the move_cost. We avoid keeping
track of this since it makes no big difference to us.

For more information, see papers:
[1] https://arxiv.org/abs/1703.07608
[2] https://arxiv.org/abs/1806.03335
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.utils import auto_reset_environment

import dm_env
from dm_env import specs
import numpy as np


class DeepSea(auto_reset_environment.Base):
  """Deep Sea environment to test for deep exploration."""

  def __init__(self,
               size: int,
               deterministic: bool = True,
               unscaled_move_cost: float = 0.01,
               randomize_actions: bool = True,
               seed: int = None):
    """Deep sea environment to test for deep exploration."""
    super(DeepSea, self).__init__()
    self._size = size
    self._deterministic = deterministic
    self._unscaled_move_cost = unscaled_move_cost
    self._rng = np.random.RandomState(seed)

    if randomize_actions:
      self._action_mapping = self._rng.binomial(1, 0.5, [size, size])
    else:
      self._action_mapping = np.ones(size)

    if not self._deterministic:  # action 'right' only succeeds (1 - 1/N)
      optimal_no_cost = (1 - 1 / self._size) ** (self._size - 1)
    else:
      optimal_no_cost = 1.
    self._optimal_return = optimal_no_cost - self._unscaled_move_cost

    self._column = 0
    self._row = 0
    self._bad_episode = False
    self._total_bad_episodes = 0
    self._reset()

  def _get_observation(self):
    obs = np.zeros(shape=(self._size, self._size), dtype=np.float32)
    if self._row >= self._size:  # End of episode null observation
      return obs
    else:
      obs[self._row, self._column] = 1.
      return obs

  def _reset(self):
    self._row = 0
    self._column = 0
    self._bad_episode = False
    return dm_env.restart(self._get_observation())

  def _step(self, action):
    reward = 0.
    action_right = action == self._action_mapping[self._row, self._column]

    # Reward calculation
    if self._column == self._size - 1 and action_right:
      reward += 1.
    if not self._deterministic:  # Noisy rewards on the 'end' of chain.
      if self._row == self._size - 1 and self._column in [0, self._size - 1]:
        reward += np.random.randn()

    # Transition dynamics
    if action_right:
      if np.random.rand() > 1 / self._size or self._deterministic:
        self._column = np.clip(self._column + 1, 0, self._size - 1)
      reward -= self._unscaled_move_cost / self._size
    else:
      if self._row == self._column:  # You were on the right path and went wrong
        self._bad_episode = True
      self._column = np.clip(self._column - 1, 0, self._size - 1)
    self._row += 1

    observation = self._get_observation()
    if self._row == self._size:
      if self._bad_episode:
        self._total_bad_episodes += 1
      return dm_env.termination(reward=reward, observation=observation)
    else:
      return dm_env.transition(reward=reward, observation=observation)

  def observation_spec(self):
    return specs.Array(shape=(self._size, self._size), dtype=np.float32)

  def action_spec(self):
    return specs.DiscreteArray(2, name='action')

  def bsuite_info(self):
    return dict(total_bad_episodes=self._total_bad_episodes)


