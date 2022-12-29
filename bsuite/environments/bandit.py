# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Simple diagnostic bandit environment.

Observation is a single pixel of 0 - this is an independent arm bandit problem!
Rewards are [0, 0.1, .. 1] assigned randomly to 11 arms and deterministic
"""

from typing import Optional

from bsuite.environments import base
from bsuite.experiments.bandit import sweep

import dm_env
from dm_env import specs
import numpy as np


class SimpleBandit(base.Environment):
  """SimpleBandit environment."""

  def __init__(self, mapping_seed: Optional[int] = None, num_actions: int = 11):
    """Builds a simple bandit environment.

    Args:
      mapping_seed: Optional integer. Seed for action mapping.
      num_actions: number of actions available, defaults to 11.
    """
    super(SimpleBandit, self).__init__()
    self._rng = np.random.RandomState(mapping_seed)
    self._num_actions = num_actions
    action_mask = self._rng.choice(
        range(self._num_actions), size=self._num_actions, replace=False)
    self._rewards = np.linspace(0, 1, self._num_actions)[action_mask]

    self._total_regret = 0.
    self._optimal_return = 1.
    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def _get_observation(self):
    return np.ones(shape=(1, 1), dtype=np.float32)

  def _reset(self) -> dm_env.TimeStep:
    observation = self._get_observation()
    return dm_env.restart(observation)

  def _step(self, action: int) -> dm_env.TimeStep:
    reward = self._rewards[action]
    self._total_regret += self._optimal_return - reward
    observation = self._get_observation()
    return dm_env.termination(reward=reward, observation=observation)

  def observation_spec(self):
    return specs.Array(shape=(1, 1), dtype=np.float32, name='observation')

  def action_spec(self):
    return specs.DiscreteArray(self._num_actions, name='action')

  def bsuite_info(self):
    return dict(total_regret=self._total_regret)
