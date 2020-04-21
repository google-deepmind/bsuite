# python3
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
"""Simple diagnostic discounting challenge.

Observation is two pixels: (context, time_to_live)

Context will only be -1 in the first step, then equal to the action selected in
the first step. For all future decisions the agent is in a "chain" for that
action. Reward of +1 come  at one of: 1, 3, 10, 30, 100

However, depending on the seed, one of these chains has a 10% bonus.
"""

from typing import Any, Dict, Optional

from bsuite.environments import base
from bsuite.experiments.discounting_chain import sweep

import dm_env
from dm_env import specs
import numpy as np


class DiscountingChain(base.Environment):
  """Discounting Chain environment."""

  def __init__(self, seed: Optional[int] = None):
    """Builds the Discounting Chain environment.

    Args:
      seed: Optional integer, if specified determines which reward is bonus.
    """
    super().__init__()
    self._episode_len = 100
    self._reward_timestep = [1, 3, 10, 30, 100]
    self._n_actions = len(self._reward_timestep)
    if seed is None:
      seed = np.random.randint(0, self._n_actions)
    else:
      seed = seed % self._n_actions

    self._rewards = np.ones(self._n_actions)
    self._rewards[seed] += 0.1

    self._timestep = 0
    self._context = -1

    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def _get_observation(self):
    obs = np.zeros(shape=(1, 2), dtype=np.float32)
    obs[0, 0] = self._context
    obs[0, 1] = self._timestep / self._episode_len
    return obs

  def _reset(self) -> dm_env.TimeStep:
    self._timestep = 0
    self._context = -1
    observation = self._get_observation()
    return dm_env.restart(observation)

  def _step(self, action: int) -> dm_env.TimeStep:
    if self._timestep == 0:
      self._context = action

    self._timestep += 1
    if self._timestep == self._reward_timestep[self._context]:
      reward = self._rewards[self._context]
    else:
      reward = 0.

    observation = self._get_observation()
    if self._timestep == self._episode_len:
      return dm_env.termination(reward=reward, observation=observation)
    return dm_env.transition(reward=reward, observation=observation)

  def observation_spec(self):
    return specs.Array(shape=(1, 2), dtype=np.float32)

  def action_spec(self):
    return specs.DiscreteArray(self._n_actions, name='action')

  def _save(self, observation):
    self._raw_observation = (observation * 255).astype(np.uint8)

  @property
  def optimal_return(self):
    # Returns the maximum total reward achievable in an episode.
    return 1.1

  def bsuite_info(self) -> Dict[str, Any]:
    return {}
