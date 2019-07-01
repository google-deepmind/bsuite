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
"""Simple diagnostic memory challenge.

Observation is given by n+1 pixels: (context, time_to_live).

Context will only be nonzero in the first step, when it will be +1 or -1 iid
by component. All actions take no effect until time_to_live=0, then the agent
must repeat the observations that it saw bit-by-bit.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.utils import auto_reset_environment
import dm_env
from dm_env import specs
import numpy as np


class MemoryChain(auto_reset_environment.Base):
  """Memory Chain environment, implementing the environment API."""

  def __init__(self,
               memory_length: int,
               num_bits: int = 1,
               perfect_bonus: float = 0.,
               seed=None):
    """Builds the memory chain environment."""
    super(MemoryChain, self).__init__()
    self._memory_length = memory_length
    self._num_bits = num_bits
    self._perfect_bonus = perfect_bonus
    self._rng = np.random.RandomState(seed)

    # Contextual information per episode
    self._timestep = 0
    self._context = self._rng.binomial(1, 0.5, num_bits)

    # Logging info
    self._episode_mistakes = 0.
    self._total_perfect = 0.
    self._total_regret = 0.

  def _get_observation(self):
    obs = np.zeros(shape=(1, self._num_bits + 1), dtype=np.float32)
    obs[0, -1] = 1 - self._timestep / self._memory_length
    if self._timestep == 0:
      obs[0, :-1] = 2 * self._context - 1
    return obs

  def _step(self, action):
    observation = self._get_observation()
    self._timestep += 1

    if self._timestep - 1 < self._memory_length:
      return dm_env.transition(reward=0., observation=observation)

    elif self._timestep == self._memory_length + self._num_bits:
      if action == self._context[-1]:
        reward = 1.
        if self._episode_mistakes == 0:
          self._total_perfect += 1
          reward += self._perfect_bonus
      else:
        reward = -1.
        self._episode_mistakes += 1
        self._total_regret += 2.
      return dm_env.termination(reward=reward, observation=observation)

    elif self._timestep < self._memory_length + self._num_bits:
      time_remainder = self._timestep - self._memory_length
      if action == self._context[time_remainder]:
        reward = 1.
      else:
        reward = -1.
        self._episode_mistakes += 1
      return dm_env.transition(reward=reward, observation=observation)

  def _reset(self):
    self._timestep = 0
    self._context = self._rng.binomial(1, 0.5, self._num_bits)
    observation = self._get_observation()
    return dm_env.restart(observation)

  def observation_spec(self):
    return specs.Array(shape=(1, self._num_bits + 1), dtype=np.float32)

  def action_spec(self):
    return specs.DiscreteArray(2, name='action')

  def _save(self, observation):
    self._raw_observation = (observation * 255).astype(np.uint8)

  def bsuite_info(self):
    return dict(total_perfect=self._total_perfect,
                total_regret=self._total_regret)
