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
"""Simple diagnostic memory challenge.

Observation is given by n+1 pixels: (context, time_to_live).

Context will only be nonzero in the first step, when it will be +1 or -1 iid
by component. All actions take no effect until time_to_live=0, then the agent
must repeat the observations that it saw bit-by-bit.
"""

from bsuite.environments import base

import dm_env
from dm_env import specs
import numpy as np


class MemoryChain(base.Environment):
  """Memory Chain environment, implementing the environment API."""

  def __init__(self,
               memory_length: int,
               num_bits: int = 1,
               seed=None):
    """Builds the memory chain environment."""
    super(MemoryChain, self).__init__()
    self._memory_length = memory_length
    self._num_bits = num_bits
    self._rng = np.random.RandomState(seed)

    # Contextual information per episode
    self._timestep = 0
    self._context = self._rng.binomial(1, 0.5, num_bits)
    self._query = self._rng.randint(num_bits)

    # Logging info
    self._total_perfect = 0
    self._total_regret = 0
    self._episode_mistakes = 0

    # bsuite experiment length.
    self.bsuite_num_episodes = 10_000  # Overridden by experiment load().

  def _get_observation(self):
    """Observation of form [time, query, num_bits of context]."""
    obs = np.zeros(shape=(1, self._num_bits + 2), dtype=np.float32)
    # Show the time, on every step.
    obs[0, 0] = 1 - self._timestep / self._memory_length
    # Show the query, on the last step
    if self._timestep == self._memory_length - 1:
      obs[0, 1] = self._query
    # Show the context, on the first step
    if self._timestep == 0:
      obs[0, 2:] = 2 * self._context - 1
    return obs

  def _step(self, action: int) -> dm_env.TimeStep:
    observation = self._get_observation()
    self._timestep += 1

    if self._timestep - 1 < self._memory_length:
      # On all but the last step provide a reward of 0.
      return dm_env.transition(reward=0., observation=observation)
    if self._timestep - 1 > self._memory_length:
      raise RuntimeError('Invalid state.')  # We shouldn't get here.

    if action == self._context[self._query]:
      reward = 1.
      self._total_perfect += 1
    else:
      reward = -1.
      self._total_regret += 2.
    return dm_env.termination(reward=reward, observation=observation)

  def _reset(self) -> dm_env.TimeStep:
    self._timestep = 0
    self._episode_mistakes = 0
    self._context = self._rng.binomial(1, 0.5, self._num_bits)
    self._query = self._rng.randint(self._num_bits)
    observation = self._get_observation()
    return dm_env.restart(observation)

  def observation_spec(self):
    return specs.Array(shape=(1, self._num_bits + 2), dtype=np.float32)

  def action_spec(self):
    return specs.DiscreteArray(2, name='action')

  def _save(self, observation):
    self._raw_observation = (observation * 255).astype(np.uint8)

  def bsuite_info(self):
    return dict(
        total_perfect=self._total_perfect,
        total_regret=self._total_regret)
