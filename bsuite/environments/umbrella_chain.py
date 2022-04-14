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
"""Simple diagnostic credit assigment challenge.

Observation is 3 + n_distractor pixels:
  (need_umbrella, have_umbrella, time_to_live, n x distractors)

Only the first action takes any effect (pick up umbrella or not).
All other actions take no effect and the reward is +1, -1 on the final step.
Distractor states are always Bernoulli sampled  iid each step.
"""

from typing import Optional

from bsuite.environments import base
from bsuite.experiments.umbrella_length import sweep

import dm_env
from dm_env import specs
import numpy as np


class UmbrellaChain(base.Environment):
  """Umbrella Chain environment."""

  def __init__(self,
               chain_length: int,
               n_distractor: int = 0,
               seed: Optional[int] = None):
    """Builds the umbrella chain environment.

    Args:
      chain_length: Integer. Length that the agent must back up.
      n_distractor: Integer. Number of distractor observations.
      seed: Optional integer. Seed for numpy's random number generator (RNG).
    """
    super().__init__()
    self._chain_length = chain_length
    self._rng = np.random.RandomState(seed)
    self._n_distractor = n_distractor
    self._timestep = 0
    self._need_umbrella = self._rng.binomial(1, 0.5)
    self._has_umbrella = 0
    self._total_regret = 0
    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def _get_observation(self):
    obs = np.zeros(shape=(1, 3 + self._n_distractor), dtype=np.float32)
    obs[0, 0] = self._need_umbrella
    obs[0, 1] = self._has_umbrella
    obs[0, 2] = 1 - self._timestep / self._chain_length
    obs[0, 3:] = self._rng.binomial(1, 0.5, size=self._n_distractor)
    return obs

  def _step(self, action: int) -> dm_env.TimeStep:
    self._timestep += 1

    if self._timestep == 1:  # you can only pick up umbrella t=1
      self._has_umbrella = action

    if self._timestep == self._chain_length:  # reward only at end.
      if self._has_umbrella == self._need_umbrella:
        reward = 1.
      else:
        reward = -1.
        self._total_regret += 2.
      observation = self._get_observation()
      return dm_env.termination(reward=reward, observation=observation)

    reward = 2. * self._rng.binomial(1, 0.5) - 1.
    observation = self._get_observation()
    return dm_env.transition(reward=reward, observation=observation)

  def _reset(self) -> dm_env.TimeStep:
    self._timestep = 0
    self._need_umbrella = self._rng.binomial(1, 0.5)
    self._has_umbrella = self._rng.binomial(1, 0.5)
    observation = self._get_observation()
    return dm_env.restart(observation)

  def observation_spec(self):
    return specs.Array(
        shape=(1, 3 + self._n_distractor), dtype=np.float32, name='observation')

  def action_spec(self):
    return specs.DiscreteArray(2, name='action')

  def bsuite_info(self):
    return dict(total_regret=self._total_regret)

  def _save(self, observation):
    self._raw_observation = (observation * 255).astype(np.uint8)

  @property
  def optimal_return(self):
    # Returns the maximum total reward achievable in an episode.
    return 1

  @property
  def context(self):
    return self._context  # pytype: disable=attribute-error  # bind-properties
