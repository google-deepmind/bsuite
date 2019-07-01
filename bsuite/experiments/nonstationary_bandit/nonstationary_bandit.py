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
"""Nonstationary bandit environment.

This approximates a run from an N-armed bandit with drift. This code is designed
to sample from a simple conjugate prior of Beta-Bernoulli with drift.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.utils import auto_reset_environment
import dm_env
from dm_env import specs
import numpy as np


class NonstationaryBandit(auto_reset_environment.Base):
  """NonstationaryBandit environment."""

  def __init__(self,
               gamma: float,
               num_arm: int = 3,
               prior_success: float = 1.,
               prior_failure: float = 1.,
               seed: int = None):
    """Bandit with drift gamma back to prior beta(success, failure) init."""
    super(NonstationaryBandit, self).__init__()
    self._gamma = gamma
    self._num_arm = num_arm
    self._prior_success = prior_success
    self._prior_failure = prior_failure
    self._posterior_success = np.array([prior_success for _ in range(num_arm)])
    self._posterior_failure = np.array([prior_failure for _ in range(num_arm)])

    self._rng = np.random.RandomState(seed)
    self._probs = np.zeros(num_arm)
    self._reset()
    self._total_regret = 0.

  def _get_observation(self):
    return np.zeros(shape=(1, 1), dtype=np.float32)

  def _reset(self):
    observation = self._get_observation()
    self._probs = np.array([
        self._rng.beta(self._posterior_success[a], self._posterior_failure[a])
        for a in range(self._num_arm)
    ])
    return dm_env.restart(observation)

  def _step(self, action):
    # Compute the regret
    self._total_regret += self.optimal_return - self._probs[action]
    reward = float(self._rng.binomial(1, self._probs[action]))

    # Sampled arm has some learning
    self._posterior_success[action] += reward
    self._posterior_failure[action] += 1 - reward

    # All arms drift back to the baseline
    self._posterior_success = (self._prior_success * self._gamma
                               + self._posterior_success * (1 - self._gamma))
    self._posterior_failure = (self._prior_failure * self._gamma
                               + self._posterior_failure * (1 - self._gamma))
    observation = self._get_observation()
    return dm_env.termination(reward=reward, observation=observation)

  def observation_spec(self):
    return specs.Array(shape=(1, 1), dtype=np.float32)

  def action_spec(self):
    return specs.DiscreteArray(self._num_arm, name='action')

  def bsuite_info(self):
    return dict(total_regret=self._total_regret)

  @property
  def optimal_return(self):
    return np.max(self._probs)
