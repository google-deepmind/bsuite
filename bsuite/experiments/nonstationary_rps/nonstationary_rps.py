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
"""Nonstationary Rock Paper Scissors environment.

The agent plays rock, paper, scissors against an adaptive, randomized adversary:
  - The environment adapts towards beating the most recent moves.
  - The environment also has some drift back towards random.

Encode actions [0, 1, 2] = [rock, paper, scissors]
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.experiments.nonstationary_rps import sweep
from bsuite.utils import auto_reset_environment
import dm_env
from dm_env import specs
import numpy as np

_NUM_ACTION = 3
_RPS_PAYOUT = (0., 1., -1.)


def _compute_reward(agent_action: int, env_action: int):
  assert agent_action in [0, 1, 2]
  assert env_action in [0, 1, 2]
  return _RPS_PAYOUT[(agent_action - env_action) % _NUM_ACTION]


class NonstationaryRPS(auto_reset_environment.Base):
  """NonstationaryRPS environment."""

  def __init__(self,
               winning_update: float = 0.1,
               random_update: float = 0.1,
               seed: int = None):
    """Rock Paper Scissors against an adversary that adapts."""
    super(NonstationaryRPS, self).__init__()
    self._winning_update = winning_update
    self._random_update = random_update
    self._rng = np.random.RandomState(seed)

    self._probs = self._sample_uniform_dirichlet()
    self._total_regret = 0.
    self._total_rescaled_regret = 0.

    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def _sample_uniform_dirichlet(self):
    w = self._rng.exponential(1, size=_NUM_ACTION)
    return w / np.sum(w)

  def _reset(self):
    return dm_env.restart(self.observation)

  def _step(self, action):
    env_action = self._rng.choice(range(_NUM_ACTION), p=self._probs)
    reward = _compute_reward(action, env_action)

    # Computing the regret of the action relative to probs - loop for clarity!
    for imagined_env_action in range(_NUM_ACTION):
      regret_of_action = 1. - _compute_reward(action, imagined_env_action)
      self._total_regret += regret_of_action * self._probs[imagined_env_action]

    # Update probs towards what *would* have beaten agent
    env_winning_move = np.zeros(_NUM_ACTION)
    env_winning_move[(action + 1) % _NUM_ACTION] = 1
    self._probs = (self._probs * (1 - self._winning_update)
                   + env_winning_move * self._winning_update)

    # Update probs towards a random target
    self._probs = (self._probs * (1 - self._random_update)
                   + self._sample_uniform_dirichlet() * self._random_update)

    return dm_env.termination(reward=reward, observation=self.observation)

  def observation_spec(self):
    return specs.Array(shape=(1, 1), dtype=np.float32)

  def action_spec(self):
    return specs.DiscreteArray(_NUM_ACTION, name='action')

  def bsuite_info(self):
    return dict(total_regret=self._total_regret)

  @property
  def observation(self):
    return np.zeros(shape=(1, 1), dtype=np.float32)

  @property
  def optimal_return(self):
    return np.max(self._probs)
