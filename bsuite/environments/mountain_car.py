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
"""Python implementation of 'Mountain Car' environment.

An underpowered car must drive up a hill, to succeed you must go back/forth.
This is a classic environment in RL research, first described by:
  A Moore, Efficient Memory-Based Learning for Robot Control,
  PhD thesis, University of Cambridge, 1990.
"""

from bsuite.environments import base
from bsuite.experiments.mountain_car import sweep

import dm_env
from dm_env import specs
import numpy as np


class MountainCar(base.Environment):
  """Mountain Car, an underpowered car must power up a hill."""

  def __init__(self,
               max_steps: int = 1000,
               seed: int = None):
    """Mountain Car, an underpowered car must power up a hill.

    Args:
      max_steps : maximum number of steps to perform per episode
      seed : randomization seed
    """
    super().__init__()
    self._min_pos = -1.2
    self._max_pos = 0.6
    self._max_speed = 0.07
    self._goal_pos = 0.5
    self._force = 0.001
    self._gravity = 0.0025

    self._max_steps = max_steps
    self._rng = np.random.RandomState(seed)
    self._timestep = 0
    self._raw_return = 0.
    self._position = 0.
    self._velocity = 0.
    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def _get_observation(self):
    obs = [self._position, self._velocity, self._timestep / self._max_steps]
    return np.array([obs], dtype=np.float32)

  def _reset(self) -> dm_env.TimeStep:
    """Random initialize in [-0.6, -0.4] and zero velocity."""
    self._timestep = 0
    self._position = self._rng.uniform(-0.6, -0.4)
    self._velocity = 0
    return dm_env.restart(self._get_observation())

  def _step(self, action: int) -> dm_env.TimeStep:
    self._timestep += 1
    reward = -1.
    self._raw_return += reward

    # Step the environment
    self._velocity += (action - 1) * self._force + np.cos(
        3 * self._position) * -self._gravity
    self._velocity = np.clip(self._velocity, -self._max_speed, self._max_speed)
    self._position += self._velocity
    self._position = np.clip(self._position, self._min_pos, self._max_pos)
    if self._position == self._min_pos:
      self._velocity = np.clip(self._velocity, 0, self._max_speed)

    observation = self._get_observation()
    if self._position >= self._goal_pos or self._timestep >= self._max_steps:
      return dm_env.termination(reward=reward, observation=observation)
    return dm_env.transition(reward=reward, observation=observation)

  def observation_spec(self):
    return specs.Array(shape=(1, 3), dtype=np.float32)

  def action_spec(self):
    """Actions [0,1,2] -> [Left, Stay, Right]."""
    return specs.DiscreteArray(3, name='action')

  def bsuite_info(self):
    return dict(raw_return=self._raw_return)


