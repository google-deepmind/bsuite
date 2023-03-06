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
"""A swing up experiment in Cartpole."""

from typing import Optional

from bsuite.environments import base
from bsuite.environments import cartpole
from bsuite.experiments.cartpole_swingup import sweep

import dm_env
from dm_env import specs
import numpy as np


class CartpoleSwingup(base.Environment):
  """A difficult 'swing up' version of the classic Cart Pole task.

  In this version of the problem the pole begins downwards, and the agent must
  swing the pole up in order to see reward. Unlike the typical cartpole task
  the agent must pay a cost for moving, which aggravates the explore-exploit
  tradedoff. Algorithms without 'deep exploration' will simply remain still.
  """

  def __init__(self,
               height_threshold: float = 0.5,
               theta_dot_threshold: float = 1.,
               x_reward_threshold: float = 1.,
               move_cost: float = 0.1,
               x_threshold: float = 3.,
               timescale: float = 0.01,
               max_time: float = 10.,
               init_range: float = 0.05,
               seed: Optional[int] = None):
    # Setup.
    self._state = cartpole.CartpoleState(0, 0, 0, 0, 0)
    super().__init__()
    self._rng = np.random.RandomState(seed)
    self._init_fn = lambda: self._rng.uniform(low=-init_range, high=init_range)

    # Logging info
    self._raw_return = 0.
    self._total_upright = 0.
    self._best_episode = 0.
    self._episode_return = 0.

    # Reward/episode logic
    self._height_threshold = height_threshold
    self._theta_dot_threshold = theta_dot_threshold
    self._x_reward_threshold = x_reward_threshold
    self._move_cost = move_cost
    self._x_threshold = x_threshold
    self._timescale = timescale
    self._max_time = max_time

    # Problem config
    self._cartpole_config = cartpole.CartpoleConfig(
        mass_cart=1.,
        mass_pole=0.1,
        length=0.5,
        force_mag=10.,
        gravity=9.8,
    )

    # Public attributes.
    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def reset(self):
    self._reset_next_step = False
    self._state = cartpole.CartpoleState(
        x=self._init_fn(),
        x_dot=self._init_fn(),
        theta=np.pi + self._init_fn(),
        theta_dot=self._init_fn(),
        time_elapsed=0.,
    )
    self._episode_return = 0.
    return dm_env.restart(self.observation)

  def step(self, action):
    if self._reset_next_step:
      return self.reset()

    self._state = cartpole.step_cartpole(
        action=action,
        timescale=self._timescale,
        state=self._state,
        config=self._cartpole_config,
    )

    # Rewards only when the pole is central and balanced
    is_upright = (np.cos(self._state.theta) > self._height_threshold
                  and np.abs(self._state.theta_dot) < self._theta_dot_threshold
                  and np.abs(self._state.x) < self._x_reward_threshold)
    reward = -1. * np.abs(action - 1) * self._move_cost

    if is_upright:
      reward += 1.
      self._total_upright += 1
    self._raw_return += reward
    self._episode_return += reward

    is_end_of_episode = (self._state.time_elapsed > self._max_time
                         or np.abs(self._state.x) > self._x_threshold)
    if is_end_of_episode:
      self._best_episode = max(self._episode_return, self._best_episode)
      self._reset_next_step = True
      return dm_env.termination(reward=reward, observation=self.observation)
    else:  # continuing transition.
      return dm_env.transition(reward=reward, observation=self.observation)

  def _step(self, action: int) -> dm_env.TimeStep:
    raise NotImplementedError('This environment implements its own auto-reset.')

  def _reset(self) -> dm_env.TimeStep:
    raise NotImplementedError('This environment implements its own auto-reset.')

  def action_spec(self):
    return specs.DiscreteArray(dtype=np.int32, num_values=3, name='action')

  def observation_spec(self):
    return specs.Array(shape=(1, 8), dtype=np.float32, name='state')

  @property
  def observation(self) -> np.ndarray:
    """Approximately normalize output."""
    obs = np.zeros((1, 8), dtype=np.float32)
    obs[0, 0] = self._state.x / self._x_threshold
    obs[0, 1] = self._state.x_dot / self._x_threshold
    obs[0, 2] = np.sin(self._state.theta)
    obs[0, 3] = np.cos(self._state.theta)
    obs[0, 4] = self._state.theta_dot
    obs[0, 5] = self._state.time_elapsed / self._max_time
    obs[0, 6] = 1. if np.abs(self._state.x) < self._x_reward_threshold else -1.
    theta_dot = self._state.theta_dot
    obs[0, 7] = 1. if np.abs(theta_dot) < self._theta_dot_threshold else -1.
    return obs

  def bsuite_info(self):
    return dict(raw_return=self._raw_return,
                total_upright=self._total_upright,
                best_episode=self._best_episode)
