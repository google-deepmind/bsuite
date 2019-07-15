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
"""The Cartpole reinforcement learning environment."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.experiments.cartpole import sweep
import dm_env
from dm_env import specs
import numpy as np


class Cartpole(dm_env.Environment):
  """A configurable Cart Pole environment: move a cart to balance a pole.

  This implements the classic Cart Pole task as described here:
  https://webdocs.cs.ualberta.ca/~sutton/papers/barto-sutton-anderson-83.pdf

  The observation is a vector representing:
    `(x, x_dot, sin(theta), sin(theta)_dot, cos(theta), cos(theta)_dot,
      time_elapsed)`
  All components are normalized to the [-1, +1] range.

  The actions are discrete, and there are three available: `stay`, `left`,
  and `right`; `left`/`right` have a cost `move_cost`, `stay` has no cost.

  Episodes start with the pole at an angle `initial_theta`.
  Episodes ends once `max_time` is reached or |x| makes it to `width_threshold`.
  """

  def __init__(self,
               seed: int = 0,
               height_threshold: float = 0.8,
               initial_theta: float = 0.,
               move_cost: float = 0.,
               timescale: float = 0.01,
               max_time: float = 10.,
               x_reward_threshold: float = 1.,
               theta_dot_threshold: float = 1.,
               width_threshold: float = 10.):
    # Setup.
    self._reset_next_step = True
    self._rng = np.random.RandomState(seed)

    # Problem constants
    self._gravity = 9.8
    self._mass_cart = 1
    self._mass_pole = 0.1
    self._total_mass = (self._mass_pole + self._mass_cart)
    self._length = 0.5  # actually half the pole's length
    self._force_mag = 10.

    # Config constants
    self._timescale = timescale
    self._max_time = max_time
    self._height_threshold = height_threshold
    self._x_reward_threshold = x_reward_threshold
    self._theta_dot_threshold = theta_dot_threshold
    self._width_threshold = width_threshold
    self._move_cost = move_cost
    self._initial_theta = initial_theta

    # Updating internal state
    self._raw_return = 0.
    self._state = np.zeros((1, 5))

    # Public attributes.
    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def reset(self):
    self._reset_next_step = False
    self._state = self._rng.uniform(
        low=-0.05, high=0.05, size=(1, 5)).astype(np.float32)
    self._state[0, 2] += self._initial_theta
    self._state[0, 4] = 0  # time elapsed always starts at zero
    return dm_env.restart(self.observation)

  def step(self, action):
    if self._reset_next_step:
      return self.reset()
    x, x_dot, theta, theta_dot, time_elapsed = self._state[0, :]
    force = (action - 1) * self._force_mag  # Converting the action to force.

    # Unpack variables into "short" names for mathematical equation
    cos = np.cos(theta)
    sin = np.sin(theta)
    pl = self._mass_pole * self._length
    l = self._length
    m_pole = self._mass_pole
    m_total = self._total_mass
    g = self._gravity

    # Compute the physical evolution
    temp = (force + pl * theta_dot**2 * sin) / m_total
    theta_acc = (g * sin - cos * temp) / (l * (4/3 - m_pole * cos**2 / m_total))
    x_acc = temp - pl * theta_acc * cos / m_total

    # Update states according to discrete dynamics
    x += self._timescale * x_dot
    x_dot += self._timescale * x_acc
    theta += self._timescale * theta_dot
    theta = np.remainder(theta, 2 * np.pi)  # theta in range (0, 2 * pi)
    theta_dot += self._timescale * theta_acc
    time_elapsed += self._timescale

    # Add singleton dimension.
    self._state = np.array(
        [[x, x_dot, theta, theta_dot, time_elapsed]], dtype=np.float32
    )

    # Rewards only when the pole is central and balanced
    is_reward = (np.cos(theta) > self._height_threshold
                 and np.abs(theta_dot) < self._theta_dot_threshold
                 and np.abs(x) < self._x_reward_threshold)
    reward = float(is_reward)
    reward -= self._move_cost * np.abs(action - 1)
    self._raw_return += reward

    if time_elapsed > self._max_time or np.abs(x) > self._width_threshold:
      self._reset_next_step = True
      return dm_env.termination(reward=reward, observation=self.observation)
    else:  # continuing transition.
      return dm_env.transition(reward=reward, observation=self.observation)

  def observation_spec(self):
    return specs.Array(shape=(1, 7), dtype=np.float32, name='state')

  def action_spec(self):
    return specs.DiscreteArray(dtype=np.int, num_values=3, name='action')

  @property
  def observation(self) -> np.ndarray:
    """Approximately normalize output."""
    x, x_dot, theta, theta_dot, time_elapsed = self._state[0, :]
    obs = np.zeros((1, 7), dtype=np.float32)
    obs[0, 0] = x / self._width_threshold
    obs[0, 1] = x_dot / self._width_threshold
    obs[0, 2] = np.sin(theta)
    obs[0, 3] = np.cos(theta) * theta_dot / np.pi
    obs[0, 4] = np.cos(theta)
    obs[0, 5] = - np.sin(theta) * theta_dot / np.pi
    obs[0, 6] = time_elapsed / self._max_time
    return obs

  def bsuite_info(self):
    return dict(raw_return=self._raw_return)
