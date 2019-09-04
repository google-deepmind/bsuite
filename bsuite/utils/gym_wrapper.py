# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""bsuite adapter for OpenAI gym run-loops."""

# Import all packages

import dm_env
from dm_env import specs
import gym
from gym import spaces
import numpy as np
from typing import Any, Dict, Optional, Text, Tuple, Union

# OpenAI gym step format = obs, reward, is_finished, other_info
_GymTimestep = Tuple[np.ndarray, float, bool, Dict[Text, Any]]


class GymWrapper(gym.Env):
  """A wrapper that converts a dm_env.Environment to an OpenAI gym.Env."""

  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self, env: dm_env.Environment):
    self._env = env  # type: dm_env.Environment
    self._last_observation = None  # type: Optional[np.ndarray]
    self.viewer = None

  def step(self, action: int) -> _GymTimestep:
    timestep = self._env.step(action)
    self._last_observation = timestep.observation
    reward = timestep.reward or 0.
    return timestep.observation, reward, timestep.last(), {}

  def reset(self) -> np.ndarray:
    timestep = self._env.reset()
    self._last_observation = timestep.observation
    return timestep.observation

  def render(self, mode: Text = 'rgb_array') -> Union[np.ndarray, bool]:
    if self._last_observation is None:
      raise ValueError('Environment not ready to render. Call reset() first.')

    if mode == 'rgb_array':
      return self._last_observation

    if mode == 'human':
      if self.viewer is None:
        from gym.envs.classic_control import rendering  # pylint: disable=g-import-not-at-top
        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(self._last_observation)
      return self.viewer.isopen

  @property
  def action_space(self) -> spaces.Discrete:
    action_spec = self._env.action_spec()  # type: specs.DiscreteArray
    return spaces.Discrete(action_spec.num_values)

  @property
  def observation_space(self) -> spaces.Box:
    obs_spec = self._env.observation_spec()  # type: specs.Array
    if isinstance(obs_spec, specs.BoundedArray):
      return spaces.Box(
          low=float(obs_spec.minimum),
          high=float(obs_spec.maximum),
          shape=obs_spec.shape,
          dtype=obs_spec.dtype)
    return spaces.Box(
        low=-float('inf'),
        high=float('inf'),
        shape=obs_spec.shape,
        dtype=obs_spec.dtype)

  @property
  def reward_range(self) -> Tuple[float, float]:
    reward_spec = self._env.reward_spec()
    if isinstance(reward_spec, specs.BoundedArray):
      return reward_spec.minimum, reward_spec.maximum
    return -float('inf'), float('inf')

  def __getattr__(self, attr):
    """Delegate attribute access to underlying environment."""
    return getattr(self._env, attr)

def space2spec(space: gym.Space, name: Text=None):
  """Convert a gym space to a dm_env spec.

  Box, MultiBinary and MultiDiscrete Openai spaces are converted to BoundedArray specs.  Discrete Openai
  spaces are converted to DiscreteArray specs.  Tuple and Dict spaces are recursively converted to tuples
  and dictionaries of specs.
  """
  if isinstance(space, spaces.Discrete):
    return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

  elif isinstance(space, spaces.Box):
    return specs.BoundedArray(shape=space.shape, dtype=space.dtype, minimum=space.low, maximum=space.high,
                              name=name)

  elif isinstance(space, spaces.MultiBinary):
    return specs.BoundedArray(shape=space.shape, dtype=space.dtype, minimum=0.0, maximum=1.0, name=name)

  elif isinstance(space, spaces.MultiDiscrete):
    return specs.BoundedArray(shape=space.shape, dtype=space.dtype, minimum=np.zeros(space.shape),
                              maximum=space.nvec, name=name)

  elif isinstance(space, spaces.Tuple):

    spec_list = []
    for _space in space.spaces:
      spec_list.append(space2spec(_space, name))

    return tuple(spec_list)

  elif isinstance(space, spaces.Dict):

    spec_dict = {}
    for k in space.spaces:
      spec_dict[k] = space2spec(space.spaces[k], name)

    return spec_dict

  else:
    raise ValueError('Unexpected gym space: {}'.format(space))

class ReverseGymWrapper(dm_env.Environment):
  """A wrapper that converts an OpenAI gym environment to a dm_env.Environment."""

  def __init__(self, gym_env: gym.Env):
    self.gym_env = gym_env
    # convert gym action and observation spaces to dm_env specs
    self._observation_spec = space2spec(self.gym_env.observation_space, 'observations')
    self._action_spec = space2spec(self.gym_env.action_space, 'actions')

  def reset(self):
    self.gym_env.reset()

  def step(self, action):
    """Convert a gym step result (observations, reward, done, info) to a dm_env TimeStep."""
    obs,reward,done,_ = self.gym_env.step(action)

    if done:
      return dm_env.termination(reward,obs)
    else:
      return dm_env.transition(reward,obs)

  def close(self):
    self.gym_env.close()

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec
