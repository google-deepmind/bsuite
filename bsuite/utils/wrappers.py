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
"""bsuite logging and image observation wrappers."""

from typing import Any, Dict, Sequence

from bsuite import environments
from bsuite.logging import base

import dm_env
from dm_env import specs
import numpy as np
from skimage import transform

# Keys that are present for all experiments. These are computed from within
# the `Logging` wrapper.
STANDARD_KEYS = frozenset(
    ['steps', 'episode', 'total_return', 'episode_len', 'episode_return'])


class Logging(dm_env.Environment):
  """Environment wrapper to track and log bsuite stats."""

  def __init__(self,
               env: environments.Environment,
               logger: base.Logger,
               log_by_step: bool = False,
               log_every: bool = False):
    """Initializes the logging wrapper.

    Args:
      env: Environment to wrap.
      logger: An object that records a row of data. This must have a `write`
        method that accepts a dictionary mapping from column name to value.
      log_by_step: Whether to log based on step or episode count (default).
      log_every: Forces logging at each step or episode, e.g. for debugging.
    """
    self._env = env
    self._logger = logger
    self._log_by_step = log_by_step
    self._log_every = log_every

    # Accumulating throughout experiment.
    self._steps = 0
    self._episode = 0
    self._total_return = 0.0

    # Most-recent-episode.
    self._episode_len = 0
    self._episode_return = 0.0

  def flush(self):
    if hasattr(self._logger, 'flush'):
      self._logger.flush()

  def reset(self):
    timestep = self._env.reset()
    self._track(timestep)
    return timestep

  def step(self, action):
    timestep = self._env.step(action)
    self._track(timestep)
    return timestep

  def action_spec(self):
    return self._env.action_spec()

  def observation_spec(self):
    return self._env.observation_spec()

  def _track(self, timestep: dm_env.TimeStep):
    # Count transitions only.
    if not timestep.first():
      self._steps += 1
      self._episode_len += 1
    if timestep.last():
      self._episode += 1
    self._episode_return += timestep.reward or 0.0
    self._total_return += timestep.reward or 0.0

    # Log statistics periodically, either by step or by episode.
    if self._log_by_step:
      if _logarithmic_logging(self._steps) or self._log_every:
        self._log_bsuite_data()

    elif timestep.last():
      if _logarithmic_logging(self._episode) or self._log_every:
        self._log_bsuite_data()

    # Perform bookkeeping at the end of episodes.
    if timestep.last():
      self._episode_len = 0
      self._episode_return = 0.0

  def _log_bsuite_data(self):
    """Log summary data for bsuite."""
    data = dict(
        # Accumulated data.
        steps=self._steps,
        episode=self._episode,
        total_return=self._total_return,
        # Most-recent-episode data.
        episode_len=self._episode_len,
        episode_return=self._episode_return,
    )
    # Environment-specific metadata used for scoring.
    data.update(self._env.bsuite_info())
    self._logger.write(data)

  @property
  def raw_env(self):
    # Recursively unwrap until we reach the true 'raw' env.
    wrapped = self._env
    if hasattr(wrapped, 'raw_env'):
      return wrapped.raw_env
    return wrapped

  def __getattr__(self, attr):
    """Delegate attribute access to underlying environment."""
    return getattr(self._env, attr)


def _logarithmic_logging(episode: int, ratios: Sequence[float] = None) -> bool:
  """Returns `True` only at specific ratios of 10**exponent."""
  if ratios is None:
    ratios = [1., 1.2, 1.4, 1.7, 2., 2.5, 3., 4., 5., 6., 7., 8., 9., 10.]
  exponent = np.floor(np.log10(np.maximum(1, episode)))
  special_vals = [10**exponent * ratio for ratio in ratios]
  return any(episode == val for val in special_vals)


class ImageObservation(dm_env.Environment):
  """Environment wrapper to convert observations to an image-like format."""

  def __init__(self, env: dm_env.Environment, shape: Sequence[int]):
    self._env = env
    self._shape = shape

  def observation_spec(self):
    spec = self._env.observation_spec()
    return specs.Array(shape=self._shape, dtype=spec.dtype, name=spec.name)

  def action_spec(self):
    return self._env.action_spec()

  def reset(self):
    timestep = self._env.reset()
    return timestep._replace(
        observation=to_image(self._shape, timestep.observation))

  def step(self, action):
    timestep = self._env.step(action)
    return timestep._replace(
        observation=to_image(self._shape, timestep.observation))

  def __getattr__(self, attr):
    """Delegate attribute access to underlying environment."""
    return getattr(self._env, attr)


def _small_state_to_image(shape: Sequence[int],
                          observation: np.ndarray) -> np.ndarray:
  """Converts a small state into an image-like format."""
  result = np.empty(shape=shape, dtype=observation.dtype)
  size = observation.size
  flattened = observation.ravel()

  # Explicitly handle small observation dimensions separately
  if size == 1:
    result[:] = flattened[0]
  elif size == 2:
    result[:, :shape[1] // 2] = flattened[0]
    result[:, shape[1] // 2:] = flattened[1]
  elif size == 3 or size == 4:
    # Top-left.
    result[:shape[0] // 2, :shape[1] // 2] = flattened[0]
    # Top-right.
    result[shape[0] // 2:, :shape[1] // 2] = flattened[1]
    # Bottom-left.
    result[:shape[0] // 2, shape[1] // 2:] = flattened[2]
    # Bottom-right.
    result[shape[0] // 2:, shape[1] // 2:] = flattened[-1]
  else:
    raise ValueError('Hand-crafted rule only for small state observation.')

  return result


def _interpolate_to_image(shape: Sequence[int],
                          observation: np.ndarray) -> np.ndarray:
  """Converts observation to desired shape using an interpolation."""
  result = np.empty(shape=shape, dtype=observation.dtype)
  if len(observation.shape) == 1:
    observation = np.expand_dims(observation, 0)

  # Interpolate the image and broadcast over all trailing channels.
  plane_image = transform.resize(observation, shape[:2], preserve_range=True)
  while plane_image.ndim < len(shape):
    plane_image = np.expand_dims(plane_image, -1)
  result[:, :] = plane_image
  return result


def to_image(shape: Sequence[int], observation: np.ndarray) -> np.ndarray:
  """Converts a bsuite observation into an image-like format.

  Example usage, converting a 3-element array into a stacked Atari-like format:

      observation = to_image((84, 84, 4), np.array([1, 2, 0]))

  Args:
    shape: A sequence containing the desired output shape (length >= 2).
    observation: A numpy array containing the observation data.

  Returns:
    A numpy array with shape `shape` and dtype matching the dtype of
    `observation`. The entries in this array are tiled from `observation`'s
    entries.
  """
  assert len(shape) >= 2

  if observation.size <= 4:
    return _small_state_to_image(shape, observation)
  elif len(observation.shape) <= 2:
    return _interpolate_to_image(shape, observation)
  else:
    raise ValueError(
        'Cannot convert observation shape {} to desired shape {}'.format(
            observation.shape, shape))


class RewardNoise(environments.Environment):
  """Reward Noise environment wrapper."""

  def __init__(self,
               env: environments.Environment,
               noise_scale: float,
               seed: int = None):
    """Builds the Reward Noise environment wrapper.

    Args:
      env: An environment whose rewards to perturb.
      noise_scale: Standard deviation of gaussian noise on rewards.
      seed: Optional seed for numpy's random number generator (RNG).
    """
    super(RewardNoise, self).__init__()
    self._env = env
    self._noise_scale = noise_scale
    self._rng = np.random.RandomState(seed)

  def reset(self):
    return self._env.reset()

  def step(self, action):
    return self._add_reward_noise(self._env.step(action))

  def _add_reward_noise(self, timestep: dm_env.TimeStep):
    if timestep.first():
      return timestep
    reward = timestep.reward + self._noise_scale * self._rng.randn()
    return dm_env.TimeStep(
        step_type=timestep.step_type,
        reward=reward,
        discount=timestep.discount,
        observation=timestep.observation)

  def observation_spec(self):
    return self._env.observation_spec()

  def action_spec(self):
    return self._env.action_spec()

  @property
  def raw_env(self):
    # Recursively unwrap until we reach the true 'raw' env.
    wrapped = self._env
    if hasattr(wrapped, 'raw_env'):
      return wrapped.raw_env
    return wrapped

  def _step(self, action: int) -> dm_env.TimeStep:
    raise NotImplementedError('Please call step() instead of _step().')

  def _reset(self) -> dm_env.TimeStep:
    raise NotImplementedError('Please call reset() instead of _reset().')

  def bsuite_info(self) -> Dict[str, Any]:
    return self._env.bsuite_info()

  def __getattr__(self, attr):
    """Delegate attribute access to underlying environment."""
    return getattr(self._env, attr)


class RewardScale(environments.Environment):
  """Reward Scale environment wrapper."""

  def __init__(self,
               env: environments.Environment,
               reward_scale: float,
               seed: int = None):
    """Builds the Reward Scale environment wrapper.

    Args:
      env: Environment whose rewards to rescale.
      reward_scale: Rescaling for rewards.
      seed: Optional seed for numpy's random number generator (RNG).
    """
    super(RewardScale, self).__init__()
    self._env = env
    self._reward_scale = reward_scale
    self._rng = np.random.RandomState(seed)

  def reset(self):
    return self._env.reset()

  def step(self, action):
    return self._rescale_rewards(self._env.step(action))

  def _rescale_rewards(self, timestep: dm_env.TimeStep):
    if timestep.first():
      return timestep
    reward = timestep.reward * self._reward_scale
    return dm_env.TimeStep(
        step_type=timestep.step_type,
        reward=reward,
        discount=timestep.discount,
        observation=timestep.observation)

  def observation_spec(self):
    return self._env.observation_spec()

  def action_spec(self):
    return self._env.action_spec()

  def _step(self, action: int) -> dm_env.TimeStep:
    raise NotImplementedError('Please call step() instead of _step().')

  def _reset(self) -> dm_env.TimeStep:
    raise NotImplementedError('Please call reset() instead of _reset().')

  @property
  def raw_env(self):
    # Recursively unwrap until we reach the true 'raw' env.
    wrapped = self._env
    if hasattr(wrapped, 'raw_env'):
      return wrapped.raw_env
    return wrapped

  def bsuite_info(self) -> Dict[str, Any]:
    return self._env.bsuite_info()

  def __getattr__(self, attr):
    """Delegate attribute access to underlying environment."""
    return getattr(self._env, attr)
