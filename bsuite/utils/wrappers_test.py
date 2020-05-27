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
"""Tests for bsuite.utils.wrapper."""

from absl.testing import absltest
from absl.testing import parameterized

from bsuite import environments
from bsuite.environments import catch
from bsuite.utils import wrappers

import dm_env
from dm_env import specs
from dm_env import test_utils
import mock
import numpy as np


class FakeEnvironment(environments.Environment):
  """An environment that returns pre-determined rewards and observations."""

  def __init__(self, time_steps):
    """Initializes a new FakeEnvironment.

    Args:
      time_steps: A sequence of time step namedtuples. This could represent
        one episode, or several. This class just repeatedly plays through the
        sequence and doesn't inspect the contents.
    """
    super().__init__()
    self._time_steps = time_steps

    obs = np.asarray(self._time_steps[0].observation)
    self._observation_spec = specs.Array(shape=obs.shape, dtype=obs.dtype)
    self._step_index = 0
    self._reset_next_step = True

  def reset(self):
    self._reset_next_step = False
    self._step_index = 0
    return self._time_steps[0]

  def step(self, action):
    del action
    if self._reset_next_step:
      return self.reset()

    self._step_index += 1
    self._step_index %= len(self._time_steps)
    return self._time_steps[self._step_index]

  def _reset(self):
    raise NotImplementedError

  def _step(self, action: int):
    raise NotImplementedError

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return specs.Array(shape=(), dtype=np.int32)

  def bsuite_info(self):
    return {}


class WrapperTest(absltest.TestCase):

  def test_wrapper(self):
    """Tests that the wrapper computes and logs the correct data."""
    mock_logger = mock.MagicMock()
    mock_logger.write = mock.MagicMock()

    # Make a fake environment that cycles through these time steps.
    timesteps = [
        dm_env.restart([]),
        dm_env.transition(1, []),
        dm_env.transition(2, []),
        dm_env.termination(3, []),
    ]
    expected_episode_return = 6
    fake_env = FakeEnvironment(timesteps)
    env = wrappers.Logging(env=fake_env, logger=mock_logger, log_every=True)  # pytype: disable=wrong-arg-types

    num_episodes = 5

    for _ in range(num_episodes):
      timestep = env.reset()
      while not timestep.last():
        timestep = env.step(action=0)

    # We count the number of transitions, hence the -1.
    expected_episode_length = len(timesteps) - 1

    expected_calls = []
    for i in range(1, num_episodes + 1):
      expected_calls.append(
          mock.call(dict(
              steps=expected_episode_length * i,
              episode=i,
              total_return=expected_episode_return * i,
              episode_len=expected_episode_length,
              episode_return=expected_episode_return,
              ))
          )
    mock_logger.write.assert_has_calls(expected_calls)

  def test_unwrap(self):
    raw_env = FakeEnvironment([dm_env.restart([])])
    scale_env = wrappers.RewardScale(raw_env, reward_scale=1.)
    noise_env = wrappers.RewardNoise(scale_env, noise_scale=1.)
    logging_env = wrappers.Logging(noise_env, logger=None)  # pytype: disable=wrong-arg-types

    unwrapped = logging_env.raw_env
    self.assertEqual(id(raw_env), id(unwrapped))


class ImageObservationTest(parameterized.TestCase):

  @parameterized.parameters(
      ((84, 84, 4), np.array([1, 2])),
      ((70, 90), np.array([[1, 0, 2, 3]])),
  )
  def test_to_image(self, shape, observation):
    image = wrappers.to_image(shape, observation)
    self.assertEqual(image.shape, shape)
    self.assertCountEqual(np.unique(image), np.unique(observation))


class ImageWrapperCatchTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    env = catch.Catch()
    return wrappers.ImageObservation(env, (84, 84, 4))

  def make_action_sequence(self):
    actions = [0, 1, 2]
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.choice(actions)


if __name__ == '__main__':
  absltest.main()
