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
"""Tests for bsuite.utils.gym_wrapper."""

from absl.testing import absltest
from bsuite.utils import gym_wrapper

from dm_env import specs
import gym
import numpy as np


class DMEnvFromGymTest(absltest.TestCase):

  def test_gym_cartpole(self):
    env = gym_wrapper.DMEnvFromGym(gym.make('CartPole-v0'))

    # Test converted observation spec.
    observation_spec = env.observation_spec()
    self.assertEqual(type(observation_spec), specs.BoundedArray)
    self.assertEqual(observation_spec.shape, (4,))
    self.assertEqual(observation_spec.minimum.shape, (4,))
    self.assertEqual(observation_spec.maximum.shape, (4,))
    self.assertEqual(observation_spec.dtype, np.dtype('float32'))

    # Test converted action spec.
    action_spec = env.action_spec()
    self.assertEqual(type(action_spec), specs.DiscreteArray)
    self.assertEqual(action_spec.shape, ())
    self.assertEqual(action_spec.minimum, 0)
    self.assertEqual(action_spec.maximum, 1)
    self.assertEqual(action_spec.num_values, 2)
    self.assertEqual(action_spec.dtype, np.dtype('int64'))

    # Test step.
    timestep = env.reset()
    self.assertTrue(timestep.first())
    timestep = env.step(1)
    self.assertEqual(timestep.reward, 1.0)
    self.assertEqual(timestep.observation.shape, (4,))
    env.close()

  def test_episode_truncation(self):
    # Pendulum has no early termination condition.
    gym_env = gym.make('Pendulum-v0')
    env = gym_wrapper.DMEnvFromGym(gym_env)
    ts = env.reset()
    while not ts.last():
      ts = env.step(env.action_spec().generate_value())
    self.assertEqual(ts.discount, 1.0)
    env.close()


if __name__ == '__main__':
  absltest.main()
