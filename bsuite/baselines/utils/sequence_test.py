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
"""Tests for sequence buffer."""

from absl.testing import absltest
from bsuite.baselines.utils import sequence

import dm_env
from dm_env import specs
import numpy as np


class BufferTest(absltest.TestCase):
  """Tests for the sequence buffer class."""

  def test_buffer(self):
    # Given a buffer and some dummy data...
    max_sequence_length = 10
    obs_shape = (3, 3)
    buffer = sequence.Buffer(
        obs_spec=specs.Array(obs_shape, dtype=np.float),
        action_spec=specs.Array((), dtype=np.int32),
        max_sequence_length=max_sequence_length)
    dummy_step = dm_env.transition(observation=np.zeros(obs_shape), reward=0.)

    # If we add `max_sequence_length` items to the buffer...
    for _ in range(max_sequence_length):
      buffer.append(dummy_step, 0, dummy_step)

    # Then the buffer should now be full.
    self.assertTrue(buffer.full())

    # Any further appends should throw an error.
    with self.assertRaises(ValueError):
      buffer.append(dummy_step, 0, dummy_step)

    # If we now drain this trajectory from the buffer...
    trajectory = buffer.drain()

    # The `observations` sequence should have length `T + 1`.
    self.assertLen(trajectory.observations, max_sequence_length + 1)

    # All other sequences should have length `T`.
    self.assertLen(trajectory.actions, max_sequence_length)
    self.assertLen(trajectory.rewards, max_sequence_length)
    self.assertLen(trajectory.discounts, max_sequence_length)

    # The buffer should now be empty.
    self.assertTrue(buffer.empty())

    # A second call to drain() should throw an error, since the buffer is empty.
    with self.assertRaises(ValueError):
      buffer.drain()

    # If we now append another transition...
    buffer.append(dummy_step, 0, dummy_step)

    # And immediately drain the buffer...
    trajectory = buffer.drain()

    # We should have a valid partial trajectory of length T=1.
    self.assertLen(trajectory.observations, 2)
    self.assertLen(trajectory.actions, 1)
    self.assertLen(trajectory.rewards, 1)
    self.assertLen(trajectory.discounts, 1)


if __name__ == '__main__':
  absltest.main()
