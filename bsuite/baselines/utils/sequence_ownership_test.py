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
"""Drained trajectories must not alias the reusable sequence buffer."""

from absl.testing import absltest
from absl.testing import parameterized
from bsuite.baselines.utils import sequence
import dm_env
from dm_env import specs
import numpy as np


def make_buffer():
  return sequence.Buffer(
      specs.Array((2,), np.float32),
      specs.DiscreteArray(3),
      max_sequence_length=3,
  )


def append_sequence(buffer, start, length, terminal=False):
  step = dm_env.restart(np.array([start, start + 0.5], np.float32))
  for index in range(length):
    observation = np.array([start + index + 1, start + index + 1.5], np.float32)
    if terminal and index == length - 1:
      following = dm_env.termination(start + index + 2.0, observation)
    else:
      following = dm_env.transition(
          start + index + 2.0, observation, discount=0.75
      )
    buffer.append(step, index % 3, following)
    step = following


class SequenceOwnershipTest(parameterized.TestCase):

  @parameterized.product(length=[1, 3], terminal=[False, True])
  def test_drained_values_survive_buffer_reuse(self, length, terminal):
    buffer = make_buffer()
    append_sequence(buffer, 1, length, terminal)
    saved = buffer.drain()
    expected = [field.copy() for field in saved]
    for iteration in range(3):
      append_sequence(buffer, 20 + 10 * iteration, 3, terminal=True)
      next_saved = buffer.drain()
      for actual, reference, latest in zip(saved, expected, next_saved):
        np.testing.assert_array_equal(actual, reference)
        self.assertFalse(np.shares_memory(actual, latest))
    self.assertEqual(saved.observations.shape, (length + 1, 2))
    self.assertEqual(saved.actions.dtype, np.int32)
    self.assertEqual(saved.rewards.dtype, np.float32)

  @parameterized.parameters(*sequence.Trajectory._fields)
  def test_mutating_drained_arrays_cannot_change_current_sequence(self, field):
    buffer = make_buffer()
    append_sequence(buffer, 1, 3)
    old = buffer.drain()
    append_sequence(buffer, 10, 3)
    # Change old data before draining the current sequence.
    getattr(old, field)[...] = -123
    latest = buffer.drain()
    reference_buffer = make_buffer()
    append_sequence(reference_buffer, 10, 3)
    reference = reference_buffer.drain()
    for actual, expected in zip(latest, reference):
      np.testing.assert_array_equal(actual, expected)

  def test_old_returns_and_bootstrap_observations_stay_consistent(self):
    buffer = make_buffer()
    append_sequence(buffer, 1, 3)
    saved = buffer.drain()

    def discounted_return(trajectory):
      result = trajectory.observations[-1, 0]
      for reward, discount in zip(
          trajectory.rewards[::-1], trajectory.discounts[::-1]
      ):
        result = reward + discount * result
      return result

    expected = discounted_return(saved)
    append_sequence(buffer, 100, 3)
    self.assertEqual(discounted_return(saved), expected)

  def test_draining_preserves_full_and_empty_guards(self):
    buffer = make_buffer()
    with self.assertRaises(ValueError):
      buffer.drain()
    append_sequence(buffer, 1, 3, terminal=True)
    self.assertTrue(buffer.full())
    trajectory = buffer.drain()
    self.assertTrue(buffer.empty())
    self.assertEqual(trajectory.discounts[-1], 0.0)
    with self.assertRaises(ValueError):
      buffer.drain()


if __name__ == "__main__":
  absltest.main()
