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
"""Tests for bsuite.baselines.replay."""

from absl.testing import absltest
from absl.testing import parameterized

from bsuite.baselines.utils import replay as replay_lib
import numpy as np


class BasicReplayTest(absltest.TestCase):

  def test_end_to_end(self):
    shapes = (10, 10, 3), ()
    capacity = 5

    def generate_sample():
      return [np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8),
              np.random.uniform(size=())]

    replay = replay_lib.Replay(capacity=capacity)

    # Does it crash if we sample when there's barely any data?
    sample = generate_sample()
    replay.add(sample)
    samples = replay.sample(size=2)
    for sample, shape in zip(samples, shapes):
      self.assertEqual(sample.shape, (2,) + shape)

    # Fill to capacity.
    for _ in range(capacity - 1):
      replay.add(generate_sample())
      samples = replay.sample(size=3)
      for sample, shape in zip(samples, shapes):
        self.assertEqual(sample.shape, (3,) + shape)

    replay.add(generate_sample())
    samples = replay.sample(size=capacity)
    for sample, shape in zip(samples, shapes):
      self.assertEqual(sample.shape, (capacity,) + shape)


class ReplayResetTest(parameterized.TestCase):

  @parameterized.parameters(0, 1, 4, 6)
  def test_reset_clears_size_and_fill_fraction(self, num_added):
    replay = replay_lib.Replay(capacity=4)
    for value in range(num_added):
      replay.add([value + 1])
    replay.reset()
    self.assertEqual(replay.size, 0)
    self.assertEqual(replay.fraction_filled, 0.0)
    replay.reset()
    self.assertEqual(replay.size, 0)

  @parameterized.parameters(1, 4, 6)
  def test_reset_then_refill_matches_fresh_buffer(self, num_added):
    replay = replay_lib.Replay(capacity=4)
    for value in range(num_added):
      replay.add([np.array([value], dtype=np.int32)])
    replay.reset()
    fresh = replay_lib.Replay(capacity=4)
    rng_state = np.random.get_state()
    try:
      # New storage has a different shape and two fields. Every sampled
      # field must belong to an actual post-reset transition.
      for step in range(7):
        item = [
            np.array([step + 10, step + 20], dtype=np.float32),
            np.int32(step + 30),
        ]
        replay.add(item)
        fresh.add(item)
        self.assertEqual(replay.size, fresh.size)
        self.assertEqual(replay.fraction_filled, fresh.fraction_filled)
        np.random.seed(123)
        actual = replay.sample(32)
        np.random.seed(123)
        expected = fresh.sample(32)
        for got, wanted in zip(actual, expected):
          np.testing.assert_array_equal(got, wanted)
    finally:
      np.random.set_state(rng_state)

  def test_partial_reset_does_not_sample_unwritten_slots(self):
    replay = replay_lib.Replay(capacity=4)
    replay.add([1.0])
    replay.add([2.0])
    replay.reset()
    replay.add([17.0])
    rng_state = np.random.get_state()
    try:
      np.random.seed(0)
      np.testing.assert_array_equal(replay.sample(64)[0], np.full(64, 17.0))
    finally:
      np.random.set_state(rng_state)


if __name__ == '__main__':
  absltest.main()
