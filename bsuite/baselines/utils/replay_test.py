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
"""Tests for bsuite.baselines.replay."""

from absl.testing import absltest

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


if __name__ == '__main__':
  absltest.main()
