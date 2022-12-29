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
"""Tests for bsuite.experiments.memory_len."""

from absl.testing import absltest
from absl.testing import parameterized
from bsuite.environments import memory_chain
from dm_env import test_utils
import numpy as np


class MemoryLengthInterfaceTest(test_utils.EnvironmentTestMixin,
                                parameterized.TestCase):

  def make_object_under_test(self):
    return memory_chain.MemoryChain(memory_length=10, num_bits=1)

  def make_action_sequence(self):
    valid_actions = [0, 1]
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.choice(valid_actions)


class MemorySizeInterfaceTest(test_utils.EnvironmentTestMixin,
                              parameterized.TestCase):

  def make_object_under_test(self):
    return memory_chain.MemoryChain(memory_length=2, num_bits=10)

  def make_action_sequence(self):
    valid_actions = [0, 1]
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.choice(valid_actions)

if __name__ == '__main__':
  absltest.main()
