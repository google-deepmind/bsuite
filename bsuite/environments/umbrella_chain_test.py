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
"""Tests for bsuite.experiments.umbrella_distract."""

from absl.testing import absltest
from bsuite.environments import umbrella_chain
from dm_env import test_utils

import numpy as np


class UmbrellaDistractInterfaceTest(test_utils.EnvironmentTestMixin,
                                    absltest.TestCase):

  def make_object_under_test(self):
    return umbrella_chain.UmbrellaChain(chain_length=20, n_distractor=22)

  def make_action_sequence(self):
    valid_actions = [0, 1]
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.choice(valid_actions)


class UmbrellaLengthInterfaceTest(test_utils.EnvironmentTestMixin,
                                  absltest.TestCase):

  def make_object_under_test(self):
    return umbrella_chain.UmbrellaChain(chain_length=10, n_distractor=0)

  def make_action_sequence(self):
    valid_actions = [0, 1]
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.choice(valid_actions)

if __name__ == '__main__':
  absltest.main()
