# pylint: disable=g-bad-file-header
# Copyright 2019 The bsuite Authors. All Rights Reserved.
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
"""Tests for bsuite.experiments.bandit_noise."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.
from absl.testing import absltest
from bsuite.experiments.nonstationary_bandit import nonstationary_bandit
from bsuite.utils import environment_test
import numpy as np


class BanditNoiseInterfaceTest(
    environment_test.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    return nonstationary_bandit.NonstationaryBandit(gamma=0.99, num_arm=3)

  def make_action_sequence(self):
    valid_actions = range(3)
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.choice(valid_actions)

if __name__ == '__main__':
  absltest.main()
