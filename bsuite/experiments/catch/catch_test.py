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
"""Tests for bsuite.experiments.catch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from bsuite.experiments.catch import catch
from bsuite.utils import environment_test

import numpy as np


class CatchInterfaceTest(
    environment_test.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    return catch.Catch(rows=10, columns=5)

  def make_action_sequence(self):
    valid_actions = [0, 1, 2]
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.choice(valid_actions)

if __name__ == '__main__':
  absltest.main()
