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
"""Tests for bsuite.sweep."""

from absl.testing import absltest
from absl.testing import parameterized
from bsuite import sweep


class SweepTest(parameterized.TestCase):

  def test_access_sweep(self):
    self.assertNotEmpty(sweep.SETTINGS)

  def test_access_experiment_constants(self):
    self.assertNotEmpty(sweep.DEEP_SEA)

  @parameterized.parameters(*sweep.SETTINGS)
  def test_sweep_name_format(self, bsuite_id):
    self.assertIn(sweep.SEPARATOR, bsuite_id)
    split = bsuite_id.split(sweep.SEPARATOR)
    self.assertTrue(len(split), 2)
    self.assertNotEmpty(split[0])
    self.assertNotEmpty(split[1])

if __name__ == '__main__':
  absltest.main()
