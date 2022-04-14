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
"""Tests for sweep.py."""

from absl.testing import absltest

from bsuite import sweep
from bsuite.experiments.bandit import sweep as bandit_sweep


class SweepTest(absltest.TestCase):
  """Simple tests for sweeps."""

  def test_sweep_contents(self):
    """Checks that all sweeps have sensible contents."""

    test_bsuite_id = 'bandit/0'
    test_bsuite_id_1 = 'bandit/1'

    # Check `test_bsuite_id` is in BANDIT, SWEEP, and TESTING sweeps.
    self.assertIn(test_bsuite_id, sweep.BANDIT)
    self.assertIn(test_bsuite_id, sweep.SWEEP)
    self.assertIn(test_bsuite_id, sweep.TESTING)

    # `test_bsuite_id_1` should *not* be included in the testing sweep.
    self.assertNotIn(test_bsuite_id_1, sweep.TESTING)

    # Check all settings present in sweep.
    self.assertLen(sweep.BANDIT, len(bandit_sweep.SETTINGS))

    # Check `test_bsuite_id` is found in the 'basic' TAG section.
    self.assertIn(test_bsuite_id, sweep.TAGS['basic'])

  def test_sweep_immutable(self):
    """Checks that all exposed sweeps are immutable."""

    with self.assertRaises(TypeError):
      # pytype: disable=attribute-error
      # pytype: disable=unsupported-operands
      sweep.BANDIT[0] = 'new_bsuite_id'
      sweep.SWEEP[0] = 'new_bsuite_id'
      sweep.TESTING[0] = 'new_bsuite_id'
      sweep.TAGS['new_tag'] = 42
      # pytype: enable=unsupported-operands
      # pytype: enable=attribute-error

if __name__ == '__main__':
  absltest.main()
