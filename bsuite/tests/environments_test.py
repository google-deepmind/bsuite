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
"""Tests that we can load all settings in sweep.py with bsuite.load."""

from absl.testing import absltest
from absl.testing import parameterized

from bsuite import bsuite
from bsuite import sweep


def _reduced_names_and_kwargs():
  """Returns a subset of sweep.SETTINGS that covers all environment types."""
  result = []

  last_name = None
  last_keywords = None

  for bsuite_id, kwargs in sweep.SETTINGS.items():
    name = bsuite_id.split(sweep.SEPARATOR)[0]
    keywords = set(kwargs)
    if name != last_name or keywords != last_keywords:
      if 'mnist' not in name:
        result.append((name, kwargs))
    last_name = name
    last_keywords = keywords
  return result


class EnvironmentsTest(parameterized.TestCase):

  @parameterized.parameters(*_reduced_names_and_kwargs())
  def test_environment(self, name, settings):
    env = bsuite.load(name, settings)
    self.assertGreater(env.action_spec().num_values, 0)
    self.assertGreater(env.bsuite_num_episodes, 0)

if __name__ == '__main__':
  absltest.main()
