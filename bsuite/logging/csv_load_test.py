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
"""Tests for bsuite.utils.csv_load."""

import random
import sys

from absl import flags
from absl.testing import absltest
from bsuite.logging import csv_load
from bsuite.logging import csv_logging

FLAGS = flags.FLAGS
_NUM_WRITES = 10


def generate_results(bsuite_id, results_dir):
  logger = csv_logging.Logger(bsuite_id, results_dir)
  steps_per_episode = 7
  total_return = 0.0
  for i in range(_NUM_WRITES):
    episode_return = random.random()
    total_return += episode_return
    data = dict(
        steps=i * steps_per_episode,
        episode=i,
        total_return=total_return,
        episode_len=steps_per_episode,
        episode_return=episode_return,
        extra=42,
    )
    logger.write(data)


class CsvLoadTest(absltest.TestCase):

  def test_logger(self):
    try:
      flags.FLAGS.test_tmpdir
    except flags.UnparsedFlagAccessError:
      # Need to initialize flags when running `pytest`.
      flags.FLAGS(sys.argv)
    results_dir = self.create_tempdir().full_path
    generate_results(bsuite_id='catch/0', results_dir=results_dir)
    generate_results(bsuite_id='catch/1', results_dir=results_dir)

    df = csv_load.load_one_result_set(results_dir=results_dir)
    self.assertLen(df, _NUM_WRITES * 2)

    # Check that sweep metadata is joined correctly.
    # Catch includes a 'seed' parameter, so we expect to see it here.
    self.assertIn('seed', df.columns)
    self.assertIn('bsuite_id', df.columns)


if __name__ == '__main__':
  absltest.main()
