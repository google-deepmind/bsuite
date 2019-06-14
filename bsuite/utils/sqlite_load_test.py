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
"""Tests for bsuite.utils.sqlite_load."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# Internal dependencies.

from absl.testing import absltest
from bsuite.utils import sqlite_load
from bsuite.utils import sqlite_logging

import sqlite3

_NUM_WRITES = 10


def generate_results(experiment_name, setting_id, connection):
  logger = sqlite_logging.Logger(db_path='unused',
                                 experiment_name=experiment_name,
                                 setting_id=setting_id,
                                 connection=connection)

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


class SqliteLoadTest(absltest.TestCase):

  def test_logger(self):
    connection = sqlite3.connect(':memory:')

    generate_results(
        experiment_name='test', setting_id=1, connection=connection)
    generate_results(
        experiment_name='test', setting_id=2, connection=connection)

    df = sqlite_load.load_bsuite(db_path='unused', connection=connection)
    self.assertLen(df, _NUM_WRITES * 2)


if __name__ == '__main__':
  absltest.main()
