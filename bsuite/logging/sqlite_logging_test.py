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
"""Tests for bsuite.utils.sqlite_logging."""

import random

from absl.testing import absltest
from absl.testing import parameterized
from bsuite.logging import sqlite_logging
import sqlite3


class SqliteLoggerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('no_custom_data', {},),
      ('one_custom_column', {'extra': 42},),
  )
  def test_logger(self, custom_data):
    connection = sqlite3.connect(':memory:')
    logger = sqlite_logging.Logger(db_path='unused',
                                   experiment_name='test',
                                   setting_index=1,
                                   connection=connection)

    num_writes = 10
    steps_per_episode = 7

    total_return = 0.0

    for i in range(num_writes):
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
      data.update(custom_data)
      logger.write(data)

    count_query = 'select count(*) from test;'
    cursor = connection.cursor()
    results = cursor.execute(count_query).fetchall()
    self.assertLen(results, 1)
    self.assertEqual(results[0][0], num_writes)

  def test_logger_raises_malformed_sql_error(self):
    # This experiment name should result in a malformed insert statement.
    experiment_name = 'test--'
    logger = sqlite_logging.Logger(db_path=':memory:',
                                   experiment_name=experiment_name,
                                   setting_index=1,
                                   skip_name_validation=True)

    data = dict(
        steps=10,
        episode=1,
        total_return=5.0,
        episode_len=10,
        episode_return=5.0,
    )
    with self.assertRaises(sqlite3.OperationalError):
      logger.write(data)

  def test_no_error_for_existing_table(self):
    connection = sqlite3.connect(':memory:')

    logger_1 = sqlite_logging.Logger(db_path='unused',
                                     experiment_name='test',
                                     setting_index=1,
                                     connection=connection)

    data = dict(
        steps=10,
        episode=1,
        total_return=5.0,
        episode_len=10,
        episode_return=5.0,
    )
    logger_1.write(data)

    logger_2 = sqlite_logging.Logger(db_path='unused',
                                     experiment_name='test',
                                     setting_index=1,
                                     connection=connection)

    data = dict(
        steps=20,
        episode=2,
        total_return=10.0,
        episode_len=10,
        episode_return=5.0,
    )
    logger_2.write(data)

  def test_invalid_name(self):
    with self.assertRaises(ValueError):
      sqlite_logging.Logger(db_path=':memory:',
                            experiment_name='test--',
                            setting_index=1)


if __name__ == '__main__':
  absltest.main()
