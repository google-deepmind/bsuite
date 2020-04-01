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
"""Read functionality for local SQLite-based experiments."""

from typing import List, Tuple

from bsuite import sweep
from bsuite.logging import logging_utils
import pandas as pd
import sqlite3


def load_one_result_set(db_path: str,
                        connection: sqlite3.Connection = None) -> pd.DataFrame:
  """Returns a pandas DataFrame of bsuite results.

  Args:
    db_path: Path to the database file.
    connection: Optional connection, for testing purposes. If supplied,
      `db_path` will be ignored.

  Returns:
    A pandas DataFrame containing bsuite results.
  """
  if connection is None:
    connection = sqlite3.connect(db_path)

  # Get a list of all table names in this database.
  query = 'select name from sqlite_master where type=\'table\';'
  with connection:
    table_names = connection.execute(query).fetchall()

  dataframes = []
  for table_name in table_names:
    dataframe = pd.read_sql_query('select * from ' + table_name[0], connection)
    dataframe['bsuite_id'] = [
        table_name[0] + sweep.SEPARATOR + str(setting_index)
        for setting_index in dataframe.setting_index]
    dataframes.append(dataframe)

  df = pd.concat(dataframes, sort=False)
  return logging_utils.join_metadata(df)


def load_bsuite(
    results_dirs: logging_utils.PathCollection
) -> Tuple[pd.DataFrame, List[str]]:
  """Returns a pandas DataFrame of bsuite results."""
  return logging_utils.load_multiple_runs(
      path_collection=results_dirs,
      single_load_fn=load_one_result_set,
  )
