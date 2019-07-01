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
"""Read functionality for local SQLite-based experiments."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite import sweep
import pandas as pd
import sqlite3
from typing import Text

_CATEGORICAL_COLUMNS = ('setting_id',)


def join_metadata(df: pd.DataFrame) -> pd.DataFrame:
  """Returns a DataFrame containing sweep metadata joined to the input."""
  # Assume we are loading in the settings via sweep.py, without any changes.
  metadata = sweep.SETTINGS

  data = []

  for bsuite_id, env_kwargs in metadata.items():
    # Add environment and id to dataframe.
    bsuite_env = bsuite_id.split(sweep.SEPARATOR)[0]
    env_kwargs['bsuite_id'] = bsuite_id
    env_kwargs['bsuite_env'] = bsuite_env
    data.append(env_kwargs)

  df_out = df.copy()
  # Convert category for bug https://github.com/pandas-dev/pandas/issues/18646.
  for categorical_column in _CATEGORICAL_COLUMNS:
    if categorical_column in df_out.columns:
      df_out[categorical_column] = df_out[categorical_column].astype(int)

  bsuite_df = pd.DataFrame(data)

  return pd.merge(df_out, bsuite_df, on='bsuite_id')


def load_bsuite(db_path: Text,
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
    dataframe['bsuite_id'] = [table_name[0] + sweep.SEPARATOR + str(setting_id)
                              for setting_id in dataframe.setting_id]
    dataframes.append(dataframe)

  df = pd.concat(dataframes, sort=False)
  return join_metadata(df)
