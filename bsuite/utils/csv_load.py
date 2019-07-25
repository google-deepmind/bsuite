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
"""Read functionality for local csv-based experiments."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

import collections
import glob
import os

from bsuite import sweep
from bsuite.utils import csv_logging
import pandas as pd
import six
from typing import Mapping, Sequence, Text, Union


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
  bsuite_df = pd.DataFrame(data)

  return pd.merge(df, bsuite_df, on='bsuite_id')


def load_one_result_set(results_dir: Text) -> pd.DataFrame:
  """Returns a pandas DataFrame of bsuite results."""
  data = []
  for file_path in glob.glob(os.path.join(results_dir, '*.csv')):
    _, name = os.path.split(file_path)
    # Rough and ready error-checking for only bsuite csv files.
    if not name.startswith(csv_logging.BSUITE_PREFIX):
      print('Warning - we recommend you use a fresh folder for bsuite results.')
      continue

    # Then we will assume that the file is actually a bsuite file
    df = pd.read_csv(file_path)
    file_bsuite_id = name.strip('.csv').split('=')[1]
    bsuite_id = file_bsuite_id.replace(
        csv_logging.SAFE_SEPARATOR, sweep.SEPARATOR)
    df['bsuite_id'] = bsuite_id
    df['results_dir'] = results_dir
    data.append(df)
  df = pd.concat(data, sort=False)
  return join_metadata(df)

DirCollection = Union[Text, Sequence[Text], Mapping[Text, Text]]


def load_bsuite(results_dirs: DirCollection) -> pd.DataFrame:
  """Returns a pandas DataFrame of bsuite results.

  Args:
    results_dirs: Paths to one or more directories with bsuite results.
      be given as one of:
        - A sequence (e.g. list, tuple) of paths
        - a mapping from agent or algorithm name to path.
        - A string containing a single path.

  Returns:
    A tuple of:
      - A pandas DataFrame containing the bsuite results.
      - A list of column names to group by, used by the provided notebook.
        When grouping by these columns, each group corresponds to one set of
        results.
  """
  # Convert any inputs to mapping format.
  if isinstance(results_dirs, six.string_types):
    results_dirs = {results_dirs: results_dirs}
  if not isinstance(results_dirs, collections.Mapping):
    results_dirs = {path: path for path in results_dirs}

  data = []
  for name, path in results_dirs.items():
    df = load_one_result_set(results_dir=path)
    df['agent_name'] = name
    data.append(df)

  sweep_vars = ['agent_name']
  return pd.concat(data, sort=False), sweep_vars
