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

import collections
import copy
from typing import Any, Callable, List, Mapping, Sequence, Tuple, Union

from bsuite import sweep
import pandas as pd
import six


def join_metadata(df: pd.DataFrame) -> pd.DataFrame:
  """Returns a DataFrame with bsuite sweep metadata joined on bsuite_id."""
  # Assume we are loading in the settings via sweep.py, without any changes.
  assert 'bsuite_id' in df.columns
  metadata = copy.deepcopy(sweep.SETTINGS)  # be careful not to change this

  data = []
  for bsuite_id, env_kwargs in metadata.items():
    # Add environment and id to dataframe.
    bsuite_env = bsuite_id.split(sweep.SEPARATOR)[0]
    bsuite_params = {
        'bsuite_id': bsuite_id,
        'bsuite_env': bsuite_env,
    }
    bsuite_params.update(env_kwargs)
    data.append(bsuite_params)
  bsuite_df = pd.DataFrame(data)

  return pd.merge(df, bsuite_df, on='bsuite_id')


PathCollection = Union[str, Sequence[str], Mapping[str, Any]]
SingleLoadFn = Callable[[str], pd.DataFrame]


def load_multiple_runs(
    path_collection: PathCollection,
    single_load_fn: SingleLoadFn) -> Tuple[pd.DataFrame, List[str]]:
  """Returns a pandas DataFrame of bsuite results.

  Args:
    path_collection: Paths to one or more locations of bsuite results.
      be given as one of: - A sequence (e.g. list, tuple) of paths - a mapping
        from agent or algorithm name to path. - A string containing a single
        path.
    single_load_fn: A function that takes in a single path (as specified in the
      path_collection and loads the bsuite results for one agent run).

  Returns:
    A tuple of:
      - A pandas DataFrame containing the bsuite results.
      - A list of column names to group by, used in ipython notebook provided.
        When grouping by these columns, each group corresponds to one set of
        results.
  """
  # Convert any inputs to dictionary format.
  if isinstance(path_collection, six.string_types):
    path_collection = {path_collection: path_collection}
  if not isinstance(path_collection, collections.Mapping):
    path_collection = {path: path for path in path_collection}

  # Loop through multiple bsuite runs, and apply single_load_fn to each.
  data = []
  for name, path in path_collection.items():
    df = single_load_fn(path)
    df['agent_name'] = name
    data.append(df)

  sweep_vars = ['agent_name']
  return pd.concat(data, sort=False), sweep_vars
