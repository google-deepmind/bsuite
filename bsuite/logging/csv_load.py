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

import glob
import os
from typing import List, Tuple

from bsuite import sweep
from bsuite.logging import csv_logging
from bsuite.logging import logging_utils
import pandas as pd


def load_one_result_set(results_dir: str) -> pd.DataFrame:
  """Returns a pandas DataFrame of bsuite results stored in results_dir."""
  data = []
  for file_path in glob.glob(os.path.join(results_dir, '*.csv')):
    _, name = os.path.split(file_path)
    # Rough and ready error-checking for only bsuite csv files.
    if not name.startswith(csv_logging.BSUITE_PREFIX):
      print('Warning - we recommend you use a fresh folder for bsuite results.')
      continue

    # Then we will assume that the file is actually a bsuite file
    df = pd.read_csv(file_path)
    file_bsuite_id = name.strip('.csv').split(csv_logging.INITIAL_SEPARATOR)[1]
    bsuite_id = file_bsuite_id.replace(csv_logging.SAFE_SEPARATOR,
                                       sweep.SEPARATOR)
    df['bsuite_id'] = bsuite_id
    df['results_dir'] = results_dir
    data.append(df)
  df = pd.concat(data, sort=False)
  return logging_utils.join_metadata(df)


def load_bsuite(
    results_dirs: logging_utils.PathCollection
) -> Tuple[pd.DataFrame, List[str]]:
  """Returns a pandas DataFrame of bsuite results."""
  return logging_utils.load_multiple_runs(
      path_collection=results_dirs,
      single_load_fn=load_one_result_set,
  )
