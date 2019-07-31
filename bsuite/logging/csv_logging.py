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
"""Logging functionality for local csv-based experiments."""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

import os

from bsuite import sweep
from bsuite.logging import base
from bsuite.utils import wrappers

import dm_env
import pandas as pd

from typing import Any, Mapping, Text

SAFE_SEPARATOR = '-'
INITIAL_SEPARATOR = '_-_'
BSUITE_PREFIX = 'bsuite_id' + INITIAL_SEPARATOR


def wrap_environment(env: dm_env.Environment,
                     bsuite_id: Text,
                     results_dir: Text,
                     overwrite: bool = False,
                     log_by_step: bool = False) -> dm_env.Environment:
  """Returns a wrapped environment that logs using csv."""
  logger = Logger(bsuite_id, results_dir, overwrite)
  return wrappers.Logging(env, logger, log_by_step=log_by_step)


class Logger(base.Logger):
  """Saves data to a CSV file via  Pandas.

  BSuite is split into multiple _experiments_. Each experiment has multiple
  _workers_, which may correspond to different random seeds or level of
  difficulty. The setting id is the index of the corresponding setting as
  defined in each experiment's sweep.py file.

  In this simplified logger, each bsuite_id logs to a unique csv index by
  bsuite_id. These are saved to a single results_dir by experiment.
  We strongly suggest that you use a *fresh* folder for each bsuite run.

  This logger, along with the corresponding load functionality, serves as a
  simple, minimal example for users who need to implement logging to a different
  storage system.
  """

  def __init__(self,
               bsuite_id: Text,
               results_dir: Text = '/tmp/bsuite',
               overwrite: bool = False):
    """Initializes a new CSV logger, saving to results_dir + overwrite data."""
    self._data = []

    if not os.path.exists(results_dir):
      try:
        os.makedirs(results_dir)
      except OSError:  # concurrent processes can makedir at same time
        pass

    # The default '/' symbol is dangerous for file systems!
    safe_bsuite_id = bsuite_id.replace(sweep.SEPARATOR, SAFE_SEPARATOR)
    filename = '{}{}.csv'.format(BSUITE_PREFIX, safe_bsuite_id)
    self._save_path = os.path.join(results_dir, filename)

    if os.path.exists(self._save_path) and not overwrite:
      msg = ('File {} already exists. Specify a different directory, or set'
             ' overwrite=True to overwrite existing data.')
      raise ValueError(msg.format(self._save_path))

  def write(self, data: Mapping[Text, Any]):
    """Writes a row to the internal list of data and saves to CSV."""
    # This code rewrites the entire CSV file each time. This is not intended to
    # be an optimized example. However, writes happen infrequently due to
    # bsuite's logarithmically-spaced logging.
    self._data.append(data)
    df = pd.DataFrame(self._data)
    df.to_csv(self._save_path, index=False)
