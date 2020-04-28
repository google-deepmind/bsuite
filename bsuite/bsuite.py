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
"""Functions to load bsuite environments."""

from typing import Any, Mapping, Tuple

from bsuite import sweep
from bsuite.environments import base
from bsuite.experiments.bandit import bandit
from bsuite.experiments.bandit_noise import bandit_noise
from bsuite.experiments.bandit_scale import bandit_scale
from bsuite.experiments.cartpole import cartpole
from bsuite.experiments.cartpole_noise import cartpole_noise
from bsuite.experiments.cartpole_scale import cartpole_scale
from bsuite.experiments.cartpole_swingup import cartpole_swingup
from bsuite.experiments.catch import catch
from bsuite.experiments.catch_noise import catch_noise
from bsuite.experiments.catch_scale import catch_scale
from bsuite.experiments.deep_sea import deep_sea
from bsuite.experiments.deep_sea_stochastic import deep_sea_stochastic
from bsuite.experiments.discounting_chain import discounting_chain
from bsuite.experiments.memory_len import memory_len
from bsuite.experiments.memory_size import memory_size
from bsuite.experiments.mnist import mnist
from bsuite.experiments.mnist_noise import mnist_noise
from bsuite.experiments.mnist_scale import mnist_scale
from bsuite.experiments.mountain_car import mountain_car
from bsuite.experiments.mountain_car_noise import mountain_car_noise
from bsuite.experiments.mountain_car_scale import mountain_car_scale
from bsuite.experiments.umbrella_distract import umbrella_distract
from bsuite.experiments.umbrella_length import umbrella_length

from bsuite.logging import csv_logging
from bsuite.logging import sqlite_logging
from bsuite.logging import terminal_logging

import dm_env
import termcolor

# Internal imports.

# Mapping from experiment name to environment constructor or load function.
# Each constructor or load function accepts keyword arguments as defined in
# each experiment's sweep.py file.
EXPERIMENT_NAME_TO_ENVIRONMENT = dict(
    bandit=bandit.load,
    bandit_noise=bandit_noise.load,
    bandit_scale=bandit_scale.load,
    cartpole=cartpole.load,
    cartpole_noise=cartpole_noise.load,
    cartpole_scale=cartpole_scale.load,
    cartpole_swingup=cartpole_swingup.CartpoleSwingup,
    catch=catch.load,
    catch_noise=catch_noise.load,
    catch_scale=catch_scale.load,
    deep_sea=deep_sea.load,
    deep_sea_stochastic=deep_sea_stochastic.load,
    discounting_chain=discounting_chain.load,
    memory_len=memory_len.load,
    memory_size=memory_size.load,
    mnist=mnist.load,
    mnist_noise=mnist_noise.load,
    mnist_scale=mnist_scale.load,
    mountain_car=mountain_car.load,
    mountain_car_noise=mountain_car_noise.load,
    mountain_car_scale=mountain_car_scale.load,
    umbrella_distract=umbrella_distract.load,
    umbrella_length=umbrella_length.load,
)


def unpack_bsuite_id(bsuite_id: str) -> Tuple[str, int]:
  """Returns the experiment name and setting index given a bsuite_id."""
  parts = bsuite_id.split(sweep.SEPARATOR)
  assert len(parts) == 2
  experiment_name = parts[0]
  setting_index = int(parts[1])
  return experiment_name, setting_index


def load(
    experiment_name: str,
    kwargs: Mapping[str, Any],
) -> base.Environment:
  """Returns a bsuite environment given an experiment name and settings."""
  return EXPERIMENT_NAME_TO_ENVIRONMENT[experiment_name](**kwargs)


def load_from_id(bsuite_id: str) -> base.Environment:
  """Returns a bsuite environment given a bsuite_id."""
  kwargs = sweep.SETTINGS[bsuite_id]
  experiment_name, _ = unpack_bsuite_id(bsuite_id)
  env = load(experiment_name, kwargs)
  termcolor.cprint(
      f'Loaded bsuite_id: {bsuite_id}.', color='white', attrs=['bold'])
  return env


def load_and_record(bsuite_id: str,
                    save_path: str,
                    logging_mode: str = 'csv',
                    overwrite: bool = False) -> dm_env.Environment:
  """Returns a bsuite environment wrapped with either CSV or SQLite logging."""
  if logging_mode == 'csv':
    return load_and_record_to_csv(bsuite_id, save_path, overwrite)
  elif logging_mode == 'sqlite':
    if not save_path.endswith('.db'):
      save_path += '.db'
    if overwrite:
      print('WARNING: overwrite option is ignored for SQLite logging.')
    return load_and_record_to_sqlite(bsuite_id, save_path)
  elif logging_mode == 'terminal':
    return load_and_record_to_terminal(bsuite_id)
  else:
    raise ValueError((f'Unrecognised logging_mode "{logging_mode}". '
                      'Must be "csv", "sqlite", or "terminal".'))


def load_and_record_to_sqlite(bsuite_id: str,
                              db_path: str) -> dm_env.Environment:
  """Returns a bsuite environment that saves results to an SQLite database.

  The returned environment will automatically save the results required for
  the analysis notebook when stepping through the environment.

  To load the results, specify the file path in the provided notebook, or to
  manually inspect the results use:

  ```python
  from bsuite.utils import sqlite_load

  results_df, sweep_vars = sqlite_load.load_bsuite('/path/to/database.db')
  ```

  Args:
    bsuite_id: The bsuite id identifying the environment to return. For example,
      "catch/0" or "deep_sea/3".
    db_path: Path to the database file for this set of results. The file will be
      created if it does not already exist. When generating results using
      multiple different processes, specify the *same* db_path for every
      bsuite_id.

  Returns:
    A bsuite environment determined by the bsuite_id.
  """
  raw_env = load_from_id(bsuite_id)
  experiment_name, setting_index = unpack_bsuite_id(bsuite_id)
  termcolor.cprint(
      f'Logging results to SQLite database in {db_path}.',
      color='yellow',
      attrs=['bold'])
  return sqlite_logging.wrap_environment(
      env=raw_env,
      db_path=db_path,
      experiment_name=experiment_name,
      setting_index=setting_index,
  )


def load_and_record_to_csv(bsuite_id: str,
                           results_dir: str,
                           overwrite: bool = False) -> dm_env.Environment:
  """Returns a bsuite environment that saves results to CSV.

  To load the results, specify the file path in the provided notebook, or to
  manually inspect the results use:

  ```python
  from bsuite.utils import csv_load

  results_df, sweep_vars = csv_load.load_bsuite(results_dir)
  ```

  Args:
    bsuite_id: The bsuite id identifying the environment to return. For example,
      "catch/0" or "deep_sea/3".
    results_dir: Path to the directory to store the resultant CSV files. Note
      that this logger will generate a separate CSV file for each bsuite_id.
    overwrite: Whether to overwrite existing CSV files if found.

  Returns:
    A bsuite environment determined by the bsuite_id.
  """
  raw_env = load_from_id(bsuite_id)
  termcolor.cprint(
      f'Logging results to CSV file for each bsuite_id in {results_dir}.',
      color='yellow',
      attrs=['bold'])
  return csv_logging.wrap_environment(
      env=raw_env,
      bsuite_id=bsuite_id,
      results_dir=results_dir,
      overwrite=overwrite,
  )


def load_and_record_to_terminal(bsuite_id: str) -> dm_env.Environment:
  """Returns a bsuite environment that logs to terminal."""
  raw_env = load_from_id(bsuite_id)
  termcolor.cprint(
      'Logging results to terminal.', color='yellow', attrs=['bold'])
  return terminal_logging.wrap_environment(raw_env)
