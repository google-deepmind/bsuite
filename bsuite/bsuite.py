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
"""Behaviour Suite.

Each environment defines extra "columns of interest" to log.
Each environment also has a basic 0-indexed discrete action space and a single
observation.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite import sweep
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
from bsuite.experiments.hierarchy_sea import hierarchy_sea
from bsuite.experiments.hierarchy_sea_explore import hierarchy_sea_explore
from bsuite.experiments.memory_len import memory_len
from bsuite.experiments.memory_size import memory_size
from bsuite.experiments.mnist import mnist
from bsuite.experiments.mnist_noise import mnist_noise
from bsuite.experiments.mnist_scale import mnist_scale
from bsuite.experiments.mountain_car import mountain_car
from bsuite.experiments.mountain_car_noise import mountain_car_noise
from bsuite.experiments.mountain_car_scale import mountain_car_scale
from bsuite.experiments.nonstationary_bandit import nonstationary_bandit
from bsuite.experiments.nonstationary_rps import nonstationary_rps
from bsuite.experiments.umbrella_distract import umbrella_distract
from bsuite.experiments.umbrella_length import umbrella_length

from bsuite.utils import sqlite_logging

import dm_env
from typing import Any, Mapping, Text, Tuple


# Mapping from experiment name to environment constructor or load function.
# Each constructor or load function accepts keyword arguments as defined in
# each experiment's sweep.py file.
EXPERIMENT_NAME_TO_ENVIRONMENT = dict(
    bandit=bandit.SimpleBandit,
    bandit_noise=bandit_noise.load,
    bandit_scale=bandit_scale.load,
    cartpole=cartpole.Cartpole,
    cartpole_noise=cartpole_noise.load,
    cartpole_scale=cartpole_scale.load,
    cartpole_swingup=cartpole_swingup.load,
    catch=catch.Catch,
    catch_noise=catch_noise.load,
    catch_scale=catch_scale.load,
    deep_sea=deep_sea.DeepSea,
    deep_sea_stochastic=deep_sea_stochastic.load,
    discounting_chain=discounting_chain.DiscountingChain,
    hierarchy_sea=hierarchy_sea.HierarchySea,
    hierarchy_sea_explore=hierarchy_sea_explore.load,
    memory_len=memory_len.MemoryChain,
    memory_size=memory_size.load,
    mnist=mnist.MNISTBandit,
    mnist_noise=mnist_noise.load,
    mnist_scale=mnist_scale.load,
    mountain_car=mountain_car.MountainCar,
    mountain_car_noise=mountain_car_noise.load,
    mountain_car_scale=mountain_car_scale.load,
    nonstationary_bandit=nonstationary_bandit.NonstationaryBandit,
    nonstationary_rps=nonstationary_rps.NonstationaryRPS,
    umbrella_distract=umbrella_distract.load,
    umbrella_length=umbrella_length.UmbrellaChain,
)


def unpack_bsuite_id(bsuite_id: Text) -> Tuple[Text, int]:
  """Returns the experiment name and setting index given a bsuite_id."""
  parts = bsuite_id.split(sweep.SEPARATOR)
  assert len(parts) == 2
  experiment_name = parts[0]
  setting_index = int(parts[1])
  return experiment_name, setting_index


def load(experiment_name: Text,
         kwargs: Mapping[Text, Any]) -> dm_env.Environment:
  """Returns a bsuite environment given an experiment name and settings."""
  return EXPERIMENT_NAME_TO_ENVIRONMENT[experiment_name](**kwargs)


def load_from_id(bsuite_id: Text) -> dm_env.Environment:
  """Returns a bsuite environment given a bsuite_id."""
  kwargs = sweep.SETTINGS[bsuite_id]
  experiment_name, _ = unpack_bsuite_id(bsuite_id)
  return load(experiment_name, kwargs)


def load_and_record_to_sqlite(bsuite_id: Text,
                              db_path: Text) -> dm_env.Environment:
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
  return sqlite_logging.wrap_environment(
      raw_env, db_path=db_path, experiment_name=experiment_name,
      setting_index=setting_index)
