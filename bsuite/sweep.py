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
"""Defines a full sweep over all bsuite experiments.

Each bsuite_id is designed to be a human readable string in the format:

    environment_name/i

where i is the index of the setting in that experiments sweep.py file.

This helps with interpretability on dashboards, and for any users manually
specifying what to run from a script or a command line.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.experiments.bandit import sweep as bandit_sweep
from bsuite.experiments.bandit_noise import sweep as bandit_noise_sweep
from bsuite.experiments.bandit_scale import sweep as bandit_scale_sweep
from bsuite.experiments.cartpole import sweep as cartpole_sweep
from bsuite.experiments.cartpole_noise import sweep as cartpole_noise_sweep
from bsuite.experiments.cartpole_scale import sweep as cartpole_scale_sweep
from bsuite.experiments.cartpole_swingup import sweep as cartpole_swingup_sweep
from bsuite.experiments.catch import sweep as catch_sweep
from bsuite.experiments.catch_noise import sweep as catch_noise_sweep
from bsuite.experiments.catch_scale import sweep as catch_scale_sweep
from bsuite.experiments.deep_sea import sweep as deep_sea_sweep
from bsuite.experiments.deep_sea_stochastic import sweep as deep_sea_stochastic_sweep
from bsuite.experiments.discounting_chain import sweep as discounting_chain_sweep
from bsuite.experiments.hierarchy_sea import sweep as hierarchy_sea_sweep
from bsuite.experiments.hierarchy_sea_explore import sweep as hierarchy_sea_explore_sweep
from bsuite.experiments.memory_len import sweep as memory_len_sweep
from bsuite.experiments.memory_size import sweep as memory_size_sweep
from bsuite.experiments.mnist import sweep as mnist_sweep
from bsuite.experiments.mnist_noise import sweep as mnist_noise_sweep
from bsuite.experiments.mnist_scale import sweep as mnist_scale_sweep
from bsuite.experiments.mountain_car import sweep as mountain_car_sweep
from bsuite.experiments.mountain_car_noise import sweep as mountain_car_noise_sweep
from bsuite.experiments.mountain_car_scale import sweep as mountain_car_scale_sweep
from bsuite.experiments.nonstationary_bandit import sweep as nonstationary_bandit_sweep
from bsuite.experiments.nonstationary_rps import sweep as nonstationary_rps_sweep
from bsuite.experiments.umbrella_distract import sweep as umbrella_distract_sweep
from bsuite.experiments.umbrella_length import sweep as umbrella_length_sweep

from typing import Text, Tuple

SEPARATOR = '/'
_SWEEP = []

# Mapping from bsuite id to keyword arguments for the corresponding environment.
SETTINGS = {}


def _parse_sweep(package) -> Tuple[Text]:
  """Returns the bsuite_id for each package."""
  results = []
  # package.__name__ is something like 'bsuite.experiments.bandit.sweep'
  experiment_name = package.__name__.split('.')[-2]
  for i, setting in enumerate(package.SETTINGS):
    bsuite_id = experiment_name + SEPARATOR + str(i)
    results.append(bsuite_id)
    SETTINGS[bsuite_id] = setting
  _SWEEP.extend(results)
  return tuple(results)

BANDIT = _parse_sweep(bandit_sweep)
BANDIT_NOISE = _parse_sweep(bandit_noise_sweep)
BANDIT_SCALE = _parse_sweep(bandit_scale_sweep)
CARTPOLE = _parse_sweep(cartpole_sweep)
CARTPOLE_NOISE = _parse_sweep(cartpole_noise_sweep)
CARTPOLE_SCALE = _parse_sweep(cartpole_scale_sweep)
CARTPOLE_SWINGUP = _parse_sweep(cartpole_swingup_sweep)
CATCH = _parse_sweep(catch_sweep)
CATCH_NOISE = _parse_sweep(catch_noise_sweep)
CATCH_SCALE = _parse_sweep(catch_scale_sweep)
DEEP_SEA = _parse_sweep(deep_sea_sweep)
DEEP_SEA_STOCHASTIC = _parse_sweep(deep_sea_stochastic_sweep)
DISCOUNTING_CHAIN = _parse_sweep(discounting_chain_sweep)
HIERARCHY_SEA = _parse_sweep(hierarchy_sea_sweep)
HIERARCHY_SEA_EXPLORE = _parse_sweep(hierarchy_sea_explore_sweep)
MEMORY_LEN = _parse_sweep(memory_len_sweep)
MEMORY_SIZE = _parse_sweep(memory_size_sweep)
MNIST = _parse_sweep(mnist_sweep)
MNIST_NOISE = _parse_sweep(mnist_noise_sweep)
MNIST_SCALE = _parse_sweep(mnist_scale_sweep)
MOUNTAIN_CAR = _parse_sweep(mountain_car_sweep)
MOUNTAIN_CAR_NOISE = _parse_sweep(mountain_car_noise_sweep)
MOUNTAIN_CAR_SCALE = _parse_sweep(mountain_car_scale_sweep)
NONSTATIONARY_BANDIT = _parse_sweep(nonstationary_bandit_sweep)
NONSTATIONARY_RPS = _parse_sweep(nonstationary_rps_sweep)
UMBRELLA_DISTRACT = _parse_sweep(umbrella_distract_sweep)
UMBRELLA_LENGTH = _parse_sweep(umbrella_length_sweep)

# Tuple containing all bsuite_id. Used for hyperparameter sweeps.
SWEEP = tuple(_SWEEP)


def num_episodes(bsuite_id: Text) -> int:
  """Returns the number of episodes needed for the corresponding experiment."""
  experiment_name = bsuite_id.split(SEPARATOR)[0]
  module = globals()[experiment_name + '_sweep']
  return module.NUM_EPISODES
