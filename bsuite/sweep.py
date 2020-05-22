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
"""This module exposes the bsuite experiment definitions in terms of bsuite_ids.

Each bsuite_id is designed to be a human readable string in the format:

    environment_name/i

where i is the index of the setting in that experiments sweep.py file.

Each bsuite_id can be used to load an environment, via the bsuite.load*
functions.

To iterate over the bsuite_ids for all experiments, use `sweep.SWEEP`.

To iterate over the bsuite_ids for a single experiment, use
`sweep.<EXPERIMENT_NAME>``. For example, `sweep.DISCOUNTING_CHAIN`.

To iterate over the bsuite_ids for a single experiment type, use
`sweep.TAGS[<EXPERIMENT_TAG>]`. For example, `sweep.TAGS['memory'].
"""

from typing import Any, Dict, Mapping, Tuple

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
from bsuite.experiments.memory_len import sweep as memory_len_sweep
from bsuite.experiments.memory_size import sweep as memory_size_sweep
from bsuite.experiments.mnist import sweep as mnist_sweep
from bsuite.experiments.mnist_noise import sweep as mnist_noise_sweep
from bsuite.experiments.mnist_scale import sweep as mnist_scale_sweep
from bsuite.experiments.mountain_car import sweep as mountain_car_sweep
from bsuite.experiments.mountain_car_noise import sweep as mountain_car_noise_sweep
from bsuite.experiments.mountain_car_scale import sweep as mountain_car_scale_sweep
from bsuite.experiments.umbrella_distract import sweep as umbrella_distract_sweep
from bsuite.experiments.umbrella_length import sweep as umbrella_length_sweep

import frozendict

# Common types aliases.
BSuiteId = str  # Experiment bsuite_ids are strings, e.g. 'deep_sea/10'.
Tag = str  # Experiment tags are strings, e.g. 'exploration'.
EnvKWargs = Dict[str, Any]  # Keyword arguments to environment constructors.

# bsuite_ids are strings of the form {environment_name}{SEPARATOR}{index}.
SEPARATOR = '/'
# Exclude environment names ending with the following from the testing sweep.
IGNORE_FOR_TESTING = ('_noise', '_scale')

_SETTINGS = {}
_SWEEP = []
_TAGS = {}
_TESTING = []
_EPISODES = {}


def _parse_sweep(experiment_package) -> Tuple[BSuiteId, ...]:
  """Returns the bsuite_ids for each experiment package."""
  results = []
  # package.__name__ is something like 'bsuite.experiments.bandit.sweep'
  experiment_name = experiment_package.__name__.split('.')[-2]
  eligible_for_test_sweep = not any(experiment_name.endswith(s)
                                    for s in IGNORE_FOR_TESTING)

  # Construct bsuite_ids for each setting defined by the experiment.
  for i, setting in enumerate(experiment_package.SETTINGS):
    bsuite_id = f'{experiment_name}{SEPARATOR}{i}'
    if i == 0 and eligible_for_test_sweep:
      # For each environment, add one `bsuite_id` to the TESTING sweep.
      _TESTING.append(bsuite_id)
    results.append(bsuite_id)
    _SETTINGS[bsuite_id] = setting
    _EPISODES[bsuite_id] = experiment_package.NUM_EPISODES

  # Add bsuite_ids to corresponding tag sweeps.
  for tag in experiment_package.TAGS:
    if tag not in _TAGS:
      _TAGS[tag] = []
    _TAGS[tag].extend(results)
  _SWEEP.extend(results)
  return tuple(results)


# bsuite_ids broken down by environment.
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
MEMORY_LEN = _parse_sweep(memory_len_sweep)
MEMORY_SIZE = _parse_sweep(memory_size_sweep)
MNIST = _parse_sweep(mnist_sweep)
MNIST_NOISE = _parse_sweep(mnist_noise_sweep)
MNIST_SCALE = _parse_sweep(mnist_scale_sweep)
MOUNTAIN_CAR = _parse_sweep(mountain_car_sweep)
MOUNTAIN_CAR_NOISE = _parse_sweep(mountain_car_noise_sweep)
MOUNTAIN_CAR_SCALE = _parse_sweep(mountain_car_scale_sweep)
UMBRELLA_DISTRACT = _parse_sweep(umbrella_distract_sweep)
UMBRELLA_LENGTH = _parse_sweep(umbrella_length_sweep)

# Mapping from bsuite id to keyword arguments for the corresponding environment.
SETTINGS: Mapping[BSuiteId, EnvKWargs] = frozendict.frozendict(**_SETTINGS)

# Tuple containing all bsuite_ids. Used for hyperparameter sweeps.
SWEEP: Tuple[BSuiteId, ...] = tuple(_SWEEP)

# Mapping from tag (e.g. 'memory') to experiment `bsuite_id`s with that tag.
TAGS: Mapping[Tag, Tuple[BSuiteId, ...]] = frozendict.frozendict(
    **{k: tuple(v) for k, v in _TAGS.items()})

# Tuple containing a representative subset bsuite_ids used for agent tests.
TESTING: Tuple[BSuiteId, ...] = tuple(_TESTING)

# Mapping from bsuite_id to bsuite_num_episodes = how many episodes to run.
EPISODES: Mapping[BSuiteId, int] = frozendict.frozendict(**_EPISODES)
