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
"""Defines a full sweep over all bsuite experiments.

Each bsuite_id is designed to be a human readable string in the format:

    environment_name/i

where i is the index of the setting in that experiments sweep.py file.

This helps with interpretability on dashboards, and for any users manually
specifying what to run from a script or a command line.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

# pylint: disable=unused-import
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
from bsuite.experiments.mountain_car import sweep as mountain_car_sweep
from bsuite.experiments.mountain_car_noise import sweep as mountain_car_noise_sweep
from bsuite.experiments.mountain_car_scale import sweep as mountain_car_scale_sweep
from bsuite.experiments.nonstationary_bandit import sweep as nonstationary_bandit_sweep
from bsuite.experiments.nonstationary_rps import sweep as nonstationary_rps_sweep
from bsuite.experiments.umbrella_distract import sweep as umbrella_distract_sweep
from bsuite.experiments.umbrella_length import sweep as umbrella_length_sweep
# pylint: enable=unused-import

SEPARATOR = '/'

# "Register" all the imported experiments, by creating a mapping from experiment
# name to the imported module.
_suffix_length = len('_sweep')
_experiment_modules = {
    name[:-_suffix_length]: module for name, module in locals().items()
    # Only select variables that _are_ experiment_name/sweep.py modules.
    if inspect.ismodule(module) and hasattr(module, 'SETTINGS')
}
# Create public containers for iterating over bsuite experiments.

# Mapping from bsuite id to keyword arguments for the corresponding environment.
SETTINGS = {}

# Sequence of all bsuite ids.
_all_bsuite_ids = []

for experiment_name, experiment_module in _experiment_modules.items():
  # Make an uppercase constant containing bsuite ids for each experiment.
  per_experiment_bsuite_ids = []
  locals()[experiment_name.upper()] = per_experiment_bsuite_ids

  for i, setting in enumerate(experiment_module.SETTINGS):
    bsuite_id = '{}/{}'.format(experiment_name, i)
    per_experiment_bsuite_ids.append(bsuite_id)
    _all_bsuite_ids.append(bsuite_id)
    SETTINGS[bsuite_id] = setting

# Sorted tuple containing all bsuite_id. Used for hyperparameter sweeps.
SWEEP = tuple(sorted(_all_bsuite_ids))
