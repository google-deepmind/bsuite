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
"""Sweep definition for deep_sea_stochastic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bsuite.experiments.deep_sea import sweep as deep_sea_sweep

NUM_EPISODES = deep_sea_sweep.NUM_EPISODES

_log_spaced = []
_log_spaced.extend(range(5, 11))
_log_spaced.extend([12, 14, 17, 20, 25])
_log_spaced.extend(range(30, 55, 5))

SETTINGS = tuple({'size': n} for n in _log_spaced)

