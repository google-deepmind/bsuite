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
"""Sweep definition for a balancing experiment in Cartpole."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bsuite.experiments.cartpole import sweep as cartpole_sweep

NUM_EPISODES = cartpole_sweep.NUM_EPISODES

SETTINGS = tuple({'seed': n, 'height_threshold': n / 20} for n in range(20))
