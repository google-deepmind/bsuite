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
"""Sweep definition for bandit_scale experiment."""

from bsuite.experiments.bandit import sweep as bandit_sweep

NUM_EPISODES = bandit_sweep.NUM_EPISODES

_settings = []
for scale in [0.001, 0.03, 1.0, 30., 1000.]:
  for seed in range(4):
    _settings.append({'reward_scale': scale, 'seed': seed})

SETTINGS = tuple(_settings)
TAGS = ('scale',)
