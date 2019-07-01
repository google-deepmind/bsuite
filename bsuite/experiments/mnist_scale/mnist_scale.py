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
"""Simple diagnostic bandit_scale challenge.

Observation is a single pixel of 0 - this is an indep arm bandit problem!
Rewards are np.linspace(0, 1, 11) with no noise, but rescaled.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bsuite.experiments.bandit import bandit
from bsuite.utils import wrappers


def load(reward_scale, seed):
  """Load a bandit_scale experiment with the prescribed settings."""
  return wrappers.RewardScale(
      env=bandit.SimpleBandit(seed=seed),
      reward_scale=reward_scale,
      seed=seed)
