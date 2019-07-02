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
"""An agent that takes uniformly random actions."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.baselines import base
import dm_env
from dm_env import specs

import numpy as np

from typing import Optional


class Random(base.Agent):
  """A random agent."""

  def __init__(self,
               action_spec: specs.DiscreteArray,
               seed: Optional[int] = None):
    self._num_actions = action_spec.num_values
    self._rng = np.random.RandomState(seed)

  def policy(self, timestep: dm_env.TimeStep) -> base.Action:
    del timestep
    return self._rng.randint(self._num_actions)

  def update(self,
             timestep: dm_env.TimeStep,
             action: base.Action,
             new_timestep: dm_env.TimeStep) -> None:
    del timestep
    del action
    del new_timestep
