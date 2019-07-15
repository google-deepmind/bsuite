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
"""Simple diagnostic memory challenge.

Observation is given by n+1 pixels: (context, time_to_live).

Context will only be nonzero in the first step, when it will be +1 or -1 iid
by component. All actions take no effect until time_to_live=0, then the agent
must repeat the observations that it saw bit-by-bit.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.experiments.memory_len import memory_len
from bsuite.experiments.memory_size import sweep


def load(num_bits):
  """Memory Chain environment, with variable number of bits."""
  env = memory_len.MemoryChain(
      memory_length=5,
      num_bits=num_bits,
      seed=73,
  )
  env.bsuite_num_episodes = sweep.NUM_EPISODES
  return env

