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

from typing import Optional

from bsuite.environments import memory_chain
from bsuite.experiments.memory_len import sweep


def load(memory_length: int, seed: Optional[int] = 0):
  """Memory Chain environment, with variable delay between cue and decision."""
  env = memory_chain.MemoryChain(
      memory_length=memory_length,
      num_bits=1,
      seed=seed,
  )
  env.bsuite_num_episodes = sweep.NUM_EPISODES
  return env
