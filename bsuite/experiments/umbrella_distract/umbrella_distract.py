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
"""Unbrella chain environment with varying distractor observations."""

from bsuite.environments import umbrella_chain
from bsuite.experiments.umbrella_distract import sweep


def load(n_distractor: int, seed=0):
  """Load a deep sea experiment with the prescribed settings."""
  env = umbrella_chain.UmbrellaChain(
      chain_length=20,
      n_distractor=n_distractor,
      seed=seed,
  )
  env.bsuite_num_episodes = sweep.NUM_EPISODES
  return env
