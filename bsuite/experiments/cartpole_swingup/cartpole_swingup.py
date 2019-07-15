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
"""A balancing experiment in Cartpole."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.experiments.cartpole import cartpole
from bsuite.experiments.cartpole_swingup import sweep
import numpy as np


def load(seed: int, height_threshold: float):
  """Load a bandit_scale experiment with the prescribed settings."""
  env = cartpole.Cartpole(
      seed=seed,
      height_threshold=height_threshold,
      initial_theta=np.pi,
      move_cost=0.)
  env.bsuite_num_episodes = sweep.NUM_EPISODES
  return env
