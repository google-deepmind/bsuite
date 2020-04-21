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
"""Cartpole environment with noisy rewards."""

from bsuite.environments import cartpole
from bsuite.experiments.cartpole_noise import sweep
from bsuite.utils import wrappers


def load(noise_scale, seed):
  """Load a cartpole experiment with the prescribed settings."""
  env = wrappers.RewardNoise(
      env=cartpole.Cartpole(seed=seed),
      noise_scale=noise_scale,
      seed=seed)
  env.bsuite_num_episodes = sweep.NUM_EPISODES
  return env
