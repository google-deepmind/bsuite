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
"""A simple agent-environment training loop."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from absl import logging

from bsuite.baselines import base
import dm_env
import numpy as np
import pandas as pd

from typing import Sequence


def run(agent: base.Agent,
        environment: dm_env.Environment,
        num_episodes: int = None,
        verbose: bool = False) -> pd.DataFrame:
  """Runs an agent on an environment and returns basic performance data."""
  if num_episodes is None:
    num_episodes = float('inf')

  results = []
  episode = 1
  total_return = 0
  while episode <= num_episodes:
    # Reset the environment.
    timestep = environment.reset()
    episode_len = 0
    episode_return = 0

    # Run an episode.
    while not timestep.last():
      # Generate an action from the agent's policy.
      action = agent.policy(timestep)

      # Step the environment.
      new_timestep = environment.step(action)

      # Tell the agent about what just happened.
      agent.update(timestep, action, new_timestep)

      # Book-keeping.
      timestep = new_timestep
      episode_len += 1
      episode_return += new_timestep.reward

    # Collect and log the results.
    total_return += episode_return
    result = {
        'episode': episode,
        'episode_len': episode_len,
        'episode_return': episode_return,
        'total_return': total_return,
    }
    if verbose:
      logging.info(result)
    if logarithmic_logging(episode):  # Only log at exponential intervals.
      results.append(result)
    episode += 1

  return pd.DataFrame(results)


def logarithmic_logging(episode: int, ratios: Sequence[float] = None) -> bool:
  """Returns `True` only at specific ratios of 10**exponent."""
  if ratios is None:
    ratios = [1., 1.2, 1.4, 1.7, 2., 2.5, 3., 4., 5., 6., 7., 8., 9., 10.]
  exponent = np.floor(np.log10(np.maximum(1, episode)))
  special_vals = [10**exponent * ratio for ratio in ratios]
  do_log = any([episode == val for val in special_vals])
  return do_log
