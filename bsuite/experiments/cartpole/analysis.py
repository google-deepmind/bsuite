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
"""Analysis for cartpole."""

from typing import Sequence

from bsuite.experiments.cartpole import sweep
from bsuite.utils import plotting
import numpy as np
import pandas as pd
import plotnine as gg

NUM_EPISODES = sweep.NUM_EPISODES
BASE_REGRET = 1000
GOOD_EPISODE = 500
TAGS = sweep.TAGS


def score(df: pd.DataFrame) -> float:
  """Output a single score for cartpole = 50% regret, 50% has a good run."""
  cp_df = cartpole_preprocess(df_in=df)
  regret_score = plotting.ave_regret_score(
      cp_df, baseline_regret=BASE_REGRET, episode=NUM_EPISODES)

  # Give 50% of score if your "best" episode > GOOD_EPISODE threshold.
  solve_score = np.mean(
      cp_df.groupby('seed')['best_episode'].max() > GOOD_EPISODE)

  return 0.5 * (regret_score + solve_score)


def cartpole_preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
  """Preprocess cartpole data for use with regret metrics."""
  df = df_in.copy()
  df = df[df.episode <= NUM_EPISODES]
  df['total_regret'] = (BASE_REGRET * df.episode) - df.raw_return
  return df


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[str] = None) -> gg.ggplot:
  """Simple learning curves for cartpole."""
  df = cartpole_preprocess(df)
  p = plotting.plot_regret_learning(
      df, sweep_vars=sweep_vars, max_episode=NUM_EPISODES)
  p += gg.geom_hline(gg.aes(yintercept=BASE_REGRET),
                     linetype='dashed', alpha=0.4, size=1.75)
  return p


def plot_seeds(df_in: pd.DataFrame,
               sweep_vars: Sequence[str] = None,
               colour_var: str = None) -> gg.ggplot:
  """Plot the returns through time individually by run."""
  df = df_in.copy()
  df['average_return'] = df.raw_return.diff() / df.episode.diff()
  p = plotting.plot_individual_returns(
      df_in=df,
      max_episode=NUM_EPISODES,
      return_column='average_return',
      colour_var=colour_var,
      yintercept=BASE_REGRET,
      sweep_vars=sweep_vars,
  )
  return p + gg.ylab('average episodic return')
