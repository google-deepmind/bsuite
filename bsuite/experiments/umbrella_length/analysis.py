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
"""Analysis for umbrella_length experiment."""

from typing import Optional, Sequence

from bsuite.experiments.umbrella_length import sweep
from bsuite.utils import plotting
import numpy as np
import pandas as pd
import plotnine as gg


NUM_EPISODES = sweep.NUM_EPISODES
REGRET_THRESH = 0.5
TAGS = sweep.TAGS


def score_by_group(df: pd.DataFrame, group_col: str) -> float:
  """Output a single score for umbrella_chain."""
  regret_list = []  # Loop to handle partially-finished runs.
  for _, sub_df in df.groupby(group_col):
    max_eps = np.minimum(sub_df.episode.max(), sweep.NUM_EPISODES)
    ave_regret = (
        sub_df.loc[sub_df.episode == max_eps, 'total_regret'].mean() / max_eps)
    regret_list.append(ave_regret)
  return np.mean(np.array(regret_list) < REGRET_THRESH)


def score(df: pd.DataFrame) -> float:
  return score_by_group(df, group_col='chain_length')


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Plots the average regret through time."""
  return plotting.plot_regret_group_nosmooth(
      df_in=df,
      group_col='chain_length',
      sweep_vars=sweep_vars,
      max_episode=sweep.NUM_EPISODES,
  )


def plot_scale(df: pd.DataFrame,
               sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Plots the average return at end of learning investigating scaling."""
  return plotting.plot_regret_ave_scaling(
      df_in=df,
      group_col='chain_length',
      episode=sweep.NUM_EPISODES,
      regret_thresh=0.5,
      sweep_vars=sweep_vars,
  )


def plot_seeds(df_in: pd.DataFrame,
               sweep_vars: Optional[Sequence[str]] = None,
               colour_var: str = 'chain_length') -> gg.ggplot:
  """Plot the returns through time individually by run."""
  df = df_in.copy()
  df['average_return'] = 1.0 - (df.total_regret.diff() / df.episode.diff())
  p = plotting.plot_individual_returns(
      df_in=df,
      max_episode=NUM_EPISODES,
      return_column='average_return',
      colour_var=colour_var,
      yintercept=1.,
      sweep_vars=sweep_vars,
  )
  return p + gg.ylab('average episodic return')

