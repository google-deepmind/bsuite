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
"""Analysis for cartpole swingup."""

from typing import Optional, Sequence

from bsuite.experiments.cartpole_swingup import sweep
from bsuite.utils import plotting
import numpy as np
import pandas as pd
import plotnine as gg

NUM_EPISODES = sweep.NUM_EPISODES
BASE_REGRET = 700
GOOD_EPISODE = 100
TAGS = sweep.TAGS


def cp_swingup_preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
  """Preprocess data for cartpole swingup."""
  df = df_in.copy()
  df = df[df.episode <= NUM_EPISODES]
  df['perfection_regret'] = df.episode * BASE_REGRET - df.total_return
  return df


def score(df: pd.DataFrame) -> float:
  """Output a single score for swingup = 50% regret, 50% does a swingup."""
  df = cp_swingup_preprocess(df_in=df)
  scores = []
  for _, sub_df in df.groupby('height_threshold'):
    regret_score = plotting.ave_regret_score(
        sub_df,
        baseline_regret=BASE_REGRET,
        episode=NUM_EPISODES,
        regret_column='perfection_regret'
    )
    swingup_score = np.mean(
        sub_df.groupby('bsuite_id')['best_episode'].max() > GOOD_EPISODE)
    scores.append(0.5 * (regret_score + swingup_score))
  return np.mean(scores)


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Plots the average return through time by cartpole swingup."""
  df = cp_swingup_preprocess(df_in=df)
  p = plotting.plot_regret_group_nosmooth(
      df_in=df,
      group_col='height_threshold',
      sweep_vars=sweep_vars,
      regret_col='perfection_regret',
      max_episode=sweep.NUM_EPISODES,
  )
  return p


def plot_scale(df: pd.DataFrame,
               sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Plots the best episode observed by height_threshold."""
  df = cp_swingup_preprocess(df_in=df)

  group_vars = ['height_threshold']
  if sweep_vars:
    group_vars += sweep_vars
  plt_df = df.groupby(group_vars)['best_episode'].max().reset_index()

  p = (gg.ggplot(plt_df)
       + gg.aes(x='factor(height_threshold)', y='best_episode',
                colour='best_episode > {}'.format(GOOD_EPISODE))
       + gg.geom_point(size=5, alpha=0.8)
       + gg.scale_colour_manual(values=['#d73027', '#313695'])
       + gg.geom_hline(gg.aes(yintercept=0.0), alpha=0)  # axis hack
       + gg.scale_x_discrete(breaks=[0, 0.25, 0.5, 0.75, 1.0])
       + gg.ylab('best return in first {} episodes'.format(NUM_EPISODES))
       + gg.xlab('height threshold')
      )
  return plotting.facet_sweep_plot(p, sweep_vars)


def plot_seeds(df_in: pd.DataFrame,
               sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Plot the returns through time individually by run."""
  df = df_in.copy()
  df['average_return'] = df.raw_return.diff() / df.episode.diff()
  p = plotting.plot_individual_returns(
      df_in=df[df.episode > 1],
      max_episode=NUM_EPISODES,
      return_column='average_return',
      colour_var='height_threshold',
      sweep_vars=sweep_vars,
  )
  return p + gg.ylab('average episodic return')
