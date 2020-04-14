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
"""Analysis for memory_len."""

from typing import Sequence

from bsuite.experiments.memory_len import sweep
from bsuite.utils import plotting
import numpy as np
import pandas as pd
import plotnine as gg

NUM_EPISODES = sweep.NUM_EPISODES
TAGS = sweep.TAGS
LEARNING_THRESH = 0.75


def memory_preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
  """Preprocess data for memory environments = regret relative to random."""
  df = df_in.copy()
  df['perfection_regret'] = df.episode - df.total_perfect
  # a random agent always has 50% chance on each episode
  # independently from memory length and number of bits.
  df['base_rate'] = 0.5
  df['regret_ratio'] = df.perfection_regret / df.base_rate
  return df


def score(df: pd.DataFrame, group_col: str = 'memory_length') -> float:
  """Output a single score for memory_len."""
  df = memory_preprocess(df_in=df)
  regret_list = []  # Loop to handle partially-finished runs.
  for _, sub_df in df.groupby(group_col):
    max_eps = np.minimum(sub_df.episode.max(), sweep.NUM_EPISODES)
    ave_perfection = (
        sub_df.loc[sub_df.episode == max_eps, 'regret_ratio'].mean() / max_eps)
    regret_list.append(ave_perfection)
  return np.mean(np.array(regret_list) < LEARNING_THRESH)


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[str] = None,
                  group_col: str = 'memory_length') -> gg.ggplot:
  """Plots the average return through time by memory_length."""
  df = memory_preprocess(df_in=df)
  p = plotting.plot_regret_group_nosmooth(
      df_in=df,
      group_col=group_col,
      sweep_vars=sweep_vars,
      regret_col='regret_ratio',
      max_episode=sweep.NUM_EPISODES,
  )
  return p + gg.ylab('average % of correct episodes compared to random.')


def plot_scale(df: pd.DataFrame,
               sweep_vars: Sequence[str] = None,
               group_col: str = 'memory_length') -> gg.ggplot:
  """Plots the regret_ratio through time by memory_length."""
  df = memory_preprocess(df_in=df)
  p = plotting.plot_regret_ave_scaling(
      df_in=df,
      group_col=group_col,
      episode=sweep.NUM_EPISODES,
      regret_thresh=LEARNING_THRESH,
      sweep_vars=sweep_vars,
      regret_col='regret_ratio'
  )
  return p + gg.ylab('% correct episodes after\n{} episodes compared to random'
                     .format(sweep.NUM_EPISODES))


def plot_seeds(df_in: pd.DataFrame,
               sweep_vars: Sequence[str] = None,
               colour_var: str = 'memory_length') -> gg.ggplot:
  """Plot the returns through time individually by run."""
  df = df_in.copy()
  df['average_return'] = df.total_return.diff() / df.episode.diff()
  p = plotting.plot_individual_returns(
      df_in=df[df.episode > 10],
      max_episode=NUM_EPISODES,
      return_column='average_return',
      colour_var=colour_var,
      sweep_vars=sweep_vars,
  )
  return p + gg.ylab('average episodic return')
