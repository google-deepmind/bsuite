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

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.experiments.memory_len import sweep
from bsuite.utils import plotting
import numpy as np
import pandas as pd
import plotnine as gg

from typing import Text, Sequence

NUM_EPISODES = sweep.NUM_EPISODES
TAGS = ('memory',)
LEARNING_THRESH = 0.75


def memory_preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
  """Preprocess data for memory environments = regret relative to random."""
  df = df_in.copy()
  df['perfection_regret'] = df.episode - df.total_perfect
  if df.bsuite_env == 'memory_len':  # Then we know it is a single bit.
    df['num_bits'] = df['num_bits'].fillna(1.)
  df['base_rate'] = 1 - 0.5 ** df.num_bits
  df['regret_ratio'] = df.perfection_regret / df.base_rate
  return df


def score(df: pd.DataFrame, group_col: Text = 'memory_length') -> float:
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
                  sweep_vars: Sequence[Text] = None,
                  group_col: Text = 'memory_length') -> gg.ggplot:
  """Plots the average return through time by memory_length."""
  df = memory_preprocess(df_in=df)
  p = plotting.plot_regret_group_nosmooth(
      df_in=df,
      group_col=group_col,
      sweep_vars=sweep_vars,
      regret_col='regret_ratio',
      max_episode=sweep.NUM_EPISODES,
  )
  return p + gg.ylab('average % of imperfect episodes compare to random.')


def plot_scale(df: pd.DataFrame,
               sweep_vars: Sequence[Text] = None,
               group_col: Text = 'memory_length') -> gg.ggplot:
  """Plots the average return through time by memory_length."""
  df = memory_preprocess(df_in=df)
  p = plotting.plot_regret_ave_scaling(
      df_in=df,
      group_col=group_col,
      episode=sweep.NUM_EPISODES,
      regret_thresh=LEARNING_THRESH,
      sweep_vars=sweep_vars,
      regret_col='regret_ratio'
  )
  return p + gg.ylab('% imperfect episodes after {} episodes compared to random'
                     .format(sweep.NUM_EPISODES))
