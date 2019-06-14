# pylint: disable=g-bad-file-header
# Copyright 2019 The bsuite Authors. All Rights Reserved.
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
"""Analysis for Umbrella Length."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite import plotting
import numpy as np
import pandas as pd
import plotnine as gg

from typing import Sequence, Text


def score_by_group(df: pd.DataFrame, group_var: Text) -> float:
  """Output a single score for umbrella_chain."""
  n_eps = 10000
  regret_list = []  # Loop to handle partially-finished runs.
  for _, sub_df in df.groupby(group_var):
    max_eps = np.minimum(sub_df.episode.max(), n_eps)
    ave_regret = (
        sub_df.loc[sub_df.episode == max_eps, 'total_regret'].mean() / max_eps)
    regret_list.append(ave_regret)
  return np.mean(np.array(regret_list) < 0.5)


def plot_learning_by_group(df_in: pd.DataFrame,  # plot_learning
                           group_var: Text,
                           sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average regret through time by group_var."""
  df = df_in.copy()
  df['average_regret'] = df.total_regret / df.episode
  df['group'] = df[group_var].astype('category')
  p = (gg.ggplot(df[df.episode >= 100])
       + gg.aes('episode', 'average_regret', group='group',
                fill=group_var, colour=group_var)
       + gg.geom_line(size=2, alpha=0.75)
       + gg.geom_hline(
           gg.aes(yintercept=1.0), linetype='dashed', alpha=0.4, size=1.75)
       + gg.ylab('average regret per episode')
       + gg.geom_hline(gg.aes(yintercept=0.0), alpha=0)  # axis hack
      )
  return plotting.facet_sweep_plot(p, sweep_vars, True)


def plot_scale_by_group(df_in: pd.DataFrame,
                        group_var: Text,
                        sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average regret at end of learning by group_var."""
  df = df_in.copy()
  n_eps = 10000
  group_vars = (sweep_vars or []) + [group_var]
  plt_df = (df[df.episode == n_eps]
            .groupby(group_vars)['total_regret'].mean().reset_index())
  plt_df['average_regret'] = plt_df.total_regret / n_eps
  p = (gg.ggplot(plt_df)
       + gg.aes(group_var, 'average_regret', colour='average_regret < 0.5')
       + gg.geom_point(size=5, alpha=0.8)
       + gg.scale_x_log10(breaks=[1, 3, 10, 30, 100])
       + gg.geom_hline(
           gg.aes(yintercept=1.0), linetype='dashed', alpha=0.4, size=1.75)
       + gg.scale_colour_manual(values=['#d73027', '#313695'])
       + gg.ylab('average regret at 10k episodes')
       + gg.geom_hline(gg.aes(yintercept=0.0), alpha=0)  # axis hack
      )
  return plotting.facet_sweep_plot(p, sweep_vars)


def score(df: pd.DataFrame) -> float:
  return score_by_group(df, 'chain_length')


def plot_learning(df_in: pd.DataFrame,
                  sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  return plot_learning_by_group(df_in, 'chain_length', sweep_vars)


def plot_scale(df_in: pd.DataFrame,
               sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  return plot_scale_by_group(df_in, 'chain_length', sweep_vars)
