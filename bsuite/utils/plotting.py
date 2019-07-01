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
"""Common plotting and analysis code.

This code is based around plotnine = python implementation of ggplot.
Typically, these plots will be imported and used within experiment analysis.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.utils import smoothers
import matplotlib.style as style
import numpy as np
import pandas as pd
import plotnine as gg

from typing import Sequence, Text

style.use('seaborn-poster')
style.use('ggplot')

FIVE_COLOURS = ['#313695', '#74add1', '#ffc832', '#f46d43', '#d73027']


def ave_regret_score(df: pd.DataFrame,
                     baseline_regret: float,
                     episode: int,
                     regret_column: Text = 'total_regret') -> float:
  """Score performance by average regret, normalized to [0,1] by baseline."""
  n_eps = np.minimum(df.episode.max(), episode)
  mean_regret = df.loc[df.episode == n_eps, regret_column].mean() / n_eps
  unclipped_score = (baseline_regret - mean_regret) / baseline_regret
  return np.clip(unclipped_score, 0, 1)


def facet_sweep_plot(base_plot: gg.ggplot,
                     sweep_vars: Sequence[Text] = None,
                     tall_plot: bool = False) -> gg.ggplot:
  """Add a facet_wrap to the plot based on sweep_vars."""
  df = base_plot.data.copy()

  if sweep_vars:
    # Work out what size the plot should be based on the hypers + add facet.
    n_hypers = df[sweep_vars].drop_duplicates().shape[0]
    base_plot += gg.facet_wrap(sweep_vars, labeller='label_both')
  else:
    n_hypers = 1

  if n_hypers == 1:
    fig_size = (10, 6)
  elif n_hypers == 2:
    fig_size = (16, 6)
  elif n_hypers == 4:
    fig_size = (16, 10)
  elif n_hypers <= 12:
    fig_size = (21, 5 * np.divide(n_hypers, 3) + 1)
  else:
    print('WARNING - comparing {} agents at once is more than recommended.'
          .format(n_hypers))
    fig_size = (21, 16)

  if tall_plot:
    fig_size = (fig_size[0], fig_size[1] * 1.4)

  return base_plot + gg.theme(figure_size=fig_size)


def plot_regret_learning(df_in: pd.DataFrame,
                         group_col: Text = None,
                         sweep_vars: Sequence[Text] = None,
                         regret_col: Text = 'total_regret',
                         max_episode: int = None) -> gg.ggplot:
  """Plots the average regret through time, grouped by group_var."""
  df = df_in.copy()
  df['average_regret'] = df[regret_col] / df.episode
  if group_col is None:
    p = _plot_regret_single(df)
  else:
    p = _plot_regret_group(df, group_col)
  p += gg.geom_hline(gg.aes(yintercept=0.0), alpha=0)  # axis hack
  p += gg.ylab('average regret per timestep')
  p += gg.coord_cartesian(xlim=(0, max_episode))
  return facet_sweep_plot(p, sweep_vars, tall_plot=True)


def _plot_regret_single(df: pd.DataFrame) -> gg.ggplot:
  """Plots the average regret through time for single variable."""
  p = (gg.ggplot(df)
       + gg.aes(x='episode', y='average_regret')
       + gg.geom_smooth(method=smoothers.mean, span=0.1, size=1.75, alpha=0.1,
                        colour='#313695', fill='#313695'))
  return p


def _plot_regret_group(df: pd.DataFrame, group_col: Text) -> gg.ggplot:
  """Plots the average regret through time when grouped."""
  group_name = group_col.replace('_', ' ')
  df[group_name] = df[group_col].astype('category')
  p = (gg.ggplot(df)
       + gg.aes(x='episode', y='average_regret',
                group=group_name, colour=group_name, fill=group_name)
       + gg.geom_smooth(method=smoothers.mean, span=0.1, size=1.75, alpha=0.1)
       + gg.scale_colour_manual(values=FIVE_COLOURS)
       + gg.scale_fill_manual(values=FIVE_COLOURS))
  return p


def plot_regret_group_nosmooth(df_in: pd.DataFrame,
                               group_col: Text,
                               sweep_vars: Sequence[Text] = None,
                               regret_col: Text = 'total_regret',
                               max_episode: int = None) -> gg.ggplot:
  """Plots the average regret through time without smoothing."""
  df = df_in.copy()
  df['average_regret'] = df[regret_col] / df.episode
  group_name = group_col.replace('_', ' ')
  df[group_name] = df[group_col]
  p = (gg.ggplot(df)
       + gg.aes(x='episode', y='average_regret',
                group=group_name, colour=group_name)
       + gg.geom_line(size=2, alpha=0.75)
       + gg.geom_hline(gg.aes(yintercept=0.0), alpha=0)  # axis hack
      )
  p += gg.coord_cartesian(xlim=(0, max_episode))
  return facet_sweep_plot(p, sweep_vars, tall_plot=True)


def _preprocess_ave_regret(df_in: pd.DataFrame,
                           group_col: Text,
                           episode: int,
                           sweep_vars: Sequence[Text] = None,
                           regret_col: Text = 'total_regret') -> gg.ggplot:
  """Preprocess the data at episode for average regret calculations."""
  df = df_in.copy()
  group_vars = (sweep_vars or []) + [group_col]
  plt_df = (df[df.episode == episode]
            .groupby(group_vars)[regret_col].mean().reset_index())
  group_name = group_col.replace('_', ' ')
  plt_df[group_name] = plt_df[group_col].astype('category')
  plt_df['average_regret'] = plt_df[regret_col] / episode
  return plt_df


def plot_regret_average(df_in: pd.DataFrame,
                        group_col: Text,
                        episode: int,
                        sweep_vars: Sequence[Text] = None,
                        regret_col: Text = 'total_regret') -> gg.ggplot:
  """Bar plot the average regret at end of learning."""
  df = _preprocess_ave_regret(df_in, group_col, episode, sweep_vars, regret_col)
  group_name = group_col.replace('_', ' ')
  p = (gg.ggplot(df)
       + gg.aes(x=group_name, y='average_regret', fill=group_name)
       + gg.geom_bar(stat='identity')
       + gg.scale_fill_manual(values=FIVE_COLOURS)
       + gg.ylab('average regret after {} episodes'.format(episode))
      )
  return facet_sweep_plot(p, sweep_vars)


def plot_regret_ave_scaling(df_in: pd.DataFrame,
                            group_col: Text,
                            episode: int,
                            regret_thresh: float,
                            sweep_vars: Sequence[Text] = None,
                            regret_col: Text = 'total_regret') -> gg.ggplot:
  """Point plot of average regret investigating scaling to threshold."""
  df = _preprocess_ave_regret(df_in, group_col, episode, sweep_vars, regret_col)
  group_name = group_col.replace('_', ' ')
  p = (gg.ggplot(df)
       + gg.aes(x=group_name, y='average_regret',
                colour='average_regret < {}'.format(regret_thresh))
       + gg.geom_point(size=5, alpha=0.8)
       + gg.scale_x_log10(breaks=[1, 3, 10, 30, 100])
       + gg.scale_colour_manual(values=['#d73027', '#313695'])
       + gg.ylab('average regret at {} episodes'.format(episode))
       + gg.geom_hline(gg.aes(yintercept=0.0), alpha=0)  # axis hack
      )
  return facet_sweep_plot(p, sweep_vars)
