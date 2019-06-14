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
"""Analysis for memory_len."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite import plotting
import numpy as np
import pandas as pd
import plotnine as gg

from typing import Text, Sequence


def score(df: pd.DataFrame) -> float:
  """Output a single score for memory_len."""
  n_eps = 10000
  return_list = []  # Loop to handle partially-finished runs.
  for _, sub_df in df.groupby('memory_length'):
    max_eps = np.minimum(sub_df.episode.max(), n_eps)
    ave_return = (
        sub_df.loc[sub_df.episode == max_eps, 'total_return'].mean() / max_eps)
    return_list.append(ave_return)
  return np.mean(np.array(return_list) > 0.5)


def plot_learning(df_in: pd.DataFrame,
                  sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average return through time by memory_length."""
  df = df_in.copy()
  df['average_regret'] = 1 - df.total_return / df.episode
  p = (gg.ggplot(df[df.episode >= 100])
       + gg.aes('episode', 'average_regret', group='factor(memory_length)',
                fill='memory_length', colour='memory_length')
       + gg.geom_line(size=2, alpha=0.75)
       + gg.geom_hline(
           gg.aes(yintercept=1.), linetype='dashed', alpha=0.4, size=1.75)
       + gg.ylab('average regret per episode')
      )
  return plotting.facet_sweep_plot(p, sweep_vars, tall_plot=True)


def plot_scale(df_in: pd.DataFrame,
               sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average return through time by memory_length."""
  df = df_in.copy()
  n_eps = 10000
  group_vars = (sweep_vars or []) + ['memory_length']
  plt_df = (df[df.episode == n_eps]
            .groupby(group_vars)['total_return'].mean().reset_index())
  plt_df['average_regret'] = 1 - plt_df.total_return / n_eps
  p = (gg.ggplot(plt_df)
       + gg.aes('memory_length', 'average_regret',
                colour='average_regret < 0.5')
       + gg.geom_point(size=5, alpha=0.8)
       + gg.scale_x_log10(breaks=[1, 3, 10, 30, 100])
       + gg.geom_hline(
           gg.aes(yintercept=1.), linetype='dashed', alpha=0.4, size=1.75)
       + gg.scale_colour_manual(values=['#d73027', '#313695'])
       + gg.ylab('average regret at 10k episodes')
      )
  return plotting.facet_sweep_plot(p, sweep_vars)
