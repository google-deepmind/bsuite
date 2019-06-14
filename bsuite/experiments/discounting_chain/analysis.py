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
"""Analysis for discounting_chain."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite import plotting
from bsuite.utils import smoothers
import numpy as np
import pandas as pd
import plotnine as gg

from typing import Sequence, Text


def score(df: pd.DataFrame) -> float:
  """Output a single score for discounting_chain."""
  n_eps = np.minimum(df.episode.max(), 10000)
  ave_return = df.loc[df.episode == n_eps, 'total_return'].mean() / n_eps
  return 1 - 10 * (1.1 - ave_return)

_HORIZONS = np.array([1, 3, 10, 30, 100])


def plot_average(df_in: pd.DataFrame,
                 sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average regret at 1k episodes by optimal_horizon."""
  df = df_in.copy()
  n_eps = np.minimum(df.episode.max(), 10000)
  df['optimal_horizon'] = _HORIZONS[(df.seed % len(_HORIZONS)).astype(int)]
  group_vars = (sweep_vars or []) + ['optimal_horizon']
  plt_df = (df[df.episode == n_eps]
            .groupby(group_vars)['total_return'].mean().reset_index())
  plt_df['average_regret'] = 1.1 - plt_df.total_return / n_eps
  plt_df['optimal_horizon'] = plt_df.optimal_horizon.astype('category')
  p = (gg.ggplot(plt_df)
       + gg.aes('optimal_horizon', 'average_regret', fill='optimal_horizon')
       + gg.geom_bar(stat='identity')
       + gg.geom_hline(
           gg.aes(yintercept=0.08), linetype='dashed', alpha=0.4, size=1.75)
       + gg.scale_fill_manual(values=plotting.FIVE_COLOURS)
       + gg.ylab('average regret after 10k episodes')
       + gg.coord_cartesian(ylim=(0, 0.1))
      )
  return plotting.facet_sweep_plot(p, sweep_vars)


def plot_learning(df_in: pd.DataFrame,
                  sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average regret through time by optimal_horizon."""
  df = df_in.copy()
  df['optimal_horizon'] = _HORIZONS[(df.seed % len(_HORIZONS)).astype(int)]
  df['average_regret'] = 1.1 - df.total_return / df.episode
  df['optimal_horizon'] = df.optimal_horizon.astype('category')

  p = (gg.ggplot(df)
       + gg.aes('episode', 'average_regret', group='optimal_horizon',
                fill='optimal_horizon', colour='optimal_horizon')
       + gg.geom_smooth(method=smoothers.mean, span=0.1, size=1.75, alpha=0.1)
       + gg.scale_colour_manual(values=plotting.FIVE_COLOURS)
       + gg.scale_fill_manual(values=plotting.FIVE_COLOURS)
       + gg.geom_hline(
           gg.aes(yintercept=0.08), linetype='dashed', alpha=0.4, size=1.75)
       + gg.ylab('average regret per episode')
       + gg.coord_cartesian(ylim=(0, 0.1))
      )
  return plotting.facet_sweep_plot(p, sweep_vars, tall_plot=True)
