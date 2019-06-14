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
"""Analysis for mountain_car."""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite import plotting
from bsuite.experiments.mountain_car import analysis
from bsuite.utils import smoothers

import numpy as np
import pandas as pd
import plotnine as gg

from typing import Sequence, Text

score = analysis.score


def plot_learning(df_in: pd.DataFrame,
                  sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average regret through time by reward_scale."""
  df = df_in.copy()
  df['average_regret'] = 1 - (df.total_return /(df.episode * df.reward_scale))
  df['reward_scale'] = df.reward_scale.astype('category')
  p = (gg.ggplot(df)
       + gg.aes('episode', 'average_regret', group='reward_scale',
                colour='reward_scale', fill='reward_scale')
       + gg.geom_smooth(method=smoothers.mean, span=0.1, size=1.75, alpha=0.1)
       + gg.scale_colour_manual(values=plotting.FIVE_COLOURS)
       + gg.scale_fill_manual(values=plotting.FIVE_COLOURS)
       + gg.scale_y_continuous(breaks=np.arange(0, 1.1, 0.1).tolist())
       + gg.theme(panel_grid_major_y=gg.element_line(size=2.5),
                  panel_grid_minor_y=gg.element_line(size=0),)
       + gg.geom_hline(
           gg.aes(yintercept=0.5), linetype='dashed', alpha=0.4, size=1.75)
       + gg.ylab('average regret per timestep')
       + gg.coord_cartesian(ylim=(0, 1))
      )
  return plotting.facet_sweep_plot(p, sweep_vars, tall_plot=True)


def plot_average(df_in: pd.DataFrame,
                 sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average regret through time by reward_scale."""
  df = df_in.copy()
  n_eps = 10000
  group_vars = (sweep_vars or []) + ['reward_scale']
  plt_df = (df[df.episode == n_eps]
            .groupby(group_vars)['total_return'].mean().reset_index())
  plt_df['average_arm'] = plt_df.total_return / (n_eps * plt_df.reward_scale)
  plt_df['average_regret'] = 1 - plt_df.average_arm
  plt_df['reward_scale'] = plt_df.reward_scale.astype('category')

  p = (gg.ggplot(plt_df)
       + gg.aes('reward_scale', 'average_regret', fill='reward_scale')
       + gg.geom_bar(stat='identity')
       + gg.scale_fill_manual(values=plotting.FIVE_COLOURS)
       + gg.scale_y_continuous(breaks=np.arange(0, 1.1, 0.1).tolist())
       + gg.theme(panel_grid_major_y=gg.element_line(size=2.5),
                  panel_grid_minor_y=gg.element_line(size=0),)
       + gg.geom_hline(
           gg.aes(yintercept=0.5), linetype='dashed', alpha=0.4, size=1.75)
       + gg.ylab('average regret after 10k episodes')
       + gg.coord_cartesian(ylim=(0, 1))
      )
  return plotting.facet_sweep_plot(p, sweep_vars)

