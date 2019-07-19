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
"""Analysis for bandit_noise."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.experiments.bandit import analysis as bandit_analysis
from bsuite.experiments.bandit import sweep
from bsuite.utils import plotting
import numpy as np
import pandas as pd
import plotnine as gg
from typing import Text, Sequence

NUM_EPISODES = sweep.NUM_EPISODES
TAGS = ('noise',)


def score(df: pd.DataFrame, scaling_var='noise_scale') -> float:
  """Output a single score for experiment = mean - std over scaling_var."""
  return plotting.score_by_scaling(
      df=df,
      score_fn=bandit_analysis.score,
      scaling_var=scaling_var,
  )


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[Text] = None,
                  group_col: Text = 'noise_scale') -> gg.ggplot:
  """Plots the average regret through time."""
  p = plotting.plot_regret_learning(
      df_in=df, group_col=group_col, sweep_vars=sweep_vars,
      max_episode=sweep.NUM_EPISODES)
  return bandit_analysis.bandit_learning_format(p)


def plot_average(df: pd.DataFrame,
                 sweep_vars: Sequence[Text] = None,
                 group_col: Text = 'noise_scale') -> gg.ggplot:
  """Plots the average regret through time by noise_scale."""
  p = plotting.plot_regret_average(
      df_in=df,
      group_col=group_col,
      episode=sweep.NUM_EPISODES,
      sweep_vars=sweep_vars
  )
  p += gg.scale_y_continuous(breaks=np.arange(0, 1.1, 0.1).tolist())
  p += gg.theme(panel_grid_major_y=gg.element_line(size=2.5),
                panel_grid_minor_y=gg.element_line(size=0),)
  p += gg.geom_hline(gg.aes(yintercept=bandit_analysis.BASE_REGRET),
                     linetype='dashed', alpha=0.4, size=1.75)
  p += gg.coord_cartesian(ylim=(0, 1))
  return p
