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
"""Analysis for num_hierarchy."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.experiments.hierarchy_sea import sweep
from bsuite.utils import plotting
import numpy as np
import pandas as pd
import plotnine as gg

from typing import Text, Sequence

TAGS = ('hierarchy',)
PERFECTION_THRESH = 0.25


def hs_preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
  """Preprocess data for memory environments."""
  df = df_in.copy()
  df['perfection_regret'] = df.episode - df.total_perfect
  return df


def score(df: pd.DataFrame) -> float:
  """Output a single score for num_hierarchy."""
  return_list = []  # Loop to handle partially-finished runs.
  for _, sub_df in df.groupby('num_hierarchy'):
    max_eps = np.minimum(sub_df.episode.max(), sweep.NUM_EPISODES)
    ave_perfection = (
        sub_df.loc[sub_df.episode == max_eps, 'total_perfect'].mean() / max_eps)
    return_list.append(ave_perfection)
  return np.mean(np.array(return_list) > PERFECTION_THRESH)


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average return through time by num_hierarchy."""
  df = hs_preprocess(df_in=df)
  p = plotting.plot_regret_group_nosmooth(
      df_in=df,
      group_col='num_hierarchy',
      sweep_vars=sweep_vars,
      regret_col='perfection_regret',
      max_episode=sweep.NUM_EPISODES,
  )
  return p + gg.ylab('average % of imperfect episodes')


def plot_scale(df: pd.DataFrame,
               sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average return through time by num_hierarchygth."""
  df = hs_preprocess(df_in=df)
  p = plotting.plot_regret_ave_scaling(
      df_in=df,
      group_col='num_hierarchy',
      episode=sweep.NUM_EPISODES,
      regret_thresh=0.5,
      sweep_vars=sweep_vars,
      regret_col='perfection_regret'
  )
  p += gg.ylab(
      '% of imperfect episodes after {} episodes'.format(sweep.NUM_EPISODES))
  return p
