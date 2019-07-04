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
"""Analysis for cartpole swingup."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.utils import plotting
import numpy as np
import pandas as pd
import plotnine as gg

from typing import Text, Sequence

EPISODE = 1000
EP_MAX = 1000
REWARD_THRESH = 10
TAGS = ('exploration', 'generalization')


def score(df: pd.DataFrame) -> float:
  """Percentage of heights for which the agent receives positive return."""
  return_list = []  # Loop to handle partially-finished runs.
  for _, sub_df in df.groupby('height_threshold'):
    max_eps = np.minimum(sub_df.episode.max(), EPISODE)
    ave_return = (
        sub_df.loc[sub_df.episode == max_eps, 'total_return'].mean() / max_eps)
    return_list.append(ave_return)
  return np.mean(np.array(return_list) > REWARD_THRESH)


def cp_swingup_preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
  """Preprocess data for cartpole swingup."""
  df = df_in.copy()
  df['perfection_regret'] = df.episode * EP_MAX - df.total_return
  return df


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average return through time by cartpole swingup."""
  df = cp_swingup_preprocess(df_in=df)
  p = plotting.plot_regret_group_nosmooth(
      df_in=df,
      group_col='height_threshold',
      sweep_vars=sweep_vars,
      regret_col='perfection_regret',
      max_episode=EPISODE,
  )
  return p


def plot_scale(df: pd.DataFrame,
               sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average return through time by cartpole swingup."""
  df = cp_swingup_preprocess(df_in=df)
  p = plotting.plot_regret_ave_scaling(
      df_in=df,
      group_col='height_threshold',
      episode=EPISODE,
      regret_thresh=0.5,
      sweep_vars=sweep_vars,
      regret_col='perfection_regret'
  )
  return p + gg.scale_x_discrete(breaks=[0, 0.25, 0.5, 0.75, 1.0])
