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

from bsuite.utils import plotting

import numpy as np
import pandas as pd
import plotnine as gg

from typing import Sequence, Text

BASE_REGRET = 0.08
EPISODE = 1000
TAGS = ('credit_assignment',)
_HORIZONS = np.array([1, 3, 10, 30, 100])


def score(df: pd.DataFrame) -> float:
  """Output a single score for discounting_chain."""
  n_eps = np.minimum(df.episode.max(), EPISODE)
  ave_return = df.loc[df.episode == n_eps, 'total_return'].mean() / n_eps
  return 1. - 10. * (1.1 - ave_return)


def dc_preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
  """Preprocess discounting chain data for use with regret metrics."""
  df = df_in.copy()
  df['optimal_horizon'] = _HORIZONS[(df.seed % len(_HORIZONS)).astype(int)]
  df['total_regret'] = 1.1 * df.episode - df.total_return
  df['optimal_horizon'] = df.optimal_horizon.astype('category')
  return df


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average regret through time by optimal_horizon."""
  df = dc_preprocess(df_in=df)
  p = plotting.plot_regret_learning(
      df_in=df,
      group_col='optimal_horizon',
      sweep_vars=sweep_vars,
      max_episode=EPISODE
  )
  p += gg.geom_hline(gg.aes(yintercept=BASE_REGRET),
                     linetype='dashed', alpha=0.4, size=1.75)
  p += gg.coord_cartesian(ylim=(0, 0.1))
  return p


def plot_average(df: pd.DataFrame,
                 sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plots the average regret at 1k episodes by optimal_horizon."""
  df = dc_preprocess(df_in=df)
  p = plotting.plot_regret_average(
      df_in=df,
      group_col='optimal_horizon',
      episode=EPISODE,
      sweep_vars=sweep_vars
  )
  p += gg.geom_hline(gg.aes(yintercept=BASE_REGRET),
                     linetype='dashed', alpha=0.4, size=1.75)
  return p

