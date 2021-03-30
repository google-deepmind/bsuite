# python3
# pylint: disable=g-bad-file-header
# Copyright 2020 #TODO ... All Rights Reserved.
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
"""Analysis for MDP Playground."""

###TODO change to mdpp stuff below
from typing import Sequence

from bsuite.experiments.mdp_playground import analysis as mdp_playground_analysis
from bsuite.experiments.mdp_playground_p_noise import sweep
from bsuite.utils import plotting
import numpy as np
import pandas as pd
import plotnine as gg

NUM_EPISODES = sweep.NUM_EPISODES
BASE_REGRET = 100
GOOD_EPISODE = 50
TAGS = sweep.TAGS



def score(df: pd.DataFrame, scaling_var='transition_noise') -> float:
  """Output a single score for experiment = mean - std over scaling_var."""
  return plotting.score_by_scaling(
      df=df,
      score_fn=mdp_playground_analysis.score,
      scaling_var=scaling_var,
  )


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[str] = None,
                  group_col: str = 'transition_noise') -> gg.ggplot:
  """Plots the average regret through time."""
  df = mdp_playground_analysis.mdpp_preprocess(df)
  p = plotting.plot_regret_learning(
      df_in=df, group_col=group_col, sweep_vars=sweep_vars,
      max_episode=sweep.NUM_EPISODES)
  p += gg.geom_hline(gg.aes(yintercept=BASE_REGRET),
                     linetype='dashed', alpha=0.4, size=1.75)
  return p


def plot_average(df: pd.DataFrame,
                 sweep_vars: Sequence[str] = None,
                 group_col: str = 'transition_noise') -> gg.ggplot:
  """Plots the average regret through time by transition_noise."""
  df = mdp_playground_analysis.mdpp_preprocess(df)
  p = plotting.plot_regret_average(
      df_in=df,
      group_col=group_col,
      episode=sweep.NUM_EPISODES,
      sweep_vars=sweep_vars
  )
  p += gg.geom_hline(gg.aes(yintercept=BASE_REGRET),
                     linetype='dashed', alpha=0.4, size=1.75)
  return p


def plot_seeds(df: pd.DataFrame,
               sweep_vars: Sequence[str] = None) -> gg.ggplot:
  """Plot the performance by individual work unit."""
  return mdp_playground_analysis.plot_seeds(
      df_in=df,
      sweep_vars=sweep_vars,
      colour_var='transition_noise'
  ) + gg.ylab('average episodic return (removing noise)')
