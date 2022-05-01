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

from bsuite.experiments.mdp_playground import sweep
from bsuite.utils import plotting
import numpy as np
import pandas as pd
import plotnine as gg

NUM_EPISODES = sweep.NUM_EPISODES
BASE_REGRET = 100
GOOD_EPISODE = 50
TAGS = sweep.TAGS

def score(df: pd.DataFrame) -> float:
  """Output a score for MDP Playground."""
  df = mdpp_preprocess(df_in=df)
  regret_score = plotting.ave_regret_score(
      df, baseline_regret=BASE_REGRET, episode=NUM_EPISODES)

  norm_score = 1.0 * regret_score # 2.5 was heuristically chosen value to get Sonnet DQN to score approx. 0.75, so that better algorithms like Rainbow can get score close to 1. With a bigger NN this would mean an unclipped score of 1.1 for Sonnet DQN, which is fair I think. However, a2c_rnn even reached 2.0 on this scale. DQN may be not performing as well because its epsilon is not annealed to 0.
  # print("unclipped score:", norm_score)
  norm_score = np.clip(norm_score, 0, 1)
  return norm_score

def mdpp_preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
  """Preprocess MDP Playground data for use with regret metrics."""
  df = df_in.copy()
  df = df[df.episode <= NUM_EPISODES]
  df['total_regret'] = (BASE_REGRET * df.episode) - df.raw_return
  return df

def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[str] = None) -> gg.ggplot:
  """Simple learning curves for MDP Playground."""
  df = mdpp_preprocess(df)
  p = plotting.plot_regret_learning(
      df, sweep_vars=sweep_vars, max_episode=NUM_EPISODES)
  p += gg.geom_hline(gg.aes(yintercept=BASE_REGRET),
                     linetype='dashed', alpha=0.4, size=1.75)
  return p

def plot_seeds(df_in: pd.DataFrame,
               sweep_vars: Sequence[str] = None,
               colour_var: str = None) -> gg.ggplot:
  """Plot the returns through time individually by run."""
  df = df_in.copy()
  df['average_return'] = df.raw_return.diff() / df.episode.diff()
  p = plotting.plot_individual_returns(
      df_in=df,
      max_episode=NUM_EPISODES,
      return_column='average_return',
      colour_var=colour_var,
      yintercept=BASE_REGRET,
      sweep_vars=sweep_vars,
  )
  return p + gg.ylab('average episodic return')
