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
"""Analysis for deep sea stochastic.

We say that a deep sea episode is 'bad' when the agent takes a move 'left'
while it on the 'optimal' trajectory. However, for the stochastic case this
means that the agent can have few 'bad' trajectories just by luck of the
environment noise. To make sure that this is not by dumb luck, we use a more
stringent threshold and only once the agent has done at least 100 episodes.
"""
# Import all packages

from bsuite.experiments.deep_sea import analysis as deep_sea_analysis
from bsuite.experiments.deep_sea_stochastic import sweep

import numpy as np
import pandas as pd
import plotnine as gg

from typing import Text, Sequence

NUM_EPISODES = sweep.NUM_EPISODES
TAGS = ('exploration', 'noise')

score = deep_sea_analysis.score
plot_scaling = deep_sea_analysis.plot_scaling
plot_scaling_log = deep_sea_analysis.plot_scaling_log
plot_regret = deep_sea_analysis.plot_regret


def find_solution(df_in: pd.DataFrame,
                  sweep_vars: Sequence[Text] = None) -> pd.DataFrame:
  """Find first solution episode, with harsher thresh for stochastic domain."""
  df = df_in.copy()
  df = df[df.episode >= 100]
  return deep_sea_analysis.find_solution(df, sweep_vars, thresh=0.8)


def plot_seeds(df: pd.DataFrame,
               sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plot the returns through time individually by run."""
  return deep_sea_analysis.plot_seeds(
      df_in=df,
      sweep_vars=sweep_vars,
      yintercept=np.exp(-1)
  ) + gg.ylab('average episodic return (excluding additive noise)')
