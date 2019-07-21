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
"""Analysis for cartpole_scale environments."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.experiments.cartpole import analysis as cartpole_analysis
from bsuite.experiments.cartpole_noise import analysis as cartpole_noise_analysis
from bsuite.experiments.cartpole_scale import sweep
import pandas as pd
import plotnine as gg

from typing import Sequence, Text

NUM_EPISODES = sweep.NUM_EPISODES
TAGS = ('scale', 'generalization')


def score(df: pd.DataFrame) -> float:
  return cartpole_noise_analysis.score(df, scaling_var='reward_scale')


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  return cartpole_noise_analysis.plot_learning(df, sweep_vars, 'reward_scale')


def plot_average(df: pd.DataFrame,
                 sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  return cartpole_noise_analysis.plot_average(df, sweep_vars, 'reward_scale')


def plot_seeds(df: pd.DataFrame,
               sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  """Plot the performance by individual work unit."""
  return cartpole_analysis.plot_seeds(
      df_in=df,
      sweep_vars=sweep_vars,
      colour_var='reward_scale'
  ) + gg.ylab('average episodic return (after rescaling)')
