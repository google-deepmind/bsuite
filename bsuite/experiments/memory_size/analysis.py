# python3
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
"""Analysis for memory_len experiment."""

from typing import Sequence

from bsuite.experiments.memory_len import analysis as memory_len_analysis
from bsuite.experiments.memory_size import sweep
import pandas as pd
import plotnine as gg

NUM_EPISODES = sweep.NUM_EPISODES
TAGS = sweep.TAGS


def score(df: pd.DataFrame) -> float:
  return memory_len_analysis.score(df, group_col='num_bits')


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[str] = None) -> gg.ggplot:
  return memory_len_analysis.plot_learning(df, sweep_vars, 'num_bits')


def plot_scale(df: pd.DataFrame,
               sweep_vars: Sequence[str] = None) -> gg.ggplot:
  return memory_len_analysis.plot_scale(df, sweep_vars, 'num_bits')


def plot_seeds(df: pd.DataFrame,
               sweep_vars: Sequence[str] = None) -> gg.ggplot:
  return memory_len_analysis.plot_seeds(
      df_in=df[df.episode > 100],
      sweep_vars=sweep_vars,
      colour_var='num_bits',
  )
