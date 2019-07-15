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
"""Analysis for catch scale environments."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.experiments.catch_noise import analysis as catch_noise_analysis
from bsuite.experiments.catch_noise import sweep
import pandas as pd
import plotnine as gg

from typing import Sequence, Text

NUM_EPISODES = sweep.NUM_EPISODES
TAGS = ('scale',)
score = catch_noise_analysis.score


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  return catch_noise_analysis.plot_learning(df, sweep_vars, 'reward_scale')


def plot_average(df: pd.DataFrame,
                 sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  return catch_noise_analysis.plot_average(df, sweep_vars, 'reward_scale')
