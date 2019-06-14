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
"""Analysis for Umbrella Distract."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.experiments.umbrella_length import analysis
import pandas as pd
import plotnine as gg

from typing import Sequence, Text


def score(df: pd.DataFrame) -> float:
  return analysis.score_by_group(df, 'n_distractor')


def plot_learning(df_in: pd.DataFrame,
                  sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  return analysis.plot_learning_by_group(df_in, 'n_distractor', sweep_vars)


def plot_scale(df_in: pd.DataFrame,
               sweep_vars: Sequence[Text] = None) -> gg.ggplot:
  return analysis.plot_scale_by_group(df_in, 'n_distractor', sweep_vars)
