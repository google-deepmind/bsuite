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
"""Common plotting and analysis code."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

import matplotlib.style as style
import numpy as np
import plotnine as gg

from typing import Sequence, Text

style.use('seaborn-poster')
style.use('ggplot')

FIVE_COLOURS = ['#313695', '#74add1', '#ffc832', '#f46d43', '#d73027']


def facet_sweep_plot(base_plot: gg.ggplot,
                     sweep_vars: Sequence[Text] = None,
                     tall_plot: bool = False) -> gg.ggplot:
  """Add a facet to the plot based on sweep_vars."""
  df = base_plot.data.copy()

  if sweep_vars:
    # Add a facet
    base_plot += gg.facet_wrap(sweep_vars, labeller='label_both')

    # Work out what size the plot should be based on the hypers.
    all_hypers = df.groupby(sweep_vars).size().reset_index().drop(0, axis=1)
    n_hypers = all_hypers.shape[0]
    print(n_hypers)
  else:
    n_hypers = 1

  if n_hypers == 1:
    fig_size = (10, 6)
  elif n_hypers == 2:
    fig_size = (16, 6)
  elif n_hypers == 4:
    fig_size = (16, 10)
  elif n_hypers <= 12:
    fig_size = (21, 5 * np.divide(n_hypers, 3) + 1)
  else:
    print('WARNING - comparing {} agents at once is more than recommended.'
          .format(n_hypers))
    fig_size = (21, 16)

  if tall_plot:
    fig_size = (fig_size[0], fig_size[1] * 1.4)

  return base_plot + gg.theme(figure_size=fig_size)
