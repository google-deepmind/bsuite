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
"""Analysis for num_hierarchy."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.experiments.hierarchy_sea import analysis as hierarchy_sea_analysis

EPISODE = hierarchy_sea_analysis.EPISODE
TAGS = ('hierarchy', 'exploration',)
score = hierarchy_sea_analysis.score
plot_learning = hierarchy_sea_analysis.plot_learning
plot_scale = hierarchy_sea_analysis.plot_scale
