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
"""A hierarchical deep sea environment."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.experiments.hierarchy_sea import hierarchy_sea
from bsuite.experiments.hierarchy_sea_explore import sweep


def load(num_hierarchy, num_mapping):
  """Load a hierarchy sea experiment with the prescribed settings."""
  intra_ds_shaping = -1. / num_hierarchy
  inter_ds_shaping = -1. / num_hierarchy
  env = hierarchy_sea.HierarchySea(
      num_hierarchy=num_hierarchy,
      num_mapping=num_mapping,
      intra_ds_shaping=intra_ds_shaping,
      inter_ds_shaping=inter_ds_shaping,
  )
  env.bsuite_num_episodes = sweep.NUM_EPISODES
  return env
