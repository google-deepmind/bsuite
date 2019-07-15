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

from bsuite.experiments.deep_sea import deep_sea
from bsuite.experiments.hierarchy_sea import sweep
from bsuite.utils import auto_reset_environment

import dm_env
from dm_env import specs
import numpy as np


class HierarchySea(auto_reset_environment.Base):
  """A hierarchical deep sea environment.

  Observation is [deep_sea_size, deep_sea_size, 2] where trailing dimension is
  a one-hot representation of the action mask. You must solve num_hierarchy
  sequential deep_sea problems, each of which has a different action_mapping
  chosen from num_mappings. If you make it through all num_hierarchy stages
  you get a complete_bonus.
  """

  def __init__(self,
               num_hierarchy: int,
               num_mapping: int,
               deep_sea_size: int = 20,
               complete_bonus: float = 10.,
               intra_ds_shaping: float = 0.1,
               inter_ds_shaping: float = 1.,
               seed: int = None):
    """Hierarchical deep sea environment."""
    super(HierarchySea, self).__init__()
    self._deep_sea_size = deep_sea_size
    self._num_hierarchy = num_hierarchy
    self._complete_bonus = complete_bonus
    self._intra_ds_shaping = intra_ds_shaping
    self._inter_ds_shaping = inter_ds_shaping

    # Make the action mappings / state representations
    self._rng = np.random.RandomState(seed)
    mappings_shape = [num_mapping, deep_sea_size, deep_sea_size]
    self._action_mappings = self._rng.binomial(1, 0.5, mappings_shape)

    # Make the hierarchical representation
    hierarchy_shape = [num_hierarchy, deep_sea_size]
    self._hierarchy_repr = self._rng.binomial(1, 0.5, hierarchy_shape)

    # And now which hierarchical levels get which mappings
    self._hierarchy_mappings = self._rng.randint(0, num_mapping, num_hierarchy)

    # Make the underlying env and counter for stages
    self._deep_sea = deep_sea.DeepSea(
        size=deep_sea_size,
        unscaled_move_cost=-self._intra_ds_shaping,
        seed=seed,
    )
    self._hierarchy_stage = 0

    # Internal counters
    self._ds_solve_thresh = 0.5  # if ds_reward > ds_solve_thresh -> solved DS.
    self._total_stages = 0  # total number of deep sea stages completed.
    self._total_perfect = 0  # total number of 'perfect' episodes = completed.

    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def _get_obs(self, deep_sea_obs):
    hierarchy_obs = np.expand_dims(
        self._hierarchy_repr[self._hierarchy_stage], 1)
    return np.concatenate(
        [deep_sea_obs, hierarchy_obs], axis=1).astype(np.float32)

  def _override_deep_sea_action_mapping_from_hierarchy(self):
    """Overrides the underlying deep sea with action mapping based on stage."""
    mapping_idx = self._hierarchy_mappings[self._hierarchy_stage]
    self._deep_sea._action_mapping = self._action_mappings[mapping_idx]  # pylint:disable=protected-access

  def _step(self, action):
    ds_step = self._deep_sea.step(action)
    ds_reward = ds_step.reward
    obs = self._get_obs(ds_step.observation)

    if not ds_step.last():
      # Step deep sea as normal
      return dm_env.transition(reward=ds_reward, observation=obs)

    elif (self._hierarchy_stage < self._num_hierarchy - 1
          and ds_reward > self._ds_solve_thresh):
      # You have succeeded at this stage of deep sea -> go to the next hierarchy
      self._hierarchy_stage += 1
      self._total_stages += 1
      self._override_deep_sea_action_mapping_from_hierarchy()
      reset_step = self._deep_sea.reset()
      obs = self._get_obs(reset_step.observation)
      # Compute a shaping at the end of each ds_scale
      reward = ds_reward * self._inter_ds_shaping
      return dm_env.transition(reward=reward, observation=obs)

    else:
      # This is the end of the hierarchy, give a bonus if you solve it.
      if ds_reward > self._ds_solve_thresh:
        ds_reward += self._complete_bonus
        self._total_perfect += 1
      return dm_env.termination(reward=ds_reward, observation=obs)

  def _reset(self):
    self._hierarchy_stage = 0
    self._override_deep_sea_action_mapping_from_hierarchy()
    ds_step = self._deep_sea.reset()
    obs = self._get_obs(ds_step.observation)
    return dm_env.restart(observation=obs)

  def observation_spec(self):
    obs_shape = (self._deep_sea_size, self._deep_sea_size + 1)
    return specs.Array(shape=obs_shape, dtype=np.float32)

  def action_spec(self):
    return specs.DiscreteArray(2, name='action')

  def bsuite_info(self):
    return dict(total_stages=self._total_stages,
                total_perfect=self._total_perfect)

