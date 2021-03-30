# pylint: disable=g-bad-file-header
# Copyright .... All Rights Reserved.
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

###TODO change to mdpp stuff below
"""Tests for bsuite.experiments.mdp_playground_p_noise."""

# Import all required packages

from absl.testing import absltest
from bsuite.environments import mdp_playground
from dm_env import test_utils

import numpy as np


class InterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    config = {}
    config["state_space_type"] = "discrete"
    config["action_space_type"] = "discrete"
    config["state_space_size"] = 8
    config["action_space_size"] = 8
    config["generate_random_mdp"] = True
    config["terminal_state_density"] = 0.25
    config["maximally_connected"] = True
    config["repeats_in_sequences"] = False
    config["reward_density"] = 0.25
    config["transition_noise"] = 0.25
    config["make_denser"] = False
    env = mdp_playground.DM_RLToyEnv(**config)
    return env

  def make_action_sequence(self):
    valid_actions = list(range(8))
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.choice(valid_actions)


if __name__ == '__main__':
  absltest.main()
