# pylint: disable=g-bad-file-header
###TODO Copyright stuff
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
"""Sweep definition for experiments in the MDP Playground."""

import copy

NUM_EPISODES = 1000
#NUM_TIMESTEPS = 20000

# Need to have full config, including: S, A,; explicitly state all of them for backward compatibility.

config = {}
# config["seed"] = 0

config["state_space_type"] = "discrete"
config["action_space_type"] = "discrete"
config["state_space_size"] = 8
config["action_space_size"] = 8
config["delay"] = 0
config["sequence_length"] = 1
config["reward_scale"] = 1
config["reward_shift"] = 0
# config["reward_noise"] = lambda a: a.normal(0, 0.5)
# config["transition_noise"] = 0.1
#config["reward_density"] = 0.25
config["make_denser"] = False
config["terminal_state_density"] = 0.25
config["completely_connected"] = True
config["repeats_in_sequences"] = False
config["generate_random_mdp"] = True
# import logging
# config["log_level"] = logging.DEBUG


## sparse reward experiement settings
_SETTINGS = []
r_density = [0.17, 0.34, 0.5, 0.67, 0.84]
num_seeds = 4
for i in range(len(r_density)):
  for j in range(num_seeds):
    config_copy = copy.deepcopy(config)
    config_copy["reward_density"] = r_density[i]
    config_copy["seed"] = j
    _SETTINGS.append(config_copy)

SETTINGS = tuple(_SETTINGS) # delays, seeds for agents or envs?
TAGS = ('mdp_playground',)#, 'sparsity', 'basic', 'generalization')
